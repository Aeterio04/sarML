"""
train_agent1_fixed.py  —  Agent 1: Fixed Model Training
=========================================================
ROOT CAUSE FIX: Per-typology MI feature selection

THE BUG (confirmed):
  Old code ran mutual_info_classif(X, y_sar_worthy) — one global MI pass
  against the sar_worthy binary label across all 7000 rows. Because
  rapid_movement is the largest typology (977 cases = 27.9% of SAR),
  its burst/velocity features dominate the MI scores. Features critical
  for structuring, TBML, shell, etc. score near-zero globally and get
  dropped. The model ends up with 7 features — all burst-derived —
  and is blind to 5 out of 6 typologies.

THE FIX:
  Run MI separately for EACH label (sar_worthy + 6 typology columns).
  Take the UNION of top-N features per label. Every typology now
  contributes its own discriminative features to the shared feature set.
  No typology gets crowded out by another's stronger signal.

  Old selected features (9):  all burst/velocity/exit — useless for structuring/TBML
  New selected features (~27): covers ALL typologies with their key signals

ARCHITECTURE (unchanged from original):
  MultiOutputClassifier wrapping XGBoost — one binary XGBoost per label.
  StandardScaler on train set only.
  70/15/15 train/val/test split, stratified on sar_worthy.

Output: model_fixed.pkl  (drop-in replacement for model.pkl)
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (classification_report, hamming_loss,
                              f1_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
import xgboost as xgb

np.random.seed(42)
os.makedirs('agent1_fixed', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH   = 'data_engineered.csv'
OUTPUT_PKL  = 'finalmodel.pkl'
OUTPUT_JSON = 'agent1_fixed/metrics_fixed.json'

LABEL_COLS = [
    'sar_worthy',
    'typology_structuring',
    'typology_rapid_movement',
    'typology_funnel_account',
    'typology_trade_based',
    'typology_shell_company',
    'typology_round_tripping',
]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("AGENT 1 FIXED TRAINING")
print("=" * 65)

df = pd.read_csv(DATA_PATH)
print(f"\nLoaded: {len(df):,} rows x {df.shape[1]} cols")
print(f"SAR rate: {df['sar_worthy'].mean():.3f}")

# Validate all label cols exist
missing = [c for c in LABEL_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Missing label columns: {missing}")

# All non-label columns are candidate features
ALL_FEAT_COLS = [c for c in df.columns if c not in LABEL_COLS]
print(f"Candidate features: {len(ALL_FEAT_COLS)}")

print("\nTypology distribution:")
for col in LABEL_COLS:
    n = df[col].sum()
    print(f"  {col:<40}: {n:>4}  ({n/len(df)*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: PER-TYPOLOGY MI FEATURE SELECTION  ← THE FIX
#
# Why this is correct:
#   Each typology has different discriminative features. Global MI against
#   sar_worthy floods the ranking with rapid_movement signals (the biggest
#   typology). Per-label MI gives each typology a voice, then we union
#   the results so the final feature set covers ALL typologies.
#
# TOP_N = 8 per label is conservative. With 7 labels that gives a union
# of ~20-27 features (less than 8×7 due to overlap), which is plenty.
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print("STEP 2: Per-typology MI feature selection (top 8 per label)")
print("=" * 65)

TOP_N = 8
X_all = df[ALL_FEAT_COLS].fillna(0).values
selected_union = set()
mi_records = {}

for label in LABEL_COLS:
    y = df[label].values
    mi_scores = mutual_info_classif(X_all, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=ALL_FEAT_COLS).sort_values(ascending=False)
    top_feats  = mi_series.head(TOP_N).index.tolist()
    selected_union.update(top_feats)
    mi_records[label] = mi_series.to_dict()
    print(f"\n  {label}  (top {TOP_N}):")
    for f in top_feats:
        print(f"    {f:<40} {mi_series[f]:.4f}")

SELECTED_FEATURES = sorted(selected_union)
print(f"\n  ─────────────────────────────────────────────────────")
print(f"  Union: {len(SELECTED_FEATURES)} features selected")
print(f"  Old approach selected: 9 features (all burst-derived)")
print(f"  New approach selected: {len(SELECTED_FEATURES)} features (covers all typologies)")
print(f"\n  Full selected list:")
for f in SELECTED_FEATURES:
    print(f"    {f}")

# Save MI scores for audit
mi_df_out = pd.DataFrame(mi_records).fillna(0)
mi_df_out.to_csv('agent1_fixed/mi_scores_per_typology.csv')
print(f"\n  MI scores saved: agent1_fixed/mi_scores_per_typology.csv")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: TRAIN / VAL / TEST SPLIT  (70 / 15 / 15, stratified on sar_worthy)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print("STEP 3: Train/val/test split  (70/15/15, stratified)")
print("=" * 65)

X = df[SELECTED_FEATURES].fillna(0).values
y = df[LABEL_COLS].values

X_tv, X_test, y_tv, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y[:, 0]
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=0.15 / 0.85, random_state=42, stratify=y_tv[:, 0]
)

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
print(f"  Features: {len(SELECTED_FEATURES)}")
for i, col in enumerate(LABEL_COLS):
    print(f"  {col:<40}: train={y_train[:,i].sum()}  val={y_val[:,i].sum()}  test={y_test[:,i].sum()}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: TRAIN MultiOutputClassifier (one XGBoost per label)
#
# XGBoost params are moderate — not over-tuned, which is appropriate
# because the training data is synthetic. Over-tuning on synthetic data
# causes brittleness on real inference inputs.
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print("STEP 4: Training MultiOutputClassifier (XGBoost per label)")
print("=" * 65)

base_clf = xgb.XGBClassifier(
    n_estimators      = 400,
    max_depth         = 5,
    learning_rate     = 0.05,
    subsample         = 0.80,
    colsample_bytree  = 0.80,
    min_child_weight  = 3,
    gamma             = 0.1,
    reg_alpha         = 0.1,
    reg_lambda        = 1.5,
    eval_metric       = 'logloss',
    random_state      = 42,
    verbosity         = 0,
)

model = MultiOutputClassifier(base_clf, n_jobs=-1)
print("  Fitting... (this takes ~60 seconds)")
model.fit(X_train_s, y_train)
print("  Done.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: EVALUATE
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print("STEP 5: Evaluation on test set")
print("=" * 65)

y_pred  = model.predict(X_test_s)
y_proba = np.array([
    est.predict_proba(X_test_s)[:, 1] for est in model.estimators_
]).T  # shape: (n_test, n_labels)

print(f"\n  {'Label':<40} {'F1':>6}  {'AUC':>6}  {'Prec':>6}  {'Rec':>6}")
print(f"  {'-'*65}")

per_label_metrics = {}
for i, col in enumerate(LABEL_COLS):
    f1   = f1_score(y_test[:, i], y_pred[:, i], zero_division=0)
    prec = float(np.mean(y_pred[:, i][y_test[:, i] == 1])) if y_test[:, i].sum() > 0 else 0
    rec  = float(y_pred[:, i][y_test[:, i] == 1].mean()) if y_test[:, i].sum() > 0 else 0
    try:
        auc = roc_auc_score(y_test[:, i], y_proba[:, i])
    except Exception:
        auc = 0.5
    per_label_metrics[col] = {'f1': round(f1, 4), 'auc': round(auc, 4)}
    flag = '  ← TARGET MET' if f1 >= 0.75 else '  ← NEEDS REVIEW'
    print(f"  {col:<40} {f1:>6.3f}  {auc:>6.3f}  {flag}")

hl       = hamming_loss(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

print(f"\n  Hamming Loss : {hl:.4f}  (lower is better, target < 0.15)")
print(f"  Macro F1     : {macro_f1:.4f}  (target > 0.70)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: QUICK SANITY CHECK — simulate inference for each typology
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print("STEP 6: Per-typology inference sanity check")
print("=" * 65)
print("  (using mean feature values of known positive cases per typology)")

for i, col in enumerate(LABEL_COLS[1:], start=1):
    positives = df[df[col] == 1][SELECTED_FEATURES].fillna(0)
    if len(positives) == 0:
        continue
    mean_row = positives.mean().values.reshape(1, -1)
    mean_row_s = scaler.transform(mean_row)
    probs = np.array([est.predict_proba(mean_row_s)[0, 1] for est in model.estimators_])
    sar_conf = probs[0]
    typ_conf = probs[i]
    result   = "SAR=TRUE ✓" if sar_conf >= 0.5 else "SAR=FALSE ✗"
    print(f"\n  {col}")
    print(f"    sar_worthy confidence : {sar_conf:.4f}  →  {result}")
    print(f"    typology confidence   : {typ_conf:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: SAVE
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print("STEP 7: Saving artefacts")
print("=" * 65)

artifact = {
    'model':          model,
    'scaler':         scaler,
    'feature_cols':   SELECTED_FEATURES,
    'label_cols':     LABEL_COLS,
    'mi_records':     mi_records,
    'top_n_per_label': TOP_N,
    'selection_method': 'per_label_MI_union',
    'note': (
        'Feature selection runs MI separately per label (sar_worthy + 6 typologies). '
        'Union of top-8 features per label is used. This ensures every typology '
        'contributes its discriminative features to the shared model feature set. '
        'The old approach (global MI vs sar_worthy only) was blind to structuring, '
        'TBML, shell company and round tripping typologies.'
    ),
}
joblib.dump(artifact, OUTPUT_PKL)
print(f"  Saved: {OUTPUT_PKL}  ({os.path.getsize(OUTPUT_PKL)//1024} KB)")

metrics_out = {
    'feature_selection': {
        'method':               'per_label_MI_union',
        'top_n_per_label':      TOP_N,
        'labels_used':          LABEL_COLS,
        'n_features_selected':  len(SELECTED_FEATURES),
        'selected_features':    SELECTED_FEATURES,
        'old_method':           'global_MI_vs_sar_worthy_only (BROKEN)',
        'old_n_features':       9,
        'old_features':         ['burst_score','burst_x_exit','burst_per_age',
                                 'burst_tier','fund_exit_ratio','fund_exit_tier',
                                 'time_to_first_outbound_minutes',
                                 'time_to_first_outbound_minutes_log',
                                 'hr_country_x_exit'],
    },
    'model_performance': {
        'hamming_loss':   round(hl, 4),
        'macro_f1':       round(macro_f1, 4),
        'per_label':      per_label_metrics,
    },
    'splits': {
        'train': len(X_train),
        'val':   len(X_val),
        'test':  len(X_test),
    },
}
with open(OUTPUT_JSON, 'w') as f:
    json.dump(metrics_out, f, indent=2)
print(f"  Saved: {OUTPUT_JSON}")

print(f"\n{'=' * 65}")
print("DONE")
print(f"  Macro F1    : {macro_f1:.4f}")
print(f"  Hamming Loss: {hl:.4f}")
print(f"  Features    : {len(SELECTED_FEATURES)} (was 9, all burst-derived)")
print(f"  Drop-in replacement: copy finalmodel.pkl → model/model.pkl")
print("=" * 65)
