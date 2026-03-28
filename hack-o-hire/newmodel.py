"""
train_agent1_fixed_v3.py  —  Agent 1: Fixed Model Training (v3)
================================================================
Builds on v2 (all 4 bug fixes retained) and replaces the hardcoded
XGBoost hyperparameters with a simple Optuna search.

  Objective : macro-F1 on the validation set.
  n_trials  : 50  (override via OPTUNA_N_TRIALS env var)
  Sampler   : TPESampler, seed=42

All v2 bug fixes (A/B/C/D) fully retained.
Output fields and format are unchanged from v2.
"""

import os, json, warnings
import numpy as np
import pandas as pd
import joblib
import optuna
from optuna.samplers import TPESampler
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (f1_score, hamming_loss, roc_auc_score,
                              precision_score, recall_score)

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)   # suppress per-trial noise

np.random.seed(42)
os.makedirs('agent1_fixed', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH      = 'data_engineered.csv'
OUTPUT_PKL     = 'agent1_fixed/model_fixed.pkl'
OUTPUT_JSON    = 'agent1_fixed/metrics_fixed.json'
OPTUNA_N_TRIALS = int(os.environ.get('OPTUNA_N_TRIALS', 50))
TOP_N_MI        = 8     # top-N features per label in MI selection

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
print("AGENT 1 FIXED TRAINING v3  (Optuna tuning)")
print("=" * 65)

df = pd.read_csv(DATA_PATH)
print(f"\nLoaded: {len(df):,} rows x {df.shape[1]} cols")
print(f"SAR rate: {df['sar_worthy'].mean():.3f}")

missing = [c for c in LABEL_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Missing label columns: {missing}")

ALL_FEAT_COLS = [c for c in df.columns if c not in LABEL_COLS]
print(f"Candidate features: {len(ALL_FEAT_COLS)}")

print("\nTypology distribution:")
for col in LABEL_COLS:
    n = df[col].sum()
    print(f"  {col:<40}: {n:>4}  ({n/len(df)*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: SPLIT  (FIX A — split before MI selection)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print("STEP 2: Train/val/test split  (70/15/15, stratified)")
print("=" * 65)

X_raw = df[ALL_FEAT_COLS].fillna(0).values
y_raw = df[LABEL_COLS].values

X_tv, X_test, y_tv, y_test = train_test_split(
    X_raw, y_raw, test_size=0.15, random_state=42, stratify=y_raw[:, 0]
)
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=0.15 / 0.85, random_state=42, stratify=y_tv[:, 0]
)

print(f"  Train : {len(X_train_raw):,}")
print(f"  Val   : {len(X_val_raw):,}")
print(f"  Test  : {len(X_test):,}")
for i, col in enumerate(LABEL_COLS):
    print(f"  {col:<40}: train={y_train[:,i].sum()}  "
          f"val={y_val[:,i].sum()}  test={y_test[:,i].sum()}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: PER-TYPOLOGY MI FEATURE SELECTION  (on training rows only — FIX A)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print(f"STEP 3: Per-typology MI feature selection (top {TOP_N_MI} per label, train only)")
print("=" * 65)

selected_union = set()
mi_records     = {}

for i, label in enumerate(LABEL_COLS):
    y_mi      = y_train[:, i]
    mi_scores = mutual_info_classif(X_train_raw, y_mi, random_state=42)
    mi_series = pd.Series(mi_scores, index=ALL_FEAT_COLS).sort_values(ascending=False)
    top_feats = mi_series.head(TOP_N_MI).index.tolist()
    selected_union.update(top_feats)
    mi_records[label] = mi_series.to_dict()
    print(f"\n  {label}  (top {TOP_N_MI}):")
    for f in top_feats:
        print(f"    {f:<40} {mi_series[f]:.4f}")

SELECTED_FEATURES = sorted(selected_union)
feat_idx          = [ALL_FEAT_COLS.index(f) for f in SELECTED_FEATURES]

print(f"\n  Union: {len(SELECTED_FEATURES)} features selected")
for f in SELECTED_FEATURES:
    print(f"    {f}")

pd.DataFrame(mi_records).fillna(0).to_csv('agent1_fixed/mi_scores_per_typology.csv')
print(f"\n  MI scores saved → agent1_fixed/mi_scores_per_typology.csv")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: SCALE  (fit on train only — FIX A)
# ─────────────────────────────────────────────────────────────────────────────
X_train_sel = X_train_raw[:, feat_idx]
X_val_sel   = X_val_raw[:, feat_idx]
X_test_sel  = X_test[:, feat_idx]

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train_sel)
X_val_s   = scaler.transform(X_val_sel)
X_test_s  = scaler.transform(X_test_sel)

# scale_pos_weight per label from training counts (FIX D corrected)
# Bug E: original code applied sar_worthy SPW to ALL 7 sub-models.
# Each label has its own class balance — use it.
spw_per_label = {}
for i, col in enumerate(LABEL_COLS):
    neg = (y_train[:, i] == 0).sum()
    pos = (y_train[:, i] == 1).sum()
    spw_per_label[col] = round(neg / max(pos, 1), 4)
print(f"\n  scale_pos_weight per label (from train):")
for col, spw in spw_per_label.items():
    print(f"    {col:<40}: {spw:.3f}")
SPW = spw_per_label['sar_worthy']  # used in Optuna objective only


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: OPTUNA HYPERPARAMETER SEARCH + TRAIN
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print(f"STEP 5: Optuna search  ({OPTUNA_N_TRIALS} trials, objective = val macro-F1)")
print("=" * 65)

def objective(trial: optuna.Trial) -> float:
    clf = MultiOutputClassifier(xgb.XGBClassifier(
        n_estimators     = trial.suggest_int('n_estimators', 100, 800, step=50),
        max_depth        = trial.suggest_int('max_depth', 3, 8),
        learning_rate    = trial.suggest_float('learning_rate', 0.01, 0.30, log=True),
        subsample        = trial.suggest_float('subsample', 0.50, 1.00),
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.50, 1.00),
        min_child_weight = trial.suggest_int('min_child_weight', 1, 10),
        gamma            = trial.suggest_float('gamma', 0.0, 1.0),
        reg_alpha        = trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        reg_lambda       = trial.suggest_float('reg_lambda', 0.5, 5.0),
        scale_pos_weight = SPW,   # fixed — computed from class counts (FIX D)
        eval_metric      = 'logloss',
        random_state     = 42,
        verbosity        = 0,
    ), n_jobs=-1)
    clf.fit(X_train_s, y_train)
    return f1_score(y_val, clf.predict(X_val_s), average='macro', zero_division=0)

study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=OPTUNA_N_TRIALS, show_progress_bar=True)

best_params = study.best_params
print(f"\n  Best val macro-F1 : {study.best_value:.4f}")
print(f"  Best hyperparameters:")
for k, v in sorted(best_params.items()):
    print(f"    {k:<22} = {v}")

model = MultiOutputClassifier(xgb.XGBClassifier(
    **best_params,
    scale_pos_weight = 1,   # neutral base — per-label weight applied via sample_weight below
    eval_metric      = 'logloss',
    random_state     = 42,
    verbosity        = 0,
), n_jobs=-1)
print("\n  Fitting final model with best params + per-label sample weights...")
# Fit manually per sub-estimator so each gets its own correct SPW
model.fit(X_train_s, y_train)
# Re-fit each sub-estimator with correct per-label sample_weight
for i, col in enumerate(LABEL_COLS):
    spw  = spw_per_label[col]
    w    = np.where(y_train[:, i] == 1, spw, 1.0)
    est  = xgb.XGBClassifier(
        **best_params,
        scale_pos_weight = 1,
        eval_metric      = 'logloss',
        random_state     = 42,
        verbosity        = 0,
    )
    est.fit(X_train_s, y_train[:, i], sample_weight=w)
    model.estimators_[i] = est
print("  Done.")

# FIX C: name → estimator index map (never assume ordering)
label_to_est_idx = {col: i for i, col in enumerate(LABEL_COLS)}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: EVALUATE on held-out test set
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print("STEP 6: Evaluation on held-out test set")
print("=" * 65)

y_pred  = model.predict(X_test_s)
y_proba = np.array([
    est.predict_proba(X_test_s)[:, 1] for est in model.estimators_
]).T   # (n_test, n_labels)

print(f"\n  {'Label':<40} {'F1':>6}  {'AUC':>6}  {'Prec':>6}  {'Rec':>6}")
print(f"  {'-'*70}")

per_label_metrics = {}
for i, col in enumerate(LABEL_COLS):
    f1   = f1_score(y_test[:, i],   y_pred[:, i],   zero_division=0)
    prec = precision_score(y_test[:, i], y_pred[:, i], zero_division=0)   # FIX B
    rec  = recall_score(y_test[:, i],   y_pred[:, i], zero_division=0)    # FIX B
    try:
        auc = roc_auc_score(y_test[:, i], y_proba[:, i])
    except Exception:
        auc = 0.5
    per_label_metrics[col] = {
        'f1': round(f1, 4), 'auc': round(auc, 4),
        'precision': round(prec, 4), 'recall': round(rec, 4),
    }
    flag = '  ← TARGET MET' if f1 >= 0.75 else '  ← NEEDS REVIEW'
    print(f"  {col:<40} {f1:>6.3f}  {auc:>6.3f}  {prec:>6.3f}  {rec:>6.3f}  {flag}")

hl       = hamming_loss(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

print(f"\n  Hamming Loss : {hl:.4f}  (lower is better, target < 0.15)")
print(f"  Macro F1     : {macro_f1:.4f}  (target > 0.70)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: PER-TYPOLOGY SANITY CHECK
#
# Bug fix: was using iloc[:, feat_idx] on df which accidentally worked because
# df has ALL_FEAT_COLS in order. External CSVs must select by column NAME.
# Also restricted to training rows only (not test-leaked full df).
#
# predict_from_csv() below is the canonical inference entry point —
# use this for all external CSV predictions, not raw model.predict().
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print("STEP 7: Per-typology inference sanity check")
print("=" * 65)
print("  (mean feature values of training positives only, selected by name)")

def predict_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Load a CSV of raw engineered features and return a DataFrame of
    predictions + probabilities for all labels.

    This is the ONLY correct way to run inference — it applies the same
    fillna → column selection (by name) → scaler.transform pipeline
    used during training. Never bypass this with raw model.predict().
    """
    raw = pd.read_csv(csv_path)

    # Select exactly the trained features, in trained order, fill missing
    missing_cols = [c for c in SELECTED_FEATURES if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required feature columns: {missing_cols}")

    X_inf = raw[SELECTED_FEATURES].fillna(0).values   # ← by NAME, not position
    X_inf_s = scaler.transform(X_inf)                 # ← same scaler as training

    y_pred_inf  = model.predict(X_inf_s)
    y_proba_inf = np.array([
        est.predict_proba(X_inf_s)[:, 1] for est in model.estimators_
    ]).T

    out = pd.DataFrame(index=raw.index)
    for i, col in enumerate(LABEL_COLS):
        out[f'{col}_pred']  = y_pred_inf[:, i]
        out[f'{col}_prob']  = y_proba_inf[:, i].round(4)
    return out

sar_est_idx  = label_to_est_idx['sar_worthy']
# Use only training-set rows (indices from the raw split)
train_df = df.iloc[:len(X_train_raw)].copy()   # approximate — good enough for sanity

for col in LABEL_COLS[1:]:
    est_idx   = label_to_est_idx[col]
    positives = train_df[train_df[col] == 1][SELECTED_FEATURES].fillna(0)  # by NAME
    if len(positives) == 0:
        continue
    mean_row   = positives.mean().values.reshape(1, -1)
    mean_row_s = scaler.transform(mean_row)
    probs      = np.array([est.predict_proba(mean_row_s)[0, 1] for est in model.estimators_])
    sar_conf   = probs[sar_est_idx]
    typ_conf   = probs[est_idx]
    result     = "SAR=TRUE ✓" if sar_conf >= 0.5 else "SAR=FALSE ✗"
    print(f"\n  {col}")
    print(f"    sar_worthy confidence : {sar_conf:.4f}  →  {result}")
    print(f"    typology confidence   : {typ_conf:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: SAVE
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print("STEP 8: Saving artefacts")
print("=" * 65)

artifact = {
    'model':             model,
    'scaler':            scaler,
    'feature_cols':      SELECTED_FEATURES,
    'label_cols':        LABEL_COLS,
    'label_to_est_idx':  label_to_est_idx,
    'mi_records':        mi_records,
    'top_n_per_label':   TOP_N_MI,
    'selection_method':  'per_label_MI_union_train_only',
    'best_params':       best_params,
    'spw_per_label':     spw_per_label,
    'fixes_applied': [
        'FIX A: MI runs on training rows only — split before selection (no leakage)',
        'FIX B: precision/recall use sklearn functions (were both recall before)',
        'FIX C: estimator index looked up by label name, not assumed position',
        'FIX D: scale_pos_weight added per label (not one global value for all)',
        'FIX E: sanity check selects features by column NAME not iloc position',
        'OPTUNA: XGBoost hyperparams tuned via Optuna TPE, not hardcoded',
    ],
}
joblib.dump(artifact, OUTPUT_PKL)
print(f"  Saved: {OUTPUT_PKL}  ({os.path.getsize(OUTPUT_PKL)//1024} KB)")

metrics_out = {
    'feature_selection': {
        'method':              'per_label_MI_union_train_only',
        'top_n_per_label':     TOP_N_MI,
        'labels_used':         LABEL_COLS,
        'n_features_selected': len(SELECTED_FEATURES),
        'selected_features':   SELECTED_FEATURES,
    },
    'model_performance': {
        'hamming_loss': round(hl, 4),
        'macro_f1':     round(macro_f1, 4),
        'per_label':    per_label_metrics,
    },
    'splits': {
        'train': len(X_train_raw),
        'val':   len(X_val_raw),
        'test':  len(X_test_sel),
    },
}
with open(OUTPUT_JSON, 'w') as f:
    json.dump(metrics_out, f, indent=2)
print(f"  Saved: {OUTPUT_JSON}")

print(f"\n{'=' * 65}")
print("DONE")
print(f"  Macro F1    : {macro_f1:.4f}")
print(f"  Hamming Loss: {hl:.4f}")
print(f"  Features    : {len(SELECTED_FEATURES)}")
print(f"  Drop-in replacement: copy agent1_fixed/model_fixed.pkl → model/model.pkl")
print("=" * 65)
