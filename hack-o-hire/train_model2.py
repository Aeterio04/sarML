"""
train_model.py  —  Agent 1 Classifier (FIXED)
Team Baymax · Barclays Hack-O-Hire

THE BUG THAT WAS KILLING YOU:
  Old code ran MI feature selection ONCE against sar_worthy across all 8,000 rows.
  Rapid movement has the biggest, cleanest burst/velocity signal (27.9% of SAR cases).
  Those burst features score highest in MI and get selected.
  Structuring features (txn_amount_cv, avg amounts near threshold) score LOW because
  they're only discriminative for 21% of cases — diluted across 8,000 rows.
  Result: 7 burst-only features. Model is blind to structuring and TBML.

THE FIX:
  Run MI feature selection SEPARATELY for each of the 7 labels.
  Take the UNION of top features across all labels.
  Each classifier now gets the features it actually needs.
  Structuring gets: txn_amount_cv, avg_txn_amount_cbrt, alert_density, etc.
  TBML gets: high_risk_country_flag, hr_country_x_exit, max_to_avg_txn_ratio, etc.

HOW TO RUN:
  1. Put this file in the same folder as data_engineered.csv
  2. pip install xgboost scikit-learn pandas numpy joblib (if not already installed)
  3. python train_model2.py
  4. It will produce model.pkl — drop that into your pipeline, done.

ARTIFACT FORMAT (matches pipeline.py expectations):
  model.pkl is a dict with keys:
    art['estimators']    — list of 7 fitted XGBClassifiers (one per label)
    art['feature_cols']  — the union feature list (the fix)
    art['scaler']        — fitted StandardScaler
    art['label_cols']    — the 7 label column names in order
    art['mi_scores']     — per-label MI scores (for debugging/explainability)
    art['metrics']       — test set performance numbers
"""

import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, hamming_loss
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# STEP 0: CONFIGURATION — tweak these if needed
# ─────────────────────────────────────────────

DATA_PATH   = "data_engineered.csv"
OUTPUT_PATH = "model.pkl"

# How many top features to pick PER LABEL in MI selection.
# 10 per label × 7 labels = up to 70, but with overlap it'll be ~25-35 unique features.
TOP_K_PER_LABEL = 10

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

XGBOOST_PARAMS = {
    "n_estimators":      300,
    "max_depth":         4,
    "learning_rate":     0.05,
    "subsample":         0.80,
    "colsample_bytree":  0.70,
    "min_child_weight":  3,
    "gamma":             0.1,
    "reg_alpha":         0.1,
    "reg_lambda":        1.5,
    "eval_metric":       "logloss",
    "random_state":      42,
    "verbosity":         0,
}

LABEL_COLS = [
    "sar_worthy",
    "typology_structuring",
    "typology_rapid_movement",
    "typology_funnel_account",
    "typology_trade_based",
    "typology_shell_company",
    "typology_round_tripping",
]

EXCLUDE_FROM_FEATURES = LABEL_COLS + ["typology_count"]


# ─────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────

print("=" * 60)
print("AGENT 1 — FIXED TRAINING PIPELINE")
print("=" * 60)

print(f"\n[1/6] Loading {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
print(f"      Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

missing_labels = [c for c in LABEL_COLS if c not in df.columns]
if missing_labels:
    raise ValueError(f"Missing label columns in CSV: {missing_labels}. "
                     f"Check you're using data_engineered.csv, not data_fixed.csv.")

ALL_FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE_FROM_FEATURES]
print(f"      Available feature columns: {len(ALL_FEATURE_COLS)}")

assert df.isnull().sum().sum() == 0, "ERROR: Data has null values."
assert np.isinf(df[ALL_FEATURE_COLS].values).sum() == 0, "ERROR: Data has infinite values."

X_all = df[ALL_FEATURE_COLS].values
y_all = df[LABEL_COLS].values

print(f"      SAR rate: {df['sar_worthy'].mean():.1%}")
for col in LABEL_COLS[1:]:
    print(f"      {col}: {df[col].sum():,} positive ({df[col].mean():.1%})")


# ─────────────────────────────────────────────
# STEP 2: TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────

print(f"\n[2/6] Splitting data ({TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%})...")

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_all, y_all, test_size=TEST_RATIO, random_state=42, stratify=y_all[:, 0]
)
val_frac = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=val_frac, random_state=42, stratify=y_trainval[:, 0]
)

print(f"      Train: {X_train.shape[0]:,}  Val: {X_val.shape[0]:,}  Test: {X_test.shape[0]:,}")


# ─────────────────────────────────────────────
# STEP 3: SCALE FEATURES
# ─────────────────────────────────────────────

print("\n[3/6] Fitting StandardScaler on training data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)
print("      Done.")


# ─────────────────────────────────────────────
# STEP 4: PER-LABEL MI FEATURE SELECTION  ← THE FIX
# ─────────────────────────────────────────────

print(f"\n[4/6] Per-label MI feature selection (top {TOP_K_PER_LABEL} per label)...")

mi_scores_per_label = {}
selected_indices = set()

for i, label_name in enumerate(LABEL_COLS):
    y_label = y_train[:, i]
    if y_label.sum() == 0:
        print(f"      WARNING: {label_name} has 0 positive cases — skipping.")
        continue

    mi_scores = mutual_info_classif(X_train_scaled, y_label, random_state=42)
    top_k_indices = np.argsort(mi_scores)[::-1][:TOP_K_PER_LABEL]
    selected_indices.update(top_k_indices.tolist())

    mi_scores_per_label[label_name] = {
        ALL_FEATURE_COLS[j]: float(mi_scores[j]) for j in range(len(ALL_FEATURE_COLS))
    }

    top_names = [ALL_FEATURE_COLS[j] for j in top_k_indices]
    print(f"      {label_name}: {', '.join(top_names[:5])} ...")

selected_indices_sorted = sorted(list(selected_indices))
SELECTED_FEATURE_COLS = [ALL_FEATURE_COLS[i] for i in selected_indices_sorted]

print(f"\n      Union: {len(SELECTED_FEATURE_COLS)} features selected (was 7 before fix)")
print(f"      {SELECTED_FEATURE_COLS}")

X_train_sel = X_train_scaled[:, selected_indices_sorted]
X_val_sel   = X_val_scaled[:, selected_indices_sorted]
X_test_sel  = X_test_scaled[:, selected_indices_sorted]


# ─────────────────────────────────────────────
# STEP 5: TRAIN — one XGBClassifier per label
#
# NOTE: We train 7 independent classifiers and store them as a list
# in art['estimators'] to match the format pipeline.py expects:
#
#   fitted_models = artifact['estimators']
#   for clf, label in zip(fitted_models, label_cols):
#       probas = clf.predict_proba(X_inf_s)[:, 1]
#
# Do NOT use MultiOutputClassifier here — its predict_proba returns
# a list of arrays which breaks the pipeline.py inference loop.
# ─────────────────────────────────────────────

print(f"\n[5/6] Training 7 independent XGBClassifiers (one per label)...")

estimators = []
val_f1_scores = []

for i, label_name in enumerate(LABEL_COLS):
    y_label_train = y_train[:, i]
    y_label_val   = y_val[:, i]

    # scale_pos_weight handles class imbalance per label
    n_neg = int((y_label_train == 0).sum())
    n_pos = int((y_label_train == 1).sum())
    spw   = n_neg / max(n_pos, 1)

    clf = XGBClassifier(**XGBOOST_PARAMS, scale_pos_weight=spw)
    clf.fit(X_train_sel, y_label_train)

    val_pred = clf.predict(X_val_sel)
    val_f1   = f1_score(y_label_val, val_pred, zero_division=0)
    val_f1_scores.append(val_f1)

    estimators.append(clf)
    print(f"      {label_name:<40} val F1 = {val_f1:.4f}  (spw={spw:.1f})")

print(f"\n      Mean val macro-F1: {np.mean(val_f1_scores):.4f}")


# ─────────────────────────────────────────────
# STEP 6: EVALUATE ON TEST SET
# ─────────────────────────────────────────────

print(f"\n[6/6] Evaluating on held-out test set...")

y_test_pred  = np.column_stack([clf.predict(X_test_sel) for clf in estimators])
test_macro_f1 = f1_score(y_test, y_test_pred, average="macro",    zero_division=0)
test_micro_f1 = f1_score(y_test, y_test_pred, average="micro",    zero_division=0)
test_hamming  = hamming_loss(y_test, y_test_pred)

print(f"\n      ── TEST SET RESULTS ──────────────────────────")
print(f"      Macro F1:     {test_macro_f1:.4f}  (target > 0.75)")
print(f"      Micro F1:     {test_micro_f1:.4f}")
print(f"      Hamming Loss: {test_hamming:.4f}  (target < 0.10)")
print(f"      ──────────────────────────────────────────────")

metrics = {
    "macro_f1":  float(test_macro_f1),
    "micro_f1":  float(test_micro_f1),
    "hamming":   float(test_hamming),
    "per_label": {}
}

print(f"\n      Per-label F1 scores:")
for i, label_name in enumerate(LABEL_COLS):
    label_f1 = f1_score(y_test[:, i], y_test_pred[:, i], zero_division=0)
    status = "✓" if label_f1 >= 0.70 else "✗ LOW"
    print(f"        {label_name:<40} F1 = {label_f1:.4f}  {status}")
    metrics["per_label"][label_name] = float(label_f1)

sar_f1 = metrics["per_label"]["sar_worthy"]
if sar_f1 < 0.80:
    print(f"\n      ⚠ sar_worthy F1 = {sar_f1:.4f} below 0.80. Try TOP_K_PER_LABEL=12.")
else:
    print(f"\n      sar_worthy F1 = {sar_f1:.4f} ✓")


# ─────────────────────────────────────────────
# SAVE ARTIFACT
#
# Format matches pipeline.py exactly:
#   artifact['estimators']   — list of XGBClassifiers
#   artifact['feature_cols'] — selected feature names
#   artifact['label_cols']   — label names in order
#   artifact['scaler']       — fitted StandardScaler
# ─────────────────────────────────────────────

art = {
    "estimators":           estimators,
    "feature_cols":         SELECTED_FEATURE_COLS,
    "scaler":               scaler,
    "label_cols":           LABEL_COLS,
    "mi_scores":            mi_scores_per_label,
    "metrics":              metrics,
    "top_k_per_label":      TOP_K_PER_LABEL,
    "n_features_total":     len(ALL_FEATURE_COLS),
    "n_features_selected":  len(SELECTED_FEATURE_COLS),
}

joblib.dump(art, OUTPUT_PATH)

print(f"\n{'=' * 60}")
print(f"Model saved to: {OUTPUT_PATH}")
print(f"  estimators:   {len(estimators)} classifiers")
print(f"  feature_cols: {len(SELECTED_FEATURE_COLS)} features (was 7)")
print(f"{'=' * 60}")


# ─────────────────────────────────────────────
# SMOKE TEST — verify pipeline.py inference loop works
# ─────────────────────────────────────────────

print("\n── Smoke test (single fake row, mimics pipeline.py inference) ──")
try:
    import numpy as np
    fake_row_full = np.zeros((1, len(ALL_FEATURE_COLS)))
    fake_scaled   = scaler.transform(fake_row_full)
    fake_sel      = fake_scaled[:, selected_indices_sorted]

    # Mimic exactly what pipeline.py does
    loaded = joblib.load(OUTPUT_PATH)
    for clf, label in zip(loaded["estimators"], loaded["label_cols"]):
        prob = clf.predict_proba(fake_sel)[:, 1]   # this is what pipeline.py calls
        assert prob.shape == (1,), f"Wrong proba shape for {label}: {prob.shape}"

    print("  predict_proba shape check: ✓")
    print("  Pipeline.py inference loop: ✓")
    print("  Smoke test passed ✓")
except Exception as e:
    print(f"  ✗ Smoke test failed: {e}")
