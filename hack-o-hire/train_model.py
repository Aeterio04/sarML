"""
train_model.py  —  Step 05: Model Training  (FINAL)
=====================================================
Input : data_engineered.csv    (7,000 rows x 57 cols)
Output: model/model.pkl
        model/metrics.json
        model/feature_importance.csv
        model/mi_scores.csv
        model/vif_scores.csv
        model/vif_history.csv
        model/selected_features.json
        model/optuna_best_params.json
        model/optuna_study.pkl

Pipeline
--------
  0.  80/20 stratified split  ← FIRST, before anything touches data
  1a. Mutual Information filter  — on X_train only, uniform across all features
  1b. VIF filter                 — on X_train only, base features only
  2.  StandardScaler             — fit on X_train only
  3.  Optuna: 50 trials x 5-fold StratifiedKFold CV on X_train
              Scaler fit inside each fold (correct — avoids fold-val leakage)
              Objective: mean F1 on sar_worthy across folds
  4.  Final training: one XGBClassifier per label
              Best Optuna params + scale_pos_weight per label
              Early stopping on a 10% slice of X_train (not the test set)
  5.  Feature importance: gain + n_splits (weight) per feature per label
  6.  Evaluation on X_test — the 20%, never seen until this step

BUGS FIXED vs original code
----------------------------
  [BUG-1] Feature selection leakage — MI and VIF ran on all 7K rows before
          the split. Fixed: split is now Step 0.
  [BUG-2] MI threshold had no effect — engineered features were unconditionally
          re-added after MI. Fixed: MI applied uniformly to all features.
  [BUG-3] Composite scores — removed upstream in feature_engineering.py.
          EXCLUDE list guards against old CSVs being used by mistake.
  [BUG-4] n_splits not reported — now reported alongside gain.
  [BUG-5] Suspiciously fast training — caused by composites trivialising the
          task; early stopping fired in < 10 rounds. With composites removed
          and Optuna tuning, training runs the proper number of rounds.

NOTE ON HIGH AUC
-----------------
  The dataset is synthetic (fix_data.py). Near-miss non-SAR cases were built
  by degrading SAR cases: alert_count was halved, fund_exit_ratio reduced to ~0.5.
  This creates near-deterministic regions (alert_count >= 10 → SAR = 1.0).
  A clean GBM with no leakage achieves AUC=0.997 on this data. High AUC is
  expected and is a property of the synthetic data, not a modelling error.
  What matters: no leakage, valid test set, honest tuning.
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection     import train_test_split, StratifiedKFold
from sklearn.preprocessing       import StandardScaler
from sklearn.feature_selection   import mutual_info_classif
from sklearn.metrics             import (f1_score, precision_score, recall_score,
                                         roc_auc_score, hamming_loss,
                                         confusion_matrix)
from statsmodels.stats.outliers_influence import variance_inflation_factor
import xgboost as xgb

np.random.seed(42)
os.makedirs('model', exist_ok=True)


# ── LOAD ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(r'C:\Users\shriy\hack-o-hire\data_engineered.csv')
print(f"Loaded : {df.shape[0]:,} rows x {df.shape[1]} cols  |  SAR rate: {df['sar_worthy'].mean():.3f}")

LABEL_COLS = [
    'sar_worthy', 'typology_structuring', 'typology_rapid_movement',
    'typology_funnel_account', 'typology_trade_based',
    'typology_shell_company', 'typology_round_tripping'
]

# ── EXCLUSIONS ────────────────────────────────────────────────────────────────
# Raw features replaced by transforms, labels derived at inference time,
# composite scores (defensive guard for old CSVs), and near-duplicate
# engineered features (r > 0.96 with a base feature already in the set).
EXCLUDE = [
    # raw: replaced by cbrt / log transforms
    'total_txn_amount', 'avg_txn_amount', 'std_txn_amount', 'max_txn_amount',
    'min_txn_amount', 'txn_count', 'account_age_days', 'incoming_sources_count',
    'txn_velocity', 'distinct_counterparties', 'time_to_first_outbound_minutes',
    # circular at inference time
    'typology_count',
    # composites: pre-bake the answer (guard for old CSV)
    'behavioral_risk_score', 'network_risk_score', 'overall_suspicion_score',
    '_behavioral_risk_score', '_network_risk_score', '_overall_suspicion_score',
    # near-duplicates removed in feature_engineering.py (guard for old CSV)
    'velocity_per_age', 'speed_exit_score', 'account_age_tier', 'velocity_tier',
]

ALL_FEATURES = [c for c in df.columns if c not in LABEL_COLS and c not in EXCLUDE]
print(f"Candidate features after exclusions: {len(ALL_FEATURES)}")

X_all = df[ALL_FEATURES].values
y_all = df[LABEL_COLS].values


# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — 80/20 STRATIFIED SPLIT
#
# Done first, before any feature selection or scaling.
# X_test is a completely held-out set — touched once at evaluation.
#
# A 10% slice of X_train is also carved for XGBoost early stopping.
# It is NOT part of Optuna CV and NOT the test set.
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 0: 80/20 stratified split")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all,
    test_size=0.20, random_state=42, stratify=y_all[:, 0]
)

print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
print(f"  SAR rate — Train: {y_train[:, 0].mean():.3f}  |  Test: {y_test[:, 0].mean():.3f}")

train_df = pd.DataFrame(X_train, columns=ALL_FEATURES)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1A — MUTUAL INFORMATION FILTER  (X_train only)
#
# Applied uniformly to ALL features — base and engineered alike.
# Previously engineered features bypassed MI via a separate list, making the
# threshold change invisible. Now raising MI_THRESHOLD actually drops features.
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 1A: Mutual Information filter  (train set only)")
print("="*60)

MI_THRESHOLD = 0.05

mi_scores = mutual_info_classif(X_train, y_train[:, 0], random_state=42)
mi_df = (
    pd.DataFrame({'feature': ALL_FEATURES, 'mi_score': mi_scores})
    .sort_values('mi_score', ascending=False)
    .reset_index(drop=True)
)

dropped_mi  = mi_df[mi_df['mi_score'] <  MI_THRESHOLD]['feature'].tolist()
features_mi = mi_df[mi_df['mi_score'] >= MI_THRESHOLD]['feature'].tolist()

print(f"  Threshold : {MI_THRESHOLD}")
print(f"  Kept : {len(features_mi)}  |  Dropped: {len(dropped_mi)}")
if dropped_mi:
    print(f"  Dropped (low MI):")
    for f in dropped_mi:
        sc = mi_df.loc[mi_df['feature'] == f, 'mi_score'].iloc[0]
        print(f"    {f:<40} MI = {sc:.4f}")

print(f"\n  Top 15 by MI score:")
for _, row in mi_df.head(15).iterrows():
    tag = '  <- DROPPED' if row['feature'] in dropped_mi else ''
    print(f"    {row['feature']:<40} MI = {row['mi_score']:.4f}{tag}")

mi_df.to_csv('model/mi_scores.csv', index=False)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1B — VIF FILTER  (X_train only, base features only)
#
# Engineered features intentionally bypass VIF — they are constructed
# combinations and will always be collinear with their base components.
# Near-duplicate engineered features were already removed in
# feature_engineering.py, so the remaining engineered features carry
# information that is genuinely distinct from their base counterparts.
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 1B: VIF filter  (train set only, base features only)")
print("="*60)

VIF_THRESHOLD = 10.0

ORIG_BASE_FEATS = [
    'burst_score', 'fund_exit_ratio', 'kyc_risk_score',
    'counterparty_diversity_score', 'alert_count',
    'account_age_days_log', 'txn_count_log',
    'time_to_first_outbound_minutes_log', 'txn_velocity_log',
    'incoming_sources_count_log', 'distinct_counterparties_log',
    'total_txn_amount_cbrt', 'avg_txn_amount_cbrt', 'std_txn_amount_cbrt',
    'max_txn_amount_cbrt', 'min_txn_amount_cbrt',
    'international_counterparty_flag', 'pep_flag',
    'high_risk_country_flag', 'historical_sar_flag'
]

base_for_vif = [f for f in ORIG_BASE_FEATS if f in features_mi]
eng_survived = [f for f in features_mi if f not in ORIG_BASE_FEATS]

print(f"  Base features entering VIF: {len(base_for_vif)}")
print(f"  Engineered features (MI-filtered, bypass VIF): {len(eng_survived)}")

features_vif = base_for_vif.copy()
X_vif        = train_df[features_vif].values.astype(float)
vif_history  = []
iteration    = 0

while True:
    vif_vals = [variance_inflation_factor(X_vif, i) for i in range(X_vif.shape[1])]
    max_vif  = max(vif_vals)
    max_feat = features_vif[int(np.argmax(vif_vals))]
    vif_history.append({
        'iteration': iteration, 'feature': max_feat,
        'vif': round(max_vif, 2),
        'action': 'kept' if max_vif <= VIF_THRESHOLD else 'dropped'
    })
    if max_vif <= VIF_THRESHOLD:
        print(f"  All {len(features_vif)} base features have VIF <= {VIF_THRESHOLD}.")
        break
    print(f"  Iter {iteration + 1}: drop '{max_feat}'  VIF = {max_vif:.2f}")
    idx          = features_vif.index(max_feat)
    features_vif.pop(idx)
    X_vif        = np.delete(X_vif, idx, axis=1)
    iteration   += 1

dropped_vif = [f for f in base_for_vif if f not in features_vif]
print(f"  Base after VIF: {len(features_vif)}  |  Dropped: {len(dropped_vif)}")
if dropped_vif:
    print(f"  Dropped: {dropped_vif}")

X_vif_final = train_df[features_vif].values.astype(float)
vif_final   = pd.DataFrame({
    'feature': features_vif,
    'vif':     [round(variance_inflation_factor(X_vif_final, i), 2)
                for i in range(len(features_vif))]
}).sort_values('vif', ascending=False)
vif_final.to_csv('model/vif_scores.csv', index=False)
pd.DataFrame(vif_history).to_csv('model/vif_history.csv', index=False)

FEATURE_COLS = features_vif + eng_survived
print(f"\n  Final feature set: {len(FEATURE_COLS)} features")
print(f"    Base (post-VIF)                  : {len(features_vif)}")
print(f"    Engineered (post-MI, bypass VIF) : {len(eng_survived)}")
print(f"\n  All selected features:")
for i, f in enumerate(FEATURE_COLS, 1):
    print(f"    {i:>2}. {f}")

feat_idx    = {f: i for i, f in enumerate(ALL_FEATURES)}
sel_idx     = [feat_idx[f] for f in FEATURE_COLS]
X_train_sel = X_train[:, sel_idx]
X_test_sel  = X_test[:,  sel_idx]


# ── SCALER (fit on X_train only) ──────────────────────────────────────────────
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train_sel)
X_test_s  = scaler.transform(X_test_sel)


# ── EARLY-STOPPING VALIDATION SLICE ──────────────────────────────────────────
# 10% of X_train, used only for XGBoost early stopping (not for CV or test eval)
X_tr_es, X_es_val, y_tr_es, y_es_val = train_test_split(
    X_train_s, y_train,
    test_size=0.10, random_state=42, stratify=y_train[:, 0]
)
print(f"\n  Early-stopping slice: {len(X_es_val):,} rows (10% of train, not used in CV)")


# ── scale_pos_weight PER LABEL  (X_train only) ────────────────────────────────
SPW = {}
for i, label in enumerate(LABEL_COLS):
    pos = y_train[:, i].sum()
    neg = len(y_train) - pos
    SPW[label] = round(neg / max(pos, 1), 2)

print(f"\n  scale_pos_weight (train set only):")
for label, w in SPW.items():
    pos = int(y_train[:, LABEL_COLS.index(label)].sum())
    print(f"    {label:<40} spw={w:>6}  pos={pos:,}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — OPTUNA HYPERPARAMETER TUNING
#
# Strategy: tune structural params on sar_worthy using 5-fold CV on X_train.
# Apply same structural params to all 7 labels; scale_pos_weight is per-label.
#
# Why tune on sar_worthy only:
#   - Primary target, balanced 50/50 — stable F1 metric across folds.
#   - Rare typology labels (round_tripping ~5%) produce high F1 variance on
#     small CV folds, making them unreliable tuning objectives on 7K rows.
#   - Structural params (depth, regularisation, learning rate) transfer well
#     across labels once scale_pos_weight handles per-label imbalance.
#
# Correct CV design:
#   - StandardScaler fit inside each fold on fold-train only.
#     Fitting outside the fold leaks fold-val distribution — common mistake.
#   - X_test is never touched in this step.
#   - n_estimators=300 inside Optuna (fast); final model uses 1000 + early stop.
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 3: Optuna hyperparameter tuning")
print("="*60)

N_TRIALS   = 50
N_FOLDS    = 5
OPT_SEED   = 42

def objective(trial):
    params = {
        'n_estimators'     : 300,
        'max_depth'        : trial.suggest_int('max_depth', 3, 8),
        'learning_rate'    : trial.suggest_float('learning_rate', 0.01, 0.30, log=True),
        'subsample'        : trial.suggest_float('subsample', 0.50, 1.00),
        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.40, 1.00),
        'min_child_weight' : trial.suggest_int('min_child_weight', 1, 30),
        'gamma'            : trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha'        : trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda'       : trial.suggest_float('reg_lambda', 0.10, 10.0, log=True),
        'scale_pos_weight' : SPW['sar_worthy'],
        'eval_metric'      : 'logloss',
        'use_label_encoder': False,
        'random_state'     : OPT_SEED,
        'verbosity'        : 0,
    }

    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=OPT_SEED)
    y_sar  = y_train[:, 0]
    scores = []

    for tr_idx, vl_idx in skf.split(X_train_sel, y_sar):
        X_ft, X_fv = X_train_sel[tr_idx], X_train_sel[vl_idx]
        y_ft, y_fv = y_sar[tr_idx],        y_sar[vl_idx]

        # Scaler fit inside the fold — prevents val distribution leaking into train
        fs = StandardScaler()
        X_ft_s = fs.fit_transform(X_ft)
        X_fv_s = fs.transform(X_fv)

        clf = xgb.XGBClassifier(**params)
        clf.fit(X_ft_s, y_ft, verbose=False)
        scores.append(f1_score(y_fv, clf.predict(X_fv_s), zero_division=0))

    return float(np.mean(scores))


study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=OPT_SEED),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
)

print(f"  {N_TRIALS} trials x {N_FOLDS}-fold CV  |  objective: mean F1 on sar_worthy")
print(f"  (~{N_TRIALS * N_FOLDS} XGB fits for tuning + 7 final fits)")

def _log(study, trial):
    if (trial.number + 1) % 10 == 0:
        print(f"  Trial {trial.number + 1:>3}/{N_TRIALS}  "
              f"best F1 = {study.best_value:.4f}  "
              f"(trial {study.best_trial.number + 1})")

study.optimize(objective, n_trials=N_TRIALS, callbacks=[_log])

best = study.best_trial.params
print(f"\n  Best trial #{study.best_trial.number + 1}  |  CV F1 = {study.best_value:.4f}")
print(f"  Best parameters:")
for k, v in best.items():
    print(f"    {k:<22} = {v}")

joblib.dump(study, 'model/optuna_study.pkl')
with open('model/optuna_best_params.json', 'w') as f:
    json.dump({'best_params': best, 'best_cv_f1': study.best_value,
               'n_trials': N_TRIALS, 'n_folds': N_FOLDS}, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — FINAL TRAINING: ONE XGBClassifier PER LABEL
#
# Trains on full X_train_s (the 80%).
# Early stopping uses the 10% ES slice carved before Optuna (not the test set).
# n_estimators=1000 ceiling; actual rounds controlled by early_stopping_rounds=50.
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 4: Final training — one XGBClassifier per label")
print("="*60)

FINAL_PARAMS = {
    'n_estimators'     : 1000,
    'max_depth'        : best['max_depth'],
    'learning_rate'    : best['learning_rate'],
    'subsample'        : best['subsample'],
    'colsample_bytree' : best['colsample_bytree'],
    'min_child_weight' : best['min_child_weight'],
    'gamma'            : best['gamma'],
    'reg_alpha'        : best['reg_alpha'],
    'reg_lambda'       : best['reg_lambda'],
    'eval_metric'      : 'aucpr',
    'use_label_encoder': False,
    'random_state'     : 42,
    'verbosity'        : 0,
}

fitted  = []
results = {}

for i, label in enumerate(LABEL_COLS):
    y_tr_full  = y_train[:, i]
    y_es_val_l = y_es_val[:, i]
    y_ts       = y_test[:,  i]

    clf = xgb.XGBClassifier(
        **FINAL_PARAMS,
        scale_pos_weight=SPW[label],
        early_stopping_rounds=50,
    )
    clf.fit(
        X_train_s, y_tr_full,
        eval_set=[(X_es_val, y_es_val_l)],
        verbose=False,
    )

    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)[:, 1]

    f1   = f1_score(y_ts, y_pred,  zero_division=0)
    prec = precision_score(y_ts, y_pred, zero_division=0)
    rec  = recall_score(y_ts, y_pred,    zero_division=0)
    try:
        auc = roc_auc_score(y_ts, y_prob)
    except Exception:
        auc = float('nan')

    fitted.append(clf)
    results[label] = dict(
        f1=round(f1, 4), precision=round(prec, 4), recall=round(rec, 4),
        roc_auc=round(auc, 4), best_round=int(clf.best_iteration + 1),
        spw=SPW[label], pos_in_test=int(y_ts.sum()),
    )

    target = 0.85 if label == 'sar_worthy' else 0.75
    status = 'PASS' if f1 >= target else 'WARN'
    print(f"  [{status}] {label:<40} F1={f1:.4f}  AUC={auc:.4f}  "
          f"P={prec:.4f}  R={rec:.4f}  rounds={clf.best_iteration + 1}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — FEATURE IMPORTANCE  (gain + n_splits per feature per label)
#
# gain     = feature_importances_  — avg loss reduction per split
# n_splits = get_score('weight')   — times feature used in any split
#
# get_score() key format: XGBoost receives a numpy array (StandardScaler output)
# so it has no feature names. Keys are positional: 'f0', 'f1', ... 'fN'.
# The mapping f'f{i}' below is correct for this code path.
# If you switch to passing a DataFrame, keys become feature names directly.
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 5: Feature importance — gain + n_splits per feature per label")
print("="*60)

gain_data   = {}
weight_data = {}
for label, clf in zip(LABEL_COLS, fitted):
    gain_data[label]   = clf.feature_importances_
    w_dict             = clf.get_booster().get_score(importance_type='weight')
    weight_data[label] = np.array(
        [w_dict.get(f'f{i}', 0) for i in range(len(FEATURE_COLS))]
    )

gain_df   = pd.DataFrame(gain_data,   index=FEATURE_COLS)
weight_df = pd.DataFrame(weight_data, index=FEATURE_COLS)
gain_df['mean_gain']       = gain_df[LABEL_COLS].mean(axis=1)
weight_df['mean_nsplits']  = weight_df[LABEL_COLS].mean(axis=1)

fi_df = pd.DataFrame(index=FEATURE_COLS)
fi_df['mean_gain']    = gain_df['mean_gain']
fi_df['mean_nsplits'] = weight_df['mean_nsplits']
for label in LABEL_COLS:
    fi_df[f'gain_{label}']    = gain_df[label]
    fi_df[f'nsplits_{label}'] = weight_df[label]

fi_df = fi_df.sort_values('mean_gain', ascending=False)
fi_df.insert(0, 'gain_rank', range(1, len(fi_df) + 1))

print(f"\n  {'Feature':<40} {'mean_gain':>10}  {'mean_nsplits':>14}  {'nsplits_sar':>12}")
print(f"  {'-'*82}")
for feat in fi_df.index:
    print(f"  {feat:<40} "
          f"{fi_df.loc[feat, 'mean_gain']:>10.4f}  "
          f"{fi_df.loc[feat, 'mean_nsplits']:>14.1f}  "
          f"{fi_df.loc[feat, 'nsplits_sar_worthy']:>12.0f}")

never_used = fi_df[fi_df['mean_nsplits'] == 0].index.tolist()
print(f"\n  Features selected by MI/VIF but never used in any split: {len(never_used)}")
if never_used:
    for f in never_used:
        print(f"    {f}  (candidate for removal in next iteration)")
else:
    print(f"    None.")

fi_df.to_csv('model/feature_importance.csv')


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — EVALUATION ON TEST SET
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 6: Evaluation on test set  (20%, never seen before)")
print("="*60)

y_pred_all = np.column_stack([clf.predict(X_test_s)             for clf in fitted])
y_prob_all = np.column_stack([clf.predict_proba(X_test_s)[:, 1] for clf in fitted])

hl       = hamming_loss(y_test, y_pred_all)
macro_f1 = f1_score(y_test, y_pred_all, average='macro', zero_division=0)
micro_f1 = f1_score(y_test, y_pred_all, average='micro', zero_division=0)
exact    = (y_pred_all == y_test).all(axis=1).mean()

print(f"\n  {'Metric':<25} {'Value':>10}  {'Target':>10}")
print(f"  {'-'*47}")
print(f"  {'Hamming Loss':<25} {hl:>10.4f}  {'< 0.10':>10}")
print(f"  {'Macro F1':<25} {macro_f1:>10.4f}  {'> 0.75':>10}")
print(f"  {'Micro F1':<25} {micro_f1:>10.4f}")
print(f"  {'Exact Match':<25} {exact:>10.4f}")

print(f"\n  {'Label':<40} {'F1':>7} {'AUC':>7} {'Prec':>7} {'Rec':>7} {'Rounds':>7} {'Status':>7}")
print(f"  {'-'*83}")
for label, m in results.items():
    target = 0.85 if label == 'sar_worthy' else 0.75
    status = 'PASS' if m['f1'] >= target else 'WARN'
    print(f"  {label:<40} {m['f1']:>7.4f} {m['roc_auc']:>7.4f} "
          f"{m['precision']:>7.4f} {m['recall']:>7.4f} "
          f"{m['best_round']:>7}  {status:>6}")

print(f"\n  Confusion matrix — sar_worthy:")
cm = confusion_matrix(y_test[:, 0], y_pred_all[:, 0])
tn, fp, fn, tp = cm.ravel()
total_pos, total_neg = tp + fn, tn + fp
print(f"    TN : {tn:>5}  ({tn / total_neg * 100:.1f}% of actual non-SAR)")
print(f"    FP : {fp:>5}  ({fp / total_neg * 100:.1f}% of actual non-SAR)")
print(f"    FN : {fn:>5}  ({fn / total_pos * 100:.1f}% of actual SAR — missed SARs)")
print(f"    TP : {tp:>5}  ({tp / total_pos * 100:.1f}% of actual SAR)")
print(f"    Recall    = {tp / (tp + fn):.4f}")
print(f"    Precision = {tp / (tp + fp):.4f}")
print(f"    In AML, FN (missed SARs) are more costly than FP (false alarms).")

print(f"\n  NOTE: AUC will be ~0.97-0.99 on this dataset.")
print(f"  This is expected — see fix_data.py. The synthetic near-miss generation")
print(f"  degraded alert_count by ~50% for non-SAR cases, creating near-deterministic")
print(f"  regions. A clean GBM with zero leakage achieves AUC=0.997 on this data.")


# ── SAVE ──────────────────────────────────────────────────────────────────────
artifact = {
    'estimators'    : fitted,
    'label_cols'    : LABEL_COLS,
    'feature_cols'  : FEATURE_COLS,
    'scaler'        : scaler,
    'best_params'   : best,
    'spw'           : SPW,
    'mi_threshold'  : MI_THRESHOLD,
    'vif_threshold' : VIF_THRESHOLD,
    'dropped_mi'    : dropped_mi,
    'dropped_vif'   : dropped_vif,
    'excluded'      : EXCLUDE,
}
joblib.dump(artifact, 'model/model.pkl')

metrics_out = {
    'aggregate': {
        'hamming_loss': round(hl, 4), 'macro_f1': round(macro_f1, 4),
        'micro_f1': round(micro_f1, 4), 'exact_match': round(exact, 4),
    },
    'confusion_sar_worthy': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
    'per_label'         : results,
    'test_rows'         : len(X_test),
    'train_rows'        : len(X_train),
    'n_features'        : len(FEATURE_COLS),
    'dropped_mi'        : dropped_mi,
    'dropped_vif'       : dropped_vif,
    'never_used_splits' : never_used,
    'optuna_best_params': best,
    'optuna_best_cv_f1' : round(study.best_value, 4),
}
with open('model/metrics.json', 'w') as f:
    json.dump(metrics_out, f, indent=2)

json.dump({'features': FEATURE_COLS, 'n': len(FEATURE_COLS)},
          open('model/selected_features.json', 'w'), indent=2)

print(f"\n{'='*60}")
print("SAVED")
print(f"  model/model.pkl")
print(f"  model/metrics.json")
print(f"  model/feature_importance.csv   (gain + n_splits, all labels)")
print(f"  model/mi_scores.csv            (train set only)")
print(f"  model/vif_scores.csv           (train set only)")
print(f"  model/vif_history.csv")
print(f"  model/selected_features.json   ({len(FEATURE_COLS)} features)")
print(f"  model/optuna_best_params.json")
print(f"  model/optuna_study.pkl")
print("="*60)


# ── INFERENCE SANITY CHECK ────────────────────────────────────────────────────
print(f"\nSANITY — inference on 3 test samples:")
for idx in [0, 1, 2]:
    sample = X_test_s[idx:idx + 1]
    preds  = [clf.predict(sample)[0] for clf in fitted]
    probas = [round(clf.predict_proba(sample)[0, 1], 3) for clf in fitted]
    actual = y_test[idx]
    match  = all(int(p) == int(a) for p, a in zip(preds, actual))
    print(f"\n  Sample {idx + 1}  ({'all correct' if match else 'errors present'}):")
    print(f"    {'Label':<40} {'Act':>5} {'Pred':>5} {'P(+)':>6}")
    for lbl, act, pred, prob in zip(LABEL_COLS, actual, preds, probas):
        flag = '' if int(pred) == int(act) else '  <- WRONG'
        print(f"    {lbl:<40} {int(act):>5} {int(pred):>5} {prob:>6.3f}{flag}")