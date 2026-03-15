"""
validate_seed.py  —  Step 03
-----------------------------
Runs all validation checks on data_fixed.csv before model training.
Prints PASS / WARN / FAIL for every check.
Exits with code 1 if any FAIL found (pipeline should stop).
"""

import pandas as pd
import numpy as np
import sys

df = pd.read_csv(r'C:\Users\shriy\hack-o-hire\data_fixed.csv')

LABEL_COLS  = ['sar_worthy', 'typology_structuring', 'typology_rapid_movement',
               'typology_funnel_account', 'typology_trade_based',
               'typology_shell_company', 'typology_round_tripping']
TYPO_COLS   = LABEL_COLS[1:]
BINARY_COLS = ['international_counterparty_flag', 'pep_flag',
               'high_risk_country_flag', 'historical_sar_flag']
RATIO_COLS  = ['burst_score', 'fund_exit_ratio', 'counterparty_diversity_score',
               'kyc_risk_score']
LOG_COLS    = [c for c in df.columns if c.endswith('_log')]
FEATURE_COLS = [c for c in df.columns if c not in LABEL_COLS]

errors   = []
warnings = []
passes   = []

def PASS(msg):  passes.append(f"  PASS  {msg}")
def WARN(msg):  warnings.append(f"  WARN  {msg}")
def FAIL(msg):  errors.append(f"  FAIL  {msg}")

print("=" * 65)
print("STEP 03 — DATA VALIDATION REPORT")
print(f"File  : data_fixed.csv   |   Shape: {df.shape}")
print("=" * 65)

# ── 1. SHAPE & COMPLETENESS ───────────────────────────────────────────────────
print("\n[1] Shape & Completeness")

if df.shape[0] >= 10000:
    PASS(f"Row count = {df.shape[0]:,}  (min 10,000 required)")
else:
    FAIL(f"Row count = {df.shape[0]:,} — must be ≥ 10,000")

if df.shape[1] == 38:
    PASS(f"Column count = {df.shape[1]}  (20 original + 11 log + 7 labels)")
else:
    WARN(f"Column count = {df.shape[1]} — expected 38")

nulls = df.isnull().sum()
total_nulls = nulls.sum()
if total_nulls == 0:
    PASS("No null values in any column")
else:
    FAIL(f"{total_nulls} null values found: {nulls[nulls>0].to_dict()}")

dupes = df.duplicated().sum()
if dupes == 0:
    PASS("No duplicate rows")
else:
    WARN(f"{dupes} duplicate rows found")

for msg in passes[-4:]: print(msg)
passes_section = passes[:]

# ── 2. LABEL CONSISTENCY ──────────────────────────────────────────────────────
print("\n[2] Label Consistency")
passes = []

# sar_worthy=1 must always have ≥1 typology flag
bad_a = df[(df['sar_worthy'] == 1) & (df[TYPO_COLS].sum(axis=1) == 0)]
if len(bad_a) == 0:
    PASS("All SAR=1 rows have ≥1 typology flag")
else:
    FAIL(f"{len(bad_a)} rows: SAR=1 but no typology flag")

# sar_worthy=0 must have all typology = 0
bad_b = df[(df['sar_worthy'] == 0) & (df[TYPO_COLS].sum(axis=1) > 0)]
if len(bad_b) == 0:
    PASS("All SAR=0 rows have all typology flags = 0")
else:
    FAIL(f"{len(bad_b)} rows: SAR=0 but typology flag > 0")

# Each typology must have ≥ 100 positive cases (SMOTE may have inflated base)
for col in TYPO_COLS:
    cnt = df[col].sum()
    if cnt >= 100:
        PASS(f"{col:<35} has {cnt:,} positive cases")
    else:
        FAIL(f"{col} only has {cnt} positive cases (need ≥100)")

# Binary labels must be exactly 0 or 1
for col in LABEL_COLS:
    bad = df[~df[col].isin([0, 1])]
    if len(bad) == 0:
        PASS(f"{col:<35} values ∈ {{0, 1}}")
    else:
        FAIL(f"{col} has {len(bad)} values outside {{0, 1}}")

for msg in passes: print(msg)

# ── 3. CLASS BALANCE ──────────────────────────────────────────────────────────
print("\n[3] Class Balance")
passes = []

sar_rate = df['sar_worthy'].mean()
if 0.65 <= sar_rate <= 0.80:
    PASS(f"sar_worthy rate = {sar_rate:.3f}  (target 0.65–0.80)")
elif 0.55 <= sar_rate < 0.65 or 0.80 < sar_rate <= 0.90:
    WARN(f"sar_worthy rate = {sar_rate:.3f}  (borderline — target 0.65–0.80)")
else:
    FAIL(f"sar_worthy rate = {sar_rate:.3f}  — severe imbalance (target 0.65–0.80)")

# Class counts
sar_counts = df['sar_worthy'].value_counts()
PASS(f"SAR=1: {sar_counts.get(1,0):,}  |  SAR=0: {sar_counts.get(0,0):,}")

for msg in passes: print(msg)

# ── 4. FEATURE VALUE RANGES ───────────────────────────────────────────────────
print("\n[4] Feature Value Ranges")
passes = []

# Ratio cols must be [0, 1]
for col in RATIO_COLS:
    out = df[(df[col] < 0) | (df[col] > 1)]
    if len(out) == 0:
        PASS(f"{col:<40} ∈ [0, 1]")
    else:
        FAIL(f"{col} has {len(out)} values outside [0, 1]  "
             f"(min={df[col].min():.4f}, max={df[col].max():.4f})")

# Binary feature cols must be 0/1
for col in BINARY_COLS:
    bad = df[~df[col].isin([0, 1])]
    if len(bad) == 0:
        PASS(f"{col:<40} values ∈ {{0, 1}}")
    else:
        FAIL(f"{col} has {len(bad)} non-binary values")

# Positive-only cols (counts, amounts) must be ≥ 0
pos_cols = ['total_txn_amount', 'avg_txn_amount', 'txn_count',
            'alert_count', 'account_age_days', 'distinct_counterparties',
            'incoming_sources_count']
for col in pos_cols:
    bad = df[df[col] < 0]
    if len(bad) == 0:
        PASS(f"{col:<40} ≥ 0")
    else:
        FAIL(f"{col} has {len(bad)} negative values")

for msg in passes: print(msg)

# ── 5. SKEWNESS CHECK ON LOG COLUMNS ─────────────────────────────────────────
print("\n[5] Skewness — Log-Transformed Features")
passes = []

for col in LOG_COLS:
    skew = df[col].skew()
    if abs(skew) <= 1.0:
        PASS(f"{col:<45} skew = {skew:>7.3f}  ✓")
    elif abs(skew) <= 1.5:
        WARN(f"{col:<45} skew = {skew:>7.3f}  (mild residual)")
    else:
        FAIL(f"{col:<45} skew = {skew:>7.3f}  — transform insufficient")

for msg in passes: print(msg)

# ── 6. TYPOLOGY SIGNATURE CHECKS ─────────────────────────────────────────────
print("\n[6] Typology Signature Checks")
passes = []

# Structuring: should have high burst_score
struct = df[df['typology_structuring'] == 1]
if len(struct) > 0:
    mean_burst = struct['burst_score'].mean()
    if mean_burst >= 0.55:
        PASS(f"Structuring cases mean burst_score       = {mean_burst:.3f}  (≥0.55)")
    else:
        WARN(f"Structuring cases mean burst_score       = {mean_burst:.3f}  (expected ≥0.55)")

# Rapid movement: should have high fund_exit_ratio
rapid = df[df['typology_rapid_movement'] == 1]
if len(rapid) > 0:
    mean_fer = rapid['fund_exit_ratio'].mean()
    if mean_fer >= 0.75:
        PASS(f"Rapid movement mean fund_exit_ratio      = {mean_fer:.3f}  (≥0.75)")
    else:
        WARN(f"Rapid movement mean fund_exit_ratio      = {mean_fer:.3f}  (expected ≥0.75)")

# Funnel: should have high incoming_sources_count
funnel = df[df['typology_funnel_account'] == 1]
if len(funnel) > 0:
    mean_isc = funnel['incoming_sources_count'].mean()
    if mean_isc >= 10:
        PASS(f"Funnel cases mean incoming_sources_count = {mean_isc:.1f}  (≥10)")
    else:
        WARN(f"Funnel cases mean incoming_sources_count = {mean_isc:.1f}  (expected ≥10)")

# Round-tripping: should have low time_to_first_outbound
rt = df[df['typology_round_tripping'] == 1]
if len(rt) > 0:
    mean_t2fo = rt['time_to_first_outbound_minutes'].mean()
    if mean_t2fo <= 2000:
        PASS(f"Round-tripping mean time_to_first_outbound = {mean_t2fo:.0f} min  (≤2000)")
    else:
        WARN(f"Round-tripping mean time_to_first_outbound = {mean_t2fo:.0f} min  (expected ≤2000)")

for msg in passes: print(msg)

# ── 7. FEATURE CORRELATION SANITY ────────────────────────────────────────────
print("\n[7] Feature Correlation Sanity")
passes = []

corr = df[FEATURE_COLS + ['sar_worthy']].corr()['sar_worthy'].drop('sar_worthy')
top_pos = corr.sort_values(ascending=False).head(3)
top_neg = corr.sort_values(ascending=True).head(3)

PASS(f"Top 3 positive corr with sar_worthy:")
for feat, val in top_pos.items():
    print(f"          {feat:<42} r = {val:>7.3f}")
PASS(f"Top 3 negative corr with sar_worthy:")
for feat, val in top_neg.items():
    print(f"          {feat:<42} r = {val:>7.3f}")

# Flag if any expected predictor has near-zero correlation
expected_predictors = ['fund_exit_ratio', 'kyc_risk_score', 'burst_score',
                       'alert_count', 'time_to_first_outbound_minutes_log']
for feat in expected_predictors:
    if feat in corr.index and abs(corr[feat]) < 0.05:
        WARN(f"{feat} has near-zero correlation with sar_worthy ({corr[feat]:.3f})")

for msg in passes: print(msg)

# ══════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("VALIDATION SUMMARY")
print("=" * 65)

all_issues = errors + warnings
if errors:
    print(f"\n  FAILs   : {len(errors)}")
    for e in errors:   print(e)
if warnings:
    print(f"\n  WARNINGs: {len(warnings)}")
    for w in warnings: print(w)

if errors:
    print("\n  RESULT: VALIDATION FAILED — fix errors before training.")
    sys.exit(1)
elif warnings:
    print(f"\n  RESULT: PASSED WITH {len(warnings)} WARNING(S) — review before training.")
else:
    print("\n  RESULT: ALL CHECKS PASSED — data_fixed.csv is ready for feature engineering.")

print("=" * 65)
