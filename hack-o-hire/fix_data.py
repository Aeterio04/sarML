"""
fix_data.py  —  Step 03: Data Preparation  (FIXED)
===================================================
Input : hackohire_aggregarted_data.csv  (10,000 rows x 27 cols)
Output: data_fixed.csv                  (~7,000 rows x 38 cols)

Problems addressed (original):
  A. Class imbalance + non-overlapping feature spaces
  B. Multi-label injection (~13% of SAR cases)
  C. Calibrated noise injection
  D. Skew correction

BUGS FIXED vs original
----------------------
  [BUG-1] CRITICAL — alert_count multiplier in near_miss() created a hard gap.
          Original: all_nonsar['alert_count'] = alert * uniform(0.35, 0.65)
          With SAR max alert=15, this caps near-miss at 15*0.65=9.75 → max=9.
          Result: alert_count >= 10 → SAR=1.0 (379 rows, 100% deterministic).
          Fix: change multiplier to uniform(0.60, 1.20) so near-miss alerts
          can reach 10–18, breaking the hard cap.

  [BUG-2] Clip ceilings on non-SAR rows left clean gaps vs SAR tail values.
          burst_score:      non-SAR clipped at 0.85, SAR reaches 0.99 → gap
          fund_exit_ratio:  non-SAR clipped at 0.90, SAR reaches 1.00 → gap
          kyc_risk_score:   non-SAR clipped at 0.85, SAR reaches 0.94 → gap
          account_age_days: non-SAR near_miss(shell) used N(950,280), SAR
                            reaches 2026 → p99 tail is 100% SAR.
          Fix: raise clip ceilings so non-SAR can occupy the SAR tail range.
          Fix: near_miss(shell) account_age now scales from the source SAR row
               rather than drawing from a fixed Normal(950, 280).

  [BUG-3] Noise levels were too low (5% std) to break the gaps created above.
          Even with raised clips, 5% std noise on alert_count adds < ±0.5 alert,
          which cannot bridge a gap of 5+ alert values.
          Fix: increase alert_count noise from 5% → 15% std.
               Add kyc_risk_score and account_age_days to NOISE_COLS.
               Increase burst_score and fund_exit_ratio noise from 5% → 8%.

RETAINED (correct as-is from original)
---------------------------------------
  - Near-miss case construction logic (near_miss() function structure)
  - Multi-label injection logic and counts
  - All original near_miss feature degradation targets (mu, sigma values)
  - Downsampling strategy (stratified by typology)
  - cbrt / log1p transform step
  - All assertion checks

References:
  [FATF-2005]  FATF Money Laundering Typologies 2004-2005
  [FATF-TBML]  FATF Trade-Based Money Laundering, 2006
  [FATF-SHELL] FATF Misuse of Corporate Vehicles, 2006
  [PMLA]       Prevention of Money Laundering Act 2002 (amended 2023)
  [RBI-KYC]    RBI KYC Master Direction 2016 (updated 2023)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── LOAD ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(r'C:\Users\shriy\hack-o-hire\hackohire_aggregarted_data.csv')
print(f"Input  : {df.shape[0]:,} rows x {df.shape[1]} cols  |  SAR rate: {df['sar_worthy'].mean():.3f}")

LABEL_COLS  = ['sar_worthy', 'typology_structuring', 'typology_rapid_movement',
               'typology_funnel_account', 'typology_trade_based',
               'typology_shell_company', 'typology_round_tripping']
TYPO_COLS   = LABEL_COLS[1:]
BINARY_FEAT = ['international_counterparty_flag', 'pep_flag',
               'high_risk_country_flag', 'historical_sar_flag']
FEAT_COLS   = [c for c in df.columns if c not in LABEL_COLS]

CBRT_COLS  = ['total_txn_amount', 'avg_txn_amount', 'std_txn_amount',
               'max_txn_amount', 'min_txn_amount']
LOG1P_COLS = ['time_to_first_outbound_minutes', 'txn_count', 'account_age_days',
               'incoming_sources_count', 'txn_velocity', 'distinct_counterparties']


# ══════════════════════════════════════════════════════════════════════════════
# STEP A — REBUILD NON-SAR CLASS AS NEAR-MISS CASES
#
# Near-miss non-SAR cases are built by sampling from SAR cases and degrading
# the typology-defining features to just-below-threshold values.
# This places non-SAR cases inside the same feature space as SAR cases,
# creating genuine classification difficulty.
#
# [BUG-1 FIX] alert_count multiplier changed from uniform(0.35, 0.65) to
#             uniform(0.60, 1.20).
#             Old multiplier: SAR alert=10 → near-miss max=6.5 (rounds to 6-7)
#             New multiplier: SAR alert=10 → near-miss = 6–12 (realistic range)
#             This breaks the hard gap at alert=10 that made 379 rows 100% SAR.
#
# [BUG-2 FIX] near_miss(shell) account_age now scales from source SAR row value
#             rather than drawing from N(950, 280). Original generation produced
#             near-miss max age ~1800 while SAR had cases up to 2026, making
#             the p99 tail of account_age_days 100% SAR.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP A: Rebuild non-SAR as near-miss cases")
print("="*60)

sar_df = df[df['sar_worthy'] == 1].copy()


def near_miss(typology, n, degrade, note):
    """
    Sample n cases from a typology group, degrade defining features to
    sub-threshold values with noise, label all as non-SAR.
    degrade: dict {col: (target_mean, target_std)}
    """
    src = sar_df[sar_df[typology] == 1].copy()
    if len(src) < n:
        src = src.sample(n=n, replace=True, random_state=42)
    else:
        src = src.sample(n=n, random_state=42)
    result = src[FEAT_COLS].copy().reset_index(drop=True)
    for col, (mu, sigma) in degrade.items():
        result[col] = np.clip(
            mu + np.random.normal(0, sigma, n),
            result[col].min() * 0.05,
            result[col].max() * 0.99
        )
    for lc in LABEL_COLS:
        result[lc] = 0
    print(f"  {note}: {n} near-miss cases")
    return result


ns_structuring = near_miss('typology_structuring', 650,
    {'burst_score':     (0.43, 0.07),
     'fund_exit_ratio': (0.52, 0.09),
     'time_to_first_outbound_minutes': (900, 350)},
    "Near-miss structuring (moderate burst+exit, slower)")

ns_rapid = near_miss('typology_rapid_movement', 800,
    {'fund_exit_ratio': (0.47, 0.10),
     'time_to_first_outbound_minutes': (600, 300)},
    "Near-miss rapid movement (high burst, incomplete exit)")

ns_funnel = near_miss('typology_funnel_account', 550,
    {'fund_exit_ratio': (0.44, 0.09),
     'burst_score':     (0.40, 0.08)},
    "Near-miss funnel (many sources, funds retained)")

ns_tbml = near_miss('typology_trade_based', 500,
    {'fund_exit_ratio': (0.54, 0.11),
     'kyc_risk_score':  (0.56, 0.10)},
    "Near-miss TBML (high value, domestic counterparty)")
ns_tbml['high_risk_country_flag'] = 0
ns_tbml['international_counterparty_flag'] = np.random.choice([0, 1], 500, p=[0.35, 0.65])

# [BUG-2 FIX] Shell near-miss: account_age now scales from source SAR row
# Original used N(950, 280) which capped near-miss age at ~1800 while
# SAR accounts reach 2026. Now account_age = src_age * uniform(0.85, 1.25)
# which follows the source distribution and allows near-miss to reach SAR extremes.
ns_shell_src = sar_df[sar_df['typology_shell_company'] == 1].sample(n=500, random_state=42)
ns_shell = ns_shell_src[FEAT_COLS].copy().reset_index(drop=True)
ns_shell['counterparty_diversity_score'] = np.clip(
    0.46 + np.random.normal(0, 0.10, 500),
    ns_shell['counterparty_diversity_score'].min() * 0.05,
    ns_shell['counterparty_diversity_score'].max() * 0.99
)
ns_shell['burst_score'] = np.clip(
    0.38 + np.random.normal(0, 0.09, 500),
    ns_shell['burst_score'].min() * 0.05,
    ns_shell['burst_score'].max() * 0.99
)
# Scale account_age from source row rather than fixed Normal
ns_shell['account_age_days'] = (
    ns_shell_src['account_age_days'].values * np.random.uniform(0.85, 1.25, 500)
).clip(35)
for lc in LABEL_COLS:
    ns_shell[lc] = 0
print(f"  Near-miss shell (high volume, normal diversity, age scaled from source): 500 near-miss cases")

ns_rt = near_miss('typology_round_tripping', 500,
    {'time_to_first_outbound_minutes': (4500, 1200),
     'fund_exit_ratio': (0.63, 0.10),
     'burst_score':     (0.36, 0.09)},
    "Near-miss round-tripping (slow exit, incomplete)")

all_nonsar = pd.concat(
    [ns_structuring, ns_rapid, ns_funnel, ns_tbml, ns_shell, ns_rt],
    ignore_index=True
)

# Clip to physically valid ranges
# [BUG-2 FIX] Raised clip ceilings so non-SAR can occupy the SAR tail range,
# breaking the hard gaps that made extreme feature values 100% deterministic.
#   Original burst_score clip:     (0.05, 0.85) → SAR reaches 0.99, gap 0.85-0.99
#   Original fund_exit_ratio clip: (0.13, 0.90) → SAR reaches 1.00, gap 0.90-1.00
#   Original kyc_risk_score clip:  (0.10, 0.85) → SAR reaches 0.94, gap 0.85-0.94
all_nonsar['burst_score']                  = all_nonsar['burst_score'].clip(0.05, 0.97)
all_nonsar['fund_exit_ratio']              = all_nonsar['fund_exit_ratio'].clip(0.10, 0.98)
all_nonsar['kyc_risk_score']               = all_nonsar['kyc_risk_score'].clip(0.10, 0.93)
all_nonsar['counterparty_diversity_score'] = all_nonsar['counterparty_diversity_score'].clip(0.01, 0.89)
all_nonsar['time_to_first_outbound_minutes'] = all_nonsar['time_to_first_outbound_minutes'].clip(5, 39000)
all_nonsar['account_age_days']             = all_nonsar['account_age_days'].clip(35)

# [BUG-1 FIX] alert_count multiplier changed from uniform(0.35, 0.65) to uniform(0.60, 1.20).
# Old range produced near-miss alert_count max = 9 regardless of SAR alert values,
# creating an absolute gap: alert_count >= 10 → SAR=1.0 (527 rows, 100% deterministic).
# New range allows near-miss to exceed the SAR mean, producing genuine overlap.
all_nonsar['alert_count'] = (
    all_nonsar['alert_count'] * np.random.uniform(0.60, 1.20, len(all_nonsar))
).clip(0).round().astype(int)

print(f"\n  Total near-miss non-SAR: {len(all_nonsar):,}")
print(f"  Feature overlap after rebuild:")
for f in ['burst_score', 'fund_exit_ratio', 'kyc_risk_score']:
    sar_min, sar_max = sar_df[f].min(), sar_df[f].max()
    ns_min,  ns_max  = all_nonsar[f].min(), all_nonsar[f].max()
    overlap = min(sar_max, ns_max) - max(sar_min, ns_min)
    print(f"    {f}: SAR [{sar_min:.3f},{sar_max:.3f}]  "
          f"non-SAR [{ns_min:.3f},{ns_max:.3f}]  "
          f"overlap={max(0,overlap):.3f}  {'OK' if overlap > 0.10 else 'WARN'}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP B — DOWNSAMPLE SAR + MULTI-LABEL INJECTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP B: Downsample SAR + multi-label injection")
print("="*60)

TARGET_SAR = 3500
parts = []
for t in TYPO_COLS:
    grp = sar_df[sar_df[t] == 1]
    n   = max(1, round(len(grp) / len(sar_df) * TARGET_SAR))
    parts.append(grp.sample(n=min(n, len(grp)), random_state=42))
sar_sampled = pd.concat(parts).drop_duplicates().reset_index(drop=True)
if len(sar_sampled) > TARGET_SAR:
    sar_sampled = sar_sampled.sample(n=TARGET_SAR, random_state=42)
elif len(sar_sampled) < TARGET_SAR:
    gap   = TARGET_SAR - len(sar_sampled)
    extra = sar_df.drop(sar_sampled.index, errors='ignore').sample(
        n=min(gap, len(sar_df) - len(sar_sampled)), random_state=42)
    sar_sampled = pd.concat([sar_sampled, extra]).reset_index(drop=True)

print(f"  SAR downsampled: {len(sar_df):,} → {len(sar_sampled):,}")


def inject(df_in, primary, secondary, cond, n, note):
    elig   = df_in[(df_in[primary] == 1) & cond & (df_in[secondary] == 0)].index
    chosen = np.random.choice(elig, size=min(n, len(elig)), replace=False)
    df_in.loc[chosen, secondary] = 1
    print(f"  {note}: {len(chosen)} cases")
    return df_in

sar_sampled = inject(sar_sampled, 'typology_structuring', 'typology_rapid_movement',
    (sar_sampled['fund_exit_ratio'] > 0.80) & (sar_sampled['burst_score'] > 0.70), 140,
    "Structuring + Rapid Movement  [FATF-2005 §2.4]")
sar_sampled = inject(sar_sampled, 'typology_shell_company', 'typology_trade_based',
    (sar_sampled['high_risk_country_flag'] == 1) & (sar_sampled['kyc_risk_score'] > 0.70), 100,
    "Shell + Trade-Based ML  [FATF-SHELL §3.1]")
sar_sampled = inject(sar_sampled, 'typology_funnel_account', 'typology_rapid_movement',
    (sar_sampled['time_to_first_outbound_minutes'] < 200) & (sar_sampled['fund_exit_ratio'] > 0.70), 100,
    "Funnel + Rapid Movement  [FATF-2005 §4.1]")
sar_sampled = inject(sar_sampled, 'typology_round_tripping', 'typology_shell_company',
    (sar_sampled['high_risk_country_flag'] == 1) & (sar_sampled['account_age_days'] > 800), 70,
    "Round-trip + Shell  [FATF-SHELL §4.2]")

sar_sampled['sar_worthy'] = 1
multi = (sar_sampled[TYPO_COLS].sum(axis=1) > 1).sum()
print(f"  Multi-label cases: {multi}  ({multi / len(sar_sampled) * 100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP C — COMBINE AND ADD CALIBRATED NOISE
#
# [BUG-3 FIX] Noise levels increased and coverage expanded:
#   alert_count:      5% std → 15% std
#     Old noise ±0.13 alert was far too small to bridge the gap at alert=10.
#     New noise ±0.40 alert creates genuine overlap between SAR and non-SAR.
#   burst_score:      5% std → 8% std
#   fund_exit_ratio:  5% std → 8% std
#   kyc_risk_score:   added to NOISE_COLS at 8% std
#   account_age_days: added to NOISE_COLS at 10% std
#     Old code never added noise to account_age, leaving the SAR p99 tail
#     (age > 1900) 100% deterministic. Adding noise breaks this.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP C: Combine and add calibrated noise")
print("="*60)

df_combined = pd.concat([sar_sampled, all_nonsar], ignore_index=True)

# noise_config: {col: fraction_of_col_std}
noise_config = {
    'burst_score'        : 0.08,   # was 0.05
    'fund_exit_ratio'    : 0.08,   # was 0.05
    'kyc_risk_score'     : 0.08,   # NEW — was not in noise list
    'counterparty_diversity_score': 0.05,
    'txn_velocity'       : 0.05,
    'alert_count'        : 0.15,   # was 0.05 — KEY FIX
    'account_age_days'   : 0.10,   # NEW — was not in noise list
}

for col, frac in noise_config.items():
    noise = np.random.normal(0, df_combined[col].std() * frac, len(df_combined))
    df_combined[col] = df_combined[col] + noise

# Clip all features back to valid ranges after noise
df_combined['burst_score']                  = df_combined['burst_score'].clip(0.05, 0.99)
df_combined['fund_exit_ratio']              = df_combined['fund_exit_ratio'].clip(0.10, 1.00)
df_combined['kyc_risk_score']               = df_combined['kyc_risk_score'].clip(0.01, 0.94)
df_combined['counterparty_diversity_score'] = df_combined['counterparty_diversity_score'].clip(0.01, 0.89)
df_combined['alert_count']                  = df_combined['alert_count'].clip(0).round().astype(int)
df_combined['txn_velocity']                 = df_combined['txn_velocity'].clip(0.01)
df_combined['account_age_days']             = df_combined['account_age_days'].clip(35)

# Shuffle
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"  Combined: {len(df_combined):,} rows  |  SAR rate: {df_combined['sar_worthy'].mean():.3f}")
print(f"  Noise applied to {len(noise_config)} features")
print(f"  Noise fractions: { {k: f'{v*100:.0f}%' for k,v in noise_config.items()} }")


# ── POST-NOISE DETERMINISM CHECK ──────────────────────────────────────────────
print(f"\n  Post-noise determinism check:")
y = df_combined['sar_worthy'].values

for col in ['alert_count', 'fund_exit_ratio', 'burst_score', 'kyc_risk_score']:
    vals = sorted(df_combined[col].unique()) if df_combined[col].nunique() <= 20 else None
    if vals is not None:
        perfect = [(v, df_combined[df_combined[col]==v]['sar_worthy'].mean(),
                    (df_combined[col]==v).sum())
                   for v in vals
                   if (df_combined[col]==v).sum() >= 30]
        det = [(v, r, n) for v, r, n in perfect if r >= 0.99 or r <= 0.01]
        if det:
            for v, r, n in det:
                print(f"    {col}=={v}: n={n}, SAR={r:.3f}  [still deterministic]")
        else:
            print(f"    {col}: no perfectly deterministic values (n>=30)  OK")
    else:
        # continuous: check tails
        p99 = df_combined[col].quantile(0.99)
        p01 = df_combined[col].quantile(0.01)
        high = df_combined[df_combined[col] >= p99]
        low  = df_combined[df_combined[col] <= p01]
        h_rate = high['sar_worthy'].mean()
        l_rate = low['sar_worthy'].mean()
        h_ok = '  [still deterministic]' if h_rate >= 0.99 else 'OK'
        l_ok = '  [still deterministic]' if l_rate <= 0.01 else 'OK'
        print(f"    {col} p99 ({p99:.3f}): n={len(high)}, SAR={h_rate:.3f}  {h_ok}")
        print(f"    {col} p01 ({p01:.3f}): n={len(low)}, SAR={l_rate:.3f}  {l_ok}")

from sklearn.metrics import roc_auc_score
print(f"\n  Single-feature AUCs after noise (target: no feature > 0.85 alone):")
for col in ['alert_count', 'fund_exit_ratio', 'burst_score', 'kyc_risk_score']:
    auc = roc_auc_score(y, df_combined[col].values)
    flag = '  <- WARNING: high single-feature AUC' if auc > 0.90 else ''
    print(f"    {col:<35} AUC = {auc:.4f}{flag}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP D — SKEW CORRECTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP D: Skew correction")
print("="*60)

for col in CBRT_COLS:
    sk_b = round(df_combined[col].skew(), 3)
    df_combined[f'{col}_cbrt'] = np.cbrt(df_combined[col])
    sk_a = round(df_combined[f'{col}_cbrt'].skew(), 3)
    print(f"  cbrt  {col:<35} {sk_b:>7} → {sk_a:>7}")

for col in LOG1P_COLS:
    sk_b = round(df_combined[col].skew(), 3)
    df_combined[f'{col}_log'] = np.log1p(df_combined[col].clip(0))
    sk_a = round(df_combined[f'{col}_log'].skew(), 3)
    print(f"  log1p {col:<35} {sk_b:>7} → {sk_a:>7}")


# ── COLUMN ORDER & INTEGRITY CHECKS ──────────────────────────────────────────
CBRT_FEAT  = [f'{c}_cbrt' for c in CBRT_COLS]
LOG1P_FEAT = [f'{c}_log'  for c in LOG1P_COLS]
final_cols = FEAT_COLS + CBRT_FEAT + LOG1P_FEAT + LABEL_COLS
df_out     = df_combined[final_cols]

bad_a = df_out[(df_out['sar_worthy'] == 1) & (df_out[TYPO_COLS].sum(axis=1) == 0)]
bad_b = df_out[(df_out['sar_worthy'] == 0) & (df_out[TYPO_COLS].sum(axis=1) > 0)]
nulls = df_out.isnull().sum().sum()
infs  = np.isinf(df_out.select_dtypes(include=[float])).sum().sum()
assert len(bad_a) == 0, f"FAIL: {len(bad_a)} SAR=1 with no typology"
assert len(bad_b) == 0, f"FAIL: {len(bad_b)} SAR=0 with typology"
assert nulls == 0,      f"FAIL: {nulls} nulls"
assert infs  == 0,      f"FAIL: {infs} infs"

print(f"\n  All integrity assertions passed.")


# ── SAVE ──────────────────────────────────────────────────────────────────────
out = r'C:\Users\shriy\hack-o-hire\data_fixed.csv'
df_out.to_csv(out, index=False)

print("\n" + "="*60)
print("DONE — data_fixed.csv")
print(f"  Rows    : {len(df_out):,}  (3,500 SAR + 3,500 near-miss non-SAR)")
print(f"  Columns : {len(df_out.columns)}  (20 orig + 5 cbrt + 6 log + 7 labels)")
print(f"  SAR rate: {df_out['sar_worthy'].mean():.3f}")
print(f"  Multi-label SAR cases: {multi}  ({multi / df_out['sar_worthy'].sum() * 100:.1f}%)")
print(f"  Nulls: {nulls}  |  Infs: {infs}  |  Label integrity: PASS")
print(f"  Typology distribution:")
for t in TYPO_COLS:
    print(f"    {t}: {df_out[t].sum()}")
print("="*60)