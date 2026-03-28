"""
ingestion.py  —  Step 00: Data Ingestion Layer
================================================
Reads raw case-level data from the hardcoded sample-inputs folder and
produces a CSV of ONE aggregate row for the entire case, which feeds
directly into feature_engineering.py.

Key change vs previous version
--------------------------------
  BEFORE: one row per account  (N rows per case)
  AFTER : one row per case     (1 row per case)

Each of the 20 feature columns is now computed at case level using the
aggregation strategy most relevant to SAR detection:

  Amounts      → sum / mean / std / max / min across ALL case transactions
  Burst        → max()   — worst-case burst window across any account
  Time-to-exit → min()   — fastest fund-exit across any account
  Fund ratio   → recomputed from case-level inbound / outbound totals
  Velocity     → total txn_count / mean(account_age / 30)
  Counterparty → unique counterparties across ALL case accounts
  Diversity    → Shannon entropy computed over ALL case transactions
  Binary flags → OR  (any() == 1)  across all accounts
  KYC risk     → max() — highest-risk entity drives case risk
  Account age  → min() — youngest account is the weakest KYC link
  Alerts       → sum() — total alert load for the case
  Hist. SAR    → OR  (any() == 1)  across all accounts

Input folder (hardcoded)
-------------------------
  sample-inputs/
      account_transactions.csv   — row-per-transaction
      customer_kyc.json          — list of KYC records (one per customer)
      case_management.json       — single case object
      transaction_alerts.csv     — row-per-alert

Output (hardcoded)
------------------
  data_aggregated.csv  (27 cols: 20 raw features + 7 label columns, 1 row)

Entry point
-----------
  df = run_ingestion()   # returns the single-row aggregated DataFrame
  python ingestion.py    # also writes data_aggregated.csv
"""

import json
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

HIGH_RISK_COUNTRIES = {
    "Afghanistan", "Albania", "Barbados", "Burkina Faso", "Cameroon",
    "Cayman Islands", "Congo", "Croatia", "DR Congo", "Gibraltar",
    "Haiti", "Iran", "Jamaica", "Jordan", "Mali", "Mozambique",
    "Myanmar", "Namibia", "Nicaragua", "Nigeria", "North Korea",
    "Pakistan", "Panama", "Philippines", "Russia", "Senegal",
    "South Africa", "South Sudan", "Syria", "Tanzania", "Trinidad",
    "Tobago", "Turkey", "Uganda", "UAE", "United Arab Emirates",
    "Vanuatu", "Vietnam", "Yemen",
}

KYC_RISK_MAP = {
    "low":    0.20,
    "medium": 0.55,
    "high":   0.85,
}

TYPOLOGY_MAP = {
    "structuring":    "typology_structuring",
    "rapid_movement": "typology_rapid_movement",
    "funnel_account": "typology_funnel_account",
    "trade_based":    "typology_trade_based",
    "shell_company":  "typology_shell_company",
    "round_tripping": "typology_round_tripping",
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

FEATURE_COLS = [
    "total_txn_amount", "avg_txn_amount", "std_txn_amount", "txn_count",
    "max_txn_amount", "min_txn_amount", "burst_score",
    "time_to_first_outbound_minutes", "fund_exit_ratio", "txn_velocity",
    "distinct_counterparties", "counterparty_diversity_score",
    "incoming_sources_count", "international_counterparty_flag", "pep_flag",
    "high_risk_country_flag", "kyc_risk_score", "account_age_days",
    "alert_count", "historical_sar_flag",
]


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _parse_dt(s):
    """Parse ISO-ish datetime string → timezone-aware UTC datetime."""
    if pd.isna(s) or s is None:
        return None
    s = str(s).strip()
    s = re.sub(r"Z$", "+00:00", s)
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        dt = pd.to_datetime(s)
        if hasattr(dt, "to_pydatetime"):
            dt = dt.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _case_counterparty_diversity(case_txns: pd.DataFrame) -> float:
    """
    Shannon entropy-based counterparty diversity score across ALL case
    transactions, normalised to [0, 1].

    Using all transactions (not per-account) gives a case-level view of
    how spread-out the money flows are — the core funnel-account signal.
    """
    cp = case_txns["counterparty_account"].dropna()
    if len(cp) == 0:
        return 0.0
    counts = cp.value_counts(normalize=True)
    entropy = -np.sum(counts * np.log(counts + 1e-9))
    max_entropy = np.log(len(counts))
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def _case_burst_score(case_txns: pd.DataFrame, window_hours: float = 24) -> float:
    """
    Worst-case burst score across ALL case accounts.

    For each account, compute the fraction of its transactions that fall
    within the single densest rolling <window_hours> window, then return
    the maximum across accounts.  Maximum is the right choice because a
    single account exhibiting rapid-movement behaviour is sufficient to
    flag the case.
    """
    account_ids = case_txns["account_id"].unique()
    max_burst = 0.0

    for acc_id in account_ids:
        acc = (
            case_txns[case_txns["account_id"] == acc_id]
            .dropna(subset=["txn_datetime"])
            .sort_values("txn_datetime")
        )
        n = len(acc)
        if n <= 1:
            max_burst = max(max_burst, float(n))
            continue

        times = acc["txn_datetime"].values
        window_ns = np.timedelta64(int(window_hours * 3600), "s")
        best = 1
        j = 0
        for i in range(n):
            while j < n and times[j] - times[i] <= window_ns:
                j += 1
            best = max(best, j - i)

        max_burst = max(max_burst, round(best / n, 4))

    return max_burst


def _time_to_first_outbound(case_txns: pd.DataFrame) -> float:
    """
    Minimum time-to-first-outbound across all case accounts (minutes).

    The minimum captures the most suspicious account — the one that moved
    money out the fastest after receiving it.
    """
    account_ids = case_txns["account_id"].unique()
    min_delta = float("inf")

    for acc_id in account_ids:
        acc = case_txns[case_txns["account_id"] == acc_id]

        inbound_times  = acc[acc["direction"] == "credit"]["txn_datetime"].dropna()
        outbound_times = acc[acc["direction"] == "debit"]["txn_datetime"].dropna()

        if len(outbound_times) == 0:
            continue

        if len(inbound_times) > 0:
            first_in  = inbound_times.min()
            first_out = outbound_times.min()
        else:
            first_in  = acc["txn_datetime"].dropna().min()
            first_out = outbound_times.min()

        delta = max(0.0, (first_out - first_in).total_seconds() / 60)
        min_delta = min(min_delta, delta)

    return round(min_delta if min_delta != float("inf") else 0.0, 2)


# ── MAIN INGESTION FUNCTION ───────────────────────────────────────────────────

def ingest_case(input_dir: str | Path) -> pd.DataFrame:
    """
    Read one case folder and return a single-row DataFrame with all 27
    columns (20 features + 7 labels) ready for feature_engineering.py.

    Parameters
    ----------
    input_dir : path to the case folder containing the four source files.

    Returns
    -------
    pd.DataFrame  — exactly 1 row, columns = FEATURE_COLS + LABEL_COLS
    """
    input_dir = Path(input_dir)

    # ── LOAD SOURCE FILES ─────────────────────────────────────────────────────
    txn_path   = input_dir / "account_transactions.csv"
    kyc_path   = input_dir / "customer_kyc.json"
    case_path  = input_dir / "case_management.json"
    alert_path = input_dir / "transaction_alerts.csv"

    for p in [txn_path, kyc_path, case_path, alert_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file missing: {p}")

    txn_df   = pd.read_csv(txn_path)
    kyc_list = json.loads(kyc_path.read_text(encoding="utf-8"))
    case_obj = json.loads(case_path.read_text(encoding="utf-8"))
    alert_df = pd.read_csv(alert_path)

    # ── NORMALISE TRANSACTIONS ────────────────────────────────────────────────
    txn_df["txn_datetime"] = pd.to_datetime(txn_df["txn_datetime"], utc=True, errors="coerce")
    txn_df["amount"]       = pd.to_numeric(txn_df["amount"], errors="coerce")
    txn_df["direction"]    = txn_df["direction"].str.strip().str.lower()

    # ── CASE METADATA ─────────────────────────────────────────────────────────
    case_created_at     = _parse_dt(case_obj.get("created_at"))
    typology_guess      = str(case_obj.get("typology_guess", "")).lower()
    sar_label           = case_obj.get("sar_worthy")
    primary_account_id  = case_obj.get("primary_account_id")
    related_account_ids = set(case_obj.get("related_account_ids", []))
    all_case_accounts   = ({primary_account_id} | related_account_ids) - {None}

    # Restrict transactions to case accounts only
    case_txns = txn_df[txn_df["account_id"].isin(all_case_accounts)].copy()

    if case_txns.empty:
        raise ValueError(
            f"No transactions found for any account in the case.\n"
            f"  Case accounts : {all_case_accounts}\n"
            f"  Accounts in CSV: {txn_df['account_id'].unique().tolist()}"
        )

    # ── KYC LOOKUP ────────────────────────────────────────────────────────────
    kyc_df = pd.DataFrame(kyc_list)

    def get_kyc_rows(account_ids):
        return kyc_df[kyc_df["account_id"].isin(account_ids)]

    case_kyc = get_kyc_rows(all_case_accounts)

    # ── ALERT LOOKUP ──────────────────────────────────────────────────────────
    # Sum alerts across ALL case accounts
    case_alerts = alert_df[alert_df["account_id"].isin(all_case_accounts)]
    total_alert_count = len(case_alerts)

    # ──────────────────────────────────────────────────────────────────────────
    # FEATURE COMPUTATION  (case-level aggregates)
    # ──────────────────────────────────────────────────────────────────────────

    amounts = case_txns["amount"].dropna()

    # 1-6  Amount statistics — all transactions in the case
    total_txn_amount = float(amounts.sum())
    avg_txn_amount   = float(amounts.mean())
    std_txn_amount   = float(amounts.std(ddof=0)) if len(amounts) > 1 else 0.0
    txn_count        = int(len(amounts))
    max_txn_amount   = float(amounts.max())
    min_txn_amount   = float(amounts.min())

    # 7   Burst score — worst-case account in the case
    burst_score = _case_burst_score(case_txns, window_hours=24)

    # 8   Time to first outbound — fastest account in the case
    time_to_first_outbound_minutes = _time_to_first_outbound(case_txns)

    # 9   Fund exit ratio — case-level inbound vs outbound totals
    total_inbound  = case_txns[case_txns["direction"] == "credit"]["amount"].sum()
    total_outbound = case_txns[case_txns["direction"] == "debit"]["amount"].sum()
    fund_exit_ratio = round(
        float(total_outbound) / (float(total_inbound) + float(total_outbound) + 1e-6), 4
    )

    # 10  Transaction velocity — txns per 30-day month averaged over account ages
    ref_dt = case_created_at if case_created_at else datetime.now(tz=timezone.utc)
    account_ages = []
    for acc_id in all_case_accounts:
        acc_rows = txn_df[txn_df["account_id"] == acc_id]
        if acc_rows.empty:
            continue
        open_date_raw = acc_rows["open_date"].iloc[0]
        if pd.notna(open_date_raw):
            open_dt = _parse_dt(str(open_date_raw))
            age = max(1, (ref_dt - open_dt).days)
        else:
            age = 365
        account_ages.append(age)

    mean_account_age = float(np.mean(account_ages)) if account_ages else 365.0
    txn_velocity = round(txn_count / (mean_account_age / 30), 4)

    # 11  Distinct counterparties — unique across ALL case accounts
    distinct_counterparties = int(case_txns["counterparty_account"].dropna().nunique())

    # 12  Counterparty diversity — Shannon entropy over all case transactions
    counterparty_diversity_score = round(_case_counterparty_diversity(case_txns), 4)

    # 13  Incoming sources count — unique inbound counterparties across case
    incoming_sources_count = int(
        case_txns[case_txns["direction"] == "credit"]["counterparty_account"]
        .dropna()
        .nunique()
    )

    # 14  International counterparty flag — OR across case
    cp_countries = case_txns["counterparty_country"].dropna()
    international_counterparty_flag = int(any(c != "India" for c in cp_countries))

    # 15  PEP flag — OR across case KYC records
    if not case_kyc.empty and "pep_flag" in case_kyc.columns:
        pep_flag = int(case_kyc["pep_flag"].astype(bool).any())
    else:
        pep_flag = 0

    # 16  High-risk country flag — OR across all counterparty countries
    high_risk_country_flag = int(any(c in HIGH_RISK_COUNTRIES for c in cp_countries))

    # 17  KYC risk score — max (highest-risk entity drives case risk)
    if not case_kyc.empty and "risk_rating" in case_kyc.columns:
        kyc_risk_score = float(
            case_kyc["risk_rating"]
            .str.lower()
            .map(KYC_RISK_MAP)
            .fillna(0.55)
            .max()
        )
    else:
        kyc_risk_score = 0.55

    # 18  Account age — min (youngest account = weakest KYC link)
    account_age_days = int(min(account_ages)) if account_ages else 365

    # 19  Alert count — total across case
    alert_count = total_alert_count

    # 20  Historical SAR flag — OR across case KYC records
    if not case_kyc.empty and "previous_sar_count" in case_kyc.columns:
        historical_sar_flag = int(
            (case_kyc["previous_sar_count"].fillna(0).astype(int) > 0).any()
        )
    else:
        historical_sar_flag = 0

    # ── LABELS ────────────────────────────────────────────────────────────────
    sar_worthy_val = int(sar_label) if sar_label is not None else 0

    typo_cols = {c: 0 for c in LABEL_COLS[1:]}
    mapped_col = TYPOLOGY_MAP.get(typology_guess)
    if mapped_col and sar_worthy_val == 1:
        typo_cols[mapped_col] = 1

    # ── ASSEMBLE SINGLE ROW ───────────────────────────────────────────────────
    row = {
        "total_txn_amount":               round(total_txn_amount, 4),
        "avg_txn_amount":                 round(avg_txn_amount, 4),
        "std_txn_amount":                 round(std_txn_amount, 4),
        "txn_count":                      txn_count,
        "max_txn_amount":                 round(max_txn_amount, 4),
        "min_txn_amount":                 round(min_txn_amount, 4),
        "burst_score":                    burst_score,
        "time_to_first_outbound_minutes": time_to_first_outbound_minutes,
        "fund_exit_ratio":                fund_exit_ratio,
        "txn_velocity":                   txn_velocity,
        "distinct_counterparties":        distinct_counterparties,
        "counterparty_diversity_score":   counterparty_diversity_score,
        "incoming_sources_count":         incoming_sources_count,
        "international_counterparty_flag": international_counterparty_flag,
        "pep_flag":                        pep_flag,
        "high_risk_country_flag":          high_risk_country_flag,
        "kyc_risk_score":                  round(kyc_risk_score, 4),
        "account_age_days":                account_age_days,
        "alert_count":                     alert_count,
        "historical_sar_flag":             historical_sar_flag,
        "sar_worthy":                      sar_worthy_val,
        **typo_cols,
    }

    return pd.DataFrame([row], columns=FEATURE_COLS + LABEL_COLS)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

_INPUT_DIR  = Path(__file__).parent / "sample-inputs"
_OUTPUT_CSV = Path(__file__).parent / "data_aggregated.csv"


def run_ingestion(input_dir: str | Path = _INPUT_DIR, output_csv: str | Path = _OUTPUT_CSV) -> pd.DataFrame:
    """
    Main ingestion entry point.

    Reads the four source files from the input folder,
    computes a single case-level aggregate row, writes to output CSV,
    and returns the resulting single-row DataFrame.

    Returns
    -------
    pd.DataFrame  — 1 row, 27 columns (20 features + 7 labels).
    """
    input_dir = Path(input_dir)
    output_csv = Path(output_csv)

    print("=" * 60)
    print("SAR Ingestion Layer — Case-Level Aggregate")
    print("=" * 60)
    print(f"\n  Input  : {input_dir}")
    print(f"  Output : {output_csv}")

    df = ingest_case(input_dir)
    df.to_csv(output_csv, index=False)

    print(f"\n{'=' * 60}")
    print(f"DONE — {output_csv.name} written  (1 row)")
    print(f"  Columns  : {len(df.columns)}  "
          f"({len(FEATURE_COLS)} features + {len(LABEL_COLS)} labels)")
    print(f"  SAR label: {int(df['sar_worthy'].iloc[0])}")
    print()
    print("  Feature snapshot:")
    snapshot_cols = [
        "total_txn_amount", "txn_count", "burst_score",
        "fund_exit_ratio", "alert_count", "kyc_risk_score",
        "distinct_counterparties", "high_risk_country_flag",
    ]
    for col in snapshot_cols:
        print(f"    {col:<38} {df[col].iloc[0]}")
    print("=" * 60)
    print("  Next step: point feature_engineering.py at data_aggregated.csv")

    return df


if __name__ == "__main__":
    run_ingestion()