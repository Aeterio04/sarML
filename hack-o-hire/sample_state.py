"""
testing/sample_state.py — Sample Initial State for Pipeline Testing
====================================================================
Uses data_engineered.csv for Agent 1 ML pipeline (NOT pre-populated).
Agent 1 will classify the case based on engineered features.
"""

SAMPLE_STATE = {

    # ── Call 1 inputs ─────────────────────────────────────────────────────────
    "case_id":         "CASE-2024-0104",
    "s3_bucket":       "",
    "s3_prefix":       "",
    "transactions_csv": "data_engineered.csv",  # Use engineered features for Agent 1 ML

    # ── Call 1 outputs (NOT pre-populated - let Agent 1 classify) ────────────────────────────────────────
    "sar_worthy":       None,
    "confidence_score": 0.0,
    "typology":         "",
    "risk_score":       0.0,

    # ── Call 2 input — structured_case ────────────────────────────────────────
    "structured_case": {
        "customer": {
            "risk_rating":        "HIGH",
            "customer_type":      "corporate",
            "nationality":        "IN",
            "pep_flag":           False,
            "previous_sar_count": 1,
        },
        "accounts": [
            {
                "account_id":        "ACC-TBM-001",
                "account_type":      "current",
                "account_age_days":  4140,
                "international_txn": True,
            }
        ],
        "transactions": [
            {
                "txn_id":               "TXN-4001",
                "txn_date":             "2024-06-01T10:00:00Z",
                "txn_type":             "WIRE_OUT",
                "amount":               500000.0,
                "currency":             "INR",
                "channel":              "SWIFT",
                "counterparty_name":    "Lagos Trade Partners Ltd",
                "counterparty_account": "NG-8800551",
                "counterparty_country": "NG",
                "is_high_value":        False,
                "velocity_score":       0.72,
            },
            {
                "txn_id":               "TXN-4002",
                "txn_date":             "2024-06-05T11:30:00Z",
                "txn_type":             "WIRE_OUT",
                "amount":               1000000.0,
                "currency":             "INR",
                "channel":              "SWIFT",
                "counterparty_name":    "Lagos Trade Partners Ltd",
                "counterparty_account": "NG-8800551",
                "counterparty_country": "NG",
                "is_high_value":        True,
                "velocity_score":       0.78,
            },
            {
                "txn_id":               "TXN-4003",
                "txn_date":             "2024-06-10T09:15:00Z",
                "txn_type":             "WIRE_OUT",
                "amount":               500000.0,
                "currency":             "INR",
                "channel":              "SWIFT",
                "counterparty_name":    "Lagos Trade Partners Ltd",
                "counterparty_account": "NG-8800551",
                "counterparty_country": "NG",
                "is_high_value":        False,
                "velocity_score":       0.74,
            },
            {
                "txn_id":               "TXN-4004",
                "txn_date":             "2024-06-14T14:45:00Z",
                "txn_type":             "WIRE_OUT",
                "amount":               1000000.0,
                "currency":             "INR",
                "channel":              "SWIFT",
                "counterparty_name":    "Lagos Trade Partners Ltd",
                "counterparty_account": "NG-8800551",
                "counterparty_country": "NG",
                "is_high_value":        True,
                "velocity_score":       0.81,
            },
            {
                "txn_id":               "TXN-4005",
                "txn_date":             "2024-06-18T10:00:00Z",
                "txn_type":             "WIRE_OUT",
                "amount":               500000.0,
                "currency":             "INR",
                "channel":              "SWIFT",
                "counterparty_name":    "Lagos Trade Partners Ltd",
                "counterparty_account": "NG-8800551",
                "counterparty_country": "NG",
                "is_high_value":        False,
                "velocity_score":       0.76,
            },
            {
                "txn_id":               "TXN-4006",
                "txn_date":             "2024-06-21T16:20:00Z",
                "txn_type":             "WIRE_OUT",
                "amount":               1000000.0,
                "currency":             "INR",
                "channel":              "SWIFT",
                "counterparty_name":    "Lagos Trade Partners Ltd",
                "counterparty_account": "NG-8800551",
                "counterparty_country": "NG",
                "is_high_value":        True,
                "velocity_score":       0.85,
            },
        ],
        "banks": [
            {
                "bank_name":  "Axis Bank",
                "country":    "IN",
                "swift_code": "AXISINBB",
            }
        ],
    },

    # ── Downstream fields ─────────────────────────────────────────────────────
    "plan":                  {},
    "requires_enrichment":   False,
    "triggered_rules":       [],
    "quantified_indicators": {},
    "cognitive_event_flow":  {},
    "enrichment_data":       {},
    "sar_draft":             "",
    "sar_context_tree":      {},
    "reasoning_traces":      [],
    "compliance_passed":     False,
    "compliance_issues":     [],
    "quality_score":         0.0,
    "revision_count":        0,
    "error_log":             [],
}
