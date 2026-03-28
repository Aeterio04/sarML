"""
testing/nodes.py — LangGraph Node Wrappers
==========================================
Thin wrappers that adapt each agent's entry point to the LangGraph
node contract (receives state dict, returns partial state dict).

Also defines the three conditional edge routing functions.
"""

import logging
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger("nodes")


# ── Agent 1 ───────────────────────────────────────────────────────────────────

def node_agent1(state: dict) -> dict:
    """
    Always run Agent 1 - do not skip even if sar_worthy is pre-populated.
    """
    from agents.agent1_ingestion import agent1_ingestion
    return agent1_ingestion(state)


def route_after_agent1(state: dict) -> str:
    """Non-SAR cases exit immediately — no LLM ever called."""
    return "masking" if state.get("sar_worthy", False) else "end"


# ── Masking Agent ─────────────────────────────────────────────────────────────

def node_masking(state: dict) -> dict:
    """
    Mask PII in structured_case before any LLM agent sees it.
    Replaces counterparty_name and counterparty_account with opaque tokens.
    Token map stored in pii_mask_map table in RDS.
    """
    from agents.masking_agent.masking_agent import mask_structured_case, get_connection, ensure_table
    from config import VECTOR_DB_URL

    case_id         = state.get("case_id", "UNKNOWN")
    structured_case = state.get("structured_case", {})

    if not structured_case:
        logger.info(f"Masking [{case_id}]: structured_case empty, skipping")
        return state

    if not VECTOR_DB_URL:
        logger.warning(f"Masking [{case_id}]: VECTOR_DB_URL not set — skipping PII masking")
        return state

    try:
        conn = get_connection(_parse_db_url(VECTOR_DB_URL))
        ensure_table(conn)
        masked = mask_structured_case(structured_case, case_id, conn)
        conn.close()
        logger.info(f"Masking [{case_id}]: PII masked successfully")
        return {**state, "structured_case": masked}
    except Exception as e:
        logger.warning(f"Masking [{case_id}]: masking failed ({e}) — continuing unmasked")
        return state


def _parse_db_url(url: str) -> dict:
    """
    Parse a postgres:// connection string into a psycopg2 config dict.
    Resolves sslrootcert to an absolute path so it works regardless of
    the working directory the process was launched from.
    """
    from urllib.parse import urlparse, parse_qs
    from pathlib import Path

    p      = urlparse(url)
    params = parse_qs(p.query)
    cfg = {
        "host":     p.hostname,
        "port":     p.port or 5432,
        "dbname":   p.path.lstrip("/"),
        "user":     p.username,
        "password": p.password,
        "sslmode":  params.get("sslmode", ["require"])[0],
    }
    if "sslrootcert" in params:
        cert_path = params["sslrootcert"][0]
        # Resolve relative paths against the project root (parent of testing/)
        resolved = Path(cert_path)
        if not resolved.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            resolved = project_root / cert_path.lstrip("./")
        cfg["sslrootcert"] = str(resolved)
    return cfg


# ── Agent 2 ───────────────────────────────────────────────────────────────────

def node_agent2(state: dict) -> dict:
    from agents.agent2_planner import agent2_planner
    return agent2_planner(state)


def route_after_agent2(state: dict) -> str:
    """Route to Agent 4 if enrichment needed, else straight to Agent 3."""
    return "enrichment" if state.get("requires_enrichment", False) else "typology"


# ── Agent 3 ───────────────────────────────────────────────────────────────────

def node_agent3(state: dict) -> dict:
    from agents.agent3_typology import run_agent3
    result = run_agent3(state)
    return {**state, **result}


# ── Agent 4 ───────────────────────────────────────────────────────────────────

def node_agent4(state: dict) -> dict:
    """
    Wraps Agent4EnrichmentAgent.run() into the LangGraph state contract.

    Builds the feat_series from quantified_indicators (best available proxy
    when data_engineered.csv isn't directly accessible at runtime).
    Extracts entity names from structured_case transactions (unmasked
    counterparty_name values — masking runs after this in the flow).
    Extracts countries from transactions counterparty_country.
    """
    import pandas as pd
    import sys
    from pathlib import Path

    _AGENT4_DIR = Path(__file__).resolve().parent.parent / "agents" / "agent_4"
    if str(_AGENT4_DIR) not in sys.path:
        sys.path.insert(0, str(_AGENT4_DIR))

    from agent4_enrichment import Agent4EnrichmentAgent

    case_id         = state.get("case_id", "UNKNOWN")
    structured_case = state.get("structured_case", {})
    transactions    = structured_case.get("transactions", [])
    qi              = state.get("quantified_indicators", {})

    # Build a feat_series from quantified_indicators as best proxy
    feat_series = pd.Series({
        "total_txn_amount":              qi.get("total_amount", 0),
        "txn_count":                     qi.get("transaction_count", 0),
        "fund_exit_ratio":               qi.get("fund_exit_ratio", 0.0),
        "burst_score":                   qi.get("avg_velocity_score", 0.0),
        "alert_count":                   qi.get("unique_rules_triggered", 0),
        "alert_density":                 qi.get("pct_flagged", 0.0) / 100,
        "alert_tier":                    1,
        "kyc_x_alert":                   0.0,
        "high_risk_country_flag":        int(qi.get("has_fatf_exposure", False)),
        "international_counterparty_flag": int(qi.get("has_fatf_exposure", False)),
        "pep_flag":                      int(structured_case.get("customer", {}).get("pep_flag", False)),
        "kyc_risk_score":                {"LOW": 0.2, "MEDIUM": 0.5, "HIGH": 0.8}.get(
                                             structured_case.get("customer", {}).get("risk_rating", "LOW"), 0.2),
        "kyc_risk_tier":                 {"LOW": 1, "MEDIUM": 2, "HIGH": 3}.get(
                                             structured_case.get("customer", {}).get("risk_rating", "LOW"), 1),
        "historical_sar_flag":           int(structured_case.get("customer", {}).get("previous_sar_count", 0) > 0),
        "high_risk_combined":            int(qi.get("has_fatf_exposure", False)),
        # Remaining features default to 0 — ML falls back to rule-based scoring
        "total_txn_amount_cbrt":         0, "avg_txn_amount_cbrt": 0,
        "std_txn_amount_cbrt":           0, "txn_count_log": 0,
        "txn_amount_cv":                 qi.get("case_cv", 0.0),
        "max_to_avg_txn_ratio":          0, "burst_per_age": 0,
        "time_to_first_outbound_minutes_log": 0, "txn_velocity_log": 0,
        "distinct_counterparties_log":   0, "incoming_sources_count_log": 0,
        "counterparty_diversity_score":  0, "counterparty_to_txn_ratio": 0,
        "incoming_to_outgoing_ratio":    0, "burst_x_exit": 0,
        "hr_country_x_exit":             0, "pep_x_intl": 0,
        "sar_history_x_kyc":             0, "fund_exit_tier": 0,
        "binary_risk_flag_count":        0,
    })

    # Extract entity names and countries from transactions
    entities  = list({t.get("counterparty_name", "") for t in transactions if t.get("counterparty_name")})
    countries = list({t.get("counterparty_country", "") for t in transactions if t.get("counterparty_country")})

    # Build agent1 and agent3 output dicts from state
    agent1_output = {
        "sar_worthy":  state.get("sar_worthy", True),
        "confidence":  state.get("confidence_score", 0.5),
        "typologies":  [{"typology": f"typology_{state.get('typology','').lower().replace(' ','_')}",
                         "confidence": state.get("confidence_score", 0.5)}],
    }
    agent3_output = {
        "predicted_typology": f"typology_{state.get('typology','').lower().replace(' ','_')}",
        "rule_triggers":      [r.get("rule_id") for r in state.get("triggered_rules", [])],
        "quantified_indicators": qi,
        "case_summary": {
            "flagged_transactions":  qi.get("flagged_txn_count", 0),
            "total_flagged_amount":  qi.get("flagged_amount", 0.0),
        },
    }

    try:
        agent4 = Agent4EnrichmentAgent()
        result = agent4.run(
            case_id       = case_id,
            feat_series   = feat_series,
            agent1_output = agent1_output,
            agent3_output = agent3_output,
            entities      = entities,
            countries     = countries,
        )

        # Map Agent 4 output to state["enrichment_data"] contract
        enrichment_data = {
            "sanctions_hits":    result.get("external_intelligence", {}).get(
                                     "sanctions", {}).get("entity_results", []) if result.get("external_intelligence") else [],
            "adverse_news":      result.get("external_intelligence", {}).get(
                                     "negative_news", {}).get("entity_results", []) if result.get("external_intelligence") else [],
            "regulatory_flags":  result.get("external_intelligence", {}).get(
                                     "regulatory_advisories", []) if result.get("external_intelligence") else [],
            "strength_label":    result.get("strength_label", "MEDIUM"),
            "priority_score":    result.get("priority_score", 0.5),
            "recommended_priority": result.get("recommended_priority", "MEDIUM"),
            "evidence_gaps":     result.get("evidence_gaps", {}),
        }

        logger.info(f"Agent 4 [{case_id}]: strength={result.get('strength_label')} "
                    f"priority={result.get('recommended_priority')}")
        return {**state, "enrichment_data": enrichment_data}

    except Exception as e:
        logger.error(f"Agent 4 [{case_id}]: failed — {e}", exc_info=True)
        return {**state, "enrichment_data": {}, "error_log": [f"Agent4 failed: {e}"]}


# ── Agent 5 ───────────────────────────────────────────────────────────────────

def node_agent5(state: dict) -> dict:
    from agents.agent5_narrative import agent5_narrative
    result = agent5_narrative(state)
    # Increment revision_count on each pass through Agent 5
    return {
        **state,
        **result,
        "revision_count": state.get("revision_count", 0) + (1 if state.get("revision_count", 0) > 0 else 0),
    }


# ── Agent 6 ───────────────────────────────────────────────────────────────────

def node_agent6(state: dict) -> dict:
    from agents.agent6_evaluation import Agent6ComplianceJudge
    judge  = Agent6ComplianceJudge()
    result = judge.evaluate(state)
    # Increment revision_count so Agent 5 knows it's a revision loop
    return {**result, "revision_count": state.get("revision_count", 0) + 1}


def route_after_agent6(state: dict) -> str:
    """
    Pass → END.
    Fail + revisions remaining → back to Agent 5.
    Fail + max revisions hit → END (flagged for human review).
    """
    from agents.agent6_evaluation import MAX_REVISIONS, compliance_router
    route = compliance_router(state)
    return "revision" if route == "revision" else "end"
