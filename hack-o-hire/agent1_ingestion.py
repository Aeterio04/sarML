"""
agents/agent1_ingestion.py — Agent 1: Case Ingestion & ML Classification
==========================================================================
Uses the trained model from agent1_fixed/model_fixed.pkl to classify cases.

This agent:
1. Loads data_engineered.csv (ML features)
2. Loads the trained XGBoost model
3. Classifies the case as SAR-worthy and assigns typology
4. Returns classification results to the pipeline
"""

import logging
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import joblib

# Setup paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from state import SARAgentState

logger = logging.getLogger(__name__)

# Model path
MODEL_PATH = _PROJECT_ROOT / "agent1_fixed" / "model_fixed.pkl"


def _ts() -> str:
    """Timestamp for logging."""
    return datetime.now(timezone.utc).isoformat()


def agent1_ingestion(state: SARAgentState) -> SARAgentState:
    """
    Agent 1: Ingestion and ML Classification
    
    Reads engineered features from CSV, classifies using trained model.
    
    Args:
        state: SARAgentState with:
            - case_id: str
            - transactions_csv: path to data_engineered.csv
            
    Returns:
        Updated state with:
            - sar_worthy: bool
            - confidence_score: float
            - typology: str
            - risk_score: float
    """
    case_id = state.get("case_id", f"CASE-UNKNOWN-{_ts()}")
    logger.info(f"Agent 1 — starting ingestion for {case_id}")
    
    errors = []
    
    # Get data source
    transactions_csv = state.get("transactions_csv", "")
    if not transactions_csv:
        msg = f"[{_ts()}] Agent1 [{case_id}]: No transactions_csv provided"
        logger.error(msg)
        errors.append(msg)
        return _error_fallback(state, errors)
    
    # Find the CSV file
    csv_path = Path(transactions_csv)
    if not csv_path.exists():
        # Try relative to project root
        csv_path = _PROJECT_ROOT / transactions_csv
    if not csv_path.exists():
        msg = f"[{_ts()}] Agent1 [{case_id}]: CSV not found: {transactions_csv}"
        logger.error(msg)
        errors.append(msg)
        return _error_fallback(state, errors)
    
    logger.info(f"Agent 1 [{case_id}] — loading data from {csv_path}")
    
    try:
        # Load the trained model
        if not MODEL_PATH.exists():
            msg = f"[{_ts()}] Agent1 [{case_id}]: Model not found at {MODEL_PATH}. Run training first."
            logger.error(msg)
            errors.append(msg)
            return _error_fallback(state, errors)
        
        logger.info(f"Agent 1 [{case_id}] — loading model from {MODEL_PATH}")
        artifact = joblib.load(MODEL_PATH)
        model = artifact['model']
        scaler = artifact['scaler']
        feature_cols = artifact['feature_cols']
        label_cols = artifact['label_cols']
        
        # Load data
        df = pd.read_csv(csv_path)
        logger.info(f"Agent 1 [{case_id}] — loaded {len(df)} records")
        
        # Select a SAR-worthy case for testing
        sar_rows = df[df['sar_worthy'] == 1]
        if len(sar_rows) > 0:
            row = sar_rows.iloc[0]
            logger.info(f"Agent 1 [{case_id}] — selected SAR-worthy case (index {sar_rows.index[0]})")
        else:
            row = df.iloc[0]
            logger.info(f"Agent 1 [{case_id}] — no SAR-worthy cases, using first row")
        
        # Extract features
        X = row[feature_cols].fillna(0).values.reshape(1, -1)
        X_scaled = scaler.transform(X)
        
        # Predict
        y_pred = model.predict(X_scaled)[0]  # Shape: (n_labels,)
        y_proba = np.array([
            est.predict_proba(X_scaled)[0, 1] for est in model.estimators_
        ])  # Shape: (n_labels,)
        
        # Extract results
        sar_worthy = bool(y_pred[0])
        sar_confidence = float(y_proba[0])
        
        # Find the predicted typology (highest confidence among typologies)
        typology_labels = label_cols[1:]  # Skip sar_worthy
        typology_probs = y_proba[1:]
        typology_preds = y_pred[1:]
        
        # Get the typology with highest confidence among predicted ones
        typology = "Unknown"
        typology_confidence = 0.0
        
        predicted_typologies = [
            (typology_labels[i].replace("typology_", "").upper(), float(typology_probs[i]))
            for i in range(len(typology_labels))
            if typology_preds[i] == 1
        ]
        
        if predicted_typologies:
            # Sort by confidence and take the highest
            predicted_typologies.sort(key=lambda x: x[1], reverse=True)
            typology, typology_confidence = predicted_typologies[0]
        
        # Calculate risk score (average of SAR confidence and typology confidence)
        risk_score = float((sar_confidence + typology_confidence) / 2.0)
        
        logger.info(
            f"Agent 1 — {case_id} | "
            f"SAR-worthy: {sar_worthy} (conf: {sar_confidence:.2f}) | "
            f"Typology: {typology} (conf: {typology_confidence:.2f}) | "
            f"Risk: {risk_score:.2f}"
        )
        
        if not sar_worthy:
            logger.info(
                f"Agent 1 — {case_id} is NON-SAR. "
                "Pipeline exits here — no LLM will be called."
            )
        
        # Return updated state with native Python types
        return {
            **state,
            "sar_worthy": bool(sar_worthy),
            "confidence_score": float(round(sar_confidence, 4)),
            "typology": str(typology),
            "risk_score": float(round(risk_score, 4)),
            "error_log": errors,
            # Initialize downstream fields
            "structured_case": state.get("structured_case", {}),
            "plan": state.get("plan", {}),
            "requires_enrichment": state.get("requires_enrichment", False),
            "triggered_rules": state.get("triggered_rules", []),
            "quantified_indicators": state.get("quantified_indicators", {}),
            "cognitive_event_flow": state.get("cognitive_event_flow", {}),
            "enrichment_data": state.get("enrichment_data", {}),
            "sar_draft": state.get("sar_draft", ""),
            "sar_context_tree": state.get("sar_context_tree", {}),
            "reasoning_traces": state.get("reasoning_traces", []),
            "compliance_passed": state.get("compliance_passed", False),
            "compliance_issues": state.get("compliance_issues", []),
            "quality_score": state.get("quality_score", 0.0),
            "revision_count": state.get("revision_count", 0),
        }
        
    except Exception as e:
        msg = f"[{_ts()}] Agent1 [{case_id}]: Classification failed — {e}"
        logger.error(msg, exc_info=True)
        errors.append(msg)
        return _error_fallback(state, errors)


def _error_fallback(state: SARAgentState, errors: list[str]) -> SARAgentState:
    """Return a safe fallback state when Agent 1 fails."""
    return {
        **state,
        "sar_worthy": False,
        "confidence_score": 0.0,
        "typology": "Unknown",
        "risk_score": 0.0,
        "error_log": errors,
        "structured_case": state.get("structured_case", {}),
        "plan": state.get("plan", {}),
        "requires_enrichment": False,
        "triggered_rules": [],
        "quantified_indicators": {},
        "cognitive_event_flow": {},
        "enrichment_data": {},
        "sar_draft": "",
        "sar_context_tree": {},
        "reasoning_traces": [],
        "compliance_passed": False,
        "compliance_issues": [],
        "quality_score": 0.0,
        "revision_count": 0,
    }
