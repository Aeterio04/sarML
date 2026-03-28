"""
pipeline.py — End-to-End Inference Pipeline
===========================================
Ties together the three ML stages:
  1. Data Ingestion (sample-inputs -> 1 aggregated case row)
  2. Feature Engineering (1 aggregated case row -> engineered feature vector)
  3. Model Inference (engineered features + model.pkl -> predictions)

Usage:
  python pipeline.py
"""

import warnings
import pandas as pd
from pathlib import Path
import os
import joblib
import sys

warnings.filterwarnings('ignore', category=UserWarning)

from ingestion import run_ingestion, _INPUT_DIR, _OUTPUT_CSV
from feature_engineering import run_feature_engineering

_HERE = Path(__file__).parent
_FINAL_CSV = _HERE / "final_predictions.csv"
_MODEL_PKL = _HERE / "agent1_fixed" / "model_fixed.pkl"

# We include the inference logic directly so this script stands entirely alone 
# and doesn't rely on side-effects or path traversals.
def run_model_inference(input_df: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print("SAR Inference Layer — Generating Predictions")
    print("="*60)

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}.")
        
    print(f"  Loading model from {model_path}...")
    artifact = joblib.load(model_path)
    
    fitted_models = artifact.get('estimators') or getattr(artifact.get('model'), 'estimators_', None)
    if fitted_models is None:
        raise KeyError("model artifact missing 'estimators' key and 'model.estimators_' attribute")
    feature_cols  = artifact['feature_cols']
    label_cols    = artifact['label_cols']
    scaler        = artifact['scaler']

    print(f"  Model expects {len(feature_cols)} features.")

    # The scaler may have been fitted on all features (model.pkl, 50 cols)
    # or on selected features only (model_fixed.pkl, 27 cols).
    # Detect which case we're in by comparing scaler input size to feature_cols size.
    if scaler.n_features_in_ == len(feature_cols):
        # Scaler fitted on selected features only — select first, then scale
        missing = [c for c in feature_cols if c not in input_df.columns]
        if missing:
            raise ValueError(f"Input DataFrame is missing required features: {missing[:5]}...")
        X_inf_s = scaler.transform(input_df[feature_cols].values)
    else:
        # Scaler fitted on ALL features — scale all first, then subset
        LABEL_COLS    = label_cols + ["typology_count"]
        all_feat_cols = [c for c in input_df.columns if c not in LABEL_COLS]
        missing = [c for c in feature_cols if c not in input_df.columns]
        if missing:
            raise ValueError(f"Input DataFrame is missing required features: {missing[:5]}...")
        X_all    = input_df[all_feat_cols].values
        X_scaled = scaler.transform(X_all)
        selected_indices = [all_feat_cols.index(c) for c in feature_cols]
        X_inf_s  = X_scaled[:, selected_indices]
    
    out_df = input_df.copy()
    import numpy as np
    
    print(f"  Running inference on {len(X_inf_s):,} row(s)...")
    for clf, label in zip(fitted_models, label_cols):
        probas = clf.predict_proba(X_inf_s)[:, 1]
        col = f"pred_prob_{label}"
        out_df[col] = np.round(probas, 4)
        print(f"    -> {col}")

    print(f"\nDONE — {len(label_cols)} prediction columns generated.")
    return out_df


def run_pipeline(input_dir: str | Path = _INPUT_DIR, output_csv: str | Path = _OUTPUT_CSV) -> dict:
    print("\n" + "#" * 70)
    print("🚀 SAR PIPELINE: END-TO-END EXECUTION".center(70))
    print("#" * 70 + "\n")

    # Step 1: Ingest raw data (from sample-inputs)
    print(">>> STEP 1: INGESTION")
    df_agg = run_ingestion(input_dir=input_dir, output_csv=output_csv)
    if df_agg.empty:
        print("Pipeline stopped: Ingestion produced no data.")
        return {}

    # Step 2: Engineer features
    print("\n>>> STEP 2: FEATURE ENGINEERING")
    df_eng = run_feature_engineering(df=df_agg)
    print("The engineered features: ")
    print(df_eng)
    # Step 3: Run inference
    print("\n>>> STEP 3: INFERENCE")
    try:
        df_final = run_model_inference(input_df=df_eng, model_path=_MODEL_PKL)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        return {}

    # Save final results
    df_final.to_csv(_FINAL_CSV, index=False)
    
    print("\n" + "═" * 70)
    print("🎉 PIPELINE COMPLETE")
    print(f"  Rows Processed : {len(df_final)}")
    print(f"  Output Saved   : {_FINAL_CSV.resolve()}")
    
    print("\n  Predictions Preview (Probabilities):")
    pred_cols = [c for c in df_final.columns if c.startswith('pred_prob_')]
    preview = df_final[pred_cols].head(5).to_string(index=False)
    print(preview)
    print("═" * 70 + "\n")

    # Format result for LangGraph Agent 1 wrapper
    sar_prob = float(df_final["pred_prob_sar_worthy"].iloc[0]) if "pred_prob_sar_worthy" in df_final.columns else 0.5
    sar_worthy = bool(sar_prob >= 0.5)
    
    typo_cols = [c for c in df_final.columns if c.startswith('pred_prob_typology_')]
    typology = "Unknown"
    if typo_cols:
        best_col = max(typo_cols, key=lambda c: df_final[c].iloc[0])
        typology = best_col.replace('pred_prob_typology_', '').replace('_', ' ').title()
        
    risk_score = float(df_final["kyc_risk_score"].iloc[0]) if "kyc_risk_score" in df_final.columns else 0.5

    return {
        "sar_worthy": sar_worthy,
        "confidence_score": sar_prob,
        "typology": typology,
        "risk_score": risk_score,
        "structured_case": {
            "customer": {},
            "accounts": [],
            "transactions": []
        }
    }

if __name__ == "__main__":
    run_pipeline()
