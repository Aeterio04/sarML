"""
inference.py — SAR Model Inference
===================================
Standalone inference module. Import run_inference from here,
not from train_model.py (which executes training code on import).
"""

import os
import joblib
import numpy as np
import pandas as pd


def run_inference(input_df: pd.DataFrame, model_path: str = 'model/model.pkl') -> pd.DataFrame:
    """
    Load the trained model artifact and run inference on an engineered DataFrame.

    Parameters
    ----------
    input_df   : pd.DataFrame  — output of run_feature_engineering()
    model_path : str           — path to model/model.pkl

    Returns
    -------
    pd.DataFrame — original df with 7 pred_prob_* columns appended
    """
    print(f"\n{'='*60}")
    print("SAR Inference Layer — Generating Predictions")
    print("="*60)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model artifact not found at '{model_path}'. "
            "Run `python train_model.py` first."
        )

    print(f"  Loading model from {model_path}...")
    artifact = joblib.load(model_path)

    fitted_models = artifact['estimators']
    feature_cols  = artifact['feature_cols']
    label_cols    = artifact['label_cols']
    scaler        = artifact['scaler']

    print(f"  Model expects {len(feature_cols)} features.")

    # ── Validate ──────────────────────────────────────────────────────────────
    missing = [c for c in feature_cols if c not in input_df.columns]
    if missing:
        raise ValueError(
            f"Input DataFrame is missing {len(missing)} required feature(s): "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    # ── Scale ─────────────────────────────────────────────────────────────────
    X_inf   = input_df[feature_cols].values
    X_inf_s = scaler.transform(X_inf)

    # ── Predict ───────────────────────────────────────────────────────────────
    out_df = input_df.copy()
    print(f"  Running inference on {len(X_inf_s):,} row(s)...")

    for clf, label in zip(fitted_models, label_cols):
        probas = clf.predict_proba(X_inf_s)[:, 1]
        col    = f"pred_prob_{label}"
        out_df[col] = np.round(probas, 4)
        print(f"    -> {col}")

    print(f"\nDONE — {len(label_cols)} prediction columns generated.")
    print("="*60)

    return out_df