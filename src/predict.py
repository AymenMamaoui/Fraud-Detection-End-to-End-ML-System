import joblib
import numpy as np
import pandas as pd
import shap

from src import config

# -------------------------------------------------------------
# Load the trained pipeline once, at import time.
# The pipeline has two parts: preprocessor -> XGBoost model.
# -------------------------------------------------------------
_MODEL_PATH = config.MODELS_DIR / "fraud_pipeline.pkl"

pipeline = joblib.load(_MODEL_PATH)

# Split the pipeline into its two components
preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]

# The ColumnTransformer (last step) knows the final feature names
column_transformer = preprocessor.named_steps["column_transformer"]
feature_names = column_transformer.get_feature_names_out()

# SHAP explainer dedicated to tree-based models (exact & fast)
explainer = shap.TreeExplainer(model)

def predict(transaction: dict, top_n: int = 5) -> dict:
    """
    Score a single transaction and explain the result.

    Args:
        transaction: raw transaction as a dict (column -> value)
        top_n: number of top contributing features to return

    Returns:
        dict with the fraud probability and the top SHAP factors
    """
    # Convert the incoming dict into a one-row DataFrame
    X = pd.DataFrame([transaction])

    # 1. Fraud probability from the full pipeline
    fraud_proba = float(pipeline.predict_proba(X)[:, 1][0])

    # 2. Transform the raw input into model-ready features
    X_transformed = preprocessor.transform(X)

    # 3. SHAP values for this single prediction
    shap_values = explainer.shap_values(X_transformed)
    shap_row = shap_values[0]  # first (and only) row

    # 4. Pair each feature name with its SHAP contribution
    contributions = list(zip(feature_names, shap_row))

    # Sort by absolute impact (largest push, up or down)
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    # Keep the top_n most influential factors
    top_factors = [
        {
            "feature": name,
            "impact": float(value),
            "direction": "increases risk" if value > 0 else "decreases risk",
        }
        for name, value in contributions[:top_n]
    ]

    return {
        "fraud_probability": round(fraud_proba, 4),
        "top_factors": top_factors,
    }