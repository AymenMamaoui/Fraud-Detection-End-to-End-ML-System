import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

from src import config
from src.data import load_data
from src.features import build_preprocessor

import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import roc_auc_score, average_precision_score


def time_based_split(df, test_size=0.2):
    """
    Split chronologically: earliest transactions for training,
    latest for validation. Prevents temporal data leakage.
    """
    df_sorted = df.sort_values("TransactionDT").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_size))

    train_df = df_sorted.iloc[:split_idx]
    valid_df = df_sorted.iloc[split_idx:]
    return train_df, valid_df


def build_model_pipeline(scale_pos_weight):
    """
    Combine the preprocessing pipeline with an XGBoost classifier
    into a single object. scale_pos_weight handles class imbalance.
    """
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,   # ~27, counters the 3.5% imbalance
        eval_metric="aucpr",                 # PR-AUC: right metric for imbalance
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )

    full_pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ])
    return full_pipeline

def main():
    # 1. Load and merge the data
    print("Loading data...")
    df = load_data()

    # 2. Chronological split (no temporal leakage)
    train_df, valid_df = time_based_split(df, test_size=0.2)

    X_train = train_df.drop(columns=[config.TARGET])
    y_train = train_df[config.TARGET]
    X_valid = valid_df.drop(columns=[config.TARGET])
    y_valid = valid_df[config.TARGET]

    print(f"Train: {X_train.shape[0]:,} rows | Valid: {X_valid.shape[0]:,} rows")

    # 3. Compute class imbalance ratio for scale_pos_weight
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos
    print(f"scale_pos_weight = {scale_pos_weight:.1f}")

    # 4. Build the full pipeline (preprocessing + model)
    pipeline = build_model_pipeline(scale_pos_weight)

    # 5. Train + evaluate, tracked with MLflow
    mlflow.set_experiment("fraud-detection")
    with mlflow.start_run():
        print("Training...")
        pipeline.fit(X_train, y_train)

        # Predicted probabilities for the positive (fraud) class
        y_proba = pipeline.predict_proba(X_valid)[:, 1]

        roc_auc = roc_auc_score(y_valid, y_proba)
        pr_auc = average_precision_score(y_valid, y_proba)

        print(f"\nROC-AUC : {roc_auc:.4f}")
        print(f"PR-AUC  : {pr_auc:.4f}")

        # Log parameters and metrics to MLflow
        mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 1))
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("max_depth", 6)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            serialization_format="cloudpickle",
        )

        # 6. Save the trained pipeline locally
        config.MODELS_DIR.mkdir(exist_ok=True)
        model_path = config.MODELS_DIR / "fraud_pipeline.pkl"
        joblib.dump(pipeline, model_path)
        print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()