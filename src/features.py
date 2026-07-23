import numpy as np
import pandas as pd
from src import config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def build_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
        Create interpretable, business-level features from transaction history.

        All statistics are computed causally: for each transaction, only
        *earlier* transactions from the same card are used.
    """
    df = df.copy()
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    # Historical mean per card, excluding the current transaction.
    # shift(1) must happen INSIDE each group, hence the lambda.
    df["card_mean_amount"] = df.groupby("card1")["TransactionAmt"].transform(
        lambda s: s.expanding().mean().shift(1)
    )

    # Number of prior transactions for this card (0 for the first one)
    df["card_txn_count"] = df.groupby("card1").cumcount()

    # Ratio of current amount to the card's usual spend
    df["amount_vs_card_mean"] = df["TransactionAmt"] / df["card_mean_amount"]

    # Time elapsed since this card's previous transaction
    df["time_since_last_txn"] = df.groupby("card1")["TransactionDT"].diff()

    # Flag transactions with no prior history
    df["is_first_transaction"] = df["card_mean_amount"].isna().astype(int)

    # Neutral values where no history exists
    df["amount_vs_card_mean"] = df["amount_vs_card_mean"].fillna(1.0)
    df["time_since_last_txn"] = df["time_since_last_txn"].fillna(-1)

    df = df.drop(columns=["card_mean_amount"])
    return df

class TimeFeatures(BaseEstimator, TransformerMixin):
    """
        Derive time-based features from TransactionDT.

        TransactionDT is a time delta in seconds from a reference point.
        We convert it into:
          - hour: position within the daily cycle (0-23)
          - is_high_risk_hour: flag for the early-cycle fraud peak (hours 4-10)
    """
    def fit(self, X, y=None):
        # Nothing to learn: the conversion is purely deterministic.
        return self

    def transform(self, X):
        # Work on a copy to avoid mutating the caller's DataFrame.
        X = X.copy()

        # Seconds -> hour of the daily cycle
        X["hour"] = ((X["TransactionDT"] / 3600) % 24).astype(int)

        # Seconds -> day of the weekly cycle
        X["day"] = ((X["TransactionDT"] / (3600 * 24)) % 7).astype(int)

        # High-risk window identified during EDA (fraud peak ~3x baseline)
        X["is_high_risk_hour"] = X["hour"].between(4, 10).astype(int)

        # Drop the raw column: its signal is now captured by the derived features.
        X = X.drop(columns=["TransactionDT"])

        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)

class FeatureSelector(BaseEstimator, TransformerMixin):
    """
        Select the columns to keep for modeling:
        - the engineered time features
        - the high-signal categorical features (from EDA)
        - the representative C columns (after correlation analysis)
        - TransactionAmt (log-transformed later)
        Everything else (near-empty columns, redundant C columns,
        anonymized V/id columns) is dropped.
    """
    def __init__(self):
        # Columns produced by the TimeFeatures transformer
        self.time_features = ["hour", "day", "is_high_risk_hour"]

        # Numeric feature we keep and log-transform downstream
        self.numeric_features = ["TransactionAmt"]

        # Pulled from config so decisions live in one place
        self.categorical_features = config.CATEGORICAL_FEATURES
        self.c_features = config.C_FEATURES


    def fit(self, X, y=None):
        wanted = (
            self.time_features
            + self.numeric_features
            + self.categorical_features
            + self.c_features
        )
        self.columns_to_keep_ = [col for col in wanted if col in X.columns]
        return self

    def transform(self, X):
        X = X.copy()
        return X[self.columns_to_keep_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_to_keep_)


class LogTransform(BaseEstimator, TransformerMixin):
    """
        Apply log(1 + x) to compress the skewed TransactionAmt distribution.
        log1p handles zeros safely (log1p(0) = 0).
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log1p(X)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)

def build_preprocessor():
    """
        Build the full preprocessing pipeline.

        Steps:
          1. TimeFeatures     -> engineer hour / day / high-risk flag
          2. FeatureSelector  -> keep only the chosen columns
          3. ColumnTransformer:
               - TransactionAmt      -> log transform + scaling
               - categorical columns -> one-hot encoding
               - remaining numerics  -> passed through unchanged
        Returns an unfitted sklearn Pipeline.
    """
    # Column groups (must match names after FeatureSelector)
    numeric_amount = ["TransactionAmt"]
    categorical = config.CATEGORICAL_FEATURES
    passthrough_numeric = ["hour", "day", "is_high_risk_hour"] + config.C_FEATURES

    # Transformer for the transaction amount: impute, log, then scale
    amount_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("log", LogTransform()),
        ("scale", StandardScaler()),
    ])

    # Transformer for categoricals: fill missing values, then one-hot encode
    categorical_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Apply the right transformer to each column group
    column_transformer = ColumnTransformer(
        transformers=[
            ("amount", amount_pipeline, numeric_amount),
            ("categorical", categorical_pipeline, categorical),
            ("numeric", "passthrough", passthrough_numeric),
        ],
        remainder="drop",
    )

    # Full pipeline: engineer -> select -> transform per column
    preprocessor = Pipeline(steps=[
        ("time_features", TimeFeatures()),
        ("selector", FeatureSelector()),
        ("column_transformer", column_transformer),
    ])

    return preprocessor