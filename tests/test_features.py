import pandas as pd
import numpy as np
from src import config
from src.features import build_preprocessor


def load_sample(n=5000):
    """Load a small merged sample for testing."""
    tx = pd.read_csv(config.TRANSACTION_FILE, nrows=n)
    idf = pd.read_csv(config.IDENTITY_FILE)
    df = tx.merge(idf, on=config.ID_COL, how="left")
    return df


def test_pipeline_runs_and_reduces_columns():
    """The preprocessor should fit/transform without error and output a numpy array."""
    df = load_sample()
    X = df.drop(columns=[config.TARGET])
    y = df[config.TARGET]

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X, y)

    # It returns a numpy array
    assert isinstance(X_transformed, np.ndarray)
    # Same number of rows as input
    assert X_transformed.shape[0] == X.shape[0]
    # Far fewer columns than the raw input (selection happened)
    assert X_transformed.shape[1] < X.shape[1]


def test_pipeline_has_no_nan_in_output():
    """After preprocessing, the engineered/encoded features should be finite."""
    df = load_sample()
    X = df.drop(columns=[config.TARGET])
    y = df[config.TARGET]

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X, y)

    # No NaNs introduced by the transformations we control
    assert not np.isnan(X_transformed).any()