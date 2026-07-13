import pandas as pd
from src import config
from src.predict import predict


def test_predict_returns_score_and_factors():
    # Take one real transaction from the data as a sample
    tx = pd.read_csv(config.TRANSACTION_FILE, nrows=1)
    idf = pd.read_csv(config.IDENTITY_FILE)
    df = tx.merge(idf, on=config.ID_COL, how="left")

    # Drop the target; convert the row to a dict
    sample = df.drop(columns=[config.TARGET]).iloc[0].to_dict()

    result = predict(sample)

    # Basic checks
    assert "fraud_probability" in result
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert len(result["top_factors"]) == 5
    print("\nPrediction result:", result)