from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_health_check():
    """The root endpoint should confirm the API is running."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_returns_score():
    """The /predict endpoint should return a valid fraud score and factors."""
    sample = {
        "TransactionDT": 86400,
        "TransactionAmt": 149.0,
        "ProductCD": "W",
        "card4": "visa",
        "card6": "debit",
        "P_emaildomain": "gmail.com",
        "C1": 1.0,
        "C3": 0.0,
        "C5": 0.0,
    }
    response = client.post("/predict", json=sample)

    assert response.status_code == 200
    body = response.json()
    assert "fraud_probability" in body
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert len(body["top_factors"]) == 5


def test_predict_rejects_invalid_input():
    """Missing required fields should be rejected with a 422 error."""
    # TransactionAmt is required but missing here
    bad_sample = {"TransactionDT": 86400}
    response = client.post("/predict", json=bad_sample)
    assert response.status_code == 422