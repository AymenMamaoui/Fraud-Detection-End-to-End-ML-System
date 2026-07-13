from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional

from src.predict import predict

# App instance

app = FastAPI(
    title="Fraud Detection API",
    description="Scores card transactions for fraud risk, with SHAP explanations.",
    version="1.0.0",
)


# Input schema: the key features the model actually uses.
# Optional fields default to None (handled by the pipeline's imputers).
class Transaction(BaseModel):
    TransactionDT: float = Field(..., description="Time delta in seconds from a reference point")
    TransactionAmt: float = Field(..., description="Transaction amount")
    ProductCD: Optional[str] = Field(None, description="Product code")
    card4: Optional[str] = Field(None, description="Card network (visa, mastercard...)")
    card6: Optional[str] = Field(None, description="Card type (credit, debit...)")
    P_emaildomain: Optional[str] = Field(None, description="Purchaser email domain")
    C1: Optional[float] = Field(None, description="Counting feature C1")
    C3: Optional[float] = Field(None, description="Counting feature C3")
    C5: Optional[float] = Field(None, description="Counting feature C5")

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


# Endpoints
@app.get("/")
def health_check():
    """Simple health check to confirm the API is running."""
    return {"status": "ok", "message": "Fraud Detection API is running"}


@app.post("/predict")
def predict_transaction(transaction: Transaction):
    """
    Score a transaction for fraud risk.
    Returns the fraud probability and the top contributing factors.
    """
    # Convert the validated Pydantic object into a plain dict
    tx_dict = transaction.model_dump()

    # Delegate all the logic to predict() — api.py stays thin
    result = predict(tx_dict)

    return result