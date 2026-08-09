import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from src.predict import predict

# Config
st.set_page_config(page_title="Fraud Detection", page_icon=None, layout="wide")

BLOCK_THRESHOLD = 0.50
REVIEW_THRESHOLD = 0.15

PRODUCT_LABELS = {
    "In-store purchase": "W",
    "Digital / online service": "C",
    "Recurring payment": "R",
    "Household goods": "H",
    "Subscription": "S",
}
CARD_NETWORK_LABELS = {
    "Visa": "visa", "Mastercard": "mastercard",
    "American Express": "american express", "Discover": "discover",
}
CARD_TYPE_LABELS = {"Debit card": "debit", "Credit card": "credit"}


def to_concept(feature_name: str) -> str:
    clean = (
        feature_name.replace("categorical__", "")
        .replace("numeric__", "").replace("amount__", "")
    )
    if clean.startswith("ProductCD_"):
        return "Type of purchase"
    if clean.startswith("card4_"):
        return "Card network"
    if clean.startswith("card6_"):
        return "Card type"
    if clean.startswith("P_emaildomain_"):
        return "Email provider"
    direct = {
        "TransactionAmt": "Transaction amount",
        "hour": "Time of day", "is_high_risk_hour": "Time of day",
        "day": "Day of week",
        "C1": "Card activity indicator",
        "C3": "Account verification signal",
        "C5": "Account history signal",
    }
    return direct.get(clean, clean.replace("_", " ").capitalize())


def aggregate_factors(factors: list) -> pd.DataFrame:
    df = pd.DataFrame(factors)
    df["concept"] = df["feature"].apply(to_concept)
    grouped = (
        df.groupby("concept", as_index=False)["impact"].sum()
        .sort_values("impact", key=abs, ascending=False)
    )
    grouped["direction"] = grouped["impact"].apply(
        lambda v: "increases risk" if v > 0 else "decreases risk"
    )
    return grouped


# Header
st.title("Fraud Detection System")
st.markdown(
    "Assess the fraud risk of a transaction and understand the reasoning "
    "behind each decision."
)
st.divider()

# Input form
st.subheader("Transaction details")
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("**Payment**")
    transaction_amt = st.number_input("Transaction amount ($)", min_value=0.0, value=149.0, step=10.0)
    transaction_hour = st.slider("Time of transaction", 0, 23, 12, format="%d:00")
    product_label = st.selectbox("Type of purchase", list(PRODUCT_LABELS.keys()))

with col2:
    st.markdown("**Card & customer**")
    card_network_label = st.selectbox("Card network", list(CARD_NETWORK_LABELS.keys()))
    card_type_label = st.selectbox("Card type", list(CARD_TYPE_LABELS.keys()))
    email_domain = st.selectbox(
        "Customer email provider",
        ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com", "anonymous.com"],
    )

st.markdown("**Account signals**")
hist1, hist2, hist3 = st.columns(3)
with hist1:
    c1 = st.number_input("Card activity indicator", min_value=0.0, value=1.0, step=1.0,
                         help="Typically 1-2. Values above 10 are associated with elevated fraud rates.")
with hist2:
    c3 = st.number_input("Account verification signal", min_value=0.0, value=0.0, step=1.0,
                         help="Usually 0. Non-zero values are strongly associated with legitimate activity.")
with hist3:
    c5 = st.number_input("Account history signal", min_value=0.0, value=0.0, step=1.0,
                         help="Usually 0. Non-zero values are associated with lower fraud rates.")

st.caption(
    "The account signals are proprietary indicators from the payment processor. "
    "Their exact definitions are not public, but they carry strong predictive "
    "signal — removing them reduces detection performance by ~45%."
)

st.divider()

# Prediction (direct call, no API)
if st.button("Analyze transaction", type="primary", use_container_width=True):
    payload = {
        "TransactionDT": transaction_hour * 3600,
        "TransactionAmt": transaction_amt,
        "ProductCD": PRODUCT_LABELS[product_label],
        "card4": CARD_NETWORK_LABELS[card_network_label],
        "card6": CARD_TYPE_LABELS[card_type_label],
        "P_emaildomain": email_domain,
        "C1": c1, "C3": c3, "C5": c5,
    }

    try:
        result = predict(payload, top_n=40)
        proba = result["fraud_probability"]
        grouped = aggregate_factors(result["top_factors"])

        st.subheader("Assessment")
        left, right = st.columns([1, 2], gap="large")
        with left:
            st.metric("Fraud risk", f"{proba:.0%}")
            st.progress(proba)
            st.caption("Baseline fraud rate: 3.5%")
        with right:
            if proba >= BLOCK_THRESHOLD:
                st.error("**Decline this transaction** — the risk of fraud is high. "
                         "We recommend blocking it and contacting the cardholder.")
            elif proba >= REVIEW_THRESHOLD:
                st.warning("**Send for manual review** — this transaction carries "
                           "meaningfully elevated risk and should be checked by an analyst.")
            else:
                st.success("**Approve this transaction** — the risk of fraud is low and "
                           "the patterns match normal customer behaviour.")

        st.subheader("What drove this assessment")
        risk_up = grouped[grouped["impact"] > 0].head(4)
        risk_down = grouped[grouped["impact"] < 0].head(4)
        max_impact = grouped["impact"].abs().max() or 1.0

        exp_left, exp_right = st.columns(2, gap="large")
        with exp_left:
            st.markdown("##### Raising the risk")
            if not risk_up.empty:
                for _, row in risk_up.iterrows():
                    st.markdown(f"**{row['concept']}**")
                    st.progress(min(abs(row["impact"]) / max_impact, 1.0))
            else:
                st.caption("No factors pushed the risk upward.")
        with exp_right:
            st.markdown("##### Lowering the risk")
            if not risk_down.empty:
                for _, row in risk_down.iterrows():
                    st.markdown(f"**{row['concept']}**")
                    st.progress(min(abs(row["impact"]) / max_impact, 1.0))
            else:
                st.caption("No factors pushed the risk downward.")

        st.caption("Longer bars indicate a stronger influence on the final assessment.")

        with st.expander("Technical details (for analysts)"):
            display = grouped.copy()
            display.columns = ["Factor", "Impact score", "Effect"]
            display["Impact score"] = display["Impact score"].round(4)
            st.dataframe(display, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Something went wrong: {e}")