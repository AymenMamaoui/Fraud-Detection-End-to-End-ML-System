import pandas as pd
from src import config


def load_data():
    """
    Load and merge the raw transaction and identity files.

    Returns a single DataFrame joined on TransactionID (left join,
    keeping every transaction even when identity data is missing).
    """
    transactions = pd.read_csv(config.TRANSACTION_FILE)
    identities = pd.read_csv(config.IDENTITY_FILE)

    df = transactions.merge(identities, on=config.ID_COL, how="left")
    return df