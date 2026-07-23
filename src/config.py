from pathlib import Path

# PATHS
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

# Raw files
TRANSACTION_FILE = DATA_RAW / "train_transaction.csv"
IDENTITY_FILE = DATA_RAW / "train_identity.csv"

# COLUMNS
TARGET = "isFraud"
ID_COL = "TransactionID"

# High-signal categorical features (validated during EDA)
CATEGORICAL_FEATURES = ["ProductCD", "card4", "card6", "P_emaildomain"]

# Representative C columns (kept after correlation analysis)
C_FEATURES = ["C1", "C3", "C5"]


# PARAMETERS
RANDOM_STATE = 42
MISSING_THRESHOLD = 0.9   # drop columns emptier than this ratio

# Behavioral features engineered from transaction history.
# NOT used in the current model: they showed no performance gain and require
# card-level history unavailable at inference time (would need a feature store).
# Kept documented for reference — see README.
BEHAVIORAL_FEATURES_EXPERIMENTAL = [
    "amount_vs_card_mean",
    "time_since_last_txn",
    "card_txn_count",
    "is_first_transaction",
]