import os

# Ścieżka do głównego folderu data/
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(DATA_DIR, exist_ok=True)

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# Domyślne ustawienia dla preprocessing
PREPROCESSING_DEFAULTS = {
    "PREDICTION_HORIZON": 1,
    "TARGET_TYPE": "price",  # 'price' lub 'return'
    "TEST_SIZE": 0.2,
    "VALIDATION_SIZE": 0.2,
    "SEQUENCE_LENGTH": 20,
    "SCALING_METHOD": "minmax",  # 'minmax' lub 'standard'
}
