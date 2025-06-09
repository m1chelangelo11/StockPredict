import os

# Ścieżka do głównego folderu data/
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(DATA_DIR, exist_ok=True)

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# Ustawienia pobierania
DOWNLOADING_DEFAULTS = {
    "TICKER": "AAPL",
    "START_DATE": "2020-01-01",
    "END_DATE": "2025-01-01"
}

# Domyślne ustawienia dla preprocessing
PREPROCESSING_DEFAULTS = {
    "PREDICTION_HORIZON": 1,
    "TARGET_TYPE": "price",  # 'price' lub 'return'
    "TEST_SIZE": 0.2,
    "VALIDATION_SIZE": 0.2,
    "SEQUENCE_LENGTH": 20,
    "SCALING_METHOD": "minmax",  # 'minmax' lub 'standard'
}

# Domyślne parametry modelu
MODEL_PARAMS = {
    "INPUT_DIM": 29,
    "HIDDEN_DIM": 128,
    "NUM_LAYERS": 2,
    "OUTPUT_DIM": 1,
}

# Domyślne parametry treningu
TRAINING_PARAMS = {
    "NUM_EPOCHS": 500,
    "LEARNING_RATE": 0.001,
    "LOAD_NAME": "model.pt",
    "SAVE_NAME": "model2.pt",
    "CONTINUE_TRAINING": True,
}
