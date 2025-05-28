import os

# Ścieżka do głównego folderu data/
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# Konfiguracja wskaźników technicznych
TECHNICAL_INDICATORS_CONFIG = {
    "SMA_PERIODS": [5, 10, 20, 30],
    "EMA_PERIODS": [12, 26],
    "RSI_PERIOD": 14,
    "MACD_FAST": 12,
    "MACD_SLOW": 26,
    "MACD_SIGNAL": 9,
    "BOLLINGER_PERIOD": 20,
    "BOLLINGER_STD": 2,
    "VOLATILITY_PERIOD": 20,
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
