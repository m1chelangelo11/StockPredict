import datetime
import pandas as pd
import numpy as np
import logging
import os
from clear_and_calc import (
    clean_missing_data,
    calculate_price_features,
    calculate_technical_indicators,
)
from loader import download_data, load_local_csv, save_data, update_stock_data
from preprocessing import (
    create_target_variable,
    scale_features,
    create_sequences,
    prepare_training_data,
)
from config import (
    DATA_DIR,
    PREPROCESSING_DEFAULTS,
)

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def data_pipeline(
    ticker: str,
    start_date: str = "2020-01-01",
    end_date: str = str(datetime.date.today()),
) -> tuple:
    """
    Pipeline wykonujący wszystkie kroki przetwarzania danych dla
    danego tickera. Najpierw pobiera lub aktualizuje istniejące dane,
    potem dodaje do nich cechy i wskaźniki, dzieli na zbiory,
    skaluje zbiory, a potem tworzy sekwencje, które zapisuje.

    Argumenty:
        ticker: Ticker akcji, dla której ma być wykonany pipeline.

    Zwraca:
        Krotkę zawierającą sekwencje przeskalowanych zbiorów:
        treningowego, walidacyjnego i testowego.
    """
    if f"{ticker}.csv" not in os.listdir(DATA_DIR):
        logger.info(f"Pobieranie danych dla {ticker}...")
        df = download_data(ticker, start_date, end_date)
        save_data(df, f"{ticker}.csv")
    else:
        update_stock_data(ticker)
        logger.info(f"Zaktualizowano lokalne dane dla {ticker}")
        df = load_local_csv(os.path.join(DATA_DIR, f"{ticker}.csv"))
        logger.info(f"Wczytano lokalne dane dla {ticker}")

    df_cleaned = clean_missing_data(df)
    logger.info(f"Usunięto brakujące dane dla {ticker}")

    df_features = calculate_price_features(df_cleaned)
    logger.info(f"Obliczono cechy cenowe dla {ticker}")

    df_technical = calculate_technical_indicators(df_features)
    logger.info(f"Obliczono wskaźniki techniczne dla {ticker}")

    df_clean_tech_feat = clean_missing_data(df_technical)
    logger.info(
        f"Usunięto brakujące dane po obliczeniu cech i wskaźników technicznych dla {ticker}"
    )

    df_target = create_target_variable(
        df_clean_tech_feat,
        prediction_horizon=PREPROCESSING_DEFAULTS["PREDICTION_HORIZON"],
        target_type=PREPROCESSING_DEFAULTS["TARGET_TYPE"],
    )
    logger.info(f"Utworzono zmienną docelową dla {ticker}")

    df_clean_target = clean_missing_data(df_target)
    logger.info(
        f"Usunięto brakujące dane po utworzeniu zmiennej docelowej dla {ticker}"
    )

    train_df, val_df, test_df = prepare_training_data(
        df_clean_target,
        test_size=PREPROCESSING_DEFAULTS["TEST_SIZE"],
        validation_size=PREPROCESSING_DEFAULTS["VALIDATION_SIZE"],
    )
    logger.info(
        f"Podzielono dane na zbiory treningowy, walidacyjny i testowy dla {ticker}"
    )

    train_df.set_index("Date", inplace=True)
    val_df.set_index("Date", inplace=True)
    test_df.set_index("Date", inplace=True)

    scaled_train_df, scaler_train = scale_features(
        train_df,
        exclude_columns=["Target"],
        method=PREPROCESSING_DEFAULTS["SCALING_METHOD"],
    )
    logger.info(f"Przeskalowano zbiory treningowe dla {ticker}")
    scaled_val_array = scaler_train.transform(val_df.drop(columns=["Target"]))
    scaled_test_array = scaler_train.transform(test_df.drop(columns=["Target"]))
    logger.info(f"Przeskalowano zbiory walidacyjne i testowe dla {ticker}")

    scaled_val_df = pd.DataFrame(
        scaled_val_array,
        columns=val_df.drop(columns=["Target"]).columns,
        index=val_df.index,
    )
    scaled_val_df["Target"] = val_df["Target"]

    scaled_test_df = pd.DataFrame(
        scaled_test_array,
        columns=test_df.drop(columns=["Target"]).columns,
        index=test_df.index,
    )
    scaled_test_df["Target"] = test_df["Target"]

    X_train, y_train = create_sequences(
        scaled_train_df, sequence_length=PREPROCESSING_DEFAULTS["SEQUENCE_LENGTH"]
    )
    X_val, y_val = create_sequences(
        scaled_val_df, sequence_length=PREPROCESSING_DEFAULTS["SEQUENCE_LENGTH"]
    )
    X_test, y_test = create_sequences(
        scaled_test_df, sequence_length=PREPROCESSING_DEFAULTS["SEQUENCE_LENGTH"]
    )
    logger.info(
        f"Utworzono sekwencje dla zbiorów treningowego, walidacyjnego i testowego dla {ticker}"
    )

    # Tworzenie folderu dla przetworzonch danych
    processed_dir = os.path.join(DATA_DIR, f"{ticker}_processed")
    os.makedirs(processed_dir, exist_ok=True)

    sequences = {
        f"X_train_{ticker}": X_train,
        f"y_train_{ticker}": y_train,
        f"X_val_{ticker}": X_val,
        f"y_val_{ticker}": y_val,
        f"X_test_{ticker}": X_test,
        f"y_test_{ticker}": y_test,
    }

    for name, data in sequences.items():
        filepath = os.path.join(processed_dir, f"{name}.npy")
        np.save(filepath, data)
        logger.info(f"Zapisano sekwencje do pliku: {filepath}")

    return (X_train, X_val, X_test, y_train, y_val, y_test)


# data_pipeline("AAPL", "2020-01-01", "2023-10-01")  # Przykładowe wywołanie funkcji
