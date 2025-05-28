import pandas as pd
import numpy as np
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import REQUIRED_COLUMNS

logger = logging.getLogger(__name__)


def clean_missing_data(df: pd.DataFrame, method: str = "forward_fill") -> pd.DataFrame:
    """
    Czyści brakujące wartości w danych giełdowych

    Argumenty:
        df: DataFrame z danymi giełdowymi
        method: 'forward_fill', 'backward_fill', 'interpolate', 'drop'

    Zwraca:
        DataFrame z danymi giełdowymi po czyszczeniu
    """
    df_cleaned = df.copy()
    no_data = df_cleaned.isnull().sum().sum()

    if no_data == 0:
        logger.info("Brak danych do wyczyszczenia.")
        return df_cleaned

    logger.info(f"Znaleziono {no_data} brakujących wartości.")
    no_data_col = df_cleaned.isnull().sum()
    for col, count in no_data_col.items():
        if count > 0:
            logger.info(f"Kolumna {col} ma {count} brakujących wartości.")

    if method == "forward_fill":
        df_cleaned = df_cleaned.ffill()
    elif method == "backward_fill":
        df_cleaned = df_cleaned.bfill()
    elif method == "interpolate":
        numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
        df_cleaned[numeric_columns] = df_cleaned[numeric_columns].interpolate(
            method="linear"
        )
    elif method == "drop":
        df_cleaned = df_cleaned.dropna()
    else:
        raise ValueError(f"Nieznana metoda: {method}")

    remaining_no_data = df_cleaned.isnull().sum().sum()
    if remaining_no_data > 0:
        logger.warning(
            f"Po czyszczeniu pozostało {remaining_no_data} brakujących wartości."
        )

    return df_cleaned


def calculate_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Oblicza cechy cenowe

    Argumenty:
        df: ramka danych

    Zwraca:
        DataFrame z nowymi kolumnami zawierającymi cechy cenowe
    """
    df_features = df.copy()
    try:
        logger.info("Obliczanie cech cenowych...")

        df_features["Daily_Return"] = df_features["Close"].pct_change()
        df_features["Volatility"] = df_features["Close"].rolling(window=20).std()

        try:
            if df_features["Low"].min() != 0 and df_features["Open"].min() != 0:
                df_features["HL_Ratio"] = df_features["High"] / df_features["Low"]
                df_features["CO_Ratio"] = df_features["Close"] / df_features["Open"]

        except ValueError:
            logger.warning(
                "Nie można obliczyć wskaźników HL_Ratio i CO_Ratio z powodu zera w kolumnie 'Low' lub 'Open'."
            )
            df_features["HL_Ratio"] = np.nan
            df_features["CO_Ratio"] = np.nan

        df_features["Price_Gap"] = df_features["Open"] - df_features["Close"].shift(1)
        df_features["Price_Range"] = df_features["High"] - df_features["Low"]
        df_features["Body_Size"] = abs(df_features["Close"] - df_features["Open"])
        df_features["Upper_Shadow"] = df_features["High"] - df_features[
            ["Open", "Close"]
        ].max(axis=1)
        df_features["Lower_Shadow"] = (
            df_features[["Open", "Close"]].min(axis=1) - df_features["Low"]
        )

        tr1 = df_features["High"] - df_features["Low"]
        tr2 = abs(df_features["High"] - df_features["Close"].shift(1))
        tr3 = abs(df_features["Low"] - df_features["Close"].shift(1))

        df_features["True_Range"] = np.maximum.reduce([tr1, tr2, tr3])

        vpt_change = df_features["Daily_Return"] * df_features["Volume"]
        df_features["Volume_Price_Trend"] = vpt_change.cumsum()

        logger.info(f"Dodano cechy cenowe: {list(df_features.columns)}")

    except Exception:
        logger.error(f"[BłĄD] Ramka nie zawiera wymaganych kolumn: {REQUIRED_COLUMNS}")
        raise ValueError(df.columns)

    return df_features


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Funkcja oblicza wskaźniki techniczne na podstawie danych giełdowych.

    Argumenty:
        df: Ramka danych z danymi giełdowymi

    Zwraca:
        DataFrame z nowymi kolumnami zawierającymi wskaźniki techniczne
    """
    df_tech_ind = df.copy()

    df_tech_ind["SMA_5"] = df_tech_ind["Close"].rolling(window=5).mean()
    df_tech_ind["SMA_10"] = df_tech_ind["Close"].rolling(window=10).mean()
    df_tech_ind["SMA_20"] = df_tech_ind["Close"].rolling(window=20).mean()
    df_tech_ind["SMA_50"] = df_tech_ind["Close"].rolling(window=50).mean()

    df_tech_ind["EMA_12"] = df_tech_ind["Close"].ewm(span=12).mean()
    df_tech_ind["EMA_26"] = df_tech_ind["Close"].ewm(span=26).mean()

    difference = df["Close"].diff()
    gain = difference.clip(lower=0)
    loss = (-difference).clip(lower=0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    df_tech_ind["RSI"] = 100 - (100 / (1 + (avg_gain / avg_loss)))

    df_tech_ind["MACD_line"] = df_tech_ind["EMA_12"] - df_tech_ind["EMA_26"]
    df_tech_ind["Signal_line"] = df_tech_ind["MACD_line"].ewm(span=9).mean()
    df_tech_ind["MacD_histogram"] = (
        df_tech_ind["MACD_line"] - df_tech_ind["Signal_line"]
    )

    df_tech_ind["Middle_band"] = df_tech_ind["Close"].rolling(window=20).mean()
    std = df_tech_ind["Close"].rolling(window=20).std()
    df_tech_ind["Upper_band"] = df_tech_ind["Middle_band"] + (std * 2)
    df_tech_ind["Lower_band"] = df_tech_ind["Middle_band"] - (std * 2)

    logger.info(f"Obliczone wskaźniki techniczne: {list(df_tech_ind.columns)}")

    return df_tech_ind
