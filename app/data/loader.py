import yfinance as yf
import pandas as pd
import os
import sys
import logging
from datetime import date, datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import REQUIRED_COLUMNS, DATA_DIR

logger = logging.getLogger(__name__)


def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Sprawdza, czy DataFrame zawiera wszystkie wymagane kolumny.

    Argumenty:
        df: DataFrame do sprawdzenia

    Zwraca:
        True jeśli ramka danych zawiera wszystkie wymagane kolumny, w przeciwnym wypadku False
    """
    if df.empty:
        logger.warning("DataFrame jest pusty")
        return False

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        logger.error(f"Brakujące kolumny: {missing_cols}")
        return False

    return True


def download_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """
    Pobiera dane giełdowe dla podanego tickera w określonym przedziale czasowym z Yahoo Finance.

    Argumenty:
        ticker: Symbol akcji (np. 'AAPL')
        start_date: Data początkowa w formacie 'YYYY-MM-DD'
        end_date: Data końcowa w formacie 'YYYY-MM-DD'

    Zwraca:
        Ramka danych z danymi giełdowymi lub None w przypadku błędu
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        if start_dt >= end_dt:
            logger.error(f"[BŁĄD] start_date > end_date: {start_date} > {end_date}")
            return None
    except ValueError:
        logger.error(f"[BŁĄD] Niepoprawny format daty: {start_date} lub {end_date}")
        return None

    try:
        logger.info(f"Pobieram dane dla {ticker} od {start_date} do {end_date}...")
        df = yf.download(ticker, start=start_date, end=end_date)
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        if df.empty:
            logger.warning(f"Brak danych dla {ticker} w podanym zakresie.")
            return None

        df.reset_index(inplace=True)

        if not validate_dataframe(df):
            logger.warning("Pobrane dane nie zawierają wszystkich kolumn.")
            return None

        logger.info(f"Wczytano dane: {df.shape[0]} wierszy dla {ticker}.")
        return df

    except Exception as e:
        logger.error(f"Błąd podczas pobierania danych: {e}")
        return None


def load_local_csv(filepath: str) -> pd.DataFrame:
    """
    Wczytuje dane z lokalnego pliku CSV.

    Argumenty:
        filepath: Ścieżka do pliku CSV

    Zwraca:
        Ramka z danymi lub pusta w przypadku błędu
    """
    if not os.path.exists(filepath):
        logger.error(f"Plik {filepath} nie istnieje!")
        return pd.DataFrame()

    try:
        logger.info(f"Wczytuję dane z pliku {filepath}...")
        df = pd.read_csv(filepath, parse_dates=["Date"])
        df.sort_values(by="Date", inplace=True)

        if not validate_dataframe(df):
            logger.warning(
                f"Dane z pliku {filepath} nie zawierają wszystkich wymaganych kolumn."
            )
            return pd.DataFrame()

        logger.info(f"Pomyslnie wczytano {len(df)} wierszy z {filepath}.")
        return df

    except Exception as e:
        logger.error(f"Błąd podczas wczytywania danych z pliku {filepath}: {e}")
        return pd.DataFrame()


def save_data(df: pd.DataFrame, filename: str, mode: str = "w") -> bool:
    """
    Zapisuje dane do pliku CSV.

    Argumenty:
        df: DataFrame z danymi do zapisania
        filename: Nazwa pliku (bez ścieżki)
        mode: Tryb zapisu - 'w' nadpisuje instniejący plik, 'a' dopisuje dane

    Zwraca:
        True jeśli operacja się powiodła, False w przypadku błędu
    """
    if df.empty:
        logger.warning("Brak danych do zapisania.")
        return False

    filepath = os.path.join(DATA_DIR, filename)

    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if mode == "w":
            logger.info(f"Zapisuję dane do pliku {filepath}...")
            df.to_csv(filepath, index=False)
        elif mode == "a":
            logger.info(f"Dopisuję dane do pliku {filepath}...")
            df.to_csv(filepath, mode="a", header=False, index=False)
        else:
            logger.error("Nieprawidłowy tryb zapisu. Użyj 'w' lub 'a'.")
            return False

        logger.info(f"Dane zostały zapisane do {filepath}.")
        return True

    except Exception as e:
        logger.error(f"Błąd podczas zapisywania danych do {filepath}: {e}")
        return False


def update_stock_data(ticker: str, filename: str = None) -> pd.DataFrame | None:
    """
    Aktualizuje dane dla danego tickera, sprawdzając, czy istnieją dane lokalnego
    i pobierając tylko brakujące dane.

    Argumenty:
        ticker: Symbol akcji (np. 'AAPL')
        filename: Nazwa pliku do odczytu/zapisu (domyślnie '{ticker}.csv')

    Zwraca:
        DataFrame z zaktualizowanymi danymi lub None w przypadku błędu.
    """
    if filename is None:
        filename = f"{ticker}.csv"

    filepath = os.path.join(DATA_DIR, filename)
    today = date.today().strftime("%Y-%m-%d")

    if os.path.exists(filepath):
        local_data = load_local_csv(filepath)
        if not local_data.empty:
            last_date = local_data["Date"].max().strftime("%Y-%m-%d")

            new_data = download_data(ticker, last_date, today)

            if new_data is not None and not new_data.empty:
                combined_data = pd.concat([local_data, new_data])
                combined_data.drop_duplicates(
                    subset=["Date"], keep="last", inplace=True
                )

                save_data(combined_data, filename)
                logger.info(f"Zaktualizowane dane dla {ticker}.")
                return combined_data
            else:
                logger.info(f"Brak nowych danych dla {ticker}.")
                return local_data

    start_date = "2000-01-01"
    data = download_data(ticker, start_date, today)

    if data is not None and not data.empty:
        save_data(data, filename)
        logger.info(f"Pobrano kompletne dane dla {ticker}.")
        return data

    logger.error(f"Nie udało się pobrać danych dla {ticker}.")
    return None
