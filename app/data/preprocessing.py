from numpy._core import numeric
import pandas as pd
import logging
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)


def create_target_variable(
    df: pd.DataFrame, prediction_horizon: int = 1, target_type: str = "price"
) -> pd.DataFrame:
    """
    Funkcja tworzy kolumnę docelową ('Target') na podstawie danych cenowych,
    która będzie używana w modelu predykcyjnym

    Argumenty:
        df: Ramka danych giełdowych
        prediction_horizon: Okres na jak daleko w przód chcemy przewidzieć wartość
        target_type: Przyjmuje parametry: 'price' - celem przewidywania jest wartość ceny,
        'return' - zmiana procentowa względem bieżącego dnia,

    Zwraca:
        DataFrame z nową kolumną
    """
    if not (isinstance(prediction_horizon, int) and prediction_horizon > 0):
        logger.error("[BłĄD] prediction_horizon nie jest dodatnią liczbą całkowitą")
        raise ValueError(prediction_horizon)

    if target_type not in ["price", "return"]:
        logger.error("[BŁĄD] target_type nie jest jednym z: 'price', 'return'")
        raise ValueError(target_type)

    df_target = df.copy()

    if target_type == "price":
        df_target["Target"] = df_target["Close"].shift(-prediction_horizon)
    elif target_type == "return":
        df_target["Target"] = df_target["Close"].pct_change().shift(-prediction_horizon)

    logger.info(f"Pomyślnie utworzono kolumnę Target typu {target_type}")
    logger.warning(
        f"Liczba wartości NaN w kolumnie 'Target': {df_target['Target'].isna().sum()}"
    )

    return df_target


def prepare_training_data(
    df: pd.DataFrame, target_column: str = "Target", test_size: float = 0.2
) -> tuple:
    """
    Funkcja dzieli ramkę na zbiory: treningowy, walidacyjny i testowy, zachowując chronologiczny
    porządek danych, przygotowywuje dane do modelowania

    Argumenty:
        df: Ramka danych zawierająca cechy i kolumnę docelową
        target_column: nazwa kolumny docelowej (domyślnie Target, która zawiera wartości
        do przewidywania)
        test_size: Proporcja danych przeznaczona na zbiór testowy (wartość od 0 do 1),
        domyślnie 0.2, czyli 20%

    Zwraca:
        Krotka zawierająca trzy ramki danych: train_df, val_df, test_df
    """
    if target_column not in df.columns:
        logger.error("Brak kolumny 'Target' w ramce danych")
        raise ValueError(f"Kolumna {target_column} nie istnieje w ramce danych")

    if not 0 < test_size < 1:
        logger.error(f"Zła wartość test_size: {test_size}")
        raise ValueError("test_size musi być w przedziale (0, 1)")

    df_prep = df.copy()

    n_test = int(len(df_prep) * test_size)
    n_val = int((len(df_prep) - n_test) * 0.2)
    n_train = len(df_prep) - n_test - n_val

    train_df = df_prep.iloc[:n_train]
    val_df = df_prep.iloc[n_train : n_train + n_val]
    test_df = df_prep.iloc[n_train + n_val :]
    logger.info(
        f"Ilość wierszy w ramkach: train = {n_train}, val = {n_val}, test = {n_test}"
    )

    return (train_df, val_df, test_df)


def scale_features(
    df: pd.DataFrame, method: str = "minmax", exclude_columns: list = None
) -> tuple:
    """
    Funkcja skaluje cechy numeryczne w ramce danych, by przygotować je do modelowania,
    z wykluczeniem kolumn określonych w exclude_columns

    Argumenty:
        df: Ramka danych z cechami i kolumną docelową
        method: Metoda skalowania (domyślnie 'minmax'): 'minmax': skalowanie do pedziału [0,1],
        'standard': standaryzacja do średniej = 0 i odchylenia standardowego = 1
        exclude_columns: Lista kolumn, które nie powinny być skalowane (np. ['Target'] lub inne
        nienumeryczne kolumny

    Zwraca:
        Krotka z przeskalowaną ramką danych i obiektem skalera
    """
    if method not in ["minmax", "standard"]:
        logger.error(f"Błędna metoda skalowania: {method}")
        raise ValueError("Metoda skalowania musi być 'minmax' lub 'standard'")

    if exclude_columns is None:
        exclude_columns = []
    elif not isinstance(exclude_columns, list):
        logger.error(f"exclude_columns nie jest listą: {exclude_columns}")
        raise ValueError("exclude_columns musi być listą")

    df_scale = df.copy()

    included_col = [col for col in df_scale.columns if col not in exclude_columns]
    numeric_col = df_scale[included_col].select_dtypes(include="number").columns
    if len(numeric_col) == 0:
        logger.error("Brak kolumn numerycznych do przeskalowania")
        return (df.copy(), None)

    if method == "minmax":
        scaler = MinMaxScaler()
    elif method == "standard":
        scaler = StandardScaler()

    scaler = scaler.fit(df_scale[numeric_col])
    scaled_data = scaler.transform(df_scale[numeric_col])
    scaled_df = pd.DataFrame(scaled_data, columns=numeric_col, index=df_scale.index)
    for col in exclude_columns:
        scaled_df[col] = df_scale[col]
    logger.info(f"Przeskalowano {len(numeric_col)} kolumn metodą {method}")

    return (scaled_df, scaler)


def create_sequences(
    df: pd.DataFrame,
    sequence_length: int,
    target_column: str = "Target",
    feature_columns: list = None,
) -> tuple:
    """
    Funkcja przygotowywuje sekwencje danych do modelowania szeregów czasowych. Sekwencje umożliwiają
    modelowi analizę wzorców w danych na przestrzeni czasu

    Argumenty:
        df: DataFrame zawierająca przeskalowane cechy i kolumnę docelową (np. 'Target')
        sequence_length: długość każdej sekwencji (liczba kroków czasowych w jednej sekwencji, np. 20 dni)
        target_column: nazwa kolumny docelowej (domyślnie 'Target'), która zawiera wartości do przewidywania
        feature_columns: lista kolumn cech do użycia w sekwencjach, jeśli None, wybiera wszystkie kolumny
        numeryczne poza target_column

    Zwraca:
        Krotka (X, y), gdzie:
        X to tablica 3D NumPy zawierająca sekwencje cech
        y to tablica 1D lub 2D NumPy zawierająca wartości docelowe dla każdej sekwencji
    """
    if target_column not in df.columns:
        logger.error(f"Brak kolumny docelowej {target_column}")
        raise ValueError(f"Brak kolumny docelowej w ramce: {df.columns}")

    if not (isinstance(sequence_length, int) and sequence_length > 0):
        logger.error(f"Zła wartość sequence_length: {sequence_length}")
        raise ValueError("sequence_length powinno być dodatnią liczbą całkowitą")

    if sequence_length >= len(df):
        logger.error(
            f"sequence_length: {sequence_length} jest większy lub równy liczbie wierszy w ramce: {len(df)}"
        )
        raise ValueError("sequence_length musi być mniejsze od liczby wierszy w ramce")

    if feature_columns is None:
        feature_columns = [
            col
            for col in df.select_dtypes(include=[np.number]).columns
            if col != target_column
        ]

    if not all(col in df.columns for col in feature_columns):
        logger.error(
            f"Niektóre wartości z feature_columns nie istnieją: {feature_columns}"
        )
        raise ValueError("Brak odpowiednich kolumn")

    X_sequences = []
    y_sequences = []
    skipped_sequences = 0

    for i in range(0, len(df) - sequence_length):
        seq = df[feature_columns].iloc[i : i + sequence_length]
        val = df[target_column].iloc[i + sequence_length]

        if np.any(pd.isna(seq)) or np.any(pd.isna(val)):
            logger.warning(f"Pominięto sekwencję o indeksie {i} z powodu wartości NaN")
            skipped_sequences += 1
            continue

        X_sequences.append(seq)
        y_sequences.append(val)

    if skipped_sequences > 0:
        logger.warning(f"Pominięto {skipped_sequences} sekwencji")

    if len(X_sequences) == 0:
        logger.error("Nie utworzono żadnych sekwencji")
        raise ValueError("Brak prawidłowych sekwencji do utworzenia")

    X = np.array(X_sequences)
    y = np.array(y_sequences).reshape(-1, 1)

    logger.info(
        f"Utworzono {len(X_sequences)} sekwencji o długości {sequence_length}. Kształt X: {X.shape}, Kształt y: {y.shape}"
    )

    return (X, y)
