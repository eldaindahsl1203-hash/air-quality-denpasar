import os
import pandas as pd

INPUT_FILE = os.path.join(
    "include",
    "data",
    "air_quality_labelled.csv",
)

OUTPUT_FILE = os.path.join(
    "include",
    "data",
    "air_quality_ml.csv",
)


def load_data():
    print("Membaca dataset labelled...")

    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        raise ValueError(
            "Dataset labelled kosong."
        )

    print(
        f"Dataset labelled: {len(df)} baris."
    )

    return df


def validate_data(df):
    print("\n=== VALIDASI DATA ===")

    required_columns = [
        "date",
        "pm25",
        "pm1",
        "relativehumidity",
        "temperature",
        "um003",
        "ispu_pm25",
        "air_quality_label",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Kolom berikut tidak ditemukan: "
            + ", ".join(missing_columns)
        )

    print("Semua kolom yang diperlukan tersedia.")


def prepare_features(df):
    print("\n=== FEATURE ENGINEERING ===")

    df = df.copy()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df = df.sort_values("date")

    # Fitur waktu
    df["day_of_week"] = (
        df["date"].dt.dayofweek
    )

    df["month"] = (
        df["date"].dt.month
    )

    df["day_of_month"] = (
        df["date"].dt.day
    )

    # Lag PM2.5
    df["pm25_lag_1"] = (
        df["pm25"].shift(1)
    )

    df["pm25_lag_2"] = (
        df["pm25"].shift(2)
    )

    df["pm25_lag_3"] = (
        df["pm25"].shift(3)
    )

    # Rolling PM2.5
    df["pm25_rolling_mean_3"] = (
        df["pm25"]
        .rolling(window=3)
        .mean()
    )

    df["pm25_rolling_std_3"] = (
        df["pm25"]
        .rolling(window=3)
        .std()
    )

    print(
        "Fitur waktu, lag, dan rolling berhasil dibuat."
    )

    return df


def clean_features(df):
    print("\n=== CLEANING FEATURE ===")

    df = df.copy()

    feature_columns = [
        "pm25",
        "pm1",
        "relativehumidity",
        "temperature",
        "um003",
        "pm25_lag_1",
        "pm25_lag_2",
        "pm25_lag_3",
        "pm25_rolling_mean_3",
        "pm25_rolling_std_3",
    ]

    before = len(df)

    df = df.dropna(
        subset=feature_columns
    )

    after = len(df)

    print(
        f"Baris sebelum cleaning : {before}"
    )

    print(
        f"Baris sesudah cleaning : {after}"
    )

    return df


def select_columns(df):
    print("\n=== PEMILIHAN KOLOM ===")

    columns = [
        "date",
        "pm25",
        "pm1",
        "relativehumidity",
        "temperature",
        "um003",
        "pm25_lag_1",
        "pm25_lag_2",
        "pm25_lag_3",
        "pm25_rolling_mean_3",
        "pm25_rolling_std_3",
        "ispu_pm25",
        "air_quality_label",
    ]

    df = df[columns]

    return df


def validate_final_dataset(df):
    print("\n=== DATASET ML ===")

    print(
        f"Jumlah baris : {len(df)}"
    )

    print(
        f"Jumlah kolom : {len(df.columns)}"
    )

    print("\nKolom:")

    print(
        df.columns.tolist()
    )

    print("\nDistribusi target:")

    print(
        df["air_quality_label"]
        .value_counts()
    )

    print("\nMissing value:")

    missing = df.isnull().sum()

    print(
        missing[missing > 0]
    )

    print("\nContoh data:")

    print(
        df.head()
    )


def save_dataset(df):
    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\nDataset ML berhasil disimpan:"
    )

    print(OUTPUT_FILE)


def main():

    print(
        "=== FEATURE ENGINEERING DENPASAR ==="
    )

    df = load_data()

    validate_data(df)

    df = prepare_features(df)

    df = clean_features(df)

    df = select_columns(df)

    validate_final_dataset(df)

    save_dataset(df)

    print(
        "\n=== FEATURE ENGINEERING SELESAI ==="
    )


if __name__ == "__main__":
    main()