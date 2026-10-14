import os
import pandas as pd

INPUT_FILE = os.path.join(
    "include",
    "data",
    "air_quality_ml.csv",
)

OUTPUT_FILE = os.path.join(
    "include",
    "data",
    "air_quality_documentation.csv",
)

def load_data():
    print("Mengambil dataset hasil feature engineering...")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"File input tidak ditemukan: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Dataset berhasil dibaca: {len(df)} baris.")
    return df


def validate_data(df):
    required_columns = [
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

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Kolom yang dibutuhkan tidak ditemukan: {missing_columns}"
        )

    print("Validasi dataset berhasil.")


def prepare_dataset(df):
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df = df.sort_values("date").reset_index(drop=True)

    return df


def save_dataset(df):
    os.makedirs(
        os.path.join("include", "data"),
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Dataset dokumentasi berhasil disimpan: "
        f"{OUTPUT_FILE}"
    )

    print(f"Jumlah baris: {len(df)}")
    print(f"Jumlah kolom: {len(df.columns)}")


def main():
    print("=" * 60)
    print("SAVE DATASET DOKUMENTASI")
    print("=" * 60)

    df = load_data()
    validate_data(df)
    df = prepare_dataset(df)
    save_dataset(df)

    print("Penyimpanan dataset dokumentasi selesai.")


if __name__ == "__main__":
    main()