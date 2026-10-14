import os
import pandas as pd

INPUT_FILE = os.path.join(
    "include",
    "data",
    "air_quality_prepared.csv",
)

OUTPUT_FILE = os.path.join(
    "include",
    "data",
    "air_quality_labelled.csv",
)


def load_data():
    print("Membaca dataset prepared...")

    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        raise ValueError(
            "Dataset prepared kosong."
        )

    print(
        f"Dataset berhasil dibaca: {len(df)} baris."
    )

    return df


def prepare_daily_data(df):
    print("\n=== AGREGASI HARIAN ===")

    df = df.copy()

    df["observed_at"] = pd.to_datetime(
        df["observed_at"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["observed_at"]
    )

    if "pm25" not in df.columns:
        raise ValueError(
            "Kolom pm25 tidak ditemukan."
        )

    df["date"] = (
        df["observed_at"]
        .dt.date
    )

    daily = (
        df.groupby("date")
        .agg(
            pm25=("pm25", "mean"),
            pm1=("pm1", "mean"),
            relativehumidity=(
                "relativehumidity",
                "mean"
            ),
            temperature=(
                "temperature",
                "mean"
            ),
            um003=("um003", "mean"),
        )
        .reset_index()
    )

    print(
        f"Jumlah hari: {len(daily)}"
    )

    return daily


def calculate_ispu_pm25(concentration):
    """
    Menghitung ISPU PM2.5 berdasarkan
    breakpoint konsentrasi PM2.5.
    """

    breakpoints = [
        (0.0, 15.5, 0, 50),
        (15.5, 55.4, 51, 100),
        (55.4, 150.4, 101, 200),
        (150.4, 250.4, 201, 300),
        (250.4, 500.4, 301, 500),
    ]

    if pd.isna(concentration):
        return None

    if concentration < 0:
        return None

    for c_low, c_high, i_low, i_high in breakpoints:

        if concentration <= c_high:

            ispu = (
                (i_high - i_low)
                / (c_high - c_low)
                * (concentration - c_low)
                + i_low
            )

            return round(ispu, 2)

    return 500.0


def create_label(ispu):
    if pd.isna(ispu):
        return None

    if ispu <= 50:
        return "Baik"

    elif ispu <= 100:
        return "Sedang"

    elif ispu <= 200:
        return "Tidak Sehat"

    elif ispu <= 300:
        return "Sangat Tidak Sehat"

    else:
        return "Berbahaya"


def create_labels(df):
    print("\n=== PEMBUATAN LABEL ===")

    df = df.copy()

    df["ispu_pm25"] = (
        df["pm25"]
        .apply(calculate_ispu_pm25)
    )

    df["air_quality_label"] = (
        df["ispu_pm25"]
        .apply(create_label)
    )

    print("\nDistribusi label:")

    print(
        df["air_quality_label"]
        .value_counts(dropna=False)
    )

    return df


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
        "\nDataset labelled berhasil disimpan:"
    )

    print(OUTPUT_FILE)


def main():

    print(
        "=== LABELLING DATA DENPASAR ==="
    )

    df = load_data()

    df = prepare_daily_data(df)

    df = create_labels(df)

    save_dataset(df)

    print(
        "\n=== LABELLING SELESAI ==="
    )


if __name__ == "__main__":
    main()