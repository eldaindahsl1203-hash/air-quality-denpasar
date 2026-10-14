import os

import mysql.connector
import pandas as pd
from dotenv import load_dotenv


load_dotenv()


# ============================================================
# KONFIGURASI
# ============================================================

DB_CONFIG = {
    "host": os.getenv(
        "MYSQL_HOST",
        "host.docker.internal",
    ),
    "port": int(
        os.getenv("MYSQL_PORT", 3306)
    ),
    "user": os.getenv(
        "MYSQL_USER",
        "root",
    ),
    "password": os.getenv(
        "MYSQL_PASSWORD"
    ),
    "database": os.getenv(
        "MYSQL_DATABASE",
        "air_quality_db",
    ),
}

TABLE_NAME = "air_quality_denpasar_raw"

OUTPUT_DIR = os.path.join("include", "data")
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "air_quality_prepared.csv",
)

# ============================================================
# LOAD DATA DARI MYSQL
# ============================================================

def load_data():

    print(
        "Mengambil data raw "
        "Denpasar dari MySQL..."
    )

    conn = mysql.connector.connect(
        **DB_CONFIG
    )

    query = f"""
        SELECT
            sensor_id,
            parameter_name,
            value,
            unit,
            observed_at
        FROM {TABLE_NAME}
        ORDER BY observed_at ASC
    """

    df = pd.read_sql(
        query,
        conn,
    )

    conn.close()

    print(
        f"Data raw berhasil diambil: "
        f"{len(df)} baris."
    )

    return df


# ============================================================
# VALIDASI DATA AWAL
# ============================================================

def validate_raw_data(df):

    print(
        "\n=== VALIDASI DATA RAW ==="
    )

    if df.empty:
        raise ValueError(
            "Data raw kosong."
        )

    required_columns = [
        "sensor_id",
        "parameter_name",
        "value",
        "unit",
        "observed_at",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Kolom berikut tidak ditemukan: "
            + ", ".join(
                missing_columns
            )
        )

    print(
        "Kolom raw lengkap."
    )


# ============================================================
# KONVERSI TIMESTAMP
# ============================================================

def prepare_timestamp(df):

    df = df.copy()

    df["observed_at"] = pd.to_datetime(
        df["observed_at"],
        errors="coerce",
    )

    invalid_timestamp = (
        df["observed_at"].isna().sum()
    )

    if invalid_timestamp > 0:

        print(
            f"Menghapus "
            f"{invalid_timestamp} "
            "timestamp tidak valid."
        )

        df = df.dropna(
            subset=["observed_at"]
        )

    return df


# ============================================================
# KONVERSI NILAI
# ============================================================

def prepare_value(df):

    df = df.copy()

    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce",
    )

    invalid_value = (
        df["value"].isna().sum()
    )

    if invalid_value > 0:

        print(
            f"Menghapus "
            f"{invalid_value} "
            "nilai tidak valid."
        )

        df = df.dropna(
            subset=["value"]
        )

    return df


# ============================================================
# PIVOT DATA
# ============================================================

def pivot_parameters(df):

    print(
        "\n=== PIVOT PARAMETER ==="
    )

    pivot = (
        df.pivot_table(
            index="observed_at",
            columns="parameter_name",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
    )

    pivot.columns.name = None

    print(
        "Parameter berhasil diubah "
        "menjadi kolom."
    )

    return pivot


# ============================================================
# NORMALISASI NAMA KOLOM
# ============================================================

def normalize_columns(df):

    df = df.copy()

    rename_columns = {
        "pm25": "pm25",
        "pm1": "pm1",
        "relativehumidity": "relativehumidity",
        "temperature": "temperature",
        "um003": "um003",
    }

    df = df.rename(
        columns=rename_columns
    )

    return df


# ============================================================
# CEK DATASET HASIL PREPARASI
# ============================================================

def validate_prepared_data(df):

    print(
        "\n=== DATASET PREPARED ==="
    )

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

    print(
        "\nMissing value:"
    )

    missing = (
        df.isnull()
        .sum()
    )

    print(
        missing[
            missing > 0
        ]
    )

    print(
        "\nContoh data:"
    )

    print(
        df.head()
    )


# ============================================================
# SIMPAN DATASET
# ============================================================

def save_dataset(df):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "\nDataset prepared "
        "berhasil disimpan:"
    )

    print(
        OUTPUT_FILE
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    validate_raw_data(
        df
    )

    df = prepare_timestamp(
        df
    )

    df = prepare_value(
        df
    )

    df = pivot_parameters(
        df
    )

    df = normalize_columns(
        df
    )

    validate_prepared_data(
        df
    )

    save_dataset(
        df
    )

    print(
        "\n=== PREPARE DATASET "
        "SELESAI ==="
    )


if __name__ == "__main__":
    main()