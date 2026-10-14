import os

import mysql.connector
import pandas as pd
from dotenv import load_dotenv


load_dotenv()


# ============================================================
# KONFIGURASI DATABASE
# ============================================================

DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "host.docker.internal"),
    "port": int(os.getenv("MYSQL_PORT", 3306)),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv(
        "MYSQL_DATABASE",
        "air_quality_db",
    ),
}

TABLE_NAME = "air_quality_denpasar_raw"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    conn = mysql.connector.connect(**DB_CONFIG)

    query = f"""
        SELECT
            id,
            location_id,
            location_name,
            sensor_id,
            parameter_id,
            parameter_name,
            value,
            unit,
            latitude,
            longitude,
            observed_at,
            ingested_at
        FROM {TABLE_NAME}
        ORDER BY observed_at ASC
    """

    df = pd.read_sql(query, conn)

    conn.close()

    return df


# ============================================================
# QUALITY CHECK
# ============================================================

def check_missing(df):
    print("\n=== 1. MISSING VALUE ===")

    missing = df.isnull().sum()

    if missing.sum() == 0:
        print("Tidak ada missing value.")
        return

    print("\nKolom dengan missing value:")

    print(
        missing[missing > 0]
        .sort_values(ascending=False)
    )

    # Latitude dan longitude boleh NULL
    # karena koordinat tidak selalu tersedia
    # pada measurement OpenAQ.
    metadata_columns = [
        "latitude",
        "longitude",
    ]

    critical_missing = missing.drop(
        labels=metadata_columns,
        errors="ignore",
    )

    if critical_missing.sum() == 0:
        print(
            "\nTidak ada missing value pada "
            "kolom data utama."
        )
    else:
        print(
            "\nTerdapat missing value pada "
            "kolom data utama."
        )


def check_duplicate(df):
    print("\n=== 2. DUPLICATE ===")

    duplicate = df.duplicated(
        subset=[
            "sensor_id",
            "observed_at",
        ]
    ).sum()

    print(
        f"Jumlah duplicate: {duplicate}"
    )

    if duplicate == 0:
        print(
            "Tidak ditemukan duplicate "
            "sensor dan timestamp."
        )


def check_value(df):
    print("\n=== 3. VALIDASI NILAI ===")

    invalid = df[
        (df["value"].notna())
        & (df["value"] < 0)
    ]

    if len(invalid) == 0:
        print(
            "Tidak ditemukan nilai negatif."
        )
    else:
        print(
            f"Ditemukan {len(invalid)} "
            "nilai negatif."
        )

        print(
            invalid[
                [
                    "sensor_id",
                    "parameter_name",
                    "value",
                    "observed_at",
                ]
            ].head(10)
        )


def check_humidity(df):
    print(
        "\n=== 4. VALIDASI "
        "RELATIVE HUMIDITY ==="
    )

    humidity = df[
        df["parameter_name"]
        == "relativehumidity"
    ].copy()

    if humidity.empty:
        print(
            "Data relativehumidity "
            "tidak ditemukan."
        )
        return

    invalid = humidity[
        (humidity["value"] < 0)
        | (humidity["value"] > 100)
    ]

    if len(invalid) == 0:
        print(
            "Rentang relative humidity valid."
        )
    else:
        print(
            f"Ditemukan {len(invalid)} "
            "data humidity tidak valid."
        )


def check_coordinates(df):
    print("\n=== 5. VALIDASI KOORDINAT ===")

    invalid_lat = df[
        (df["latitude"].notna())
        & (
            (df["latitude"] < -90)
            | (df["latitude"] > 90)
        )
    ]

    invalid_lon = df[
        (df["longitude"].notna())
        & (
            (df["longitude"] < -180)
            | (df["longitude"] > 180)
        )
    ]

    print(
        f"Latitude tidak valid : "
        f"{len(invalid_lat)}"
    )

    print(
        f"Longitude tidak valid: "
        f"{len(invalid_lon)}"
    )


def check_time_series(df):
    print("\n=== 6. TIME SERIES ===")

    df = df.sort_values(
        [
            "sensor_id",
            "observed_at",
        ]
    ).copy()

    print(
        "Data pertama :",
        df["observed_at"].min(),
    )

    print(
        "Data terakhir:",
        df["observed_at"].max(),
    )

    print(
        "\nInterval antar data per sensor:"
    )

    for sensor_id, group in df.groupby(
        "sensor_id"
    ):
        group = group.sort_values(
            "observed_at"
        )

        time_diff = (
            group["observed_at"]
            .diff()
            .dropna()
        )

        print(
            f"\nSensor {sensor_id}:"
        )

        if time_diff.empty:
            print(
                "Tidak cukup data untuk "
                "menghitung interval."
            )
        else:
            print(
                time_diff
                .value_counts()
                .head(5)
            )


def check_parameter_distribution(df):
    print(
        "\n=== 7. DISTRIBUSI PARAMETER ==="
    )

    distribution = (
        df.groupby("parameter_name")
        .agg(
            jumlah_data=("value", "count"),
            minimum=("value", "min"),
            maksimum=("value", "max"),
            rata_rata=("value", "mean"),
        )
        .round(3)
    )

    print(distribution)


def check_sensor_distribution(df):
    print(
        "\n=== 8. DISTRIBUSI SENSOR ==="
    )

    distribution = (
        df.groupby(
            [
                "sensor_id",
                "parameter_name",
            ]
        )
        .agg(
            jumlah_data=("value", "count"),
            data_pertama=(
                "observed_at",
                "min",
            ),
            data_terakhir=(
                "observed_at",
                "max",
            ),
        )
        .sort_index()
    )

    print(distribution)


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Mengambil data Denpasar "
        "dari MySQL Docker..."
    )

    print(
        f"Host MySQL : {DB_CONFIG['host']}"
    )

    print(
        f"Port MySQL : {DB_CONFIG['port']}"
    )

    print(
        f"Database   : {DB_CONFIG['database']}"
    )

    df = load_data()

    print("\n=== DATASET INFO ===")

    print(
        f"Jumlah baris : {len(df)}"
    )

    print(
        f"Jumlah kolom : {len(df.columns)}"
    )

    if df.empty:
        print(
            "\nDataset masih kosong."
        )
        return

    print("\nKolom:")

    print(
        df.columns.tolist()
    )

    check_missing(df)

    check_duplicate(df)

    check_value(df)

    check_humidity(df)

    check_coordinates(df)

    check_time_series(df)

    check_parameter_distribution(df)

    check_sensor_distribution(df)

    print(
        "\n=== QUALITY CHECK SELESAI ==="
    )


if __name__ == "__main__":
    main()