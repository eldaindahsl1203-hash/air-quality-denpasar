import os
import requests
import pandas as pd
import geopandas as gpd

from datetime import datetime, timedelta, timezone
from shapely.geometry import Point
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

BASE_FOLD = os.getenv(
    "BASE_FOLD",
    "C:/Users/VICTUS/airflow/air-quality-denpasar"
)

OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")

if not OPENAQ_API_KEY:
    raise ValueError("OPENAQ_API_KEY belum tersedia di file .env")

OPENAQ_BASE_URL = "https://api.openaq.org/v3"

REQUEST_TIMEOUT = int(
    os.getenv("REQUEST_TIMEOUT", "60")
)

BACKFILL_DAYS = int(
    os.getenv("BACKFILL_DAYS", "30")
)

# Jika START_DATE dan END_DATE diisi, periode ini dipakai.
# Jika tidak, program mengambil BACKFILL_DAYS terakhir.
START_DATE = os.getenv("START_DATE")
END_DATE = os.getenv("END_DATE")


# ============================================================
# DENPASAR BOUNDARY
# ============================================================

DENPASAR_BOUNDARY_PATH = os.getenv(
    "DENPASAR_BOUNDARY_PATH",
    os.path.join(
        BASE_FOLD,
        "data",
        "boundary",
        "denpasar.shp"
    )
)

# BBOX hanya untuk mencari kandidat OpenAQ.
# Batas final tetap ditentukan oleh polygon Denpasar.
DENPASAR_BBOX = (
    115.10,
    -8.80,
    115.30,
    -8.50
)

# ============================================================
# HEADERS
# ============================================================

OPENAQ_HEADERS = {
    "X-API-Key": OPENAQ_API_KEY
}


# ============================================================
# HTTP HELPER
# ============================================================

def get_json(
    url,
    params=None,
    headers=None,
    timeout=REQUEST_TIMEOUT
):
    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=timeout
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# TIME RANGE
# ============================================================

def get_backfill_range():
    """
    Menghasilkan satu periode waktu yang dipakai bersama oleh
    OpenAQ, Open-Meteo, NASA POWER, dan transform.

    Interval internal:
        [start_time, end_time)

    Artinya start termasuk, end tidak termasuk.
    """

    if START_DATE and END_DATE:
        start_time = datetime.strptime(
            START_DATE,
            "%Y-%m-%d"
        ).replace(tzinfo=timezone.utc)

        end_time = datetime.strptime(
            END_DATE,
            "%Y-%m-%d"
        ).replace(tzinfo=timezone.utc)

        if end_time <= start_time:
            raise ValueError(
                "END_DATE harus lebih besar dari START_DATE."
            )

        return start_time, end_time

    end_time = datetime.now(timezone.utc)
    start_time = (
        end_time
        - timedelta(days=BACKFILL_DAYS)
    )

    return start_time, end_time


# ============================================================
# LOAD DENPASAR BOUNDARY
# ============================================================

def load_denpasar_boundary():

    print("\n================================")
    print("LOAD BOUNDARY DENPASAR")
    print("================================")

    boundary = gpd.read_file(
        DENPASAR_BOUNDARY_PATH
    )

    if boundary.empty:
        raise ValueError(
            "Boundary Kota Denpasar kosong."
        )

    print(
        f"CRS asli       : {boundary.crs}"
    )

    # Koordinat OpenAQ/API berbentuk latitude-longitude,
    # sehingga boundary harus berada pada WGS84.
    boundary = boundary.to_crs(
        "EPSG:4326"
    )

    print(
        f"CRS setelahnya : {boundary.crs}"
    )

    denpasar_polygon = boundary.geometry.union_all()

    print(
        f"Jumlah polygon : {len(boundary)}"
    )

    print(
        "Boundary berhasil digabung."
    )

    return denpasar_polygon


# ============================================================
# CHECK POINT
# ============================================================

def is_inside_denpasar(
    latitude,
    longitude,
    denpasar_polygon
):

    if latitude is None or longitude is None:
        return False

    try:
        point = Point(
            float(longitude),
            float(latitude)
        )

        return denpasar_polygon.covers(
            point
        )

    except (TypeError, ValueError):
        return False


# ============================================================
# OPENAQ: GET LOCATIONS
# ============================================================

def get_candidate_locations():

    print("\n================================")
    print("OPENAQ: CARI KANDIDAT LOCATION")
    print("================================")

    params = {
        "bbox": ",".join(
            map(str, DENPASAR_BBOX)
        ),
        "limit": 1000,
        "page": 1
    }

    all_locations = []

    while True:

        data = get_json(
            f"{OPENAQ_BASE_URL}/locations",
            params=params,
            headers=OPENAQ_HEADERS
        )

        results = data.get(
            "results",
            []
        )

        if not results:
            break

        all_locations.extend(
            results
        )

        meta = data.get(
            "meta",
            {}
        )

        found = meta.get("found")
        page = meta.get(
            "page",
            params["page"]
        )
        limit = meta.get(
            "limit",
            params["limit"]
        )

        print(
            f"page={page} | "
            f"rows={len(results)} | "
            f"total={found}"
        )

        if found is None:
            break

        if page * limit >= found:
            break

        params["page"] += 1

    print(
        f"Total kandidat: {len(all_locations)}"
    )

    return all_locations


def filter_denpasar_locations(
    locations,
    denpasar_polygon
):

    denpasar_locations = []

    for location in locations:

        coordinates = location.get(
            "coordinates",
            {}
        )

        latitude = coordinates.get(
            "latitude"
        )

        longitude = coordinates.get(
            "longitude"
        )

        inside = is_inside_denpasar(
            latitude,
            longitude,
            denpasar_polygon
        )

        print(
            f"{location.get('id')} | "
            f"{location.get('name')} | "
            f"{latitude}, {longitude} | "
            f"Denpasar={inside}"
        )

        if inside:
            denpasar_locations.append(
                location
            )

    print(
        f"\nLocation di Denpasar: "
        f"{len(denpasar_locations)}"
    )

    return denpasar_locations


# ============================================================
# OPENAQ: GET SENSORS
# ============================================================

def get_location_sensors(location_id):

    url = (
        f"{OPENAQ_BASE_URL}/locations/"
        f"{location_id}/sensors"
    )

    data = get_json(
        url,
        headers=OPENAQ_HEADERS
    )

    return data.get(
        "results",
        []
    )


# ============================================================
# OPENAQ: GET MEASUREMENTS
# ============================================================

def get_sensor_measurements(
    sensor_id,
    start_time,
    end_time
):

    url = (
        f"{OPENAQ_BASE_URL}/sensors/"
        f"{sensor_id}/measurements"
    )

    params = {
        "datetime_from": (
            start_time.isoformat()
        ),
        "datetime_to": (
            end_time.isoformat()
        ),
        "limit": 1000,
        "page": 1
    }

    all_measurements = []

    while True:

        data = get_json(
            url,
            params=params,
            headers=OPENAQ_HEADERS
        )

        results = data.get(
            "results",
            []
        )

        if not results:
            break

        all_measurements.extend(
            results
        )

        meta = data.get(
            "meta",
            {}
        )

        found = meta.get("found")
        page = meta.get(
            "page",
            params["page"]
        )
        limit = meta.get(
            "limit",
            params["limit"]
        )

        print(
            f"    sensor={sensor_id} | "
            f"page={page} | "
            f"rows={len(results)}"
        )

        if found is None:
            break

        if page * limit >= found:
            break

        params["page"] += 1

    return all_measurements


# ============================================================
# API 1: OPENAQ
# ============================================================

def extract_openaq(
    start_time,
    end_time
):

    print("\n================================")
    print("EXTRACT API 1: OPENAQ")
    print("================================")

    denpasar_polygon = (
        load_denpasar_boundary()
    )

    candidate_locations = (
        get_candidate_locations()
    )

    locations = (
        filter_denpasar_locations(
            candidate_locations,
            denpasar_polygon
        )
    )

    if not locations:
        raise ValueError(
            "Tidak ditemukan location OpenAQ "
            "di dalam batas Kota Denpasar."
        )

    print(
        f"\nPeriode OpenAQ:\n"
        f"{start_time}\n"
        f"sampai\n"
        f"{end_time}"
    )

    records = []

    for location in locations:

        location_id = location.get("id")
        location_name = location.get("name")

        coordinates = location.get(
            "coordinates",
            {}
        )

        latitude = coordinates.get(
            "latitude"
        )

        longitude = coordinates.get(
            "longitude"
        )

        print(
            f"\nLOCATION: "
            f"{location_id} | "
            f"{location_name}"
        )

        sensors = get_location_sensors(
            location_id
        )

        print(
            f"Jumlah sensor: {len(sensors)}"
        )

        for sensor in sensors:

            sensor_id = sensor.get("id")

            parameter = sensor.get(
                "parameter",
                {}
            )

            parameter_id = parameter.get("id")
            parameter_name = parameter.get("name")
            unit = parameter.get("units")

            print(
                f"  Sensor {sensor_id} | "
                f"{parameter_name}"
            )

            measurements = (
                get_sensor_measurements(
                    sensor_id,
                    start_time,
                    end_time
                )
            )

            for measurement in measurements:

                value = measurement.get("value")

                period = measurement.get(
                    "period",
                    {}
                )

                datetime_from = period.get(
                    "datetimeFrom",
                    {}
                )

                observed_at = datetime_from.get(
                    "utc"
                )

                if observed_at is None:
                    continue

                records.append({
                    "source_api": "openaq",
                    "dataset_type": "air_quality",
                    "location_id": location_id,
                    "location_name": location_name,
                    "latitude": latitude,
                    "longitude": longitude,
                    "sensor_id": sensor_id,
                    "parameter_id": parameter_id,
                    "parameter_name": parameter_name,
                    "value": value,
                    "unit": unit,
                    "observed_at": observed_at,
                    "data_type": "observation"
                })

    print(
        f"\nTotal raw OpenAQ records: "
        f"{len(records)}"
    )

    return records

# ============================================================
# TRANSFORM
# ============================================================

def transform_data(
    openaq_records,
    open_meteo_records,
    nasa_records,
    start_time,
    end_time
):

    print("\n================================")
    print("TRANSFORM DATA")
    print("================================")

    all_records = (
        openaq_records
        + open_meteo_records
        + nasa_records
    )

    if not all_records:
        print(
            "Tidak ada record untuk ditransform."
        )
        return pd.DataFrame()

    df = pd.DataFrame(
        all_records
    )

    # Pastikan semua timestamp UTC.
    df["observed_at"] = pd.to_datetime(
        df["observed_at"],
        errors="coerce",
        utc=True
    )

    # Nilai numerik.
    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce"
    )

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce"
    )

    # Buang record tanpa timestamp/value valid.
    df = df.dropna(
        subset=[
            "observed_at",
            "value"
        ]
    )

    # Terapkan SATU periode yang sama untuk semua API.
    df = df[
        (df["observed_at"] >= start_time)
        & (df["observed_at"] < end_time)
    ].copy()

    # Hilangkan duplicate record.
    df = df.drop_duplicates(
        subset=[
            "source_api",
            "location_id",
            "sensor_id",
            "parameter_name",
            "observed_at"
        ]
    )

    # Urutkan.
    df = df.sort_values(
        [
            "location_id",
            "observed_at",
            "source_api",
            "parameter_name"
        ]
    ).reset_index(
        drop=True
    )

    # Kolom standar ETL.
    preferred_columns = [
        "source_api",
        "dataset_type",
        "location_id",
        "location_name",
        "latitude",
        "longitude",
        "sensor_id",
        "parameter_id",
        "parameter_name",
        "value",
        "unit",
        "observed_at",
        "data_type"
    ]

    existing_columns = [
        column
        for column in preferred_columns
        if column in df.columns
    ]

    df = df[
        existing_columns
    ]

    print(
        f"Total record setelah transform: "
        f"{len(df)}"
    )

    print("\nRecord per source:")
    print(
        df["source_api"]
        .value_counts()
        .to_string()
    )

    print("\nRecord per parameter:")
    print(
        df["parameter_name"]
        .value_counts()
        .to_string()
    )

    return df


# ============================================================
# EXTRACT ALL
# ============================================================

def extract_all(
    start_time,
    end_time
):

    # API 1
    openaq_records = extract_openaq(
        start_time,
        end_time
    )

    return (
        openaq_records,
    )


# ============================================================
# SAVE CSV
# ============================================================

def save_processed_data(df):

    if df.empty:
        print(
            "DataFrame kosong. CSV tidak dibuat."
        )
        return None

    output_dir = os.path.join(
        BASE_FOLD,
        "data",
        "processed"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    output_path = os.path.join(
        output_dir,
        "denpasar_3api_30days.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nCSV berhasil disimpan:\n"
        f"{output_path}"
    )

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    start_time, end_time = (
        get_backfill_range()
    )

    print("\n================================")
    print("ETL TRANSFORM DENPASAR")
    print("================================")

    print(
        f"Periode UTC:\n"
        f"{start_time}\n"
        f"sampai\n"
        f"{end_time}"
    )

    openaq_records, open_meteo_records, nasa_records = (
        extract_all(
            start_time,
            end_time
        )
    )

    df = transform_data(
        openaq_records,
        open_meteo_records,
        nasa_records,
        start_time,
        end_time
    )

    if df.empty:
        raise ValueError(
            "Hasil transform kosong."
        )

    print("\n================================")
    print("SAMPLE HASIL TRANSFORM")
    print("================================")

    print(
        df.head(10).to_string(
            index=False
        )
    )

    save_processed_data(df)

    return df


if __name__ == "__main__":
    main()
