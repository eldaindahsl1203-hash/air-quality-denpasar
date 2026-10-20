"""
Air Quality Denpasar - NASA POWER DAG

Self-contained:
API -> transform -> CSV

Parameter:
ALLSKY_SFC_SW_DWN

Backfill:
30 hari data valid terbaru per lokasi.

Catatan:
- Tidak menggunakan batas pencarian 60 hari.
- Data dicari mundur secara bertahap sampai 30 hari data hourly
  yang valid ditemukan.
- Pencarian dapat berjalan jauh ke belakang hingga awal histori NASA POWER.
- Timestamp diproses menggunakan Python datetime, tanpa pandas.
"""

import os
from datetime import datetime, timedelta, timezone

from airflow.decorators import dag, task


# ============================================================
# NASA POWER CONFIGURATION
# ============================================================

NASA_URL = (
    "https://power.larc.nasa.gov/api/temporal/hourly/point"
)

PARAMETER = "ALLSKY_SFC_SW_DWN"

# 30 hari x 24 jam
BACKFILL_HOURS = 30 * 24

# Ukuran pencarian setiap request.
SEARCH_CHUNK_DAYS = 90

# Batas awal histori yang diperiksa.
NASA_HISTORY_START = datetime(
    2001,
    1,
    1,
    tzinfo=timezone.utc,
)

OUTPUT_DIR = "/usr/local/airflow/data/processed"


# ============================================================
# LOCATIONS
# ============================================================

LOCATIONS = [
    {
        "id": "6494341",
        "name": "Shiva Industries",
        "lat": -8.6418293,
        "lon": 115.1994208,
    },
    {
        "id": "6529627",
        "name": "Segara Ayu Denpasar",
        "lat": -8.68279,
        "lon": 115.26208,
    },
    {
        "id": "6536246",
        "name": "Cactus House",
        "lat": -8.70913,
        "lon": 115.24927,
    },
]


# ============================================================
# NASA POWER VALUE VALIDATION
# ============================================================

def is_valid_value(value):
    """
    Memeriksa apakah nilai NASA POWER valid.

    Nilai None dan nilai <= -900 dianggap tidak valid.
    """

    if value is None:
        return False

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return False

    return numeric_value > -900


# ============================================================
# NASA POWER API REQUEST
# ============================================================

def fetch_nasa_range(
    requests,
    location,
    start_date,
    end_date,
):
    """
    Mengambil data NASA POWER untuk satu lokasi
    dan satu rentang tanggal.
    """

    params = {
        "parameters": PARAMETER,
        "community": "RE",
        "longitude": location["lon"],
        "latitude": location["lat"],
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "format": "JSON",
        "time-standard": "UTC",
    }

    print(
        f"NASA POWER request | "
        f"{location['name']} | "
        f"{start_date} -> {end_date}"
    )

    response = requests.get(
        NASA_URL,
        params=params,
        timeout=120,
    )

    print(
        f"NASA POWER HTTP {response.status_code} | "
        f"{location['name']}"
    )

    response.raise_for_status()

    data = response.json()

    parameter_data = (
        data
        .get("properties", {})
        .get("parameter", {})
        .get(PARAMETER, {})
    )

    if not isinstance(parameter_data, dict):
        print(
            f"Tidak ada data parameter {PARAMETER} "
            f"untuk {location['name']}."
        )
        return []

    rows = []

    for timestamp, raw_value in parameter_data.items():

        try:
            observed = datetime.strptime(
                timestamp,
                "%Y%m%d%H",
            ).replace(tzinfo=timezone.utc)

        except ValueError:
            print(
                f"Timestamp tidak dikenali: {timestamp}"
            )
            continue

        if not is_valid_value(raw_value):
            continue

        rows.append(
            {
                "location_id": location["id"],
                "location_name": location["name"],
                "latitude": location["lat"],
                "longitude": location["lon"],
                "parameter_id": PARAMETER,
                "parameter_name": PARAMETER,
                "value": float(raw_value),
                "unit": "kWh/m2",
                "observed_at": observed,
            }
        )

    return rows


# ============================================================
# FIND LATEST 30 DAYS VALID DATA
# ============================================================

def get_latest_valid_data(
    requests,
    location,
):
    """
    Mencari 30 hari data hourly valid terbaru
    untuk satu lokasi.

    Jika satu chunk belum menyediakan 720 jam valid,
    pencarian bergerak mundur 90 hari sampai:
    - 720 jam valid ditemukan, atau
    - mencapai awal histori NASA POWER.
    """

    print("=" * 70)
    print(
        f"Mencari {BACKFILL_HOURS} jam data valid terbaru:"
    )
    print(f"Location : {location['name']}")
    print(f"Latitude : {location['lat']}")
    print(f"Longitude: {location['lon']}")
    print("=" * 70)

    today = datetime.now(timezone.utc).date()

    current_end = today

    current_start = (
        current_end
        - timedelta(days=SEARCH_CHUNK_DAYS - 1)
    )

    collected = {}

    while True:

        if current_start < NASA_HISTORY_START.date():
            current_start = NASA_HISTORY_START.date()

        if current_start > current_end:
            break

        try:

            chunk_rows = fetch_nasa_range(
                requests,
                location,
                current_start,
                current_end,
            )

        except Exception as exc:

            print(
                f"NASA POWER request gagal untuk "
                f"{location['name']}: {exc}"
            )

            chunk_rows = []

        print(
            f"Data valid dari chunk: {len(chunk_rows)}"
        )

        # Dictionary digunakan untuk mencegah
        # timestamp duplikat antar chunk.
        for row in chunk_rows:

            timestamp = row["observed_at"]

            collected[timestamp] = row

        print(
            f"Total data valid terkumpul: "
            f"{len(collected)} jam"
        )

        # Sudah memenuhi target 720 jam.
        if len(collected) >= BACKFILL_HOURS:

            sorted_rows = sorted(
                collected.values(),
                key=lambda x: x["observed_at"],
                reverse=True,
            )

            latest_rows = sorted_rows[:BACKFILL_HOURS]

            latest_rows.sort(
                key=lambda x: x["observed_at"]
            )

            print("=" * 70)
            print(
                f"BERHASIL menemukan "
                f"{len(latest_rows)} jam data valid."
            )

            print(
                f"Data terbaru : "
                f"{latest_rows[-1]['observed_at']}"
            )

            print(
                f"Data terlama : "
                f"{latest_rows[0]['observed_at']}"
            )

            print("=" * 70)

            return latest_rows

        # Sudah mencapai awal histori.
        if current_start <= NASA_HISTORY_START.date():
            break

        next_end = current_start - timedelta(days=1)

        next_start = (
            next_end
            - timedelta(days=SEARCH_CHUNK_DAYS - 1)
        )

        current_end = next_end
        current_start = next_start

    # Histori habis tetapi data yang ditemukan masih kurang
    # dari 720 jam.
    if collected:

        sorted_rows = sorted(
            collected.values(),
            key=lambda x: x["observed_at"],
        )

        print("=" * 70)
        print(
            f"PERINGATAN: hanya ditemukan "
            f"{len(sorted_rows)} jam data valid "
            f"untuk {location['name']}."
        )
        print("=" * 70)

        return sorted_rows

    raise RuntimeError(
        "NASA POWER tidak menemukan data valid "
        f"untuk lokasi {location['name']} "
        "sepanjang rentang histori yang diperiksa."
    )


# ============================================================
# DAG
# ============================================================

@dag(
    dag_id="denpasar_nasa_power",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=[
        "denpasar",
        "etl",
        "nasa-power",
    ],
    description=(
        "NASA POWER Denpasar: "
        "extract, transform, save to CSV"
    ),
)
def denpasar_nasa_power():

    @task
    def extract_transform():

        import csv
        import requests

        all_rows = []

        print("=" * 70)
        print("NASA POWER EXTRACTION")
        print("=" * 70)

        print(f"Parameter : {PARAMETER}")
        print(
            f"Target    : {BACKFILL_HOURS} jam "
            f"(30 hari)"
        )
        print(
            "Search    : unlimited sampai histori NASA POWER"
        )

        print("=" * 70)

        # Ambil 30 hari data valid terbaru
        # untuk setiap lokasi.
        for location in LOCATIONS:

            rows = get_latest_valid_data(
                requests,
                location,
            )

            print(
                f"{location['name']} -> "
                f"{len(rows)} rows"
            )

            all_rows.extend(rows)

        if not all_rows:
            raise RuntimeError(
                "NASA POWER tidak mengembalikan "
                "data valid."
            )

        all_rows.sort(
            key=lambda x: (
                x["location_id"],
                x["observed_at"],
            )
        )

        os.makedirs(
            OUTPUT_DIR,
            exist_ok=True,
        )

        output = os.path.join(
            OUTPUT_DIR,
            "denpasar_nasa_power.csv",
        )

        with open(
            output,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                [
                    "location_id",
                    "location_name",
                    "latitude",
                    "longitude",
                    "parameter_id",
                    "parameter_name",
                    "value",
                    "unit",
                    "observed_at",
                ]
            )

            for row in all_rows:

                writer.writerow(
                    [
                        row["location_id"],
                        row["location_name"],
                        row["latitude"],
                        row["longitude"],
                        row["parameter_id"],
                        row["parameter_name"],
                        row["value"],
                        row["unit"],
                        row["observed_at"].isoformat(),
                    ]
                )

        print("=" * 70)
        print(
            f"NASA POWER transformed rows: "
            f"{len(all_rows)}"
        )
        print(f"Output CSV: {output}")
        print("=" * 70)

        return output

    extract_transform()


denpasar_nasa_power()