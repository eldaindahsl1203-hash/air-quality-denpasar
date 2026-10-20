"""
Air Quality Denpasar - Open-Meteo DAG
Self-contained: API -> transform -> CSV.
"""

import os
from datetime import datetime, timedelta, timezone

from airflow.decorators import dag, task


OPENMETEO_URL = (
    "https://historical-forecast-api.open-meteo.com/v1/forecast"
)

LOCATIONS = [
    {
        "id": 6494341,
        "name": "Shiva Industries",
        "lat": -8.6418293,
        "lon": 115.1994208,
    },
    {
        "id": 6529627,
        "name": "Segara Ayu Denpasar",
        "lat": -8.68279,
        "lon": 115.26208,
    },
    {
        "id": 6536246,
        "name": "Cactus House",
        "lat": -8.70913,
        "lon": 115.24927,
    },
]

HOURLY = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
]

UNITS = {
    "temperature_2m": "°C",
    "relative_humidity_2m": "%",
    "precipitation": "mm",
    "wind_speed_10m": "m/s",
    "wind_direction_10m": "°",
}

OUTPUT_DIR = "/usr/local/airflow/data/processed"


def get_range():
    end = datetime.now(timezone.utc).replace(microsecond=0)

    start = end - timedelta(
        days=int(os.getenv("BACKFILL_DAYS", "30"))
    )

    return start, end


@dag(
    dag_id="denpasar_openmeteo",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["denpasar", "etl", "open-meteo"],
    description="Open-Meteo Denpasar: extract, transform, save to CSV",
)
def denpasar_openmeteo():

    @task
    def extract_transform():

        import csv
        import requests

        start, end = get_range()

        rows = []

        # Open-Meteo menggunakan tanggal kalender lokal
        # karena timezone=Asia/Makassar.
        local_timezone = timezone(timedelta(hours=8))

        local_start = start.astimezone(local_timezone).date()
        local_end = end.astimezone(local_timezone).date()

        for loc in LOCATIONS:

            params = {
                "latitude": loc["lat"],
                "longitude": loc["lon"],
                "start_date": local_start.isoformat(),
                "end_date": local_end.isoformat(),
                "hourly": ",".join(HOURLY),
                "timezone": "Asia/Makassar",
            }

            response = requests.get(
                OPENMETEO_URL,
                params=params,
                timeout=60,
            )

            response.raise_for_status()

            data = response.json()

            hourly = data.get("hourly", {})
            times = hourly.get("time", [])

            for i, local_time in enumerate(times):

                local_dt = datetime.fromisoformat(
                    local_time
                ).replace(tzinfo=local_timezone)

                observed = local_dt.astimezone(timezone.utc)

                # Batasi kembali ke rentang UTC yang diminta.
                if not (start <= observed < end):
                    continue

                for parameter in HOURLY:

                    values = hourly.get(parameter, [])

                    value = (
                        values[i]
                        if i < len(values)
                        else None
                    )

                    if value is None:
                        continue

                    rows.append(
                        (
                            loc["id"],
                            loc["name"],
                            loc["lat"],
                            loc["lon"],
                            parameter,
                            parameter,
                            value,
                            UNITS[parameter],
                            observed.isoformat(),
                        )
                    )

        if not rows:
            raise RuntimeError(
                "Open-Meteo tidak mengembalikan data."
            )

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        output = os.path.join(
            OUTPUT_DIR,
            "denpasar_openmeteo_30days.csv",
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

            writer.writerows(rows)

        print(
            f"Open-Meteo transformed rows: {len(rows)}"
        )
        print(f"Output CSV: {output}")

        return output

    extract_transform()


denpasar_openmeteo()