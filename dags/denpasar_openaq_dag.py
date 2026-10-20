"""
Air Quality Denpasar - OpenAQ DAG
Self-contained: API -> transform -> CSV.
"""

import os
from datetime import datetime, timedelta, timezone

from airflow.decorators import dag, task


OPENAQ_BASE = "https://api.openaq.org/v3"

# Known Denpasar OpenAQ locations already validated during project setup.
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

OUTPUT_DIR = "/usr/local/airflow/data/processed"


def get_range():
    end = datetime.now(timezone.utc).replace(microsecond=0)
    start = end - timedelta(days=int(os.getenv("BACKFILL_DAYS", "30")))
    return start, end


@dag(
    dag_id="denpasar_openaq",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["denpasar", "etl", "openaq"],
    description="OpenAQ Denpasar: extract, transform, save to CSV",
)
def denpasar_openaq():

    @task
    def extract_transform():

        import csv
        import requests

        api_key = os.getenv("OPENAQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "OPENAQ_API_KEY belum tersedia di environment Airflow."
            )

        start, end = get_range()

        headers = {
            "X-API-Key": api_key
        }

        rows = []

        for loc in LOCATIONS:

            # Ambil daftar sensor pada lokasi
            sensor_url = f"{OPENAQ_BASE}/locations/{loc['id']}/sensors"

            response = requests.get(
                sensor_url,
                headers=headers,
                params={"limit": 100},
                timeout=60,
            )

            response.raise_for_status()

            sensors = response.json().get("results", [])

            for sensor in sensors:

                sensor_id = sensor.get("id")
                parameter = sensor.get("parameter") or {}

                parameter_id = parameter.get("id")
                parameter_name = parameter.get("name")
                unit = parameter.get("units")

                if not sensor_id or not parameter_name:
                    continue

                page = 1

                while True:

                    params = {
                        "datetime_from": start.isoformat().replace(
                            "+00:00", "Z"
                        ),
                        "datetime_to": end.isoformat().replace(
                            "+00:00", "Z"
                        ),
                        "limit": 1000,
                        "page": page,
                    }

                    url = (
                        f"{OPENAQ_BASE}/sensors/"
                        f"{sensor_id}/measurements"
                    )

                    response = requests.get(
                        url,
                        headers=headers,
                        params=params,
                        timeout=60,
                    )

                    response.raise_for_status()

                    payload = response.json()
                    results = payload.get("results", [])

                    if not results:
                        break

                    for item in results:

                        value = item.get("value")

                        period = item.get("period") or {}
                        datetime_from = period.get("datetimeFrom") or {}
                        observed_at = datetime_from.get("utc")

                        if value is None or not observed_at:
                            continue

                        rows.append(
                            (
                                loc["id"],
                                loc["name"],
                                loc["lat"],
                                loc["lon"],
                                sensor_id,
                                parameter_id,
                                parameter_name,
                                value,
                                unit,
                                observed_at,
                            )
                        )

                    meta = payload.get("meta") or {}
                    found = meta.get("found")

                    if len(results) < 1000:
                        break

                    if isinstance(found, int) and page * 1000 >= found:
                        break

                    page += 1

        if not rows:
            raise RuntimeError(
                "OpenAQ tidak mengembalikan data "
                "untuk rentang waktu tersebut."
            )

        # Pastikan folder output tersedia
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        output = os.path.join(
            OUTPUT_DIR,
            "denpasar_openaq_30days.csv",
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
                    "sensor_id",
                    "parameter_id",
                    "parameter_name",
                    "value",
                    "unit",
                    "observed_at",
                ]
            )

            writer.writerows(rows)

        print(f"OpenAQ transformed rows: {len(rows)}")
        print(f"Output CSV: {output}")

        return output

    extract_transform()


denpasar_openaq()