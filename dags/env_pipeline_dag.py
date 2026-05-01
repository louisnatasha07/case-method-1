import json
import logging
import os
import time
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras
import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, RequestException, Timeout
from airflow import DAG
from airflow.operators.python import PythonOperator

log = logging.getLogger(__name__)

# =========================
# CONFIG
# =========================
OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "envdata")
POSTGRES_USER = os.getenv("POSTGRES_USER", "envuser")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "envpass")

# Gunakan range yang sudah lewat, jangan future date.
START_DATE = "2026-01-19"
END_DATE = "2026-04-30"

# 8 lokasi Singapore, CGB tidak dipakai karena baru mulai Maret 2026.
LOCATIONS = [
    {"id": 3023432, "name": "NASA GSFC Rutgers Calib. N7", "lat": 1.2976, "lon": 103.7803},
    {"id": 3038744, "name": "Ocean Park", "lat": 1.3094745, "lon": 103.9178515},
    {"id": 3040714, "name": "Midwood", "lat": 1.3641, "lon": 103.7637},
    {"id": 3400991, "name": "Potong Pasir Singapore", "lat": 1.330931, "lon": 103.868663},
    {"id": 5905179, "name": "Shelford", "lat": 1.324983, "lon": 103.812506},
    {"id": 6119237, "name": "Singapore", "lat": 1.3521, "lon": 103.8198},
    {"id": 6191434, "name": "Joo Chiat Place", "lat": 1.313748, "lon": 103.90176},
    {"id": 6273498, "name": "461B AQ", "lat": 1.3554983141500208, "lon": 103.74032475704472},
]

# =========================
# HELPERS
# =========================
def get_db_connection():
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )


def openaq_get(path: str, params: dict | None = None, max_retries: int = 5) -> dict:
    if not OPENAQ_API_KEY:
        raise RuntimeError("OPENAQ_API_KEY belum diset di .env")

    headers = {
        "Accept": "application/json",
        "X-API-Key": OPENAQ_API_KEY,
    }
    url = f"https://api.openaq.org/v3{path}"

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=90)
            if resp.status_code in (401, 403):
                raise RuntimeError(f"OpenAQ API key tidak valid: {resp.text[:200]}")
            resp.raise_for_status()
            return resp.json()
        except (ChunkedEncodingError, ConnectionError, Timeout, RequestException) as exc:
            log.warning("[OpenAQ retry %d/%d] path=%s params=%s error=%s", attempt, max_retries, path, params, exc)
            if attempt == max_retries:
                raise
            time.sleep(2 * attempt)


def ts_to_datetime_utc(ts_raw: str) -> datetime:
    return datetime.fromisoformat(ts_raw.replace("Z", "+00:00")).astimezone(timezone.utc)


def clean_value(v, lo, hi):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if lo <= f <= hi else None


def paginate_sensor_measurements(sensor_id: int, date_from: str, date_to: str) -> list[dict]:
    page = 1
    all_results = []

    while True:
        raw = openaq_get(
            f"/sensors/{sensor_id}/measurements/hourly",
            {"date_from": date_from, "date_to": date_to, "limit": 100, "page": page},
        )
        results = raw.get("results", [])
        if not results:
            break

        all_results.extend(results)
        if len(results) < 100:
            break

        page += 1

    return all_results


def fetch_open_meteo(loc: dict) -> list[dict]:
    resp = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": loc["lat"],
            "longitude": loc["lon"],
            "hourly": "temperature_2m,relative_humidity_2m,apparent_temperature",
            "temperature_unit": "celsius",
            "timezone": "UTC",
            "start_date": START_DATE,
            "end_date": END_DATE,
        },
        timeout=90,
    )
    resp.raise_for_status()
    hourly = resp.json().get("hourly", {})

    rows = []
    for i, ts in enumerate(hourly.get("time", [])):
        dt = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
        rows.append({
            "source": "Open-Meteo",
            "location_id": loc["id"],
            "location_name": loc["name"],
            "latitude": loc["lat"],
            "longitude": loc["lon"],
            "temperature_c": hourly.get("temperature_2m", [None])[i] if i < len(hourly.get("temperature_2m", [])) else None,
            "apparent_temperature_c": hourly.get("apparent_temperature", [None])[i] if i < len(hourly.get("apparent_temperature", [])) else None,
            "relative_humidity_pct": hourly.get("relative_humidity_2m", [None])[i] if i < len(hourly.get("relative_humidity_2m", [])) else None,
            "datetime_utc": dt,
            "raw_payload": json.dumps({
                "time": ts,
                "temperature_2m": hourly.get("temperature_2m", [None])[i] if i < len(hourly.get("temperature_2m", [])) else None,
                "apparent_temperature": hourly.get("apparent_temperature", [None])[i] if i < len(hourly.get("apparent_temperature", [])) else None,
                "relative_humidity_2m": hourly.get("relative_humidity_2m", [None])[i] if i < len(hourly.get("relative_humidity_2m", [])) else None,
            }),
        })
    return rows


# =========================
# MAIN ETL
# =========================
def run_env_pipeline():
    dag_run_id = f"manual_{datetime.now(timezone.utc).isoformat()}"
    run_ts = datetime.now(timezone.utc)

    log.info("=== START PIPELINE MEDALLION ===")
    log.info("Range data: %s s.d. %s", START_DATE, END_DATE)

    bronze_aq_rows = []
    bronze_weather_rows = []

    # -------------------------
    # 1. EXTRACT AQ -> BRONZE
    # -------------------------
    for loc in LOCATIONS:
        log.info("[AQ] Ambil sensors untuk %s (%s)", loc["name"], loc["id"])
        sensor_data = openaq_get(f"/locations/{loc['id']}/sensors")
        sensors = [
            {
                "sensor_id": s["id"],
                "param_name": s.get("parameter", {}).get("name", "unknown"),
                "param_unit": s.get("parameter", {}).get("units", ""),
            }
            for s in sensor_data.get("results", [])
            if s.get("parameter", {}).get("name") == "pm25"
        ]

        log.info("[AQ] %s punya %d sensor pm25", loc["name"], len(sensors))

        for sensor in sensors:
            log.info("[AQ] Ambil measurements sensor_id=%s untuk %s", sensor["sensor_id"], loc["name"])
            measurements = paginate_sensor_measurements(sensor["sensor_id"], START_DATE, END_DATE)

            for item in measurements:
                period = item.get("period", {})
                ts_raw = period.get("datetimeFrom", {}).get("utc") or item.get("datetime", {}).get("utc")
                if not ts_raw:
                    continue

                value = item.get("value") or item.get("summary", {}).get("mean")
                value = clean_value(value, 0, 1000)
                if value is None:
                    continue

                bronze_aq_rows.append({
                    "source": "OpenAQ",
                    "location_id": loc["id"],
                    "location_name": loc["name"],
                    "latitude": loc["lat"],
                    "longitude": loc["lon"],
                    "parameter": sensor["param_name"],
                    "value": round(float(value), 4),
                    "unit": sensor["param_unit"],
                    "datetime_utc": ts_to_datetime_utc(ts_raw),
                    "raw_payload": json.dumps(item),
                })

    if not bronze_aq_rows:
        raise RuntimeError("OpenAQ tidak mengembalikan data PM2.5 untuk semua lokasi.")

    log.info("[Bronze] AQ rows: %d", len(bronze_aq_rows))

    # -------------------------
    # 2. EXTRACT WEATHER -> BRONZE
    # -------------------------
    for loc in LOCATIONS:
        log.info("[Weather] Ambil Open-Meteo untuk %s", loc["name"])
        bronze_weather_rows.extend(fetch_open_meteo(loc))

    log.info("[Bronze] Weather rows: %d", len(bronze_weather_rows))

    # -------------------------
    # 3. LOAD BRONZE + TRANSFORM SILVER
    # -------------------------
    conn = get_db_connection()

    try:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                """
                INSERT INTO bronze.air_quality_raw
                    (source, location_id, location_name, latitude, longitude, parameter,
                     value, unit, datetime_utc, raw_payload)
                VALUES
                    (%(source)s, %(location_id)s, %(location_name)s, %(latitude)s, %(longitude)s,
                     %(parameter)s, %(value)s, %(unit)s, %(datetime_utc)s, %(raw_payload)s)
                ON CONFLICT (source, location_id, parameter, datetime_utc) DO UPDATE SET
                    location_name = EXCLUDED.location_name,
                    latitude = EXCLUDED.latitude,
                    longitude = EXCLUDED.longitude,
                    value = EXCLUDED.value,
                    unit = EXCLUDED.unit,
                    raw_payload = EXCLUDED.raw_payload,
                    ingested_at = NOW()
                """,
                bronze_aq_rows,
                page_size=500,
            )

            psycopg2.extras.execute_batch(
                cur,
                """
                INSERT INTO bronze.weather_raw
                    (source, location_id, location_name, latitude, longitude,
                     temperature_c, apparent_temperature_c, relative_humidity_pct,
                     datetime_utc, raw_payload)
                VALUES
                    (%(source)s, %(location_id)s, %(location_name)s, %(latitude)s, %(longitude)s,
                     %(temperature_c)s, %(apparent_temperature_c)s, %(relative_humidity_pct)s,
                     %(datetime_utc)s, %(raw_payload)s)
                ON CONFLICT (source, location_id, datetime_utc) DO UPDATE SET
                    location_name = EXCLUDED.location_name,
                    latitude = EXCLUDED.latitude,
                    longitude = EXCLUDED.longitude,
                    temperature_c = EXCLUDED.temperature_c,
                    apparent_temperature_c = EXCLUDED.apparent_temperature_c,
                    relative_humidity_pct = EXCLUDED.relative_humidity_pct,
                    raw_payload = EXCLUDED.raw_payload,
                    ingested_at = NOW()
                """,
                bronze_weather_rows,
                page_size=500,
            )

            # Rebuild silver untuk range yang sedang diproses agar idempotent.
            cur.execute("DELETE FROM silver.air_quality_clean WHERE date BETWEEN %s AND %s", (START_DATE, END_DATE))
            cur.execute("DELETE FROM silver.weather_clean WHERE date BETWEEN %s AND %s", (START_DATE, END_DATE))

            cur.execute(
                """
                INSERT INTO silver.air_quality_clean
                    (location_id, location_name, latitude, longitude, pm25, datetime_utc, date, hour_utc)
                SELECT
                    location_id,
                    location_name,
                    latitude,
                    longitude,
                    AVG(value) AS pm25,
                    date_trunc('hour', datetime_utc) AS datetime_utc,
                    DATE(date_trunc('hour', datetime_utc)) AS date,
                    EXTRACT(HOUR FROM date_trunc('hour', datetime_utc))::SMALLINT AS hour_utc
                FROM bronze.air_quality_raw
                WHERE parameter = 'pm25'
                  AND DATE(datetime_utc) BETWEEN %s AND %s
                GROUP BY location_id, location_name, latitude, longitude, date_trunc('hour', datetime_utc)
                ON CONFLICT (location_id, datetime_utc) DO UPDATE SET
                    location_name = EXCLUDED.location_name,
                    latitude = EXCLUDED.latitude,
                    longitude = EXCLUDED.longitude,
                    pm25 = EXCLUDED.pm25,
                    date = EXCLUDED.date,
                    hour_utc = EXCLUDED.hour_utc,
                    processed_at = NOW()
                """,
                (START_DATE, END_DATE),
            )

            cur.execute(
                """
                INSERT INTO silver.weather_clean
                    (location_id, location_name, latitude, longitude, temperature_c,
                     apparent_temperature_c, relative_humidity_pct, datetime_utc, date, hour_utc)
                SELECT
                    location_id,
                    location_name,
                    latitude,
                    longitude,
                    AVG(temperature_c) AS temperature_c,
                    AVG(apparent_temperature_c) AS apparent_temperature_c,
                    AVG(relative_humidity_pct) AS relative_humidity_pct,
                    date_trunc('hour', datetime_utc) AS datetime_utc,
                    DATE(date_trunc('hour', datetime_utc)) AS date,
                    EXTRACT(HOUR FROM date_trunc('hour', datetime_utc))::SMALLINT AS hour_utc
                FROM bronze.weather_raw
                WHERE DATE(datetime_utc) BETWEEN %s AND %s
                GROUP BY location_id, location_name, latitude, longitude, date_trunc('hour', datetime_utc)
                ON CONFLICT (location_id, datetime_utc) DO UPDATE SET
                    location_name = EXCLUDED.location_name,
                    latitude = EXCLUDED.latitude,
                    longitude = EXCLUDED.longitude,
                    temperature_c = EXCLUDED.temperature_c,
                    apparent_temperature_c = EXCLUDED.apparent_temperature_c,
                    relative_humidity_pct = EXCLUDED.relative_humidity_pct,
                    date = EXCLUDED.date,
                    hour_utc = EXCLUDED.hour_utc,
                    processed_at = NOW()
                """,
                (START_DATE, END_DATE),
            )

            cur.execute("SELECT COUNT(*) FROM silver.air_quality_clean WHERE date BETWEEN %s AND %s", (START_DATE, END_DATE))
            silver_aq_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM silver.weather_clean WHERE date BETWEEN %s AND %s", (START_DATE, END_DATE))
            silver_weather_count = cur.fetchone()[0]

            cur.execute(
                """
                INSERT INTO metadata.pipeline_runs
                    (dag_run_id, date_from, date_to, bronze_aq_records, bronze_weather_records,
                     silver_aq_records, silver_weather_records, status, finished_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'success', %s)
                """,
                (
                    dag_run_id,
                    START_DATE,
                    END_DATE,
                    len(bronze_aq_rows),
                    len(bronze_weather_rows),
                    silver_aq_count,
                    silver_weather_count,
                    run_ts,
                ),
            )

        conn.commit()
        log.info(
            "[Load] selesai | bronze_aq=%d | bronze_weather=%d | silver_aq=%d | silver_weather=%d",
            len(bronze_aq_rows), len(bronze_weather_rows), silver_aq_count, silver_weather_count
        )

    except Exception as exc:
        conn.rollback()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO metadata.pipeline_runs
                        (dag_run_id, date_from, date_to, status, error_message, finished_at)
                    VALUES (%s, %s, %s, 'failed', %s, %s)
                    """,
                    (dag_run_id, START_DATE, END_DATE, str(exc)[:1000], run_ts),
                )
            conn.commit()
        except Exception:
            pass
        raise
    finally:
        conn.close()


# =========================
# DAG
# =========================
with DAG(
    dag_id="env_data_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["env", "singapore", "medallion"],
) as dag:
    run_pipeline = PythonOperator(
        task_id="run_env_pipeline",
        python_callable=run_env_pipeline,
    )