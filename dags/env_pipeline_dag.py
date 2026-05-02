import json
import logging
import os
from datetime import datetime, timezone, timedelta

import psycopg2
import psycopg2.extras
import requests
from airflow import DAG
from airflow.operators.python import PythonOperator

log = logging.getLogger(__name__)

# CONFIG
OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "envdata")
POSTGRES_USER = os.getenv("POSTGRES_USER", "envuser")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "envpass")

_today = datetime.now(timezone.utc)
END_DATE   = (_today - timedelta(days=7)).strftime("%Y-%m-%d")   # 7 hari lalu (batas aman Open-Meteo)
START_DATE = (_today - timedelta(days=37)).strftime("%Y-%m-%d")  # 30 hari sebelum END_DATE

LOCATIONS = [
    {"id": 6273498, "name": "461B AQ",                     "lat": 1.3555,   "lon": 103.74033},
    {"id": 3040714, "name": "Midwood",                     "lat": 1.36411,  "lon": 103.76370},
    {"id": 3023432, "name": "NASA GSFC Rutgers Calib. N7", "lat": 1.29761,  "lon": 103.78031},
    {"id": 5905179, "name": "Shelford",                    "lat": 1.32498,  "lon": 103.81252},
    {"id": 6289675, "name": "CGB",                         "lat": 1.28484,  "lon": 103.82913},
    {"id": 3400991, "name": "Potong Pasir Singapore",      "lat": 1.33092,  "lon": 103.86867},
    {"id": 6191434, "name": "Joo Chiat Place",             "lat": 1.31374,  "lon": 103.90176},
    {"id": 3038744, "name": "Ocean Park",                  "lat": 1.30947,  "lon": 103.91785},
]

AQ_EXCLUDE_PARAMS = {"temperature", "relativehumidity"}


def get_db_connection():
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )


def openaq_get(path: str, params: dict | None = None) -> dict:
    headers = {
        "Accept": "application/json",
        "X-API-Key": OPENAQ_API_KEY,
    }
    resp = requests.get(
        f"https://api.openaq.org/v3{path}",
        params=params,
        headers=headers,
        timeout=60,
    )
    if resp.status_code in (401, 403):
        raise RuntimeError(f"OpenAQ API key tidak valid: {resp.text[:200]}")
    resp.raise_for_status()
    return resp.json()


def ts_to_hour_key(ts_raw: str) -> str:
    dt = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
    return dt.strftime("%Y-%m-%dT%H:00:00+00:00")


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
            {
                "date_from": date_from,
                "date_to": date_to,
                "limit": 100,
                "page": page,
            },
        )
        results = raw.get("results", [])
        if not results:
            break

        all_results.extend(results)

        if len(results) < 100:
            break

        page += 1

    return all_results


def run_env_pipeline():
    dag_run_id = f"manual_{datetime.now(timezone.utc).isoformat()}"
    run_ts = datetime.now(timezone.utc)

    log.info("=== START PIPELINE ===")
    log.info("Range data: %s s.d. %s", START_DATE, END_DATE)

    # 1. EXTRACT AIR QUALITY
    aq_records_raw = []

    for loc in LOCATIONS:
        location_id = loc["id"]
        location_name = loc["name"]
        lat = loc["lat"]
        lon = loc["lon"]

        log.info("[AQ] Ambil sensors untuk %s (%s)", location_name, location_id)

        sensor_data = openaq_get(f"/locations/{location_id}/sensors")
        sensors = [
            {
                "sensor_id": s["id"],
                "param_name": s.get("parameter", {}).get("name", "unknown"),
                "param_unit": s.get("parameter", {}).get("units", ""),
            }
            for s in sensor_data.get("results", [])
            if s.get("parameter", {}).get("name") == "pm25"
        ]
        
        log.info("[AQ] %s punya %d sensor pm25", location_name, len(sensors))

        all_readings = {}

        for s in sensors:
            if s["param_name"] in AQ_EXCLUDE_PARAMS:
                continue

            log.info(
                "[AQ] Ambil measurements sensor_id=%s param=%s untuk%s",
                s["sensor_id"],
                s["param_name"],
                location_name,
            )
            
            measurements = paginate_sensor_measurements(
                sensor_id=s["sensor_id"],
                date_from=START_DATE,
                date_to=END_DATE,
            )

            for item in measurements:
                period = item.get("period", {})
                ts_raw = (
                    period.get("datetimeFrom", {}).get("utc")
                    or item.get("datetime", {}).get("utc")
                )

                if not ts_raw:
                    continue

                hour_key = ts_to_hour_key(ts_raw)
                value = item.get("value") or item.get("summary", {}).get("mean")

                if value is not None:
                    all_readings.setdefault(hour_key, {})[s["param_name"]] = {
                        "value": round(float(value), 4),
                        "unit": s["param_unit"],
                    }

        for hk in sorted(all_readings):
            dt = datetime.fromisoformat(hk)
            aq_records_raw.append({
                "datetime_utc": hk,
                "date": hk[:10],
                "hour_utc": dt.hour,
                "location_name": location_name,
                "latitude": lat,
                "longitude": lon,
                "parameters": all_readings[hk],
            })

    if not aq_records_raw:
        raise RuntimeError("OpenAQ tidak mengembalikan data apapun untuk semua lokasi.")

    log.info("[AQ] Total raw AQ records: %d", len(aq_records_raw))

    # 2. EXTRACT WEATHER
    weather_records_raw = []

    for loc in LOCATIONS:
        location_name = loc["name"]
        lat = loc["lat"]
        lon = loc["lon"]

        log.info("[Weather] Ambil Open-Meteo untuk %s", location_name)

        resp = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m,relative_humidity_2m,apparent_temperature",
                "temperature_unit": "celsius",
                "timezone": "UTC",
                "start_date": START_DATE,
                "end_date": END_DATE,
            },
            timeout=60,
        )
        resp.raise_for_status()
        h = resp.json().get("hourly", {})

        for i, ts in enumerate(h.get("time", [])):
            dt = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
            hour_key = dt.strftime("%Y-%m-%dT%H:00:00+00:00")

            weather_records_raw.append({
                "datetime_utc": hour_key,
                "date": dt.strftime("%Y-%m-%d"),
                "hour_utc": dt.hour,
                "location_name": location_name,
                "latitude": lat,
                "longitude": lon,
                "temperature_c": h["temperature_2m"][i] if i < len(h.get("temperature_2m", [])) else None,
                "apparent_temperature_c": h["apparent_temperature"][i] if i < len(h.get("apparent_temperature", [])) else None,
                "relative_humidity_pct": h["relative_humidity_2m"][i] if i < len(h.get("relative_humidity_2m", [])) else None,
            })

    log.info("[Weather] Total raw weather records: %d", len(weather_records_raw))

    # 3. TRANSFORM
    KNOWN_PARAMS = {"pm1", "pm25", "pm10", "um003"}
    AQ_VALID_RANGES = {
        "pm1": (0, 1000),
        "pm25": (0, 1000),
        "pm10": (0, 2000),
        "um003": (0, 100000),
    }

    aq_normalized = []
    for row in aq_records_raw:
        params = row.get("parameters", {})
        extra = {}

        flat = {
            "datetime_utc": row["datetime_utc"],
            "date": row["date"],
            "hour_utc": row["hour_utc"],
            "location_name": row["location_name"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "pm1": None,
            "pm25": None,
            "pm10": None,
            "um003": None,
        }

        for pname, pdata in params.items():
            val = pdata.get("value")
            if pname in KNOWN_PARAMS:
                lo, hi = AQ_VALID_RANGES[pname]
                flat[pname] = clean_value(val, lo, hi)
            else:
                extra[pname] = pdata

        flat["extra_params"] = extra if extra else None
        aq_normalized.append(flat)

    weather_normalized = []
    for row in weather_records_raw:
        weather_normalized.append({
            **row,
            "temperature_c": clean_value(row["temperature_c"], -10, 60),
            "apparent_temperature_c": clean_value(row["apparent_temperature_c"], -20, 70),
            "relative_humidity_pct": clean_value(row["relative_humidity_pct"], 0, 100),
        })

    log.info("[Transform] AQ normalized: %d", len(aq_normalized))
    log.info("[Transform] Weather normalized: %d", len(weather_normalized))

    # 4. LOAD
    conn = get_db_connection()

    try:
        dt_loc_to_meta = {}

        for r in aq_normalized:
            dt_loc_to_meta.setdefault((r["datetime_utc"], r["location_name"]), r)

        for r in weather_normalized:
            dt_loc_to_meta.setdefault((r["datetime_utc"], r["location_name"]), r)

        slot_rows = [
            {
                "datetime_utc": dt,
                "date": meta["date"],
                "hour_utc": meta["hour_utc"],
                "location_name": loc_name,
                "latitude": meta["latitude"],
                "longitude": meta["longitude"],
            }
            for (dt, loc_name), meta in dt_loc_to_meta.items()
        ]

        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                """
                INSERT INTO measurement_slots
                    (datetime_utc, date, hour_utc, location_name, latitude, longitude)
                VALUES
                    (%(datetime_utc)s, %(date)s, %(hour_utc)s,
                     %(location_name)s, %(latitude)s, %(longitude)s)
                ON CONFLICT (datetime_utc, location_name) DO UPDATE SET
                    date      = EXCLUDED.date,
                    hour_utc  = EXCLUDED.hour_utc,
                    latitude  = EXCLUDED.latitude,
                    longitude = EXCLUDED.longitude
                """,
                slot_rows,
                page_size=500,
            )

            cur.execute(
                """
                SELECT id, datetime_utc, location_name
                FROM measurement_slots
                WHERE date BETWEEN %s AND %s
                """,
                (START_DATE, END_DATE),
            )

            slot_id_map = {
                (
                    row[1].strftime("%Y-%m-%dT%H:00:00+00:00"),
                    row[2],
                ): row[0]
                for row in cur.fetchall()
            }

        aq_rows = []
        for r in aq_normalized:
            sid = slot_id_map.get((r["datetime_utc"], r["location_name"]))
            if sid is None:
                continue

            aq_rows.append({
                "slot_id": sid,
                "pm1": r.get("pm1"),
                "pm25": r.get("pm25"),
                "pm10": r.get("pm10"),
                "um003": r.get("um003"),
                "extra_params": json.dumps(r["extra_params"]) if r.get("extra_params") else None,
            })

        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                """
                INSERT INTO air_quality_raw
                    (slot_id, pm1, pm25, pm10, um003, extra_params)
                VALUES
                    (%(slot_id)s, %(pm1)s, %(pm25)s, %(pm10)s, %(um003)s, %(extra_params)s)
                ON CONFLICT (slot_id) DO UPDATE SET
                    pm1          = EXCLUDED.pm1,
                    pm25         = EXCLUDED.pm25,
                    pm10         = EXCLUDED.pm10,
                    um003        = EXCLUDED.um003,
                    extra_params = EXCLUDED.extra_params,
                    ingested_at  = NOW()
                """,
                aq_rows,
                page_size=500,
            )

        weather_rows = []
        for r in weather_normalized:
            sid = slot_id_map.get((r["datetime_utc"], r["location_name"]))
            if sid is None:
                continue

            weather_rows.append({
                "slot_id": sid,
                "temperature_c": r.get("temperature_c"),
                "apparent_temperature_c": r.get("apparent_temperature_c"),
                "relative_humidity_pct": r.get("relative_humidity_pct"),
            })

        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                """
                INSERT INTO temperature_raw
                    (slot_id, temperature_c, apparent_temperature_c, relative_humidity_pct)
                VALUES
                    (%(slot_id)s, %(temperature_c)s, %(apparent_temperature_c)s, %(relative_humidity_pct)s)
                ON CONFLICT (slot_id) DO UPDATE SET
                    temperature_c          = EXCLUDED.temperature_c,
                    apparent_temperature_c = EXCLUDED.apparent_temperature_c,
                    relative_humidity_pct  = EXCLUDED.relative_humidity_pct,
                    ingested_at            = NOW()
                """,
                weather_rows,
                page_size=500,
            )

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO pipeline_runs
                    (dag_run_id, date_from, date_to, slot_records,
                     aq_records, temp_records, status, finished_at)
                VALUES (%s, %s, %s, %s, %s, %s, 'success', %s)
                """,
                (
                    dag_run_id,
                    START_DATE,
                    END_DATE,
                    len(slot_rows),
                    len(aq_rows),
                    len(weather_rows),
                    run_ts,
                ),
            )

        conn.commit()
        log.info(
            "[Load] selesai | range=%s s.d. %s | slots=%d | aq=%d | weather=%d",
            START_DATE, END_DATE, len(slot_rows), len(aq_rows), len(weather_rows)
        )

    except Exception as exc:
        conn.rollback()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO pipeline_runs
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


# DAG
with DAG(
    dag_id="env_data_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["env", "singapore"],
) as dag:
    run_pipeline = PythonOperator(
        task_id="run_env_pipeline",
        python_callable=run_env_pipeline,
    )