-- =========================================================
-- MEDALLION DATABASE STRUCTURE
-- Bronze  : raw/staging data from API
-- Silver  : cleaned and standardized hourly data
-- Gold    : final analytics layer for dashboard / forecast
-- =========================================================

CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;

-- =========================
-- BRONZE LAYER
-- =========================
-- Measurement slot tetap dipakai sebagai master waktu + lokasi
CREATE TABLE IF NOT EXISTS bronze.measurement_slots (
    id SERIAL PRIMARY KEY,
    datetime_utc TIMESTAMPTZ NOT NULL,
    date DATE NOT NULL,
    hour_utc SMALLINT NOT NULL,
    location_id BIGINT,
    location_name TEXT NOT NULL,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    CONSTRAINT uq_bronze_slot UNIQUE (datetime_utc, location_name)
);

CREATE INDEX IF NOT EXISTS idx_bronze_slot_date ON bronze.measurement_slots (date);
CREATE INDEX IF NOT EXISTS idx_bronze_slot_datetime ON bronze.measurement_slots (datetime_utc);
CREATE INDEX IF NOT EXISTS idx_bronze_slot_location ON bronze.measurement_slots (location_name);

-- Raw air quality dari OpenAQ
CREATE TABLE IF NOT EXISTS bronze.air_quality_raw (
    id SERIAL PRIMARY KEY,
    slot_id INTEGER NOT NULL REFERENCES bronze.measurement_slots(id) ON DELETE CASCADE,
    source TEXT DEFAULT 'OpenAQ',
    pm1 DOUBLE PRECISION,
    pm25 DOUBLE PRECISION,
    pm10 DOUBLE PRECISION,
    um003 DOUBLE PRECISION,
    extra_params JSONB,
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_bronze_aq_slot UNIQUE (slot_id)
);

-- Raw weather dari Open-Meteo
CREATE TABLE IF NOT EXISTS bronze.weather_raw (
    id SERIAL PRIMARY KEY,
    slot_id INTEGER NOT NULL REFERENCES bronze.measurement_slots(id) ON DELETE CASCADE,
    source TEXT DEFAULT 'Open-Meteo',
    temperature_c DOUBLE PRECISION,
    apparent_temperature_c DOUBLE PRECISION,
    relative_humidity_pct DOUBLE PRECISION,
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_bronze_weather_slot UNIQUE (slot_id)
);

-- =========================
-- SILVER LAYER
-- =========================
-- Cleaned air quality, 1 row per location per hour
CREATE TABLE IF NOT EXISTS silver.air_quality_clean (
    id SERIAL PRIMARY KEY,
    datetime_utc TIMESTAMPTZ NOT NULL,
    date DATE NOT NULL,
    hour_utc SMALLINT NOT NULL,
    location_id BIGINT,
    location_name TEXT NOT NULL,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    pm1 DOUBLE PRECISION,
    pm25 DOUBLE PRECISION,
    pm10 DOUBLE PRECISION,
    um003 DOUBLE PRECISION,
    extra_params JSONB,
    processed_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_silver_aq UNIQUE (datetime_utc, location_name)
);

CREATE INDEX IF NOT EXISTS idx_silver_aq_datetime ON silver.air_quality_clean (datetime_utc);
CREATE INDEX IF NOT EXISTS idx_silver_aq_location ON silver.air_quality_clean (location_name);

-- Cleaned weather, 1 row per location per hour
CREATE TABLE IF NOT EXISTS silver.weather_clean (
    id SERIAL PRIMARY KEY,
    datetime_utc TIMESTAMPTZ NOT NULL,
    date DATE NOT NULL,
    hour_utc SMALLINT NOT NULL,
    location_id BIGINT,
    location_name TEXT NOT NULL,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    temperature_c DOUBLE PRECISION,
    apparent_temperature_c DOUBLE PRECISION,
    relative_humidity_pct DOUBLE PRECISION,
    processed_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_silver_weather UNIQUE (datetime_utc, location_name)
);

CREATE INDEX IF NOT EXISTS idx_silver_weather_datetime ON silver.weather_clean (datetime_utc);
CREATE INDEX IF NOT EXISTS idx_silver_weather_location ON silver.weather_clean (location_name);

-- =========================
-- GOLD LAYER
-- =========================
-- Final dataset untuk dashboard dan forecast
CREATE OR REPLACE VIEW gold.env_merged AS
SELECT
    aq.datetime_utc,
    aq.date,
    aq.hour_utc,
    aq.location_id,
    aq.location_name,
    aq.latitude,
    aq.longitude,
    aq.pm1,
    aq.pm25,
    aq.pm10,
    aq.um003,
    aq.extra_params,
    w.temperature_c,
    w.apparent_temperature_c,
    w.relative_humidity_pct,
    GREATEST(aq.processed_at, w.processed_at) AS ingested_at
FROM silver.air_quality_clean aq
JOIN silver.weather_clean w
    ON aq.datetime_utc = w.datetime_utc
    AND aq.location_name = w.location_name;

-- Summary agregat untuk dashboard
CREATE OR REPLACE VIEW gold.location_summary AS
SELECT
    location_name,
    COUNT(*) AS total_records,
    ROUND(AVG(pm25)::numeric, 2) AS avg_pm25,
    ROUND(MAX(pm25)::numeric, 2) AS max_pm25,
    ROUND(MIN(pm25)::numeric, 2) AS min_pm25,
    ROUND(AVG(temperature_c)::numeric, 2) AS avg_temperature_c,
    ROUND(AVG(relative_humidity_pct)::numeric, 2) AS avg_humidity_pct
FROM gold.env_merged
GROUP BY location_name;

-- Summary harian untuk dashboard / forecast sederhana
CREATE OR REPLACE VIEW gold.daily_summary AS
SELECT
    date,
    location_name,
    ROUND(AVG(pm25)::numeric, 2) AS avg_pm25,
    ROUND(AVG(temperature_c)::numeric, 2) AS avg_temperature_c,
    ROUND(AVG(relative_humidity_pct)::numeric, 2) AS avg_humidity_pct
FROM gold.env_merged
GROUP BY date, location_name
ORDER BY date, location_name;

-- =========================
-- PIPELINE LOG
-- =========================
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id SERIAL PRIMARY KEY,
    dag_run_id TEXT NOT NULL,
    date_from DATE NOT NULL,
    date_to DATE NOT NULL,
    slot_records INTEGER,
    aq_records INTEGER,
    temp_records INTEGER,
    silver_aq_records INTEGER,
    silver_weather_records INTEGER,
    status TEXT NOT NULL,
    error_message TEXT,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    finished_at TIMESTAMPTZ
);

-- =========================
-- BACKWARD COMPATIBILITY VIEWS
-- Biar query lama Streamlit tetap aman.
-- =========================
CREATE OR REPLACE VIEW measurement_slots AS
SELECT * FROM bronze.measurement_slots;

CREATE OR REPLACE VIEW air_quality_raw AS
SELECT * FROM bronze.air_quality_raw;

CREATE OR REPLACE VIEW temperature_raw AS
SELECT * FROM bronze.weather_raw;

CREATE OR REPLACE VIEW env_merged AS
SELECT
    ROW_NUMBER() OVER () AS slot_id,
    datetime_utc,
    date,
    hour_utc,
    location_name,
    latitude,
    longitude,
    pm1,
    pm25,
    pm10,
    um003,
    extra_params,
    temperature_c,
    apparent_temperature_c,
    relative_humidity_pct,
    ingested_at
FROM gold.env_merged;
