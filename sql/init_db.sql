CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;
CREATE SCHEMA IF NOT EXISTS metadata;

CREATE TABLE IF NOT EXISTS bronze.air_quality_raw (
    id SERIAL PRIMARY KEY,
    source TEXT NOT NULL DEFAULT 'OpenAQ',
    location_id BIGINT NOT NULL,
    location_name TEXT NOT NULL,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    parameter TEXT NOT NULL,
    value DOUBLE PRECISION,
    unit TEXT,
    datetime_utc TIMESTAMPTZ NOT NULL,
    raw_payload JSONB,
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_bronze_aq UNIQUE (source, location_id, parameter, datetime_utc)
);

CREATE INDEX IF NOT EXISTS idx_bronze_aq_datetime ON bronze.air_quality_raw (datetime_utc);
CREATE INDEX IF NOT EXISTS idx_bronze_aq_location ON bronze.air_quality_raw (location_id, location_name);

CREATE TABLE IF NOT EXISTS bronze.weather_raw (
    id SERIAL PRIMARY KEY,
    source TEXT NOT NULL DEFAULT 'Open-Meteo',
    location_id BIGINT NOT NULL,
    location_name TEXT NOT NULL,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    temperature_c DOUBLE PRECISION,
    apparent_temperature_c DOUBLE PRECISION,
    relative_humidity_pct DOUBLE PRECISION,
    datetime_utc TIMESTAMPTZ NOT NULL,
    raw_payload JSONB,
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_bronze_weather UNIQUE (source, location_id, datetime_utc)
);

CREATE INDEX IF NOT EXISTS idx_bronze_weather_datetime ON bronze.weather_raw (datetime_utc);
CREATE INDEX IF NOT EXISTS idx_bronze_weather_location ON bronze.weather_raw (location_id, location_name);

CREATE TABLE IF NOT EXISTS silver.air_quality_clean (
    id SERIAL PRIMARY KEY,
    location_id BIGINT NOT NULL,
    location_name TEXT NOT NULL,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    pm25 DOUBLE PRECISION,
    datetime_utc TIMESTAMPTZ NOT NULL,
    date DATE NOT NULL,
    hour_utc SMALLINT NOT NULL,
    processed_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_silver_aq UNIQUE (location_id, datetime_utc)
);

CREATE TABLE IF NOT EXISTS silver.weather_clean (
    id SERIAL PRIMARY KEY,
    location_id BIGINT NOT NULL,
    location_name TEXT NOT NULL,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    temperature_c DOUBLE PRECISION,
    apparent_temperature_c DOUBLE PRECISION,
    relative_humidity_pct DOUBLE PRECISION,
    datetime_utc TIMESTAMPTZ NOT NULL,
    date DATE NOT NULL,
    hour_utc SMALLINT NOT NULL,
    processed_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_silver_weather UNIQUE (location_id, datetime_utc)
);

CREATE OR REPLACE VIEW gold.env_merged AS
SELECT
    aq.datetime_utc,
    aq.date,
    aq.hour_utc,
    aq.location_id,
    aq.location_name,
    aq.latitude,
    aq.longitude,
    aq.pm25,
    w.temperature_c,
    w.apparent_temperature_c,
    w.relative_humidity_pct,
    GREATEST(aq.processed_at, w.processed_at) AS processed_at
FROM silver.air_quality_clean aq
LEFT JOIN silver.weather_clean w
    ON aq.location_id = w.location_id
   AND aq.datetime_utc = w.datetime_utc;

CREATE OR REPLACE VIEW gold.location_summary AS
SELECT
    location_id,
    location_name,
    COUNT(*) AS total_records,
    ROUND(AVG(pm25)::numeric, 2) AS avg_pm25,
    ROUND(MAX(pm25)::numeric, 2) AS max_pm25,
    ROUND(MIN(pm25)::numeric, 2) AS min_pm25,
    ROUND(AVG(temperature_c)::numeric, 2) AS avg_temperature_c,
    ROUND(AVG(relative_humidity_pct)::numeric, 2) AS avg_humidity_pct
FROM gold.env_merged
GROUP BY location_id, location_name;

CREATE TABLE IF NOT EXISTS metadata.pipeline_runs (
    id SERIAL PRIMARY KEY,
    dag_run_id TEXT NOT NULL,
    date_from DATE NOT NULL,
    date_to DATE NOT NULL,
    bronze_aq_records INTEGER,
    bronze_weather_records INTEGER,
    silver_aq_records INTEGER,
    silver_weather_records INTEGER,
    status TEXT NOT NULL,
    error_message TEXT,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    finished_at TIMESTAMPTZ
);

-- Compatibility views supaya Streamlit lama masih bisa SELECT FROM env_merged;
CREATE OR REPLACE VIEW env_merged AS SELECT * FROM gold.env_merged;
CREATE OR REPLACE VIEW location_summary AS SELECT * FROM gold.location_summary;