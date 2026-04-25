CREATE TABLE IF NOT EXISTS measurement_slots (
    id SERIAL PRIMARY KEY,
    datetime_utc TIMESTAMPTZ NOT NULL,
    date DATE NOT NULL,
    hour_utc SMALLINT NOT NULL,
    location_name TEXT NOT NULL,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    CONSTRAINT uq_slot UNIQUE (datetime_utc, location_name)
);

CREATE INDEX IF NOT EXISTS idx_slot_date ON measurement_slots (date);
CREATE INDEX IF NOT EXISTS idx_slot_datetime ON measurement_slots (datetime_utc);

CREATE TABLE IF NOT EXISTS air_quality_raw (
    id SERIAL PRIMARY KEY,
    slot_id INTEGER NOT NULL REFERENCES measurement_slots(id) ON DELETE CASCADE,
    pm1 DOUBLE PRECISION,
    pm25 DOUBLE PRECISION,
    pm10 DOUBLE PRECISION,
    um003 DOUBLE PRECISION,
    extra_params JSONB,
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_aq_slot UNIQUE (slot_id)
);

CREATE TABLE IF NOT EXISTS temperature_raw (
    id SERIAL PRIMARY KEY,
    slot_id INTEGER NOT NULL REFERENCES measurement_slots(id) ON DELETE CASCADE,
    temperature_c DOUBLE PRECISION,
    apparent_temperature_c DOUBLE PRECISION,
    relative_humidity_pct DOUBLE PRECISION,
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_temp_slot UNIQUE (slot_id)
);

CREATE OR REPLACE VIEW env_merged AS
SELECT
    s.id AS slot_id,
    s.datetime_utc,
    s.date,
    s.hour_utc,
    s.location_name,
    s.latitude,
    s.longitude,
    aq.pm1,
    aq.pm25,
    aq.pm10,
    aq.um003,
    aq.extra_params,
    t.temperature_c,
    t.apparent_temperature_c,
    t.relative_humidity_pct,
    GREATEST(aq.ingested_at, t.ingested_at) AS ingested_at
FROM measurement_slots s
JOIN air_quality_raw aq ON aq.slot_id = s.id
JOIN temperature_raw t ON t.slot_id = s.id;

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id SERIAL PRIMARY KEY,
    dag_run_id TEXT NOT NULL,
    date_from DATE NOT NULL,
    date_to DATE NOT NULL,
    slot_records INTEGER,
    aq_records INTEGER,
    temp_records INTEGER,
    status TEXT NOT NULL,
    error_message TEXT,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    finished_at TIMESTAMPTZ
);