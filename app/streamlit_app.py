import streamlit as st
import pandas as pd
import psycopg2
import os

st.title("Singapore Air Quality & Weather Dashboard")

conn = psycopg2.connect(
    host=os.getenv("POSTGRES_HOST", "postgres"),
    port=os.getenv("POSTGRES_PORT", "5432"),
    dbname=os.getenv("POSTGRES_DB", "envdata"),
    user=os.getenv("POSTGRES_USER", "envuser"),
    password=os.getenv("POSTGRES_PASSWORD", "12345678")
)

query = """
SELECT
    datetime_utc,
    location_name,
    pm25,
    temperature_c,
    relative_humidity_pct
FROM env_merged
ORDER BY datetime_utc DESC
LIMIT 500;
"""

df = pd.read_sql(query, conn)

st.dataframe(df)

if not df.empty:
    st.subheader("Rata-rata PM2.5 per Lokasi")
    st.bar_chart(df.groupby("location_name")["pm25"].mean())

    st.subheader("Temperature vs PM2.5")
    st.scatter_chart(df[["temperature_c", "pm25"]])

    st.subheader("Humidity vs PM2.5")
    st.scatter_chart(df[["relative_humidity_pct", "pm25"]])