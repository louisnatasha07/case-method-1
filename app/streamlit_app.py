import streamlit as st
import pandas as pd
import psycopg2
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk

st.set_page_config(
    page_title="Singapore Air Quality Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

COLOR_SEQ = [
    "#60A5FA",  # blue
    "#F87171",  # red
    "#4ADE80",  # green
    "#FBBF24",  # amber
    "#A78BFA",  # violet
    "#22D3EE",  # cyan
    "#F472B6",  # pink
    "#A3E635",  # lime
]

BG_BASE    = "#0f1117"
BG_SURFACE = "#161b27"
BG_RAISED  = "#1e2535"
BORDER     = "#2a3347"
TEXT_PRI   = "#e2e8f0"
TEXT_SEC   = "#64748b"
TEXT_DIM   = "#475569"

def hex_to_rgba_css(hex_color, alpha=0.18):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def hex_to_rgba_list(hex_color, alpha=210):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return [r, g, b, alpha]

def pm25_label(val):
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "N/A"
    if v < 12:   return "Good"
    if v < 35.4: return "Moderate"
    if v < 55.4: return "Unhealthy (Sensitive)"
    return "Unhealthy"

def pm25_severity_color(val):
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "#475569"
    if v < 12:   return "#4ADE80"
    if v < 35.4: return "#FBBF24"
    if v < 55.4: return "#FB923C"
    return "#F87171"

def pm25_dot_color_rgb(val):
    try:
        v = float(val)
    except (TypeError, ValueError):
        return [100, 116, 139, 180]
    if v < 12:   return [74, 222, 128, 230]
    if v < 35.4: return [251, 191, 36, 230]
    if v < 55.4: return [251, 146, 60, 230]
    return [248, 113, 113, 230]

@st.cache_data(ttl=600)
def ml_forecast(pm25_json: str, horizon: int = 12):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_absolute_error, r2_score

    df = pd.read_json(pm25_json)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], errors="coerce")
    df = df.sort_values("datetime_utc").dropna(subset=["datetime_utc"]).reset_index(drop=True)

    if len(df) < 48:
        return None, None, None

    LAGS = [1, 2, 3, 6, 12, 24]
    for lag in LAGS:
        df[f"lag_{lag}"] = df["pm25"].shift(lag)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow"]      = df["datetime_utc"].dt.dayofweek
    df["dow_sin"]  = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["dow"] / 7)

    df = df.dropna().reset_index(drop=True)
    feat_cols = [f"lag_{l}" for l in LAGS] + ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]

    X, y = df[feat_cols].values, df["pm25"].values
    split = max(1, int(len(df) * 0.8))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    pipe = Pipeline([("sc", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    pipe.fit(X_tr, y_tr)
    y_hat = pipe.predict(X_te)

    mae  = float(mean_absolute_error(y_te, y_hat)) if len(y_te) else float("nan")
    r2   = float(r2_score(y_te, y_hat))            if len(y_te) > 1 else float("nan")
    rstd = float(np.std(y_te - y_hat))             if len(y_te) else 5.0

    # Iterative multi-step forecast
    buf     = list(df["pm25"].values[-24:])
    last_dt = df["datetime_utc"].iloc[-1]
    rows    = []
    for h in range(1, horizon + 1):
        ndt  = last_dt + pd.Timedelta(hours=h)
        lv   = [max(0.0, buf[-l]) for l in LAGS]
        hr, dow = ndt.hour, ndt.dayofweek
        feats = lv + [
            np.sin(2 * np.pi * hr  / 24), np.cos(2 * np.pi * hr  / 24),
            np.sin(2 * np.pi * dow /  7), np.cos(2 * np.pi * dow /  7),
        ]
        p = max(0.0, float(pipe.predict([feats])[0]))
        rows.append({
            "datetime_utc":  ndt.isoformat(),
            "pm25_forecast": round(p, 3),
            "lower":         round(max(0.0, p - 1.96 * rstd), 3),
            "upper":         round(p + 1.96 * rstd, 3),
        })
        buf.append(p)

    test_slice = df.iloc[split:][["datetime_utc", "pm25"]].copy()
    test_slice["pm25_pred"]    = y_hat
    test_slice["datetime_utc"] = test_slice["datetime_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    metrics = {
        "mae": mae, "r2": r2, "rstd": rstd,
        "n_train": int(split), "n_test": int(len(y_te)),
    }
    return pd.DataFrame(rows).to_json(orient="records"), test_slice.to_json(), metrics


@st.cache_data(ttl=600)
def ml_anomaly(pm25_json: str, contamination: float = 0.05):
    """IsolationForest anomaly detection pada PM2.5."""
    from sklearn.ensemble import IsolationForest

    df = pd.read_json(pm25_json)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], errors="coerce")
    df = df.sort_values("datetime_utc").dropna(subset=["datetime_utc", "pm25"]).reset_index(drop=True)

    if len(df) < 20:
        df["is_anomaly"] = False
    else:
        clf = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
        df["is_anomaly"] = clf.fit_predict(df[["pm25", "hour"]].values) == -1

    out = df[["pm25", "is_anomaly"]].copy()
    out["datetime_utc"] = df["datetime_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return out.to_json()


@st.cache_data(ttl=600)
def ml_importance(data_json: str):
    """RandomForest feature importance + correlation matrix."""
    from sklearn.ensemble import RandomForestRegressor

    df = pd.read_json(data_json)
    feat_map = {
        "temperature_c":         "Temperature (°C)",
        "relative_humidity_pct": "Humidity (%)",
        "hour":                  "Hour of Day",
        "pm10":                  "PM10",
        "pm1":                   "PM1",
    }
    avail = [c for c in feat_map if c in df.columns]
    if "pm25" not in df.columns or len(avail) < 2:
        return None, None

    df_c = df[avail + ["pm25"]].dropna()
    if len(df_c) < 50:
        return None, None

    rf = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(df_c[avail].values, df_c["pm25"].values)

    imp_df = pd.DataFrame({
        "feature":    [feat_map[f] for f in avail],
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=True)

    corr_labels = {
        "pm25":                  "PM2.5",
        "pm10":                  "PM10",
        "pm1":                   "PM1",
        "temperature_c":         "Temp (°C)",
        "relative_humidity_pct": "Humidity (%)",
        "hour":                  "Hour",
    }
    corr_cols = [c for c in corr_labels if c in df.columns]
    corr_mx   = df[corr_cols].dropna().corr()
    corr_mx.index   = [corr_labels[c] for c in corr_mx.index]
    corr_mx.columns = [corr_labels[c] for c in corr_mx.columns]

    return imp_df.to_json(orient="records"), corr_mx.to_json(orient="split")


PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG_SURFACE,
    plot_bgcolor=BG_BASE,
    font=dict(family="IBM Plex Sans", color=TEXT_PRI, size=12),
    title_font=dict(family="IBM Plex Mono", color=TEXT_PRI, size=13),
    legend=dict(
        bgcolor=BG_RAISED,
        bordercolor=BORDER,
        borderwidth=1,
        font=dict(color=TEXT_PRI, size=11),
    ),
    xaxis=dict(
        gridcolor=BG_RAISED,
        linecolor=BORDER,
        tickfont=dict(color=TEXT_SEC),
        zerolinecolor=BORDER,
    ),
    yaxis=dict(
        gridcolor=BG_RAISED,
        linecolor=BORDER,
        tickfont=dict(color=TEXT_SEC),
        zerolinecolor=BORDER,
    ),
    margin=dict(l=20, r=20, t=48, b=20),
)

PLOTLY_LAYOUT_NO_LEGEND = {k: v for k, v in PLOTLY_LAYOUT.items() if k != "legend"}

@st.cache_resource
def get_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "envdata"),
        user=os.getenv("POSTGRES_USER", "envuser"),
        password=os.getenv("POSTGRES_PASSWORD", "12345678"),
    )

@st.cache_data(ttl=300)
def load_data():
    conn = get_connection()
    query = """
        SELECT
            datetime_utc,
            location_name,
            latitude,
            longitude,
            pm25,
            pm10,
            pm1,
            temperature_c,
            relative_humidity_pct,
            apparent_temperature_c
        FROM env_merged
        ORDER BY datetime_utc DESC;
    """
    df = pd.read_sql(query, conn)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df["date"] = df["datetime_utc"].dt.date
    df["hour"] = df["datetime_utc"].dt.hour
    return df

try:
    df = load_data()
    data_ok = not df.empty
except Exception as e:
    st.error(f"Database connection failed: {e}")
    data_ok = False
    df = pd.DataFrame()

if data_ok:
    all_locations_sorted = sorted(df["location_name"].unique().tolist())
    loc_color_map = {
        loc: COLOR_SEQ[i % len(COLOR_SEQ)]
        for i, loc in enumerate(all_locations_sorted)
    }
else:
    all_locations_sorted = []
    loc_color_map = {}

dynamic_tag_css = ""
if loc_color_map:
    for loc, color in loc_color_map.items():
        bg     = hex_to_rgba_css(color, 0.20)
        border = hex_to_rgba_css(color, 0.50)
        safe   = loc.replace('"', '\\"')
        dynamic_tag_css += f"""
        [data-testid="stMultiSelect"] span[data-baseweb="tag"][title="{safe}"] {{
            background-color: {bg} !important;
            border: 1px solid {border} !important;
            color: {color} !important;
        }}
        [data-testid="stMultiSelect"] span[data-baseweb="tag"][title="{safe}"] span {{
            color: {color} !important;
            font-weight: 600 !important;
        }}
        [data-testid="stMultiSelect"] span[data-baseweb="tag"][title="{safe}"] button svg {{
            fill: {color} !important;
        }}
        """

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"], .stApp {{
        font-family: 'IBM Plex Sans', sans-serif !important;
        background-color: {BG_BASE} !important;
        color: {TEXT_PRI} !important;
    }}
    .block-container {{
        background-color: {BG_BASE} !important;
        padding-top: 2rem;
    }}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {{
        background-color: {BG_SURFACE} !important;
        border-right: 1px solid {BORDER} !important;
    }}
    [data-testid="stSidebar"] * {{ color: {TEXT_PRI} !important; }}
    [data-testid="stSidebar"] hr {{ border-color: {BORDER} !important; }}

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: transparent !important;
        border-bottom: 1px solid {BORDER} !important;
        gap: 0px;
        padding-bottom: 0;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 11px !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        color: {TEXT_SEC} !important;
        background-color: transparent !important;
        border-radius: 0 !important;
        padding: 10px 24px !important;
        border-bottom: 2px solid transparent !important;
    }}
    .stTabs [aria-selected="true"] {{
        color: {TEXT_PRI} !important;
        border-bottom: 2px solid #60A5FA !important;
        background-color: transparent !important;
    }}
    .stTabs [data-baseweb="tab-highlight"] {{
        display: none !important;
    }}
    .stTabs [data-baseweb="tab-panel"] {{
        padding-top: 28px !important;
    }}

    /* ── Metric cards ── */
    [data-testid="metric-container"] {{
        background-color: {BG_SURFACE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
        padding: 20px !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.4) !important;
    }}
    [data-testid="metric-container"] label {{
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 10px !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        color: {TEXT_SEC} !important;
    }}
    [data-testid="stMetricValue"] {{
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 26px !important;
        color: {TEXT_PRI} !important;
    }}

    /* ── Section headers ── */
    .section-header {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: {TEXT_DIM};
        border-bottom: 1px solid {BORDER};
        padding-bottom: 8px;
        margin: 36px 0 20px 0;
    }}

    .dashboard-title {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 22px;
        font-weight: 600;
        color: {TEXT_PRI};
        letter-spacing: -0.01em;
    }}
    .dashboard-subtitle {{
        font-size: 13px;
        color: {TEXT_SEC};
        margin-top: 2px;
        font-weight: 300;
    }}

    /* ── ML badge ── */
    .ml-badge {{
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 9px;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #A78BFA;
        background: rgba(167,139,250,0.12);
        border: 1px solid rgba(167,139,250,0.30);
        border-radius: 4px;
        padding: 2px 8px;
        margin-left: 8px;
        vertical-align: middle;
    }}

    /* ── Model info card ── */
    .model-info-card {{
        background: {BG_SURFACE};
        border: 1px solid {BORDER};
        border-left: 3px solid #A78BFA;
        border-radius: 6px;
        padding: 14px 18px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        color: {TEXT_SEC};
        line-height: 1.8;
        margin-bottom: 12px;
    }}
    .model-info-card b {{
        color: {TEXT_PRI};
        font-weight: 600;
    }}

    .chart-container {{
        background: {BG_SURFACE};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 8px;
        margin-bottom: 4px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.35);
    }}

    hr {{ border-color: {BORDER} !important; }}

    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }}

    .stButton button {{
        background-color: {BG_RAISED} !important;
        color: {TEXT_PRI} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 6px !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 11px !important;
        letter-spacing: 0.05em !important;
    }}
    .stButton button:hover {{
        background-color: {BORDER} !important;
    }}

    [data-testid="stMultiSelect"] span[data-baseweb="tag"] {{
        border-radius: 4px !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 11px !important;
    }}

    [data-baseweb="popover"] {{
        background-color: {BG_SURFACE} !important;
        border: 1px solid {BORDER} !important;
    }}
    [data-baseweb="menu"] li {{ color: {TEXT_PRI} !important; }}
    [data-baseweb="menu"] li:hover {{ background-color: {BG_RAISED} !important; }}

    [data-baseweb="input"], [data-baseweb="select"] {{
        background-color: {BG_RAISED} !important;
        border-color: {BORDER} !important;
        color: {TEXT_PRI} !important;
    }}

    [data-testid="stExpander"] {{
        background-color: {BG_SURFACE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
    }}

    .map-legend {{
        display: flex;
        gap: 16px;
        align-items: center;
        flex-wrap: wrap;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {TEXT_SEC};
        margin-bottom: 8px;
    }}
    .map-legend-item {{
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    .map-legend-dot {{
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
    }}

    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: {BG_BASE}; }}
    ::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}

    {dynamic_tag_css}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<p class="dashboard-title">SG Air Quality</p>', unsafe_allow_html=True)
    st.markdown('<p class="dashboard-subtitle">Singapore Environmental Monitor</p>', unsafe_allow_html=True)
    st.markdown("---")

    if data_ok:
        selected_locs = st.multiselect(
            "Locations",
            options=all_locations_sorted,
            default=all_locations_sorted,
        )

        min_date = df["date"].min()
        max_date = df["date"].max()
        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        st.markdown("---")
        st.markdown("**Dataset**")
        st.markdown(f"Records: `{len(df):,}`")
        st.markdown(f"Active stations: `{df['location_name'].nunique()}`")
        if len(df) > 0:
            latest = df["datetime_utc"].max()
            st.markdown(f"Latest: `{latest.strftime('%Y-%m-%d %H:%M')} UTC`")

        st.markdown("---")
        if st.button("Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

if data_ok:
    if len(date_range) == 2:
        start_d, end_d = date_range
    else:
        start_d = end_d = date_range[0]

    mask = (
        df["location_name"].isin(selected_locs) &
        (df["date"] >= start_d) &
        (df["date"] <= end_d)
    )
    fdf = df[mask].copy()
else:
    fdf = pd.DataFrame()

st.markdown('<p class="dashboard-title">Singapore Air Quality & Weather</p>', unsafe_allow_html=True)
st.markdown(
    f'<p class="dashboard-subtitle">Real-time environmental monitoring — '
    f'{df["location_name"].nunique() if data_ok else 0} stations · OpenAQ v3 + Open-Meteo Archive</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

if not data_ok or fdf.empty:
    st.warning("No data available for the selected filters. Make sure the Airflow pipeline has been run.")
    st.stop()

tab1, tab2 = st.tabs(["📊  Dashboard", "🤖  ML Insights"])

# TAB 1 — DASHBOARD (existing content, unchanged)
with tab1:

    # ── KPI METRICS ──
    st.markdown('<p class="section-header">Overview</p>', unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    avg_pm25  = fdf["pm25"].mean()
    max_pm25  = fdf["pm25"].max()
    avg_temp  = fdf["temperature_c"].mean()
    avg_hum   = fdf["relative_humidity_pct"].mean()
    n_records = len(fdf)

    with k1: st.metric("Avg PM2.5",       f"{avg_pm25:.1f} µg/m³"  if pd.notna(avg_pm25) else "N/A")
    with k2: st.metric("Max PM2.5",       f"{max_pm25:.1f} µg/m³"  if pd.notna(max_pm25) else "N/A")
    with k3: st.metric("Avg Temperature", f"{avg_temp:.1f} °C"      if pd.notna(avg_temp)  else "N/A")
    with k4: st.metric("Avg Humidity",    f"{avg_hum:.1f} %"        if pd.notna(avg_hum)   else "N/A")
    with k5: st.metric("Total Records",   f"{n_records:,}")

    # ── CHART 1: TIME SERIES ──
    st.markdown('<p class="section-header">PM2.5 Time Series</p>', unsafe_allow_html=True)

    ts_df = fdf.dropna(subset=["pm25"]).sort_values("datetime_utc")
    fig_ts = px.line(
        ts_df,
        x="datetime_utc", y="pm25", color="location_name",
        labels={"datetime_utc": "", "pm25": "PM2.5 (µg/m³)", "location_name": "Station"},
        color_discrete_map=loc_color_map,
        height=420,
    )
    fig_ts.update_layout(**PLOTLY_LAYOUT, title="Hourly PM2.5 per Station")
    fig_ts.update_traces(line=dict(width=1.8))
    fig_ts.add_hline(
        y=15, line_dash="dot", line_color=TEXT_DIM, line_width=1,
        annotation_text="WHO Annual Guideline — 15 µg/m³",
        annotation_font=dict(color=TEXT_SEC, size=10),
        annotation_position="top right",
    )
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(fig_ts, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── CHART 2: STATION MAP ──
    st.markdown('<p class="section-header">Station Map</p>', unsafe_allow_html=True)

    map_df = fdf.groupby(["location_name", "latitude", "longitude"]).agg(
        pm25=("pm25", "mean"),
    ).reset_index().dropna(subset=["latitude", "longitude"])

    map_df["color"]         = map_df["pm25"].apply(pm25_dot_color_rgb)
    map_df["station_color"] = map_df["location_name"].apply(
        lambda loc: hex_to_rgba_list(loc_color_map.get(loc, "#64748b"), 220)
    )
    map_df["pm25_label"] = map_df["pm25"].apply(lambda v: f"{v:.1f} µg/m³" if pd.notna(v) else "N/A")
    map_df["severity"]   = map_df["pm25"].apply(pm25_label)

    severity_layer = pdk.Layer(
        "ScatterplotLayer", data=map_df,
        get_position=["longitude", "latitude"],
        get_fill_color="color",
        get_radius=650, pickable=False, opacity=0.28, stroked=False,
    )
    station_layer = pdk.Layer(
        "ScatterplotLayer", data=map_df,
        get_position=["longitude", "latitude"],
        get_fill_color="station_color",
        get_radius=230, pickable=True, opacity=1.0,
        stroked=True, get_line_color=[15, 17, 23, 255], line_width_min_pixels=2,
    )

    sg_lat = map_df["latitude"].mean()  if not map_df.empty else 1.3521
    sg_lon = map_df["longitude"].mean() if not map_df.empty else 103.8198

    deck = pdk.Deck(
        layers=[severity_layer, station_layer],
        initial_view_state=pdk.ViewState(latitude=sg_lat, longitude=sg_lon, zoom=11, pitch=0),
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        tooltip={
            "html": (
                "<b style='font-family:IBM Plex Mono;font-size:12px;color:#e2e8f0'>{location_name}</b><br/>"
                "<span style='font-family:IBM Plex Sans;font-size:12px;color:#cbd5e1'>Avg PM2.5: {pm25_label}</span><br/>"
                "<span style='font-family:IBM Plex Sans;font-size:11px;color:#64748b'>Status: {severity}</span>"
            ),
            "style": {
                "backgroundColor": "#161b27",
                "border": "1px solid #2a3347",
                "borderRadius": "6px",
                "padding": "10px 14px",
                "boxShadow": "0 4px 16px rgba(0,0,0,0.6)",
            }
        },
    )

    st.markdown("""
    <div class="map-legend">
        <span style="color:#475569;letter-spacing:0.1em">PM2.5 SEVERITY:</span>
        <div class="map-legend-item">
            <span class="map-legend-dot" style="background:#4ADE80;box-shadow:0 0 7px #4ADE80"></span> Good &lt;12
        </div>
        <div class="map-legend-item">
            <span class="map-legend-dot" style="background:#FBBF24;box-shadow:0 0 7px #FBBF24"></span> Moderate &lt;35.4
        </div>
        <div class="map-legend-item">
            <span class="map-legend-dot" style="background:#FB923C;box-shadow:0 0 7px #FB923C"></span> Unhealthy Sensitive &lt;55.4
        </div>
        <div class="map-legend-item">
            <span class="map-legend-dot" style="background:#F87171;box-shadow:0 0 7px #F87171"></span> Unhealthy ≥55.4
        </div>
    </div>
    <p style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#475569;letter-spacing:0.06em;margin-bottom:6px;">
        Outer glow = PM2.5 severity &nbsp;·&nbsp; Inner dot = station identity color
    </p>
    """, unsafe_allow_html=True)

    st.pydeck_chart(deck, use_container_width=True, height=460)

    # ── CHART 3: BAR — severity gradient ──
    st.markdown('<p class="section-header">Average PM2.5 per Station</p>', unsafe_allow_html=True)

    bar_df = fdf.groupby("location_name").agg(
        pm25=("pm25", "mean"),
    ).reset_index().sort_values("pm25", ascending=True)

    bar_df["severity"]   = bar_df["pm25"].apply(pm25_label)
    bar_df["pm25_label"] = bar_df["pm25"].apply(lambda v: f"{v:.1f}" if pd.notna(v) else "N/A")

    HEATMAP_COLORSCALE = [
        [0.0,  "#1e3a5f"],
        [0.5,  "#92400e"],
        [1.0,  "#7f1d1d"],
    ]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=bar_df["pm25"], y=bar_df["location_name"],
        orientation="h",
        marker=dict(
            color=bar_df["pm25"],
            colorscale=HEATMAP_COLORSCALE,
            opacity=0.92,
            line=dict(color="rgba(0,0,0,0.2)", width=1),
            showscale=True,
            colorbar=dict(
                title=dict(text="µg/m³", font=dict(family="IBM Plex Mono", size=10, color=TEXT_SEC)),
                tickfont=dict(family="IBM Plex Mono", size=10, color=TEXT_SEC),
                bgcolor=BG_RAISED,
                bordercolor=BORDER,
                borderwidth=1,
                thickness=14,
                len=0.85,
                x=1.01,
            ),
        ),
        text=bar_df["pm25_label"], textposition="outside",
        textfont=dict(color=TEXT_SEC, size=11, family="IBM Plex Mono"),
        customdata=bar_df["severity"],
        hovertemplate="<b>%{y}</b><br>PM2.5: %{x:.1f} µg/m³<br>Status: %{customdata}<extra></extra>",
    ))

    layout_bar = {**PLOTLY_LAYOUT}
    layout_bar["legend"] = dict(
        bgcolor=BG_RAISED, bordercolor=BORDER, borderwidth=1,
        font=dict(color=TEXT_PRI, size=11),
    )
    fig_bar.update_layout(
        **layout_bar,
        title="Mean PM2.5 Concentration by Station",
        xaxis_title="PM2.5 (µg/m³)", yaxis_title="",
        height=max(380, len(bar_df) * 42 + 100),
        showlegend=False,
    )
    fig_bar.add_vline(
        x=15, line_dash="dot", line_color=TEXT_DIM, line_width=1,
        annotation_text="WHO 15 µg/m³",
        annotation_font=dict(color=TEXT_SEC, size=10),
    )
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── CHART 4: HEATMAP ──
    st.markdown('<p class="section-header">PM2.5 by Station and Hour (UTC)</p>', unsafe_allow_html=True)

    heat_df = fdf.dropna(subset=["pm25"]).groupby(
        ["location_name", "hour"]
    )["pm25"].mean().reset_index()

    if not heat_df.empty:
        heat_pivot = heat_df.pivot(index="location_name", columns="hour", values="pm25")
        fig_heat = px.imshow(
            heat_pivot,
            color_continuous_scale=[[0, "#1e3a5f"], [0.5, "#92400e"], [1, "#7f1d1d"]],
            aspect="auto",
            labels=dict(x="Hour (UTC)", y="", color="µg/m³"),
            height=360,
        )
        fig_heat.update_layout(**PLOTLY_LAYOUT, title="Diurnal PM2.5 Pattern — Station x Hour")
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(fig_heat, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── CHART 5: TEMPERATURE vs PM2.5 ──
    st.markdown('<p class="section-header">Temperature vs PM2.5</p>', unsafe_allow_html=True)

    scatter_df = fdf.dropna(subset=["pm25", "temperature_c", "relative_humidity_pct"])

    fig_sc1 = px.scatter(
        scatter_df, x="temperature_c", y="pm25", color="location_name",
        trendline="ols",
        labels={"temperature_c": "Temperature (°C)", "pm25": "PM2.5 (µg/m³)", "location_name": "Station"},
        color_discrete_map=loc_color_map, opacity=0.65, height=420,
    )
    fig_sc1.update_layout(**PLOTLY_LAYOUT, title="Temperature vs PM2.5 Concentration")
    fig_sc1.update_traces(marker=dict(size=5))
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(fig_sc1, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── CHART 6: HUMIDITY vs PM2.5 ──
    st.markdown('<p class="section-header">Relative Humidity vs PM2.5</p>', unsafe_allow_html=True)

    fig_sc2 = px.scatter(
        scatter_df, x="relative_humidity_pct", y="pm25", color="location_name",
        trendline="ols",
        labels={"relative_humidity_pct": "Relative Humidity (%)", "pm25": "PM2.5 (µg/m³)", "location_name": "Station"},
        color_discrete_map=loc_color_map, opacity=0.65, height=420,
    )
    fig_sc2.update_layout(**PLOTLY_LAYOUT, title="Relative Humidity vs PM2.5 Concentration")
    fig_sc2.update_traces(marker=dict(size=5))
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(fig_sc2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── TABLE ──
    st.markdown('<p class="section-header">Station Summary</p>', unsafe_allow_html=True)

    summary = fdf.groupby("location_name").agg(
        PM25_avg=("pm25", "mean"),
        PM25_max=("pm25", "max"),
        Temp_avg=("temperature_c", "mean"),
        Humid_avg=("relative_humidity_pct", "mean"),
        Records=("pm25", "count"),
    ).reset_index()
    summary.columns = ["Station", "PM2.5 Avg", "PM2.5 Max", "Temp Avg (°C)", "Humidity Avg (%)", "Records"]
    summary["Status"] = summary["PM2.5 Avg"].apply(pm25_label)
    for col in ["PM2.5 Avg", "PM2.5 Max", "Temp Avg (°C)", "Humidity Avg (%)"]:
        summary[col] = summary[col].apply(lambda v: f"{v:.1f}" if pd.notna(v) else "N/A")
    summary = summary.sort_values("PM2.5 Avg")

    st.dataframe(
        summary, use_container_width=True, hide_index=True,
        column_config={
            "Station": st.column_config.TextColumn(width="medium"),
            "Status":  st.column_config.TextColumn(width="small"),
            "Records": st.column_config.NumberColumn(format="%d"),
        }
    )

    with st.expander("Raw Data"):
        st.dataframe(
            fdf.sort_values("datetime_utc", ascending=False).head(500),
            use_container_width=True, hide_index=True,
        )

    st.markdown("---")
    st.markdown(
        f"<p style='text-align:center; color:{TEXT_DIM}; font-size:11px; font-family: IBM Plex Mono'>"
        "Data sources: OpenAQ v3 &nbsp;·&nbsp; Open-Meteo Historical Archive &nbsp;·&nbsp; Built with Streamlit"
        "</p>",
        unsafe_allow_html=True,
    )


# TAB 2 — ML INSIGHTS
with tab2:

    st.markdown(
        '<p class="dashboard-subtitle" style="margin-bottom:20px;">'
        'Machine learning models trained on historical data — Ridge Regression · IsolationForest · RandomForest'
        '<span class="ml-badge">scikit-learn</span>'
        '</p>',
        unsafe_allow_html=True,
    )

    # ── Controls ──
    ctrl_left, ctrl_mid, ctrl_right = st.columns([2, 1, 1])
    with ctrl_left:
        ml_station = st.selectbox(
            "Station for ML Analysis",
            options=all_locations_sorted,
            index=0,
            key="ml_station_sel",
        )
    with ctrl_mid:
        horizon = st.selectbox(
            "Forecast Horizon (hours)",
            options=[6, 12, 24, 48],
            index=1,
            help="Semakin panjang horizon, semakin besar ketidakpastian prediksi.",
            key="forecast_horizon_sel",
        )
    with ctrl_right:
        contamination = st.slider(
            "Anomaly Sensitivity",
            min_value=0.01, max_value=0.15, value=0.05, step=0.01,
            help="Fraction of data points treated as anomalies by IsolationForest.",
            key="contamination_slider",
        )

    sta_color  = loc_color_map.get(ml_station, "#60A5FA")
    ml_df      = fdf[fdf["location_name"] == ml_station].sort_values("datetime_utc").copy()
    n_pm25_pts = ml_df["pm25"].dropna().shape[0]

    if n_pm25_pts < 48:
        st.warning(
            f"Not enough PM2.5 data for **{ml_station}** in the selected date range "
            f"({n_pm25_pts} points found, minimum 48 required). "
            "Try extending the date range in the sidebar."
        )
    else:
        # ── Prepare JSON inputs (cache-safe) ──
        def _to_iso(s):
            return s.dt.strftime("%Y-%m-%dT%H:%M:%S")

        pm25_json = (
            ml_df[["datetime_utc", "pm25", "hour"]]
            .dropna(subset=["pm25"])
            .assign(datetime_utc=_to_iso(ml_df["datetime_utc"]))
            .to_json()
        )
        full_json = (
            ml_df[["datetime_utc", "pm25", "pm10", "pm1",
                   "temperature_c", "relative_humidity_pct", "hour"]]
            .assign(datetime_utc=_to_iso(ml_df["datetime_utc"]))
            .to_json()
        )

        # ── Run ML (all cached) ──
        with st.spinner("Running ML models…"):
            fc_records_json, test_json, fc_metrics = ml_forecast(pm25_json, horizon=horizon)
            anom_json                               = ml_anomaly(pm25_json, contamination)
            imp_records_json, corr_split_json       = ml_importance(full_json) or (None, None)

        # ── Parse results ──
        fc_df = pd.read_json(fc_records_json, orient="records") if fc_records_json else None
        if fc_df is not None:
            fc_df["datetime_utc"] = pd.to_datetime(fc_df["datetime_utc"])

        anom_df = pd.read_json(anom_json) if anom_json else None
        if anom_df is not None:
            anom_df["datetime_utc"] = pd.to_datetime(anom_df["datetime_utc"])

        n_anomalies = int(anom_df["is_anomaly"].sum()) if anom_df is not None else 0
        n_total     = len(anom_df)                     if anom_df is not None else 0
        anom_pct    = n_anomalies / n_total * 100      if n_total > 0 else 0.0

        # ──────────────────────────────────────
        # MODEL PERFORMANCE CARDS
        # ──────────────────────────────────────
        st.markdown('<p class="section-header">Model Performance</p>', unsafe_allow_html=True)

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            val = f"{fc_metrics['mae']:.2f} µg/m³" if fc_metrics and not np.isnan(fc_metrics["mae"]) else "N/A"
            st.metric("Forecast MAE", val, help="Mean Absolute Error on hold-out test set (lower = better)")
        with mc2:
            val = f"{fc_metrics['r2']:.3f}" if fc_metrics and not np.isnan(fc_metrics["r2"]) else "N/A"
            st.metric("R² Score", val, help="Coefficient of determination (1.0 = perfect, 0 = baseline mean)")
        with mc3:
            st.metric("Anomalies Detected", f"{n_anomalies}  ({anom_pct:.1f}%)",
                      help="Points flagged by IsolationForest as statistically unusual")
        with mc4:
            val = f"{fc_metrics['n_train']:,}" if fc_metrics else "N/A"
            st.metric("Training Points", val, help="Hourly observations used to train the forecast model")

        # ──────────────────────────────────────────────────────────
        # SECTION 1 — FORECASTING
        # ──────────────────────────────────────────────────────────
        st.markdown(
            f'<p class="section-header">PM2.5 Forecast — Next {horizon} Hours</p>',
            unsafe_allow_html=True,
        )

        col_fc, col_info = st.columns([3, 1])
        with col_info:
            rstd_str = f"{fc_metrics['rstd']:.2f}" if fc_metrics else "—"
            n_tr_str = f"{fc_metrics['n_train']:,}" if fc_metrics else "—"
            n_te_str = f"{fc_metrics['n_test']:,}"  if fc_metrics else "—"
            st.markdown(
                f"""<div class="model-info-card">
                <b>Model</b><br/>Ridge Regression<br/><br/>
                <b>Features</b><br/>
                PM2.5 lags (1h, 2h, 3h, 6h, 12h, 24h)<br/>
                Hour sin/cos · Day-of-week sin/cos<br/><br/>
                <b>Horizon</b><br/>{horizon} hours ahead<br/><br/>
                <b>Residual σ</b><br/>{rstd_str} µg/m³<br/><br/>
                <b>Train / Test</b><br/>{n_tr_str} / {n_te_str} points<br/><br/>
                <b>CI Band</b><br/>± 1.96 σ (95%)
                </div>""",
                unsafe_allow_html=True,
            )

        with col_fc:
            if fc_df is not None:
                cutoff  = ml_df["datetime_utc"].max() - pd.Timedelta(hours=72)
                hist_df = ml_df[ml_df["datetime_utc"] >= cutoff][["datetime_utc", "pm25"]].dropna()

                fig_fc = go.Figure()

                # Confidence band
                fig_fc.add_trace(go.Scatter(
                    x=pd.concat([fc_df["datetime_utc"], fc_df["datetime_utc"][::-1]]),
                    y=pd.concat([fc_df["upper"], fc_df["lower"][::-1]]),
                    fill="toself",
                    fillcolor=hex_to_rgba_css(sta_color, 0.13),
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    name="95% CI",
                    showlegend=True,
                ))

                # Historical
                fig_fc.add_trace(go.Scatter(
                    x=hist_df["datetime_utc"], y=hist_df["pm25"],
                    mode="lines",
                    name="Actual PM2.5",
                    line=dict(color=sta_color, width=1.8),
                ))

                # Forecast
                fig_fc.add_trace(go.Scatter(
                    x=fc_df["datetime_utc"], y=fc_df["pm25_forecast"],
                    mode="lines+markers",
                    name="Forecast",
                    line=dict(color=sta_color, width=2, dash="dash"),
                    marker=dict(size=6, color=BG_BASE, symbol="circle",
                                line=dict(color=sta_color, width=2)),
                ))

                # WHO guideline
                fig_fc.add_hline(
                    y=15, line_dash="dot", line_color=TEXT_DIM, line_width=1,
                    annotation_text="WHO 15 µg/m³",
                    annotation_font=dict(color=TEXT_SEC, size=10),
                    annotation_position="top right",
                )

                # "Now" divider
                last_ts = ml_df["datetime_utc"].max()
                fig_fc.add_vline(
                    x=last_ts.timestamp() * 1000,
                    line_dash="dot", line_color=TEXT_SEC, line_width=1,
                    annotation_text="Now",
                    annotation_font=dict(color=TEXT_SEC, size=10),
                    annotation_position="top left",
                )

                fig_fc.update_layout(
                    **PLOTLY_LAYOUT_NO_LEGEND,
                    title=f"PM2.5 Forecast — {ml_station} ({horizon}h)",
                    xaxis_title="", yaxis_title="PM2.5 (µg/m³)",
                    height=400,
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.01,
                        xanchor="right", x=1,
                        bgcolor=BG_RAISED, bordercolor=BORDER, borderwidth=1,
                        font=dict(color=TEXT_PRI, size=11),
                    ),
                )

                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.plotly_chart(fig_fc, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                if not fc_df.empty and not hist_df.empty:
                    latest_forecast = float(fc_df["pm25_forecast"].iloc[-1])
                    recent_avg = float(hist_df["pm25"].tail(min(24, len(hist_df))).mean())
                    forecast_peak = float(fc_df["pm25_forecast"].max())
                    direction = "naik" if latest_forecast > recent_avg else "turun/stabil"
                    risk_status = pm25_label(latest_forecast)

                    if forecast_peak >= 35.4:
                        risk_msg = "⚠️ Ada potensi kenaikan PM2.5 cukup tinggi pada horizon prediksi ini."
                    elif latest_forecast >= 15:
                        risk_msg = "🟡 PM2.5 diprediksi berada di atas guideline WHO 15 µg/m³."
                    else:
                        risk_msg = "🟢 PM2.5 diprediksi relatif aman pada horizon ini."

                    st.info(
                        f"📈 **Forecast insight:** Untuk **{ml_station}**, PM2.5 diprediksi **{direction}** "
                        f"dalam **{horizon} jam** ke depan. Estimasi akhir horizon: "
                        f"**{latest_forecast:.1f} µg/m³** ({risk_status}). {risk_msg}"
                    )
            else:
                st.info("Not enough data to train the forecast model (minimum 48 hours required).")

        # Forecast table (collapsible)
        if fc_df is not None:
            with st.expander(f"Forecast Values (next {horizon}h)", expanded=False):
                disp = fc_df.copy()
                disp["Time (UTC)"]         = disp["datetime_utc"].dt.strftime("%Y-%m-%d %H:%M")
                disp["Forecast (µg/m³)"]   = disp["pm25_forecast"].round(2)
                disp["Lower 95%"]          = disp["lower"].round(2)
                disp["Upper 95%"]          = disp["upper"].round(2)
                disp["Status"]             = disp["pm25_forecast"].apply(pm25_label)
                st.dataframe(
                    disp[["Time (UTC)", "Forecast (µg/m³)", "Lower 95%", "Upper 95%", "Status"]],
                    use_container_width=True, hide_index=True,
                )

        # ──────────────────────────────────────────────────────────
        # SECTION 2 — ANOMALY DETECTION
        # ──────────────────────────────────────────────────────────
        st.markdown(
            '<p class="section-header">Anomaly Detection</p>',
            unsafe_allow_html=True,
        )

        col_anom, col_ainfo = st.columns([3, 1])

        with col_ainfo:
            st.markdown(
                f"""<div class="model-info-card">
                <b>Model</b><br/>IsolationForest<br/><br/>
                <b>Features</b><br/>PM2.5 · Hour of Day<br/><br/>
                <b>Contamination</b><br/>{contamination:.0%} of data<br/><br/>
                <b>Logic</b><br/>Points in low-density regions of the feature space are flagged as anomalies<br/><br/>
                <b>Flagged</b><br/>{n_anomalies} / {n_total} readings<br/>({anom_pct:.1f}%)
                </div>""",
                unsafe_allow_html=True,
            )

        with col_anom:
            if anom_df is not None and not anom_df.empty:
                normal_df  = anom_df[~anom_df["is_anomaly"]]
                anomaly_df = anom_df[anom_df["is_anomaly"]]

                fig_anom = go.Figure()

                # Normal line
                fig_anom.add_trace(go.Scatter(
                    x=normal_df["datetime_utc"], y=normal_df["pm25"],
                    mode="lines",
                    name="Normal",
                    line=dict(color=sta_color, width=1.6),
                    opacity=0.85,
                ))

                # Anomaly dots
                if not anomaly_df.empty:
                    fig_anom.add_trace(go.Scatter(
                        x=anomaly_df["datetime_utc"], y=anomaly_df["pm25"],
                        mode="markers",
                        name=f"Anomaly ({n_anomalies})",
                        marker=dict(
                            color="#F87171",
                            size=9,
                            symbol="circle",
                            line=dict(color="#7f1d1d", width=1.5),
                        ),
                    ))

                fig_anom.add_hline(
                    y=15, line_dash="dot", line_color=TEXT_DIM, line_width=1,
                    annotation_text="WHO 15 µg/m³",
                    annotation_font=dict(color=TEXT_SEC, size=10),
                    annotation_position="top right",
                )
                fig_anom.update_layout(
                    **PLOTLY_LAYOUT_NO_LEGEND,
                    title=f"Anomaly Detection — {ml_station}",
                    xaxis_title="", yaxis_title="PM2.5 (µg/m³)",
                    height=380,
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.01,
                        xanchor="right", x=1,
                        bgcolor=BG_RAISED, bordercolor=BORDER, borderwidth=1,
                        font=dict(color=TEXT_PRI, size=11),
                    ),
                )
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.plotly_chart(fig_anom, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                if not anomaly_df.empty:
                    highest_anomaly = anomaly_df["pm25"].max()
                    st.warning(
                        f"🚨 **Anomaly insight:** Terdeteksi **{n_anomalies} anomaly readings** "
                        f"({anom_pct:.1f}% dari data {ml_station}). "
                        f"Nilai anomaly tertinggi: **{highest_anomaly:.1f} µg/m³**. "
                        "Anomali bisa menunjukkan spike polusi, perubahan kondisi lingkungan, atau noise sensor."
                    )
                else:
                    st.success("✅ Tidak ada anomaly reading yang terdeteksi pada filter dan sensitivitas saat ini.")
            else:
                st.info("Not enough data for anomaly detection.")

        # Anomaly table
        if anom_df is not None and n_anomalies > 0:
            with st.expander(f"Anomalous Readings ({n_anomalies} detected)", expanded=False):
                an_disp = anom_df[anom_df["is_anomaly"]][["datetime_utc", "pm25"]].copy()
                an_disp["Time (UTC)"]    = an_disp["datetime_utc"].dt.strftime("%Y-%m-%d %H:%M")
                an_disp["PM2.5 (µg/m³)"] = an_disp["pm25"].round(2)
                an_disp["Status"]         = an_disp["pm25"].apply(pm25_label)
                st.dataframe(
                    an_disp[["Time (UTC)", "PM2.5 (µg/m³)", "Status"]]
                    .sort_values("PM2.5 (µg/m³)", ascending=False),
                    use_container_width=True, hide_index=True,
                )

        # ── Footer ──
        st.markdown("---")
        st.markdown(
            f"<p style='color:{TEXT_DIM};font-size:11px;font-family:IBM Plex Mono;text-align:center'>"
            "Forecast: Ridge Regression + lag features &nbsp;·&nbsp; "
            "Anomaly: IsolationForest &nbsp;·&nbsp; "
            "Importance: RandomForest &nbsp;·&nbsp; "
            "Built with scikit-learn"
            "</p>",
            unsafe_allow_html=True,
        )