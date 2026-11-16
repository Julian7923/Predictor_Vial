# =====================================================================
# INTELLIGENT ROAD RISK PREDICTOR — BOGOTÁ D.C.
# UNICAFAM · Introduction to Data Analytics · Deliverable 3 (BI & Storytelling)
# Authors: José Julián Herrera Pineda & Deglym Sneider Ossa Parra
# Advisor: Omar Otalora Barreto, PhD
# =====================================================================

import io
import unicodedata
import numpy as np
import pandas as pd
import requests
import sympy as sp
import matplotlib.pyplot as plt
import streamlit as st

# ---------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CONSTANTS
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Road Risk BI — Bogotá D.C.",
    page_icon="🚑",
    layout="wide"
)

SAMPLE_LIMIT = 8000
POLY_DEGREE = 4
WARNING_WINDOW_MIN = 30

BOGOTA_LOCALITIES = {
    1: {"name": "Usaquén", "lat": 4.7420, "lon": -74.0300},
    2: {"name": "Chapinero", "lat": 4.6540, "lon": -74.0500},
    3: {"name": "Santa Fe", "lat": 4.6020, "lon": -74.0650},
    4: {"name": "San Cristóbal", "lat": 4.5650, "lon": -74.0800},
    5: {"name": "Usme", "lat": 4.4700, "lon": -74.1100},
    6: {"name": "Tunjuelito", "lat": 4.5750, "lon": -74.1350},
    7: {"name": "Bosa", "lat": 4.6200, "lon": -74.1900},
    8: {"name": "Kennedy", "lat": 4.6300, "lon": -74.1500},
    9: {"name": "Fontibón", "lat": 4.6700, "lon": -74.1400},
    10: {"name": "Engativá", "lat": 4.7100, "lon": -74.1100},
    11: {"name": "Suba", "lat": 4.7600, "lon": -74.0900},
    12: {"name": "Barrios Unidos", "lat": 4.6680, "lon": -74.0750},
    13: {"name": "Teusaquillo", "lat": 4.6450, "lon": -74.0850},
    14: {"name": "Los Mártires", "lat": 4.6050, "lon": -74.0880},
    15: {"name": "Antonio Nariño", "lat": 4.5900, "lon": -74.1000},
    16: {"name": "Puente Aranda", "lat": 4.6200, "lon": -74.1150},
    17: {"name": "La Candelaria", "lat": 4.5960, "lon": -74.0720},
    18: {"name": "Rafael Uribe Uribe", "lat": 4.5600, "lon": -74.1200},
    19: {"name": "Ciudad Bolívar", "lat": 4.5200, "lon": -74.1500},
    20: {"name": "Sumapaz", "lat": 4.0100, "lon": -74.3200},
}

def normalize_text(text):
    clean = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return clean.upper().strip()

CODE_TO_NAME = {k: v.get("name") for k, v in BOGOTA_LOCALITIES.items()}
NAME_TO_LAT = {v.get("name"): v.get("lat") for v in BOGOTA_LOCALITIES.values()}
NAME_TO_LON = {v.get("name"): v.get("lon") for v in BOGOTA_LOCALITIES.values()}
NORM_TO_NAME = {normalize_text(v.get("name")): v.get("name") for v in BOGOTA_LOCALITIES.values()}
NORM_TO_NAME.update({
    "RAFAEL URIBE": "Rafael Uribe Uribe",
    "CANDELARIA": "La Candelaria",
    "MARTIRES": "Los Mártires",
    "SANTAFE": "Santa Fe"
})

def find_column(df, keywords, exclude=()):
    for col in df.columns:
        if any(k in col for k in keywords) and not any(x in col for x in exclude):
            return col
    return None

def parse_hour(series):
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().mean() > 0.5 and num.median() > 1e11:
        dt = pd.to_datetime(num, unit="ms", utc=True).dt.tz_convert("America/Bogota")
        return dt.dt.hour.astype(float)
    txt = series.astype(str).str.strip()
    hr = txt.str.extract(r"(\d{1,2}):").iloc[:, 0]
    hr = hr.fillna(txt.str.extract(r"^(\d{1,2})(?:\.0)?$").iloc[:, 0])
    return pd.to_numeric(hr, errors="coerce")

def parse_date(series):
    try:
        num = pd.to_numeric(series, errors="coerce")
        if num.notna().mean() > 0.5 and num.median() > 1e11:
            d = pd.to_datetime(num, unit="ms", utc=True).dt.tz_convert("America/Bogota").dt.tz_localize(None)
        else:
            d = pd.to_datetime(series, errors="coerce")
            if getattr(d.dt, "tz", None) is not None:
                d = d.dt.tz_convert("America/Bogota").dt.tz_localize(None)
        return d.where(d.dt.year.between(2000, 2035))
    except Exception:
        return pd.Series(pd.NaT, index=series.index)

# ---------------------------------------------------------------------
# 2. DATA LINEAGE, ETL & QUALITY AUDIT METRICS
# ---------------------------------------------------------------------
DATA_SOURCES = [
    {"type": "curated_csv", "label": "GitHub Curated D2 Dataset", "url": "https://githubusercontent.com"},
    {"type": "curated_csv", "label": "GitHub Historical Backup", "url": "https://githubusercontent.com"},
    {"type": "soda_api", "label": "Bogotá Open Data SODA API", "url": "https://datos.gov.co", "params": {"$limit": SAMPLE_LIMIT}}
]

def fetch_raw_data():
    headers = {"User-Agent": "Mozilla/5.0"}
    for src in DATA_SOURCES:
        try:
            if src.get("type") == "curated_csv":
                r = requests.get(src.get("url"), headers=headers, timeout=10)
                if r.status_code == 200:
                    df = pd.read_csv(io.StringIO(r.text), low_memory=False)
                    if len(df) > 0:
                        return df, src.get("label"), "Observed (Official SDM Records)"
            elif src.get("type") == "soda_api":
                r = requests.get(src.get("url"), params=src.get("params"), headers=headers, timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    if isinstance(data, list) and len(data) > 0:
                        return pd.DataFrame(data), src.get("label"), "Observed (Official SDM Records)"
        except Exception:
            continue

    # Resilient empirical fallback
    rng = np.random.default_rng(42)
    hours = np.arange(24)
    weights = np.array((108, 90, 55, 62, 78, 215, 310, 415, 388, 405, 425, 440,
                        520, 555, 545, 540, 510, 475, 395, 420, 412, 275, 210, 142), dtype=float)
    weights /= weights.sum()
    names = [v.get("name") for v in BOGOTA_LOCALITIES.values()]
    dates = pd.date_range("2021-01-01", "2021-10-31", periods=SAMPLE_LIMIT)
    df_fb = pd.DataFrame({
        "FECHA_OCURRENCIA": dates,
        "HORA_NUMERICA": rng.choice(hours, SAMPLE_LIMIT, p=weights),
        "NOMBRE_LOCALIDAD": rng.choice(names, SAMPLE_LIMIT),
        "ACCIDENTE_CON": rng.choice(("SOLO DAÑOS", "CON HERIDOS", "CON MUERTOS"), SAMPLE_LIMIT, p=(0.61, 0.37, 0.02)),
        "CLASE_ACCIDENTE": rng.choice(("CHOQUE", "ATROPELLO", "VOLCAMIENTO", "CAIDA DE OCUPANTE"), SAMPLE_LIMIT),
        "LATITUD": rng.uniform(4.55, 4.72, SAMPLE_LIMIT),
        "LONGITUD": rng.uniform(-74.18, -74.03, SAMPLE_LIMIT)
    })
    return df_fb, "Empirical Fallback Distribution", "Synthetic (Empirical Distribution)"

def clean_and_audit(df_raw, source_label, provenance_type):
    df = df_raw.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]
    raw_count = len(df)
    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates()

    if "HORA_NUMERICA" in df.columns:
        hour = pd.to_numeric(df["HORA_NUMERICA"], errors="coerce")
    else:
        candidates = [c for c in df.columns if "HORA" in c]
        hour = None
        for c in candidates:
            h = parse_hour(df[c])
            if h.between(0, 23).mean() > 0.5:
                hour = h
                break
        if hour is None:
            hour = pd.Series(12, index=df.index)

    valid_mask = hour.between(0, 23)
    invalid_hours = int((~valid_mask).sum())
    df, hour = df[valid_mask], hour[valid_mask].astype(int)

    out = pd.DataFrame(index=df.index)
    out["HORA_NUMERICA"] = hour

    for c in [c for c in df.columns if "FECHA" in c]:
        f = parse_date(df[c])
        if f.notna().mean() > 0.5:
            out["FECHA"] = f
            break
    if "FECHA" not in out.columns or out["FECHA"].isna().all():
        out["FECHA"] = pd.date_range("2021-01-01", "2021-10-31", periods=len(out))

    days_en = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
    out["DAY_OF_WEEK"] = out["FECHA"].dt.dayofweek.map(days_en)

    col_loc = find_column(df, ("LOCALIDAD",), exclude=("COD", "ID_"))
    if col_loc is None:
        out["LOCALITY"] = "Unknown"
    else:
        vals = df[col_loc]
        nums = pd.to_numeric(vals, errors="coerce")
        if nums.notna().mean() > 0.5:
            names = nums.map(lambda x: CODE_TO_NAME.get(int(x)) if pd.notna(x) else None)
        else:
            names = vals.map(lambda x: NORM_TO_NAME.get(normalize_text(x)))
        out["LOCALITY"] = names.fillna("Unknown")

    c_lat = out["LOCALITY"].map(lambda n: NAME_TO_LAT.get(n, 4.65))
    c_lon = out["LOCALITY"].map(lambda n: NAME_TO_LON.get(n, -74.10))
    out["latitude"], out["longitude"] = c_lat, c_lon

    col_lat = find_column(df, ("LATITUD", "LATITUDE")) or ("Y" if "Y" in df.columns else None)
    col_lon = find_column(df, ("LONGITUD", "LONGITUDE")) or ("X" if "X" in df.columns else None)
    imputed_coords = 0
    if col_lat and col_lon:
        lat = pd.to_numeric(df[col_lat].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        lon = pd.to_numeric(df[col_lon].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        geo_ok = lat.between(3.5, 5.0) & lon.between(-75.0, -73.5)
        imputed_coords = int((~geo_ok).sum())
        out["latitude"] = lat.where(geo_ok, c_lat)
        out["longitude"] = lon.where(geo_ok, c_lon)

    sev_map = {
        "SOLO DAÑOS": "Property Damage Only",
        "CON HERIDOS": "With Casualties (Injured)",
        "CON MUERTOS": "With Fatalities"
    }
    col_grav = find_column(df, ("GRAVEDAD", "ACCIDENTE_CON", "SEVERIDAD"))
    if col_grav:
        raw_sev = df[col_grav].astype(str).str.strip().str.upper()
        out["SEVERITY"] = raw_sev.map(lambda s: sev_map.get(s, s.title()))
    else:
        out["SEVERITY"] = "Unspecified"

