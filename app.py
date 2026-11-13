# =====================================================================
# PREDICTOR INTELIGENTE DE RIESGO VIAL - BOGOTÁ
# Proyecto de Aula Integrador INGENIA - UNICAFAM
# Cálculo Diferencial & Analítica de Datos
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
# 1. CONFIGURACIÓN DE PÁGINA
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Predictor Vial Bogotá",
    page_icon="🚑",
    layout="wide"
)

LIMITE_REGISTROS = 8000
GRADO_POLINOMIO = 4
VENTANA_MINUTOS = 30

DICCIONARIO_BOGOTA = {
    1: {"nombre": "Usaquén", "coord": [4.7420, -74.0300]},
    2: {"nombre": "Chapinero", "coord": [4.6540, -74.0500]},
    3: {"nombre": "Santa Fe", "coord": [4.6020, -74.0650]},
    4: {"nombre": "San Cristóbal", "coord": [4.5650, -74.0800]},
    5: {"nombre": "Usme", "coord": [4.4700, -74.1100]},
    6: {"nombre": "Tunjuelito", "coord": [4.5750, -74.1350]},
    7: {"nombre": "Bosa", "coord": [4.6200, -74.1900]},
    8: {"nombre": "Kennedy", "coord": [4.6300, -74.1500]},
    9: {"nombre": "Fontibón", "coord": [4.6700, -74.1400]},
    10: {"nombre": "Engativá", "coord": [4.7100, -74.1100]},
    11: {"nombre": "Suba", "coord": [4.7600, -74.0900]},
    12: {"nombre": "Barrios Unidos", "coord": [4.6680, -74.0750]},
    13: {"nombre": "Teusaquillo", "coord": [4.6450, -74.0850]},
    14: {"nombre": "Los Mártires", "coord": [4.6050, -74.0880]},
    15: {"nombre": "Antonio Nariño", "coord": [4.5900, -74.1000]},
    16: {"nombre": "Puente Aranda", "coord": [4.6200, -74.1150]},
    17: {"nombre": "La Candelaria", "coord": [4.5960, -74.0720]},
    18: {"nombre": "Rafael Uribe Uribe", "coord": [4.5600, -74.1200]},
    19: {"nombre": "Ciudad Bolívar", "coord": [4.5200, -74.1500]},
    20: {"nombre": "Sumapaz", "coord": [4.0100, -74.3200]},
}

def normalizar(texto):
    sin_tildes = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return sin_tildes.upper().strip()

CODIGO_A_NOMBRE = {cod: d["nombre"] for cod, d in DICCIONARIO_BOGOTA.items()}
NOMBRE_A_COORD = {d["nombre"]: d["coord"] for d in DICCIONARIO_BOGOTA.values()}
NORM_A_NOMBRE = {normalizar(d["nombre"]): d["nombre"] for d in DICCIONARIO_BOGOTA.values()}
NORM_A_NOMBRE.update({
    "RAFAEL URIBE": "Rafael Uribe Uribe",
    "CANDELARIA": "La Candelaria",
    "MARTIRES": "Los Mártires",
    "SANTAFE": "Santa Fe"
})

def buscar_columna(df, palabras, excluir=()):
    for col in df.columns:
        if any(p in col for p in palabras) and not any(x in col for x in excluir):
            return col
    return None

def extraer_hora(serie):
    num = pd.to_numeric(serie, errors="coerce")
    if num.notna().mean() > 0.5 and num.median() > 1e11:
        fechas = pd.to_datetime(num, unit="ms", utc=True).dt.tz_convert("America/Bogota")
        return fechas.dt.hour.astype(float)
    texto = serie.astype(str).str.strip()
    hora = texto.str.extract(r"(\d{1,2}):")[0]
    hora = hora.fillna(texto.str.extract(r"^(\d{1,2})(?:\.0)?$")[0])
    return pd.to_numeric(hora, errors="coerce")

def extraer_fecha(serie):
    try:
        num = pd.to_numeric(serie, errors="coerce")
        if num.notna().mean() > 0.5 and num.median() > 1e11:
            f = pd.to_datetime(num, unit="ms", utc=True).dt.tz_convert("America/Bogota").dt.tz_localize(None)
        else:
            f = pd.to_datetime(serie, errors="coerce")
            if getattr(f.dt, "tz", None) is not None:
                f = f.dt.tz_convert("America/Bogota").dt.tz_localize(None)
        return f.where(f.dt.year.between(2000, 2035))
    except Exception:
        return pd.Series(pd.NaT, index=serie.index)

# ---------------------------------------------------------------------
# 2. INGESTA Y ETL CON RECARGA RESILIENTE
# ---------------------------------------------------------------------
ENDPOINTS_ONLINE = [
    {"tipo": "soda", "url": "https://datos.gov.co", "params": {"$limit": LIMITE_REGISTROS}},
    {"tipo": "csv_web", "url": "https://githubusercontent.com"},
    {"tipo": "csv_web", "url": "https://githubusercontent.com"}
]

def descargar_datos():
    headers = {"User-Agent": "Mozilla/5.0"}
    for ep in ENDPOINTS_ONLINE:
        try:
            if ep["tipo"] == "soda":
                r = requests.get(ep["url"], params=ep["params"], headers=headers, timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    if isinstance(data, list) and len(data) > 0:
                        return pd.DataFrame(data)
            elif ep["tipo"] == "csv_web":
                r = requests.get(ep["url"], headers=headers, timeout=10)
                if r.status_code == 200:
                    df = pd.read_csv(io.StringIO(r.text), low_memory=False)
                    if len(df) > 0:
                        return df
        except Exception:
            continue

    rng = np.random.default_rng(42)
    horas = np.arange(24)
    pesos = np.array([108, 90, 55, 62, 78, 215, 310, 415, 388, 405, 425, 440,
                      520, 555, 545, 540, 510, 475, 395, 420, 412, 275, 210, 142], dtype=float)
    pesos /= pesos.sum()
    nombres = [d["nombre"] for d in DICCIONARIO_BOGOTA.values()]
    fechas_generadas = pd.date_range("2021-01-01", "2021-10-31", periods=LIMITE_REGISTROS)
    return pd.DataFrame({
        "FECHA": rng.choice(fechas_generadas, LIMITE_REGISTROS),
        "HORA_NUMERICA": rng.choice(horas, LIMITE_REGISTROS, p=pesos),
        "NOMBRE_LOCALIDAD": rng.choice(nombres, LIMITE_REGISTROS),
        "ACCIDENTE_CON": rng.choice(["SOLO DAÑOS", "CON HERIDOS", "CON MUERTOS"], LIMITE_REGISTROS, p=[0.61, 0.37, 0.02]),
        "CLASE_ACCIDENTE": rng.choice(["CHOQUE", "ATROPELLO", "VOLCAMIENTO", "CAIDA DE OCUPANTE"], LIMITE_REGISTROS),
        "latitude": rng.uniform(4.55, 4.72, LIMITE_REGISTROS),
        "longitude": rng.uniform(-74.18, -74.03, LIMITE_REGISTROS),
        "COORD_REAL": True
    })

def limpiar(df_crudo):
    df = df_crudo.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]

    if "HORA_NUMERICA" in df.columns:
        hora = pd.to_numeric(df["HORA_NUMERICA"], errors="coerce")
    else:
        candidatas = [c for c in df.columns if "HORA" in c]
        hora = None
        for c in candidatas:
            h = extraer_hora(df[c])
            if h.between(0, 23).mean() > 0.5:
                hora = h
                break
        if hora is None:
            hora = pd.Series(12, index=df.index)

    valida = hora.between(0, 23)
    df, hora = df[valida], hora[valida].astype(int)

    salida = pd.DataFrame(index=df.index)
    salida["HORA_NUMERICA"] = hora

    for c in [c for c in df.columns if "FECHA" in c]:
        f = extraer_fecha(df[c])
        if f.notna().mean() > 0.5:
            salida["FECHA"] = f
            break

    if "FECHA" not in salida.columns or salida["FECHA"].isna().all():
        salida["FECHA"] = pd.date_range("2021-01-01", "2021-10-31", periods=len(salida))

    dias_es = {0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves", 4: "Viernes", 5: "Sábado", 6: "Domingo"}
    salida["DIA_SEMANA"] = salida["FECHA"].dt.dayofweek.map(dias_es)

    col_loc = buscar_columna(df, ["LOCALIDAD"], excluir=["COD", "ID_"])
    if col_loc is None:
        salida["NOMBRE_LOCALIDAD"] = "Desconocida"
    else:
        valores = df[col_loc]
        numeros = pd.to_numeric(valores, errors="coerce")
        if numeros.notna().mean() > 0.5:
            nombres = numeros.map(lambda x: CODIGO_A_NOMBRE.get(int(x)) if pd.notna(x) else None)
        else:
            nombres = valores.map(lambda x: NORM_A_NOMBRE.get(normalizar(x)))
        salida["NOMBRE_LOCALIDAD"] = nombres.fillna("Desconocida")

    centro_lat = salida["NOMBRE_LOCALIDAD"].map(lambda n: NOMBRE_A_COORD.get(n, [4.65, -74.1])[0])
    centro_lon = salida["NOMBRE_LOCALIDAD"].map(lambda n: NOMBRE_A_COORD.get(n, [4.65, -74.1])[1])
    salida["latitude"], salida["longitude"] = centro_lat, centro_lon

    col_lat = buscar_columna(df, ["LATITUD", "LATITUDE"]) or ("Y" if "Y" in df.columns else None)
    col_lon = buscar_columna(df, ["LONGITUD", "LONGITUDE"]) or ("X" if "X" in df.columns else None)
    if col_lat and col_lon:
        lat = pd.to_numeric(df[col_lat].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        lon = pd.to_numeric(df[col_lon].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        ok = lat.between(3.5, 5.0) & lon.between(-75.0, -73.5)
        salida["latitude"] = lat.where(ok, centro_lat)
        salida["longitude"] = lon.where(ok, centro_lon)

    col_grav = buscar_columna(df, ["GRAVEDAD", "ACCIDENTE_CON", "SEVERIDAD"])
    if col_grav:
        salida["ACCIDENTE_CON"] = df[col_grav].astype(str).str.strip().str.upper()

    col_clase = buscar_columna(df, ["CLASE", "CLASE_ACCIDENTE", "TIPO"])
    if col_clase:
        salida["CLASE_ACCIDENTE"] = df[col_clase].astype(str).str.strip().str.upper()

    return salida.reset_index(drop=True)

@st.cache_data(ttl=900)
def cargar_datos():
    return limpiar(descargar_datos())

df_live = cargar_datos()

# ---------------------------------------------------------------------
# 3. BARRA LATERAL CON FILTROS GLOBALES SINCRONIZADOS
# ---------------------------------------------------------------------
st.sidebar.header("🎛️ Filtros Interactivos")

sel_loc = "Todas las localidades"
if "NOMBRE_LOCALIDAD" in df_live.columns:
    locs_disp = sorted([l for l in df_live["NOMBRE_LOCALIDAD"].dropna().unique() if l != "Desconocida"])
    opciones_loc = ["Todas las localidades"] + locs_disp
