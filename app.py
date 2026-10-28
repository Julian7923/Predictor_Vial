# =====================================================================
# PREDICTOR INTELIGENTE DE RIESGO VIAL - BOGOTÁ
# Proyecto de aula integrador - INGENIA - UNICAFAM
#
# IDEA EN UNA FRASE:
#   Contamos los siniestros por hora, los convertimos en una curva R(t),
#   usamos derivadas para hallar la hora de mayor riesgo y avisamos
#   30 minutos antes para enviar unidades de emergencia.
#
# CÓMO EJECUTARLO (en la terminal de VS Code):
#   pip install streamlit pandas numpy sympy matplotlib requests openpyxl
#   streamlit run app.py
# =====================================================================

# ---------------------------------------------------------------------
# 1. LIBRERÍAS
# ---------------------------------------------------------------------
import io                          # trata los bytes de un archivo subido como si fuera un archivo
import os                          # verificación de existencia de archivos locales
import unicodedata                 # permite quitar tildes (Usaquén -> USAQUEN)
import numpy as np                 # cálculo numérico: ajuste del polinomio y arreglos
import pandas as pd                # manejo de tablas (DataFrame)
import requests                    # hace la consulta HTTP a la API del portal
import sympy as sp                 # cálculo simbólico: derivadas exactas
import matplotlib.pyplot as plt    # gráfica de la curva de riesgo
import streamlit as st             # convierte el código en una página web interactiva

# ---------------------------------------------------------------------
# 2. CONFIGURACIÓN DE LA PÁGINA (debe ser la primera instrucción de Streamlit)
# ---------------------------------------------------------------------
st.set_page_config(page_title="Predictor Vial Bogotá", layout="wide")  # título de pestaña y ancho completo
st.title("🚑 Predictor Inteligente de Riesgo Vial - Bogotá")           # título principal
st.caption("Anticipamos la hora de mayor riesgo para enviar unidades de emergencia antes del pico.")

# ---------------------------------------------------------------------
# 3. CONSTANTES DEL PROYECTO (se cambian aquí y afectan todo el programa)
# ---------------------------------------------------------------------
URLS_API = [
    "https://datosabiertos.bogota.gov.co/api/3/action/datastore_search",
    "https://datosabiertos.bogota.gov.co/en/api/3/action/datastore_search",
    "https://datosabiertos.bogota.gov.co/es/api/3/action/datastore_search",
]
URL_CSV_DIRECTO = ("https://datosabiertos.bogota.gov.co/dataset/8624f916-1db2-4c17-b669-19a19b35d1ca/"
                   "resource/f5862aaa-4e1c-463e-94d5-f04db8164360/download/historico_siniestros_bogota_d.c_-.csv")
ID_RECURSO = "f5862aaa-4e1c-463e-94d5-f04db8164360"   # identificador del CSV "Histórico Siniestros Bogotá D.C."
LIMITE_REGISTROS = 8000        # cuántos registros descargamos (coincide con la muestra del documento)
GRADO_POLINOMIO = 4            # grado del polinomio R(t) (cuártico, como en la metodología)
VENTANA_MINUTOS = 30           # minutos de anticipación para la alerta

# Nombres de las tres fuentes de datos que el usuario puede elegir
FUENTE_API = "API en línea / Dataset Automático"
FUENTE_ARCHIVO = "Subir archivo (Excel o CSV)"
FUENTE_SIM = "Datos simulados (solo demo)"


# ---------------------------------------------------------------------
# 4. FUNCIÓN AUXILIAR: quitar tildes y pasar a mayúsculas
# ---------------------------------------------------------------------
def normalizar(texto):
    """Convierte 'Usaquén ' en 'USAQUEN' para poder comparar nombres sin errores."""
    sin_tildes = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return sin_tildes.upper().strip()


# ---------------------------------------------------------------------
# 5. DICCIONARIO MAESTRO DE LOCALIDADES (código -> nombre y coordenadas del centro)
# ---------------------------------------------------------------------
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

CODIGO_A_NOMBRE = {cod: d["nombre"] for cod, d in DICCIONARIO_BOGOTA.items()}
NOMBRE_A_COORD = {d["nombre"]: d["coord"] for d in DICCIONARIO_BOGOTA.values()}
NORM_A_NOMBRE = {normalizar(d["nombre"]): d["nombre"] for d in DICCIONARIO_BOGOTA.values()}
NORM_A_NOMBRE.update({
    "RAFAEL URIBE": "Rafael Uribe Uribe",
    "CANDELARIA": "La Candelaria",
    "MARTIRES": "Los Mártires",
    "SANTAFE": "Santa Fe",
})


# ---------------------------------------------------------------------
# 6. FUNCIONES AUXILIARES DE COLUMNAS Y FORMATO
# ---------------------------------------------------------------------
def buscar_columna(df, palabras, excluir=()):
    """Devuelve el nombre de la primera columna que contenga alguna de las palabras."""
    for col in df.columns:
        if any(p in col for p in palabras) and not any(x in col for x in excluir):
            return col
    return None


def extraer_hora(serie):
    """Intenta sacar la hora (0-23) de una columna."""
    num = pd.to_numeric(serie, errors="coerce")
    if num.notna().mean() > 0.5 and num.median() > 1e11:
        fechas = pd.to_datetime(num, unit="ms", utc=True).dt.tz_convert("America/Bogota")
        return fechas.dt.hour.astype(float)
    texto = serie.astype(str).str.strip()
    hora = texto.str.extract(r"(\d{1,2}):")[0]
    hora = hora.fillna(texto.str.extract(r"^(\d{1,2})(?:\.0)?$")[0])
    return pd.to_numeric(hora, errors="coerce")


def extraer_fecha(serie):
    """Convierte una columna a fechas (texto, ISO o milisegundos)."""
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
# 7. LECTURA DE DATOS
# ---------------------------------------------------------------------
def consultar_ckan(url):
    """Consulta UNA ruta de la API CKAN y devuelve las últimas filas como tabla."""
    cabeceras = {"User-Agent": "Mozilla/5.0"}
    r1 = requests.get(url, params={"resource_id": ID_RECURSO, "limit": 1}, headers=cabeceras, timeout=20)
    r1.raise_for_status()
    total = r1.json()["result"]["total"]
    desde = max(0, total - LIMITE_REGISTROS)
    r2 = requests.get(url, params={"resource_id": ID_RECURSO, "limit": LIMITE_REGISTROS, "offset": desde},
                      headers=cabeceras, timeout=60)
    r2.raise_for_status()
    registros = r2.json()["result"]["records"]
    if not registros:
        raise ValueError("la API respondió sin registros")
    return pd.DataFrame(registros)


def descargar_api():
    """Prioriza archivos curados existentes antes de consultar endpoints externos."""
    # Prioridad 1: Archivo local curado en la carpeta de trabajo
    archivos_locales = [
        "UNICAFAM_DataAnalytics_D2_Data.csv",
        "data/UNICAFAM_DataAnalytics_D2_Data.csv",
        "UNICAFAM_DataAnalytics_Data.csv",
        "raw_sample_snapshot.csv",
        "historico_siniestros_bogota_d.c_-.csv"
    ]
    for ruta in archivos_locales:
        if os.path.exists(ruta):
            tabla = pd.read_csv(ruta, low_memory=False, encoding_errors="replace")
            if tabla.shape[1] == 1:
                tabla = pd.read_csv(ruta, sep=";", low_memory=False, encoding_errors="replace")
            return tabla.tail(LIMITE_REGISTROS)

    # Prioridad 2: Consulta a CKAN Datos Abiertos Bogotá
    errores = []
    for url in URLS_API:
        try:
            return consultar_ckan(url)
        except Exception as e:
            errores.append(f"API {url.split('.co')[1][:8]}...: {str(e)[:60]}")

    # Prioridad 3: Descarga directa del CSV oficial de Bogotá
    try:
        r = requests.get(URL_CSV_DIRECTO, headers={"User-Agent": "Mozilla/5.0"}, timeout=180)
        r.raise_for_status()
        tabla = pd.read_csv(io.BytesIO(r.content), low_memory=False, encoding_errors="replace")
        if tabla.shape[1] == 1:
            tabla = pd.read_csv(io.BytesIO(r.content), sep=";", low_memory=False, encoding_errors="replace")
        return tabla.tail(LIMITE_REGISTROS)
    except Exception as e:
        errores.append(f"CSV directo: {str(e)[:60]}")

    raise ValueError("No se pudo descargar de ninguna ruta. " + " | ".join(errores)