# =====================================================================
# PREDICTOR INTELIGENTE DE RIESGO VIAL - BOGOTÁ
# Proyecto de aula integrador - INGENIA - UNICAFAM
#
# IDEA EN UNA FRASE:
#   Contamos los siniestros por hora, los convertimos en una curva R(t),
#   usamos derivadas para hallar la hora de mayor riesgo y avisamos
#   30 minutos antes para enviar unidades de emergencia.
# =====================================================================

# ---------------------------------------------------------------------
# 1. LIBRERÍAS
# ---------------------------------------------------------------------
import unicodedata
import numpy as np
import pandas as pd
import requests
import sympy as sp
import matplotlib.pyplot as plt
import streamlit as st

# ---------------------------------------------------------------------
# 2. CONFIGURACIÓN DE LA PÁGINA
# ---------------------------------------------------------------------
st.set_page_config(page_title="Predictor Vial Bogotá", layout="wide")
st.title("🚑 Predictor Inteligente de Riesgo Vial - Bogotá")
st.caption("Anticipamos la hora de mayor riesgo para enviar unidades de emergencia antes del pico.")

# ---------------------------------------------------------------------
# 3. CONSTANTES Y ENDPOINTS OFICIALES EN LÍNEA
# ---------------------------------------------------------------------
# Endpoints vivos de la Secretaría Distrital de Movilidad y el portal nacional
ENDPOINTS_ONLINE = [
    # 1. API ArcGIS REST de la Secretaría Distrital de Movilidad de Bogotá (Siniestros Georreferenciados)
    {
        "tipo": "arcgis",
        "url": "https://datos.movilidadbogota.gov.co/server/rest/services/Siniestralidad/FeatureServer/0/query",
        "params": {
            "where": "1=1",
            "outFields": "*",
            "resultRecordCount": 8000,
            "f": "json",
            "returnGeometry": "true"
        }
    },
    # 2. API SODA de datos.gov.co (Siniestros consolidados Bogotá)
    {
        "tipo": "soda",
        "url": "https://www.datos.gov.co/resource/vftc-p83w.json",
        "params": {
            "$limit": 8000
        }
    },
    # 3. API CKAN Datastore Bogotá (Consulta general por paquete)
    {
        "tipo": "ckan",
        "url": "https://datosabiertos.bogota.gov.co/api/3/action/datastore_search",
        "params": {
            "resource_id": "54970af1-57ac-486e-8bf6-52092b9fcbd0",
            "limit": 8000
        }
    }
]

LIMITE_REGISTROS = 8000
GRADO_POLINOMIO = 4
VENTANA_MINUTOS = 30

# ---------------------------------------------------------------------
# 4. FUNCIONES AUXILIARES DE NORMALIZACIÓN
# ---------------------------------------------------------------------
def normalizar(texto):
    """Convierte 'Usaquén ' en 'USAQUEN' para homologar texto."""
    sin_tildes = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return sin_tildes.upper().strip()

# ---------------------------------------------------------------------
# 5. DICCIONARIO MAESTRO DE LOCALIDADES DE BOGOTÁ
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
# 6. PARSEO DE CAMPOS
# ---------------------------------------------------------------------
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
# 7. EXTRACCIÓN 100% VÍA API (SIN ARCHIVOS)
# ---------------------------------------------------------------------
def consultar_servicio_en_linea():
    """Consulta secuencialmente las APIs distritales y nacionales disponibles."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    errores = []

    for ep in ENDPOINTS_ONLINE:
        try:
            r = requests.get(ep["url"], params=ep["params"], headers=headers, timeout=25)
            if r.status_code != 200:
                continue

            data = r.json()

            # Caso A: API ArcGIS FeatureServer
            if ep["tipo"] == "arcgis" and "features" in data and len(data["features"]) > 0:
                filas = []
                for f in data["features"]:
                    attrs = f.get("attributes", {})
                    geom = f.get("geometry", {})
                    if "x" in geom and "y" in geom:
                        attrs["LONGITUDE"] = geom["x"]
                        attrs["LATITUDE"] = geom["y"]
                    filas.append(attrs)
                return pd.DataFrame(filas)

            # Caso B: API SODA (datos.gov.co)
            elif ep["tipo"] == "soda" and isinstance(data, list) and len(data) > 0:
                return pd.DataFrame(data)

            # Caso C: API CKAN Datastore
            elif ep["tipo"] == "ckan" and "result" in data and "records" in data["result"]:
                records = data["result"]["records"]
                if len(records) > 0:
                    return pd.DataFrame(records)

        except Exception as e:
            errores.append(str(e)[:50])

    raise RuntimeError("No se pudo obtener respuesta de las APIs en línea de Bogotá: " + " | ".join(errores))

# ---------------------------------------------------------------------
# 8. TRANSFORMACIÓN Y LIMPIEZA (ETL)
# ---------------------------------------------------------------------
def limpiar(df_crudo):
    df = df_crudo.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]

    candidatas = ([c for c in df.columns if "HORA" in c and "FECHA" not in c]
                  + [c for c in df.columns if "HORA" in c and "FECHA" in c]
                  + [c for c in df.columns if "FECHA" in c and "HORA" not in c])
    hora = None
    for c in candidatas:
        h = extraer_hora(df[c])
        if h.between(0, 23).mean() > 0.5 and (h == 0).mean() < 0.9:
            hora = h
            break

    if hora is None:
        raise ValueError("La API no entregó un campo horario utilizable. Columnas: " + ", ".join(df.columns[:10]))

    valida = hora.between(0, 23)
    df, hora = df[valida], hora[valida].astype(int)

    salida = pd.DataFrame(index=df.index)
    salida["HORA_NUMERICA"] = hora

    for c in [c for c in df.columns if "FECHA" in c]:
        f = extraer_fecha(df[c])
        if f.notna().mean() > 0.5:
            salida["FECHA"] = f
            break

    col_loc = (buscar_columna(df, ["LOCALIDAD"], excluir=["COD", "ID_"])
               or buscar_columna(df, ["COD_LOC", "CODIGO_LOCALIDAD", "ID_LOC"]))
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

    centro_lat = salida["NOMBRE_LOCALIDAD"].map(lambda n: NOMBRE_A_COORD.get(n, [np.nan, np.nan])[0])
    centro_lon = salida["NOMBRE_LOCALIDAD"].map(lambda n: NOMBRE_A_COORD.get(n, [np.nan, np.nan])[1])
    salida["latitude"], salida["longitude"] = centro_lat, centro_lon
    salida["COORD_REAL"] = False

    col_lat = buscar_columna(df, ["LATITUD", "LATITUDE"]) or ("Y" if "Y" in df.columns else None)
    col_lon = buscar_columna(df, ["LONGITUD", "LONGITUDE"]) or ("X" if "X" in df.columns else None)
    if col_lat and col_lon:
        lat = pd.to_numeric(df[col_lat].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        lon = pd.to_numeric(df[col_lon].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        ok = lat.between(3.5, 5.0) & lon.between(-75.0, -73.5)
        salida["latitude"] = lat.where(ok, centro_lat)
        salida["longitude"] = lon.where(ok, centro_lon)
        salida["COORD_REAL"] = ok

    equivalencias = [
        ("ACCIDENTE_CON", ["GRAVEDAD"]),
        ("CLASE_ACCIDENTE", ["CLASE"]),
        ("SEXO", ["SEXO", "GENERO"]),
        ("CONDICION", ["CONDICION"]),
    ]
    for destino, palabras in equivalencias:
        col = buscar_columna(df, palabras)
        if col is not None:
            serie = df[col].astype(str).str.strip().str.upper()
            salida[destino] = serie.replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})

    return salida.reset_index(drop=True)

# ---------------------------------------------------------------------
# 9. CARGA CON CACHÉ DE MEMORIA (15 MINUTOS)
# ---------------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner="Consultando API oficial de Bogotá en tiempo real...")
def obtener_datos_en_linea():
    crudo = consultar_servicio_en_linea()
    return limpiar(crudo)

# ---------------------------------------------------------------------
# 10. BARRA LATERAL
# ---------------------------------------------------------------------
st.sidebar.header("📡 Conexión API")
st.sidebar.success("🟢 Modo en línea activo")

if st.sidebar.button("🔄 Actualizar desde API"):
    st.cache_data.clear()
    st.rerun()

# ---------------------------------------------------------------------
# 11. INGESTA EN VIVO
# ---------------------------------------------------------------------
df_live = obtener_datos_en_linea()
pct_coord = df_live["COORD_REAL"].mean() * 100

st.caption(f"✅ Conectado a la API pública oficial de Movilidad Bogotá · "
           f"{len(df_live):,} eventos viales descargados · {pct_coord:.0f}% con ubicación GPS precisa.")

# ---------------------------------------------------------------------
# 12. FILTROS DINÁMICOS
# ---------------------------------------------------------------------
st.sidebar.write("---")
st.sidebar.header("🎛️ Filtros de análisis")
etiquetas = {
    "ACCIDENTE_CON": "Gravedad del siniestro:",
    "CLASE_ACCIDENTE": "Clase de siniestro:",
    "SEXO": "Género del implicado:",
}
df_filtrado = df_live.copy()
for columna, etiqueta in etiquetas.items():
    if columna in df_live.columns:
        opciones = ["Todos"] + sorted(df_live[columna].dropna().unique())
        elegido = st.sidebar.selectbox(etiqueta, opciones)
        if elegido != "Todos":
            df_filtrado = df_filtrado[df_filtrado[columna] == elegido]

hora_mapa = st.sidebar.slider("Hora a visualizar en el mapa:", 0, 23, 14)

# ---------------------------------------------------------------------
# 13. AGREGACIÓN HORARIA
# ---------------------------------------------------------------------
conteo = df_filtrado.groupby("HORA_NUMERICA").size().reindex(range(24), fill_value=0)
X = conteo.index.to_numpy()
Y = conteo.to_numpy()

if (Y > 0).sum() < 5:
    st.warning("Pocos datos para los filtros seleccionados. Amplía la selección.")
    st.stop()

# ---------------------------------------------------------------------
# 14. REGRESIÓN POLINÓMICA CUÁRTICA R(t)
# ---------------------------------------------------------------------
coeficientes = np.polyfit(X, Y, GRADO_POLINOMIO)
f_riesgo = np.poly1d(coeficientes)

ss_res = np.sum((Y - f_riesgo(X)) ** 2)
ss_tot = np.sum((Y - Y.mean()) ** 2)
r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

# ---------------------------------------------------------------------
# 15. DERIVACIÓN SIMBÓLICA CON SYMPY (MÁXIMOS Y ALERTA TEMPRANA)
# ---------------------------------------------------------------------
t = sp.Symbol("t")
R = sum(sp.Float(float(c)) * t ** (GRADO_POLINOMIO - i) for i, c in enumerate(coeficientes))
R1 = sp.diff(R, t)
R2 = sp.diff(R, t, 2)

raices = [complex(r) for r in sp.Poly(R1, t).nroots()]
reales = [r.real for r in raices if abs(r.imag) < 1e-6 and 0 <= r.real <= 23]
maximos = [r for r in reales if float(R2.subs(t, r)) < 0]

candidatos = maximos + [0.0, 23.0]
hora_critica = max(candidatos, key=lambda x: float(f_riesgo(x)))
min_pico = round(hora_critica * 60)
h_pico, m_pico = divmod(min_pico, 60)

min_alerta = (min_pico - VENTANA_MINUTOS) % 1440
h_alerta, m_alerta = divmod(min_alerta, 60)
hora_observada = int(conteo.idxmax())

# ---------------------------------------------------------------------
# 16. INDICADORES
# ---------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("🕰️ Hora pico (Modelo)", f"{h_pico:02d}:{m_pico:02d}")
m2.metric(f"🚑 Hora de alerta ({VENTANA_MINUTOS} min antes)", f"{h_alerta:02d}:{m_alerta:02d}")
m3.metric("📐 Ajuste de curva (r²)", f"{r2:.2f}")
m4.metric("📊 Hora con más eventos (Datos)", f"{hora_observada:02d}:00")

st.error(f"🚨 **ALERTA PREVENTIVA:** Enviar unidades de rescate a las **{h_alerta:02d}:{m_alerta:02d} h**.")

# ---------------------------------------------------------------------
# 17. GRÁFICA Y GEORREFERENCIACIÓN
# ---------------------------------------------------------------------
col_izq, col_der = st.columns(2)

with col_izq:
    st.write("### 📈 Curva de riesgo R(t)")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.scatter(X, Y, color="black", label="Eventos observados por hora")
    x_curva = np.linspace(0, 23, 500)
    ax.plot(x_curva, f_riesgo(x_curva), color="#0077B6", linewidth=2.5, label="R(t) Ajustada")
    ax.axvline(x=hora_critica, color="darkred", linestyle="--", label="Pico crítico")
    ax.axvline(x=min_alerta / 60, color="orange", linestyle=":", label="Momento de alerta")
    ax.set_xticks(range(0, 24, 2))
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Hora del día (t)")
    ax.set_ylabel("Cantidad de siniestros")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

with col_der:
    st.write(f"### 🗺️ Concentración a las {hora_mapa:02d}:00 h")
    datos_hora = df_filtrado[df_filtrado["HORA_NUMERICA"] == hora_mapa].dropna(subset=["latitude", "longitude"])
    if datos_hora.empty:
        st.warning("Sin eventos registrados en esta franja.")
    else:
        mapa_df = (datos_hora.assign(latitude=datos_hora["latitude"].round(3),
                                     longitude=datos_hora["longitude"].round(3))
                   .groupby(["latitude", "longitude"]).size().reset_index(name="count"))
        mapa_df["point_size"] = 150 + 1200 * mapa_df["count"] / mapa_df["count"].max()
        st.map(mapa_df, size="point_size", color="#D90429AA", zoom=10.1)

        ranking = (datos_hora[datos_hora["NOMBRE_LOCALIDAD"] != "Desconocida"]
                   ["NOMBRE_LOCALIDAD"].value_counts().head(3))
        if not ranking.empty:
            lineas = [f"{i + 1}. **{n}**: {c} eventos" for i, (n, c) in enumerate(ranking.items())]
            st.success("🚑 **Despacho territorial prioritario:**\n\n" + "\n\n".join(lineas))

# ---------------------------------------------------------------------
# 18. PIE DE PÁGINA
# ---------------------------------------------------------------------
st.write("---")
st.caption("Conexión en línea directa con el Portal de Datos de la Secretaría Distrital de Movilidad de Bogotá.")