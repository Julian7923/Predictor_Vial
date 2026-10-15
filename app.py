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
#   streamlit run predictor_vial.py
# =====================================================================

# ---------------------------------------------------------------------
# 1. LIBRERÍAS
# ---------------------------------------------------------------------
import io                          # trata los bytes de un archivo subido como si fuera un archivo
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
# API del portal Datos Abiertos Bogotá (CKAN). El portal responde 404 en algunas rutas,
# así que probamos varias en orden hasta que una funcione.
URLS_API = [
    "https://datosabiertos.bogota.gov.co/api/3/action/datastore_search",
    "https://datosabiertos.bogota.gov.co/en/api/3/action/datastore_search",
    "https://datosabiertos.bogota.gov.co/es/api/3/action/datastore_search",
]
# Plan B: descarga directa del archivo CSV completo (pesa ~38 MB; luego nos quedamos con las últimas filas)
URL_CSV_DIRECTO = ("https://datosabiertos.bogota.gov.co/dataset/8624f916-1db2-4c17-b669-19a19b35d1ca/"
                   "resource/f5862aaa-4e1c-463e-94d5-f04db8164360/download/historico_siniestros_bogota_d.c_-.csv")
ID_RECURSO = "f5862aaa-4e1c-463e-94d5-f04db8164360"   # identificador del CSV "Histórico Siniestros Bogotá D.C."
LIMITE_REGISTROS = 8000        # cuántos registros descargamos (coincide con la muestra del documento)
GRADO_POLINOMIO = 4            # grado del polinomio R(t) (cuártico, como en la metodología)
VENTANA_MINUTOS = 30           # minutos de anticipación para la alerta

# Nombres de las tres fuentes de datos que el usuario puede elegir
FUENTE_API = "API en línea (Datos Abiertos Bogotá)"
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
#    Sirve para traducir códigos a nombres y, si faltan coordenadas reales, ubicar en el mapa
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

# Tres "diccionarios derivados" para buscar rápido según el dato que tengamos
CODIGO_A_NOMBRE = {cod: d["nombre"] for cod, d in DICCIONARIO_BOGOTA.items()}      # 8 -> "Kennedy"
NOMBRE_A_COORD = {d["nombre"]: d["coord"] for d in DICCIONARIO_BOGOTA.values()}   # "Kennedy" -> [lat, lon]
NORM_A_NOMBRE = {normalizar(d["nombre"]): d["nombre"] for d in DICCIONARIO_BOGOTA.values()}  # "KENNEDY" -> "Kennedy"
# Variantes con las que algunas bases escriben ciertas localidades
NORM_A_NOMBRE.update({
    "RAFAEL URIBE": "Rafael Uribe Uribe",
    "CANDELARIA": "La Candelaria",
    "MARTIRES": "Los Mártires",
    "SANTAFE": "Santa Fe",
})


# ---------------------------------------------------------------------
# 6. FUNCIÓN AUXILIAR: encontrar una columna por palabras clave
#    (los portales llaman distinto a la misma columna: HORA, HORA_ACC, Hora_Acc...)
# ---------------------------------------------------------------------
def buscar_columna(df, palabras, excluir=()):
    """Devuelve el nombre de la primera columna que contenga alguna de las palabras."""
    for col in df.columns:
        if any(p in col for p in palabras) and not any(x in col for x in excluir):
            return col
    return None  # si no encuentra ninguna, devuelve None


def extraer_hora(serie):
    """Intenta sacar la hora (0-23) de una columna, sin importar si viene como hora o como fecha con hora."""
    num = pd.to_numeric(serie, errors="coerce")                           # ¿la columna es numérica?
    if num.notna().mean() > 0.5 and num.median() > 1e11:                  # números gigantes = fecha en milisegundos (formato ArcGIS)
        fechas = pd.to_datetime(num, unit="ms", utc=True).dt.tz_convert("America/Bogota")  # pasamos a hora de Bogotá
        return fechas.dt.hour.astype(float)
    texto = serie.astype(str).str.strip()                                 # todo a texto para analizarlo igual
    hora = texto.str.extract(r"(\d{1,2}):")[0]                            # "14:30" o "2018-05-04 14:30:00" -> toma "14"
    hora = hora.fillna(texto.str.extract(r"^(\d{1,2})(?:\.0)?$")[0])      # "14" o "14.0" -> toma "14"
    return pd.to_numeric(hora, errors="coerce")                           # texto -> número (lo ilegible queda NaN)


def extraer_fecha(serie):
    """Convierte una columna a fechas (texto, ISO o milisegundos). Lo que no se pueda leer queda vacío (NaT)."""
    try:
        num = pd.to_numeric(serie, errors="coerce")                       # ¿la columna es numérica?
        if num.notna().mean() > 0.5 and num.median() > 1e11:              # números gigantes = fecha en milisegundos (formato ArcGIS)
            f = pd.to_datetime(num, unit="ms", utc=True).dt.tz_convert("America/Bogota").dt.tz_localize(None)
        else:
            f = pd.to_datetime(serie, errors="coerce")                    # texto -> fecha
            if getattr(f.dt, "tz", None) is not None:                     # si trae zona horaria, la pasamos a Bogotá y la quitamos
                f = f.dt.tz_convert("America/Bogota").dt.tz_localize(None)
        return f.where(f.dt.year.between(2000, 2035))                     # descartamos fechas absurdas (ej. 1900 o 2099)
    except Exception:
        return pd.Series(pd.NaT, index=serie.index)                       # si algo falla, devolvemos fechas vacías


# ---------------------------------------------------------------------
# 7. LECTURA DE DATOS: tres fuentes posibles
# ---------------------------------------------------------------------
def consultar_ckan(url):
    """Consulta UNA ruta de la API CKAN y devuelve las últimas filas como tabla."""
    cabeceras = {"User-Agent": "Mozilla/5.0"}  # algunos portales rechazan consultas sin identificarse
    # Primera consulta liviana: solo para saber cuántos registros existen en total
    r1 = requests.get(url, params={"resource_id": ID_RECURSO, "limit": 1}, headers=cabeceras, timeout=20)
    r1.raise_for_status()                                   # lanza error si el portal responde mal (404, 500...)
    total = r1.json()["result"]["total"]                    # número total de filas del recurso
    # Segunda consulta: pedimos las ÚLTIMAS filas (normalmente las más recientes)
    desde = max(0, total - LIMITE_REGISTROS)                # posición desde la que empezamos a leer
    r2 = requests.get(url, params={"resource_id": ID_RECURSO, "limit": LIMITE_REGISTROS, "offset": desde},
                      headers=cabeceras, timeout=60)
    r2.raise_for_status()
    registros = r2.json()["result"]["records"]              # lista de diccionarios (una fila cada uno)
    if not registros:
        raise ValueError("la API respondió sin registros")
    return pd.DataFrame(registros)                          # lista de filas -> tabla


def descargar_api():
    """E (Extracción) del ETL: prueba cada ruta de la API y, si todas fallan, descarga el CSV directo."""
    errores = []                                            # aquí anotamos por qué falló cada intento
    for url in URLS_API:
        try:
            return consultar_ckan(url)                      # si funciona, terminamos aquí
        except Exception as e:
            errores.append(f"API {url.split('.co')[1][:8]}...: {str(e)[:60]}")
    try:                                                    # Plan B: archivo CSV completo
        r = requests.get(URL_CSV_DIRECTO, headers={"User-Agent": "Mozilla/5.0"}, timeout=180)
        r.raise_for_status()
        tabla = pd.read_csv(io.BytesIO(r.content), low_memory=False, encoding_errors="replace")
        if tabla.shape[1] == 1:                             # si quedó en una columna, el separador era ";"
            tabla = pd.read_csv(io.BytesIO(r.content), sep=";", low_memory=False, encoding_errors="replace")
        return tabla.tail(LIMITE_REGISTROS)                 # nos quedamos con las últimas filas
    except Exception as e:
        errores.append(f"CSV directo: {str(e)[:60]}")
    raise ValueError("No se pudo descargar de ninguna ruta. " + " | ".join(errores)
                     + " -> usa la opción 'Subir archivo'.")


def leer_archivo(contenido, nombre):
    """Lee un archivo que el usuario subió (Excel o CSV)."""
    if nombre.lower().endswith(".xlsx"):
        hojas = pd.read_excel(io.BytesIO(contenido), sheet_name=None)   # lee TODAS las hojas en un diccionario
        for hoja in hojas.values():
            if any("HORA" in str(c).upper() for c in hoja.columns):     # nos quedamos con la hoja que trae la hora
                return hoja
        raise ValueError("Ninguna hoja del Excel tiene una columna de hora.")
    # Si es CSV: primero con coma; si queda en una sola columna, probamos con punto y coma
    tabla = pd.read_csv(io.BytesIO(contenido), low_memory=False, encoding_errors="replace")
    if tabla.shape[1] == 1:
        tabla = pd.read_csv(io.BytesIO(contenido), sep=";", low_memory=False, encoding_errors="replace")
    return tabla


def generar_simulado(n=3000):
    """Datos inventados con picos de mañana y tarde. SOLO para probar la interfaz."""
    rng = np.random.default_rng(42)                                       # semilla fija: siempre salen los mismos datos
    horas = np.arange(24)                                                 # 0, 1, ..., 23
    pesos = 1 + 6 * np.exp(-((horas - 7.5) ** 2) / 4) + 8 * np.exp(-((horas - 17.5) ** 2) / 5)  # dos "montañas": ~7:30 y ~17:30
    pesos = pesos / pesos.sum()                                           # convertimos pesos en probabilidades (suman 1)
    nombres = [d["nombre"] for d in DICCIONARIO_BOGOTA.values()]
    return pd.DataFrame({
        "HORA": rng.choice(horas, n, p=pesos),
        "LOCALIDAD": rng.choice(nombres, n),
        "GRAVEDAD": rng.choice(["SOLO DAÑOS", "CON HERIDOS", "CON MUERTOS"], n, p=[0.70, 0.27, 0.03]),
        "CLASE": rng.choice(["CHOQUE", "ATROPELLO", "VOLCAMIENTO", "CAIDA DE OCUPANTE"], n),
        "SEXO": rng.choice(["MASCULINO", "FEMENINO"], n),
        "CONDICION": rng.choice(["CONDUCTOR", "PEATON", "MOTOCICLISTA", "PASAJERO"], n),
    })


# ---------------------------------------------------------------------
# 8. LIMPIEZA (la T de ETL): deja una tabla estándar sin importar la fuente
# ---------------------------------------------------------------------
def limpiar(df_crudo):
    df = df_crudo.copy()                                                  # trabajamos sobre una copia, no sobre el original
    df.columns = [str(c).strip().upper() for c in df.columns]             # nombres de columna: sin espacios y en MAYÚSCULAS

    # --- 8.1 Hora de ocurrencia (variable independiente t) ---
    # Probamos columnas en este orden: 1) HORA sola  2) FECHA_HORA  3) cualquier FECHA que traiga la hora
    candidatas = ([c for c in df.columns if "HORA" in c and "FECHA" not in c]
                  + [c for c in df.columns if "HORA" in c and "FECHA" in c]
                  + [c for c in df.columns if "FECHA" in c and "HORA" not in c])
    hora = None
    for c in candidatas:
        h = extraer_hora(df[c])
        # Aceptamos la columna si la mayoría de valores dan una hora válida y NO son casi todos 00:00
        # (una fecha sin hora se vería como medianoche en todas las filas y falsearía el análisis)
        if h.between(0, 23).mean() > 0.5 and (h == 0).mean() < 0.9:
            hora = h
            break
    if hora is None:
        raise ValueError("La base no trae una hora utilizable (no hay columna de hora o solo trae la fecha). "
                         "Columnas disponibles: " + ", ".join(df.columns))
    valida = hora.between(0, 23)                                          # solo horas entre 0 y 23 (NaN da False)
    df, hora = df[valida], hora[valida].astype(int)                       # descartamos filas inválidas (NO las convertimos en 12)

    salida = pd.DataFrame(index=df.index)                                 # tabla limpia final, vacía por ahora
    salida["HORA_NUMERICA"] = hora

    # --- 8.1b Fecha del siniestro (solo para informar de cuándo a cuándo son los datos) ---
    for c in [c for c in df.columns if "FECHA" in c]:                     # probamos cada columna que tenga FECHA en el nombre
        f = extraer_fecha(df[c])
        if f.notna().mean() > 0.5:                                        # la aceptamos si la mayoría de valores son fechas válidas
            salida["FECHA"] = f
            break

    # --- 8.2 Localidad (nombre) ---
    col_loc = (buscar_columna(df, ["LOCALIDAD"], excluir=["COD", "ID_"])  # primero buscamos la columna con el NOMBRE
               or buscar_columna(df, ["COD_LOC", "CODIGO_LOCALIDAD", "ID_LOC"]))  # si no hay, usamos la del CÓDIGO
    if col_loc is None:
        salida["NOMBRE_LOCALIDAD"] = "Desconocida"
    else:
        valores = df[col_loc]
        numeros = pd.to_numeric(valores, errors="coerce")                 # intentamos leerla como número
        if numeros.notna().mean() > 0.5:                                  # si más de la mitad son números, son códigos 1-20
            nombres = numeros.map(lambda x: CODIGO_A_NOMBRE.get(int(x)) if pd.notna(x) else None)
        else:                                                             # si no, son nombres: los normalizamos y buscamos
            nombres = valores.map(lambda x: NORM_A_NOMBRE.get(normalizar(x)))
        salida["NOMBRE_LOCALIDAD"] = nombres.fillna("Desconocida")        # lo que no se reconoce queda "Desconocida"

    # --- 8.3 Coordenadas: reales si existen, si no el centro de la localidad ---
    centro_lat = salida["NOMBRE_LOCALIDAD"].map(lambda n: NOMBRE_A_COORD.get(n, [np.nan, np.nan])[0])
    centro_lon = salida["NOMBRE_LOCALIDAD"].map(lambda n: NOMBRE_A_COORD.get(n, [np.nan, np.nan])[1])
    salida["latitude"], salida["longitude"] = centro_lat, centro_lon      # por defecto: centro de la localidad
    salida["COORD_REAL"] = False                                          # marca para saber qué filas tienen coordenada real
    col_lat = buscar_columna(df, ["LATITUD", "LATITUDE"]) or ("Y" if "Y" in df.columns else None)   # algunas bases llaman Y a la latitud
    col_lon = buscar_columna(df, ["LONGITUD", "LONGITUDE"]) or ("X" if "X" in df.columns else None)  # y X a la longitud
    if col_lat and col_lon:
        lat = pd.to_numeric(df[col_lat].astype(str).str.replace(",", ".", regex=False), errors="coerce")  # "4,60" -> 4.60
        lon = pd.to_numeric(df[col_lon].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        ok = lat.between(3.5, 5.0) & lon.between(-75.0, -73.5)            # caja geográfica de Bogotá: filtra valores absurdos
        salida["latitude"] = lat.where(ok, centro_lat)                    # si la coordenada es válida la usamos; si no, el centro
        salida["longitude"] = lon.where(ok, centro_lon)
        salida["COORD_REAL"] = ok

    # --- 8.4 Variables de filtro (solo se crean si la base las trae) ---
    equivalencias = [
        ("ACCIDENTE_CON", ["GRAVEDAD"]),            # severidad: solo daños / heridos / muertos
        ("CLASE_ACCIDENTE", ["CLASE"]),             # choque, atropello, volcamiento...
        ("SEXO", ["SEXO", "GENERO"]),               # género del actor vial
        ("CONDICION", ["CONDICION"]),               # conductor, peatón, motociclista...
    ]
    for destino, palabras in equivalencias:
        col = buscar_columna(df, palabras)
        if col is not None:
            serie = df[col].astype(str).str.strip().str.upper()           # texto limpio en mayúsculas
            salida[destino] = serie.replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})  # vacíos -> NaN real

    return salida.reset_index(drop=True)                                  # reiniciamos el índice 0,1,2...


# ---------------------------------------------------------------------
# 9. CARGA CON CACHÉ: no vuelve a descargar mientras no pasen 10 minutos
# ---------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner="Cargando y limpiando datos...")
def cargar_datos(fuente, contenido=None, nombre=""):
    if fuente == FUENTE_API:
        crudo = descargar_api()
    elif fuente == FUENTE_ARCHIVO:
        crudo = leer_archivo(contenido, nombre)
    else:
        crudo = generar_simulado()
    return limpiar(crudo)


# ---------------------------------------------------------------------
# 10. BARRA LATERAL: fuente de datos
# ---------------------------------------------------------------------
st.sidebar.header("📡 Fuente de datos")
fuente = st.sidebar.radio("¿De dónde vienen los datos?", [FUENTE_API, FUENTE_ARCHIVO, FUENTE_SIM])

archivo = None
if fuente == FUENTE_ARCHIVO:
    archivo = st.sidebar.file_uploader("Sube el Excel o CSV de siniestros", type=["xlsx", "csv"])
    if archivo is None:                                                   # sin archivo no hay nada que analizar
        st.info("👈 Sube un archivo en la barra lateral para comenzar.")
        st.stop()                                                         # detiene la ejecución de la página

if st.sidebar.button("🔄 Recargar datos"):
    st.cache_data.clear()                                                 # borra la caché
    st.rerun()                                                            # vuelve a ejecutar la app

# ---------------------------------------------------------------------
# 11. CARGA REAL: si algo falla, avisamos y usamos datos simulados (nunca en silencio)
# ---------------------------------------------------------------------
datos_reales = fuente != FUENTE_SIM
detalle_error = None                                                      # aquí guardamos el motivo técnico si algo falla
try:
    if archivo is not None:
        df_live = cargar_datos(fuente, archivo.getvalue(), archivo.name)
    else:
        df_live = cargar_datos(fuente)
except Exception as error:
    detalle_error = str(error)                                            # guardamos el error para mostrarlo después, sin asustar
    df_live = cargar_datos(FUENTE_SIM)                                    # la demo sigue funcionando con datos simulados
    datos_reales = False

if df_live.empty:
    st.error("Después de la limpieza no quedaron registros válidos.")
    st.stop()

def texto_periodo():
    """Devuelve 'dd/mm/aaaa a dd/mm/aaaa' con la fecha del primer y último siniestro de la muestra."""
    if "FECHA" in df_live.columns and df_live["FECHA"].notna().any():
        return f"{df_live['FECHA'].min():%d/%m/%Y} a {df_live['FECHA'].max():%d/%m/%Y}"
    return "no disponible en la base"


def mostrar_pie():
    """Pie de página: periodo de los datos, fuente y aclaración de que no son datos en tiempo real."""
    st.write("---")
    if not datos_reales:                                                  # datos simulados
        st.caption("⚠️ **Fuente:** datos simulados generados por el programa, solo para probar la interfaz. "
                   "No representan la realidad de Bogotá.")
        return
    if fuente == FUENTE_ARCHIVO and archivo is not None:                  # archivo subido por el usuario
        origen = f"archivo cargado por el usuario ({archivo.name})"
    else:                                                                 # descarga desde el portal
        origen = ("«Histórico Siniestros Bogotá D.C.», Secretaría Distrital de Movilidad, "
                  "Portal de Datos Abiertos Bogotá (datosabiertos.bogota.gov.co)")
    st.caption(f"📅 **Periodo de los datos:** {texto_periodo()}  ·  🗂️ **Fuente:** {origen}  ·  "
               f"🔢 **Muestra:** {len(df_live):,} registros")
    st.caption("ℹ️ Son datos históricos oficiales: el portal los publica con rezago y no se actualizan en tiempo real. "
               "El modelo muestra el patrón típico por hora, no la situación de hoy.")


if datos_reales:
    pct_coord = df_live["COORD_REAL"].mean() * 100                        # % de filas con coordenadas reales
    st.caption(f"✅ Datos reales: {len(df_live):,} registros válidos · periodo {texto_periodo()} · "
               f"{pct_coord:.0f}% con coordenadas exactas (el resto se ubica en el centro de su localidad).")
elif detalle_error:                                                       # falló la carga: UN solo aviso, claro y humano
    st.warning("⚠️ **No pudimos conectar con el portal de Datos Abiertos de Bogotá.** "
               "Por ahora ves datos **simulados** para mostrar cómo funciona la herramienta. "
               "Para trabajar con datos reales, elige **«Subir archivo»** en la barra lateral "
               "y sube el Excel del Anuario de Siniestralidad.")
    with st.expander("Ver detalle técnico del error"):                    # desplegable: solo lo abre quien quiera
        st.caption(detalle_error)                                           # caption hace salto de línea (st.code se cortaba)
else:                                                                     # el usuario eligió simulados a propósito
    st.warning("⚠️ **DATOS SIMULADOS**: sirven para probar la interfaz, no representan la realidad de Bogotá.")

# ---------------------------------------------------------------------
# 12. FILTROS (solo aparecen los que la base realmente tiene)
# ---------------------------------------------------------------------
st.sidebar.write("---")
st.sidebar.header("🎛️ Filtros")
etiquetas = {
    "ACCIDENTE_CON": "Gravedad del siniestro:",
    "CLASE_ACCIDENTE": "Clase de siniestro:",
    "SEXO": "Género del actor vial:",
}
df_filtrado = df_live.copy()                                              # copia: la tabla original no se altera
for columna, etiqueta in etiquetas.items():
    if columna in df_live.columns:                                        # ¿la base trae esta variable?
        opciones = ["Todos"] + sorted(df_live[columna].dropna().unique())  # lista de valores posibles
        elegido = st.sidebar.selectbox(etiqueta, opciones)
        if elegido != "Todos":
            df_filtrado = df_filtrado[df_filtrado[columna] == elegido]    # nos quedamos solo con esa categoría

hora_mapa = st.sidebar.slider("Hora a mostrar en el mapa:", 0, 23, 14)    # línea de tiempo del mapa

# ---------------------------------------------------------------------
# 13. AGREGACIÓN: pares (t, n_t) = hora y cantidad de siniestros
# ---------------------------------------------------------------------
conteo = df_filtrado.groupby("HORA_NUMERICA").size().reindex(range(24), fill_value=0)  # horas sin datos cuentan como 0
X = conteo.index.to_numpy()                                               # eje horizontal: horas 0..23
Y = conteo.to_numpy()                                                     # eje vertical: siniestros por hora

if (Y > 0).sum() < 5:                                                     # mínimo 5 horas con datos para ajustar grado 4
    st.warning("Con estos filtros hay menos de 5 horas con datos. Prueba con filtros más amplios.")
    mostrar_pie()                                                         # el pie se muestra aunque la página se detenga aquí
    st.stop()

# ---------------------------------------------------------------------
# 14. AJUSTE POLINÓMICO R(t) POR MÍNIMOS CUADRADOS  (ecuaciones 2 y 3 del documento)
# ---------------------------------------------------------------------
coeficientes = np.polyfit(X, Y, GRADO_POLINOMIO)                          # [a, b, c, d, e] que minimizan el error cuadrático
f_riesgo = np.poly1d(coeficientes)                                        # convierte los coeficientes en una función evaluable

ss_res = np.sum((Y - f_riesgo(X)) ** 2)                                   # error que deja la curva
ss_tot = np.sum((Y - Y.mean()) ** 2)                                      # variación total de los datos
r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0                           # coeficiente de determinación (1 = ajuste perfecto)

# ---------------------------------------------------------------------
# 15. CÁLCULO DIFERENCIAL CON SYMPY  (ecuaciones 4, 5 y 6 del documento)
# ---------------------------------------------------------------------
t = sp.Symbol("t")                                                        # t = hora del día (variable simbólica)
R = sum(sp.Float(float(c)) * t ** (GRADO_POLINOMIO - i) for i, c in enumerate(coeficientes))  # arma R(t) = a·t⁴ + b·t³ + ...
R1 = sp.diff(R, t)                                                        # primera derivada R'(t): ¿el riesgo sube o baja?
R2 = sp.diff(R, t, 2)                                                     # segunda derivada R''(t): ¿es un pico o un valle?

raices = [complex(r) for r in sp.Poly(R1, t).nroots()]                    # resuelve R'(t) = 0 (puntos críticos)
reales = [r.real for r in raices if abs(r.imag) < 1e-6 and 0 <= r.real <= 23]  # dejamos solo raíces reales dentro de 0-23 h
maximos = [r for r in reales if float(R2.subs(t, r)) < 0]                 # criterio 2ª derivada: R''(t) < 0 => máximo

candidatos = maximos + [0.0, 23.0]                                        # también se comparan los extremos del intervalo
hora_critica = max(candidatos, key=lambda x: float(f_riesgo(x)))          # el candidato donde R(t) es mayor = pico del día
pico_en_extremo = hora_critica in (0.0, 23.0)                             # ¿el máximo cayó en el borde del día?

min_pico = round(hora_critica * 60)                                       # hora del pico en minutos desde medianoche
h_pico, m_pico = divmod(min_pico, 60)                                     # minutos -> (horas, minutos)
min_alerta = (min_pico - VENTANA_MINUTOS) % 1440                          # restamos 30 min; % 1440 evita horas negativas
h_alerta, m_alerta = divmod(min_alerta, 60)

hora_observada = int(conteo.idxmax())                                     # hora con más siniestros en los datos reales

# ---------------------------------------------------------------------
# 16. INDICADORES PRINCIPALES
# ---------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("🕰️ Hora de mayor riesgo (modelo)", f"{h_pico:02d}:{m_pico:02d}")
m2.metric(f"🚑 Hora de alerta ({VENTANA_MINUTOS} min antes)", f"{h_alerta:02d}:{m_alerta:02d}")
m3.metric("📐 Ajuste de la curva (r²)", f"{r2:.2f}")
m4.metric("📊 Hora con más siniestros (datos)", f"{hora_observada:02d}:00")
st.caption("r² cercano a 1 significa que la curva sigue bien los datos. "
           "Si la hora del modelo y la observada se parecen, la predicción es consistente.")

if pico_en_extremo:
    st.info("El máximo del modelo quedó en el borde del día (0 h o 23 h). Interprétalo con cuidado.")

st.error(f"🚨 **ALERTA TEMPRANA:** enviar unidades preventivas a las {h_alerta:02d}:{m_alerta:02d} h.")

# ---------------------------------------------------------------------
# 17. DOS COLUMNAS: gráfica a la izquierda, mapa y ranking a la derecha
# ---------------------------------------------------------------------
col_izq, col_der = st.columns(2)

with col_izq:
    st.write("### 📈 Curva de riesgo R(t)")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.scatter(X, Y, color="black", label="Siniestros observados por hora")           # puntos reales
    x_curva = np.linspace(0, 23, 500)                                                  # 500 puntos para dibujar una curva suave
    ax.plot(x_curva, f_riesgo(x_curva), color="#0077B6", linewidth=2.5, label="Curva R(t) ajustada")
    ax.axvline(x=hora_critica, color="darkred", linestyle="--", label="Hora de mayor riesgo")      # línea del pico
    ax.axvline(x=min_alerta / 60, color="orange", linestyle=":", label="Hora de alerta")           # línea de la alerta
    ax.set_xticks(range(0, 24, 2))                                                     # marcas cada 2 horas
    ax.set_ylim(bottom=0)                                                              # el eje no baja de 0 siniestros
    ax.set_xlabel("Hora del día (t)")
    ax.set_ylabel("Cantidad de siniestros")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)                                                                     # muestra la gráfica en la página
    plt.close(fig)                                                                     # libera memoria

with col_der:
    st.write(f"### 🗺️ Mapa de siniestros a las {hora_mapa:02d}:00 h")
    datos_hora = df_filtrado[df_filtrado["HORA_NUMERICA"] == hora_mapa].dropna(subset=["latitude", "longitude"])
    if datos_hora.empty:
        st.warning("Sin siniestros para esta combinación de filtros en esa hora.")
    else:
        mapa_df = (datos_hora.assign(latitude=datos_hora["latitude"].round(3),          # redondeo ≈ 110 m: agrupa puntos cercanos
                                     longitude=datos_hora["longitude"].round(3))
                   .groupby(["latitude", "longitude"]).size().reset_index(name="count"))
        mapa_df["point_size"] = 150 + 1200 * mapa_df["count"] / mapa_df["count"].max()  # tamaño del círculo proporcional a la cantidad
        st.map(mapa_df, size="point_size", color="#D90429AA", zoom=10.1)               # círculos rojos semitransparentes

        # --- Ranking de localidades (asignación prioritaria de unidades) ---
        st.write("### 🎯 Dónde enviar las unidades primero")
        ranking = (datos_hora[datos_hora["NOMBRE_LOCALIDAD"] != "Desconocida"]          # ignoramos las no reconocidas
                   ["NOMBRE_LOCALIDAD"].value_counts().head(3))                         # las 3 con más siniestros
        if ranking.empty:
            st.info("No hay localidades reconocidas para esta hora.")
        else:
            nombres, cantidades = list(ranking.index), list(ranking.values)
            roles = ["Sector principal", "Sector secundario", "Sector de resguardo"]
            lineas = [f"{i + 1}. **{roles[i]}:** {nombres[i]} ({cantidades[i]} siniestros registrados)"
                      for i in range(len(nombres))]
            st.success("🚑 **Asignación prioritaria de recursos**\n\n" + "\n\n".join(lineas))

# ---------------------------------------------------------------------
# 18. DESGLOSE DEL SEGMENTO FILTRADO
# ---------------------------------------------------------------------
st.write("---")
if "CONDICION" in df_filtrado.columns:
    st.write("### 👥 Siniestros por condición del implicado")
    st.bar_chart(df_filtrado["CONDICION"].value_counts(), color="#4A90E2")
elif "CLASE_ACCIDENTE" in df_filtrado.columns:
    st.write("### 🚗 Siniestros por clase")
    st.bar_chart(df_filtrado["CLASE_ACCIDENTE"].value_counts(), color="#4A90E2")
else:
    st.info("La base cargada no trae variables de condición ni de clase para el desglose.")

# ---------------------------------------------------------------------
# 19. PIE DE PÁGINA: periodo y fuente de los datos
# ---------------------------------------------------------------------
mostrar_pie()