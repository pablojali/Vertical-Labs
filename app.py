import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import gpxpy
import re
import traceback
import requests
from trail_metrics_config import INDEX_CONFIG, SPEED_METRICS, display_metric_documentation
from data.gpx_loader import (
    build_cascading_selector,
    get_gpx_path,
    get_checkpoints,
    get_carrera_info,
    get_carreras,
)

# 1. Configuración de la interfaz estilo VertLabs
st.set_page_config(page_title="VertLabs - Trail Analytics", page_icon="🏃‍♂️", layout="wide")

st.title("🏃‍♂️ VertLabs - Motor de Análisis de Carreras")
st.caption("Backend de análisis: segmentación geométrica de GPX + métricas oficiales de corredores.")
st.markdown("---")

# Umbrales fijos (hardcodeados para simplificar la app).
umbral_pendiente_fuerte = 12    # >= 12% subida fuerte | <= -12% bajada fuerte
umbral_pendiente_mod_min = 5    # entre 5% y 12% = subida/bajada moderada
umbral_pendiente_mod_max = 12
umbral_altitud = 1800           # metros sobre el nivel del mar


# ============================================================
# 2. FUNCIONES DEL MOTOR (backend) - sin lógica de interfaz
# ============================================================

def procesar_gpx_avanzado(file):
    """Parsea un GPX y devuelve un DataFrame punto a punto con
    distancia acumulada, altitud y pendiente instantánea."""
    gpx = gpxpy.parse(file)
    puntos_datos = []
    distancia_acumulada = 0.0
    punto_previo = None

    for track in gpx.tracks:
        for segment in track.segments:
            for punto in segment.points:
                if punto_previo:
                    dist = punto.distance_2d(punto_previo)
                    distancia_acumulada += dist

                    desnivel = punto.elevation - punto_previo.elevation
                    # Evitar divisiones por cero en puntos idénticos
                    pendiente = (desnivel / dist) * 100 if dist > 0 else 0

                    puntos_datos.append({
                        "Distancia (km)": distancia_acumulada / 1000.0,
                        "Altitud (m)": punto.elevation,
                        "Pendiente (%)": pendiente
                    })
                punto_previo = punto
    return pd.DataFrame(puntos_datos)


def clasificar_pendiente(pendiente):
    """Categorías mutuamente excluyentes que cubren todo el rango
    sin huecos: fuerte -> moderada -> rodante."""
    if pendiente >= umbral_pendiente_fuerte:
        return "Subida Fuerte (≥12%)"
    elif pendiente <= -umbral_pendiente_fuerte:
        return "Bajada Fuerte (≤-12%)"
    elif pendiente > umbral_pendiente_mod_min:
        return "Subida Moderada (5-12%)"
    elif pendiente < -umbral_pendiente_mod_min:
        return "Bajada Moderada (-5 a -12%)"
    else:
        return "Tramo Rodante (-5 a +5%)"


def analizar_carrera(archivo_gpx):
    """Motor completo de análisis geométrico: recibe el archivo GPX
    subido y devuelve el DataFrame enriquecido con clasificación de
    pendiente y altitud, listo para graficar/mostrar."""
    df_gpx = procesar_gpx_avanzado(archivo_gpx)

    df_gpx["Tipo Pendiente"] = df_gpx["Pendiente (%)"].apply(clasificar_pendiente)

    # Clasificación por ALTITUD (independiente de la pendiente: un tramo
    # puede ser "Subida Fuerte" Y estar "Sobre 1800m" al mismo tiempo)
    df_gpx["Zona Altitud"] = df_gpx["Altitud (m)"].apply(
        lambda alt: f"Sobre {umbral_altitud}m" if alt > umbral_altitud else f"Bajo {umbral_altitud}m"
    )
    return df_gpx


def matchear_checkpoints_con_gpx(df_gpx, checkpoints_km):
    """Recibe el DataFrame del GPX ya analizado y una lista de checkpoints
    [{'punto': int, 'km': float}, ...] ingresados a mano, y devuelve un
    DataFrame con un tramo por cada par de checkpoints consecutivos:
    distancia real, desnivel positivo/negativo y pendiente promedio,
    según el relieve oficial del GPX en ese rango de km.

    Esto es lo que después se cruza con los tiempos reales del corredor
    (Pestaña 2) para calcular VPI, DMI y ER."""
    checkpoints_ordenados = sorted(checkpoints_km, key=lambda c: c["km"])
    filas = []

    for i in range(len(checkpoints_ordenados) - 1):
        cp_inicio = checkpoints_ordenados[i]
        cp_fin = checkpoints_ordenados[i + 1]

        tramo = df_gpx[
            (df_gpx["Distancia (km)"] >= cp_inicio["km"]) &
            (df_gpx["Distancia (km)"] <= cp_fin["km"])
        ]

        if tramo.empty:
            desnivel_positivo = None
            desnivel_negativo = None
            pendiente_promedio = None
        else:
            diffs_altitud = tramo["Altitud (m)"].diff().dropna()
            desnivel_positivo = diffs_altitud[diffs_altitud > 0].sum()
            desnivel_negativo = diffs_altitud[diffs_altitud < 0].sum()  # queda negativo
            pendiente_promedio = tramo["Pendiente (%)"].mean()

        filas.append({
            "Punto Inicio": cp_inicio["punto"],
            "Punto Fin": cp_fin["punto"],
            "Km Inicio": cp_inicio["km"],
            "Km Fin": cp_fin["km"],
            "Distancia Tramo (km)": round(cp_fin["km"] - cp_inicio["km"], 3),
            "Desnivel Positivo (m)": round(desnivel_positivo, 1) if desnivel_positivo is not None else None,
            "Desnivel Negativo (m)": round(desnivel_negativo, 1) if desnivel_negativo is not None else None,
            "Pendiente Promedio (%)": round(pendiente_promedio, 2) if pendiente_promedio is not None else None,
        })

    return pd.DataFrame(filas)


def parsear_tiempo_a_horas(tiempo_str):
    """Convierte un string 'H:MM:SS' (o 'HH:MM:SS') en horas decimales.
    Si viene vacío/None (ej. el checkpoint de largada), se toma como 0."""
    if pd.isna(tiempo_str) or tiempo_str is None or str(tiempo_str).strip() == "":
        return 0.0
    partes = [int(p) for p in str(tiempo_str).split(":")]
    while len(partes) < 3:
        partes.insert(0, 0)
    horas, minutos, segundos = partes[-3], partes[-2], partes[-1]
    return horas + minutos / 60 + segundos / 3600


def calcular_desnivel_positivo_total(df_gpx_completo):
    """Desnivel positivo acumulado de TODO el recorrido (no solo los
    tramos entre checkpoints), usado para Total_Km_E en el índice ER."""
    diffs = df_gpx_completo["Altitud (m)"].diff().dropna()
    return diffs[diffs > 0].sum()


def calcular_indices_corredor(df_segmentos, df_atleta, total_km, desnivel_positivo_total,
                               coef_distance_weighting=1.0):
    """Cruza los tramos oficiales de la carrera (df_segmentos, calculados
    en la Pestaña 1 a partir de los checkpoints ingresados a mano) con los
    tiempos reales del corredor (df_atleta) para calcular VPI, DMI y ER.

    El cruce se hace por el número de 'Punto' (checkpoint), que debe
    coincidir entre ambas tablas."""

    if "Punto" not in df_atleta.columns:
        raise ValueError("La tabla del corredor no tiene columna 'Punto' para matchear checkpoints.")

    tiempos_por_punto = {
        fila["Punto"]: parsear_tiempo_a_horas(fila.get("Tiempo Acumulado"))
        for _, fila in df_atleta.iterrows()
    }

    filas_cruce = []
    for _, seg in df_segmentos.iterrows():
        p_ini, p_fin = seg["Punto Inicio"], seg["Punto Fin"]
        if p_ini in tiempos_por_punto and p_fin in tiempos_por_punto:
            tiempo_corredor_h = tiempos_por_punto[p_fin] - tiempos_por_punto[p_ini]
        else:
            tiempo_corredor_h = None
        fila = seg.to_dict()
        fila["Tiempo Corredor (h)"] = tiempo_corredor_h
        filas_cruce.append(fila)

    df_cruce = pd.DataFrame(filas_cruce)
    segmentos_sin_match = df_cruce["Tiempo Corredor (h)"].isna().sum()

    df_valido = df_cruce.dropna(subset=["Tiempo Corredor (h)"]).copy()
    df_valido = df_valido[df_valido["Tiempo Corredor (h)"] > 0]

    if df_valido.empty:
        raise ValueError(
            "Ningún checkpoint de la carrera guardada coincide con los puntos del corredor. "
            "Revisá que los números de 'Punto' sean los mismos en ambas tablas."
        )

    # --- VPI: Subida Fuerte (pendiente promedio del tramo >= 12%) ---
    tramos_subida_fuerte = df_valido[df_valido["Pendiente Promedio (%)"] >= umbral_pendiente_fuerte]
    tiempo_subida_h = tramos_subida_fuerte["Tiempo Corredor (h)"].sum()
    gain_subida_m = tramos_subida_fuerte["Desnivel Positivo (m)"].sum()
    vpi = (gain_subida_m / tiempo_subida_h) if tiempo_subida_h > 0 else None

    # --- DMI: Bajada Fuerte (pendiente promedio del tramo <= -12%) ---
    tramos_bajada_fuerte = df_valido[df_valido["Pendiente Promedio (%)"] <= -umbral_pendiente_fuerte]
    tiempo_bajada_h = tramos_bajada_fuerte["Tiempo Corredor (h)"].sum()
    dist_bajada_km = tramos_bajada_fuerte["Distancia Tramo (km)"].sum()
    dmi = (dist_bajada_km / tiempo_bajada_h) if tiempo_bajada_h > 0 else None

    # --- ER: Endurance Rating (degradación de ritmo entre 1ra y 2da mitad) ---
    total_km_e = total_km + (desnivel_positivo_total / 100)
    df_valido = df_valido.sort_values("Km Inicio").reset_index(drop=True)
    df_valido["Effort Km Segmento"] = df_valido["Distancia Tramo (km)"] + (
        df_valido["Desnivel Positivo (m)"].fillna(0) / 100
    )
    df_valido["Effort Km Acumulado"] = df_valido["Effort Km Segmento"].cumsum()
    mitad_effort_km = total_km_e / 2

    primera_mitad = df_valido[df_valido["Effort Km Acumulado"] <= mitad_effort_km]
    segunda_mitad = df_valido[df_valido["Effort Km Acumulado"] > mitad_effort_km]

    def _effort_pace(segmentos):
        tiempo_min = segmentos["Tiempo Corredor (h)"].sum() * 60
        effort_km = segmentos["Effort Km Segmento"].sum()
        return (tiempo_min / effort_km) if effort_km > 0 else None

    pace_1 = _effort_pace(primera_mitad)
    pace_2 = _effort_pace(segunda_mitad)

    if pace_1 and pace_2 and pace_1 > 0:
        pacing_decay_pct = ((pace_2 / pace_1) - 1) * 100
        er = 100 - (pacing_decay_pct * coef_distance_weighting)
    else:
        pacing_decay_pct = None
        er = None

    resultado = {
        "VPI": round(vpi, 1) if vpi is not None else None,
        "DMI": round(dmi, 2) if dmi is not None else None,
        "ER": round(er, 1) if er is not None else None,
        "Pacing_Decay_%": round(pacing_decay_pct, 1) if pacing_decay_pct is not None else None,
        "segmentos_sin_match": int(segmentos_sin_match),
    }
    return resultado, df_valido


def calcular_indices_por_tramo(df_gpx_completo, df_segmentos, df_atleta):
    """Calcula VPI y DMI de forma INDEPENDIENTE para cada tramo (matriz de
    degradación), en vez de un solo valor global para toda la carrera.

    Para cada tramo entre 2 checkpoints:
    - Recorta el sub-GPX de ese tramo (por rango de km).
    - VPI: filtra solo los puntos con pendiente >= 12% dentro del tramo y
      suma su desnivel positivo real, dividido por el tiempo que tardó el
      corredor en ese tramo específico.
    - DMI: filtra solo los puntos con pendiente <= -12% dentro del tramo y
      suma la distancia real recorrida en esos puntos, dividido por el
      mismo tiempo.

    Al final normaliza ambos índices contra el Tramo 1 del corredor
    (Tramo 1 = 100), para poder graficar la curva de degradación en una
    escala 0-100 comparable entre corredores."""

    df_gpx_completo = df_gpx_completo.sort_values("Distancia (km)").reset_index(drop=True)
    # Reconstruimos el "mini-tramo" punto a punto (distancia y desnivel
    # entre cada punto consecutivo del GPX), a partir de las columnas
    # acumuladas que ya tenemos.
    dist_incremental_m = df_gpx_completo["Distancia (km)"].diff() * 1000
    desnivel_incremental_m = df_gpx_completo["Altitud (m)"].diff()

    tiempos_por_punto = {
        fila["Punto"]: parsear_tiempo_a_horas(fila.get("Tiempo Acumulado"))
        for _, fila in df_atleta.iterrows()
    }

    df_segmentos_ordenado = df_segmentos.sort_values("Km Inicio").reset_index(drop=True)
    filas = []

    for i, seg in df_segmentos_ordenado.iterrows():
        p_ini, p_fin = seg["Punto Inicio"], seg["Punto Fin"]
        km_ini, km_fin = seg["Km Inicio"], seg["Km Fin"]

        if p_ini not in tiempos_por_punto or p_fin not in tiempos_por_punto:
            tiempo_tramo_h = None
        else:
            tiempo_tramo_h = tiempos_por_punto[p_fin] - tiempos_por_punto[p_ini]

        mascara_tramo = (df_gpx_completo["Distancia (km)"] >= km_ini) & (df_gpx_completo["Distancia (km)"] <= km_fin)
        pendiente_tramo = df_gpx_completo.loc[mascara_tramo, "Pendiente (%)"]

        vpi_raw, dmi_raw = None, None
        if tiempo_tramo_h and tiempo_tramo_h > 0:
            mascara_subida_fuerte = mascara_tramo & (df_gpx_completo["Pendiente (%)"] >= umbral_pendiente_fuerte)
            gain_vpi_m = desnivel_incremental_m[mascara_subida_fuerte].sum()
            vpi_raw = gain_vpi_m / tiempo_tramo_h if gain_vpi_m > 0 else None

            mascara_bajada_fuerte = mascara_tramo & (df_gpx_completo["Pendiente (%)"] <= -umbral_pendiente_fuerte)
            dist_dmi_km = dist_incremental_m[mascara_bajada_fuerte].sum() / 1000
            dmi_raw = dist_dmi_km / tiempo_tramo_h if dist_dmi_km > 0 else None

        filas.append({
            "Tramo": f"P{p_ini}→P{p_fin}",
            "Km Inicio": km_ini,
            "Km Fin": km_fin,
            "Tiempo Corredor (h)": round(tiempo_tramo_h, 2) if tiempo_tramo_h is not None else None,
            "VPI Crudo (m/h)": round(vpi_raw, 1) if vpi_raw is not None else None,
            "DMI Crudo (km/h)": round(dmi_raw, 2) if dmi_raw is not None else None,
        })

    df_tramos = pd.DataFrame(filas)

    # Normalización contra el Tramo 1 válido de este corredor (Tramo 1 = 100)
    def _normalizar(serie):
        valores_validos = serie.dropna()
        if valores_validos.empty:
            return pd.Series([None] * len(serie), index=serie.index)
        baseline = valores_validos.iloc[0]
        if not baseline:
            return pd.Series([None] * len(serie), index=serie.index)
        return (serie / baseline) * 100

    df_tramos["VPI Índice (0-100)"] = _normalizar(df_tramos["VPI Crudo (m/h)"]).round(1)
    df_tramos["DMI Índice (0-100)"] = _normalizar(df_tramos["DMI Crudo (km/h)"]).round(1)

    return df_tramos


# NUEVA FUNCIÓN: Web Scraping automático de tablas de cronometraje
# La página live.utmb.world es una app Next.js: el HTML que devuelve el
# servidor llega vacío y los datos se piden por JS a una API interna.
# Gracias a inspeccionar el Network tab del navegador, encontramos el
# endpoint real que usa la propia web:
#   https://utmblive-api.utmb.world/runners/<ID>?locale=en
# Con esto alcanza un simple requests.get(): no hace falta navegador,
# no hace falta Playwright/Chromium ni packages.txt.

def extraer_id_corredor(url):
    """Extrae el ID numérico del corredor desde una URL tipo
    https://live.utmb.world/aranbyutmb/2026/runners/5"""
    match = re.search(r"/runners/(\d+)", url)
    return match.group(1) if match else None


def extraer_tenant(url):
    """Extrae 'carrera_año' desde una URL tipo
    https://live.utmb.world/aranbyutmb/2026/runners/5 -> 'aranbyutmb_2026'
    La API lo exige como header X-Tenant para saber a qué edición
    de la carrera pertenece el corredor."""
    match = re.search(r"live\.utmb\.world/([a-zA-Z0-9]+)/(\d{4})/runners/", url)
    if match:
        carrera, anio = match.groups()
        return f"{carrera}_{anio}"
    return None


def scrapear_tiempos_paso(url):
    corredor_id = extraer_id_corredor(url)
    if not corredor_id:
        raise ValueError(
            "No pude encontrar el ID del corredor en esa URL. "
            "Verificá que tenga el formato '.../runners/<numero>' "
            "(ej: https://live.utmb.world/aranbyutmb/2026/runners/5)."
        )

    tenant = extraer_tenant(url)
    if not tenant:
        raise ValueError(
            "No pude identificar la carrera/año en esa URL. "
            "Verificá que tenga el formato "
            "'https://live.utmb.world/<carrera>/<año>/runners/<numero>'."
        )

    api_url = f"https://utmblive-api.utmb.world/runners/{corredor_id}?locale=en"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "*/*",
        "Origin": "https://live.utmb.world",
        "Referer": "https://live.utmb.world/",
        "X-Tenant": tenant,
    }

    respuesta = requests.get(api_url, headers=headers, timeout=15)
    respuesta.raise_for_status()
    data = respuesta.json()

    resume = data.get("resume", {}) or {}
    info = resume.get("info", {}) or {}
    ranking = resume.get("ranking", {}) or {}
    pais = data.get("country", {}) or {}

    info_general = {
        "Nombre": info.get("fullname"),
        "Dorsal": resume.get("bib"),
        "Edad": info.get("age"),
        "Categoría": info.get("category"),
        "Club": info.get("club"),
        "País": pais.get("name"),
        "Tiempo Final": resume.get("raceTime"),
        "Puesto Scratch": ranking.get("scratch"),
        "Puesto Sexo": ranking.get("sex"),
        "Puesto Categoría": ranking.get("category"),
        "Estado": resume.get("status"),
    }

    passings = (data.get("detail", {}) or {}).get("passings", []) or []
    df_passings = pd.DataFrame(passings)

    # Nos quedamos solo con las columnas útiles (sacamos las crudas/redundantes
    # como timeSeconds, datetimeIn/Out, y las de predicción en vivo que no
    # aplican a un corredor ya terminado) y les ponemos nombres en español.
    columnas_utiles = {
        "pointId": "Punto",
        "cumulatedTime": "Tiempo Acumulado",
        "time": "Tiempo Tramo",
        "speed": "Velocidad (km/h)",
        "pace": "Ritmo (min/km)",
        "rank": "Puesto",
        "restTime": "Descanso",
    }
    columnas_presentes = [c for c in columnas_utiles if c in df_passings.columns]
    df_passings = df_passings[columnas_presentes].rename(columns=columnas_utiles)

    return info_general, df_passings


# ============================================================
# 3. INTERFAZ: dos pestañas independientes
# ============================================================

# "Biblioteca" de carreras analizadas, en memoria durante la sesión.
# Estructura: { "Nombre Carrera": {"df": DataFrame, "total_km": float, ...} }
if 'carreras_guardadas' not in st.session_state:
    st.session_state['carreras_guardadas'] = {}

tab_carrera, tab_corredor, tab_metodologia = st.tabs(
    ["🗺️ Análisis Carrera", "🏃 Métricas Corredor", "📖 Índices & Metodología"]
)

# ---------------------------------------------
# PESTAÑA 1: Análisis geométrico del GPX oficial
# ---------------------------------------------
with tab_carrera:
    st.header("🗺️ Análisis Geométrico de la Carrera (GPX)")
    st.caption(
        f"Pendiente fuerte: ≥{umbral_pendiente_fuerte}% subida / ≤-{umbral_pendiente_fuerte}% bajada · "
        f"Moderada: {umbral_pendiente_mod_min}-{umbral_pendiente_mod_max}% · "
        f"Altitud: >{umbral_altitud}m"
    )

    carrera_slug, anio, distancia = build_cascading_selector(st, key_prefix="tab1_selector")

    if not (carrera_slug and anio and distancia):
        st.info("👋 Elegí carrera, año y distancia arriba para cargar el análisis geométrico.")
    else:
        try:
            gpx_path = get_gpx_path(carrera_slug, anio, distancia)
            error_gpx = None
        except FileNotFoundError as e:
            gpx_path = None
            error_gpx = str(e)

        if error_gpx:
            st.error(f"❌ {error_gpx}")
        else:
            checkpoints_registro = get_checkpoints(carrera_slug, anio, distancia)
            info_registro = get_carrera_info(carrera_slug, anio, distancia)
            nombre_carrera_visible = dict(get_carreras()).get(carrera_slug, carrera_slug)

            # --- Panel de confirmación (antes de correr el análisis) ---
            with st.container(border=True):
                st.markdown(f"**GPX encontrado:** `{info_registro['gpx_file']}`")
                colA, colB = st.columns(2)
                colA.metric("Checkpoints en el registro", len(checkpoints_registro))
                colB.metric("Slug API (X-Tenant)", info_registro.get("race_slug_api", carrera_slug))

                if checkpoints_registro:
                    st.markdown("**Checkpoints:**")
                    st.dataframe(
                        checkpoints_registro,
                        column_config={
                            "id": "ID",
                            "nombre": "Nombre",
                            "km": st.column_config.NumberColumn("Km", format="%.2f"),
                        },
                        hide_index=True,
                        use_container_width=True,
                    )
                else:
                    st.warning(
                        "Esta combinación todavía no tiene checkpoints en el registro. El análisis "
                        "geométrico funciona igual, pero no vas a poder calcular VPI/DMI por tramo "
                        "hasta cargarlos en `data/races_registry.json`."
                    )

            usar_carrera = st.button(
                "✅ Usar esta carrera para el análisis", type="primary", use_container_width=True
            )
            if usar_carrera:
                st.session_state["carrera_activa_tab1"] = (carrera_slug, anio, distancia)

            carrera_activa = st.session_state.get("carrera_activa_tab1")

            if not carrera_activa:
                st.info("Hacé clic en '✅ Usar esta carrera para el análisis' para correr el motor geométrico.")
            elif carrera_activa != (carrera_slug, anio, distancia):
                st.info(
                    "Seleccionaste una combinación distinta a la que está activa. Hacé clic en "
                    "'✅ Usar esta carrera para el análisis' para actualizarla."
                )
            else:
                carrera_s, anio_a, distancia_a = carrera_activa
                gpx_path_activo = get_gpx_path(carrera_s, anio_a, distancia_a)
                checkpoints_activos = get_checkpoints(carrera_s, anio_a, distancia_a)

                with open(gpx_path_activo, "r", encoding="utf-8") as f:
                    df_gpx = analizar_carrera(f)

                # Métricas resumen. Cada punto del GPX representa la misma distancia
                # promedio (total_km / cantidad de puntos), así que contamos puntos
                # por categoría y los convertimos a km.
                total_km = df_gpx["Distancia (km)"].max()
                km_por_punto = total_km / len(df_gpx)

                km_subida_fuerte = (df_gpx["Tipo Pendiente"] == "Subida Fuerte (≥12%)").sum() * km_por_punto
                km_bajada_fuerte = (df_gpx["Tipo Pendiente"] == "Bajada Fuerte (≤-12%)").sum() * km_por_punto
                km_subida_moderada = (df_gpx["Tipo Pendiente"] == "Subida Moderada (5-12%)").sum() * km_por_punto
                km_bajada_moderada = (df_gpx["Tipo Pendiente"] == "Bajada Moderada (-5 a -12%)").sum() * km_por_punto
                km_rodante = (df_gpx["Tipo Pendiente"] == "Tramo Rodante (-5 a +5%)").sum() * km_por_punto
                km_sobre_altitud = (df_gpx["Zona Altitud"] == f"Sobre {umbral_altitud}m").sum() * km_por_punto

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Distancia Total", f"{total_km:.2f} km")
                col2.metric("Subida Fuerte (≥12%)", f"{km_subida_fuerte:.2f} km")
                col3.metric("Bajada Fuerte (≤-12%)", f"{km_bajada_fuerte:.2f} km")
                col4.metric(f"Sobre {umbral_altitud}m", f"{km_sobre_altitud:.2f} km")

                col5, col6, col7 = st.columns(3)
                col5.metric("Subida Moderada (5-12%)", f"{km_subida_moderada:.2f} km")
                col6.metric("Bajada Moderada (-5 a -12%)", f"{km_bajada_moderada:.2f} km")
                col7.metric("Tramo Rodante (-5 a +5%)", f"{km_rodante:.2f} km")

                # Renderizar Gráfica con Clasificación de Colores Multi-Capa
                st.markdown("---")
                st.subheader("📈 Mapa de Esfuerzo Biomecánico del Relieve")
                st.write("El motor ha aislado los tramos de la carrera según pendiente y altitud:")

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=df_gpx["Distancia (km)"], y=df_gpx["Altitud (m)"],
                    mode='lines', name='Perfil Base',
                    line=dict(color='#444444', width=1.5)
                ))

                capas = [
                    ("Subida Fuerte (≥12%)", "#ff4b4b", 3.5),
                    ("Bajada Fuerte (≤-12%)", "#00bfff", 3.5),
                    ("Subida Moderada (5-12%)", "#ffa500", 3.0),
                    ("Bajada Moderada (-5 a -12%)", "#7dd3fc", 3.0),
                    ("Tramo Rodante (-5 a +5%)", "#4ade80", 2.5),
                ]
                for nombre_categoria, color, ancho in capas:
                    df_capa = df_gpx.copy()
                    df_capa.loc[df_capa["Tipo Pendiente"] != nombre_categoria, "Altitud (m)"] = None
                    fig.add_trace(go.Scatter(
                        x=df_capa["Distancia (km)"], y=df_capa["Altitud (m)"],
                        mode='lines', name=nombre_categoria,
                        line=dict(color=color, width=ancho)
                    ))

                fig.add_hline(
                    y=umbral_altitud,
                    line_dash="dash",
                    line_color="#a78bfa",
                    annotation_text=f"{umbral_altitud}m",
                    annotation_position="top left",
                )

                fig.update_layout(
                    template="plotly_dark",
                    xaxis_title="Distancia (Kilómetros)",
                    yaxis_title="Altitud (Metros)",
                    height=450,
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("Ver tabla completa punto por punto"):
                    st.dataframe(df_gpx, use_container_width=True)

                st.session_state['df_gpx_analytics'] = df_gpx

                # --- Matcheo de checkpoints del registro contra el GPX ---
                st.markdown("---")
                st.subheader("📍 Checkpoints Oficiales de la Carrera")

                checkpoints_validos = []
                ids_invalidos = []
                for cp in checkpoints_activos:
                    try:
                        checkpoints_validos.append({"punto": int(cp["id"]), "km": float(cp["km"])})
                    except (ValueError, TypeError):
                        ids_invalidos.append(cp.get("id"))

                if ids_invalidos:
                    st.warning(
                        f"Los checkpoints con id {ids_invalidos} no son numéricos y se excluyeron "
                        "del matcheo (el 'id' debe coincidir con el 'pointId' de UTMB Live)."
                    )

                df_segmentos = None
                if len(checkpoints_validos) >= 2:
                    df_segmentos = matchear_checkpoints_con_gpx(df_gpx, checkpoints_validos)
                    st.markdown("##### Vista previa del matcheo (tramo por tramo)")
                    st.dataframe(df_segmentos, use_container_width=True)
                else:
                    st.info(
                        "Esta carrera tiene menos de 2 checkpoints numéricos válidos en el registro: "
                        "el análisis geométrico general queda disponible, pero no se puede calcular "
                        "VPI/DMI por tramo en la pestaña 'Métricas Corredor'."
                    )

                # --- Guardado automático en la biblioteca (Pestaña 2) ---
                nombre_carrera_guardada = f"{nombre_carrera_visible} {anio_a} - {distancia_a}K"
                st.session_state['carreras_guardadas'][nombre_carrera_guardada] = {
                    "df": df_gpx,
                    "total_km": total_km,
                    "km_subida_fuerte": km_subida_fuerte,
                    "km_bajada_fuerte": km_bajada_fuerte,
                    "km_subida_moderada": km_subida_moderada,
                    "km_bajada_moderada": km_bajada_moderada,
                    "km_rodante": km_rodante,
                    "km_sobre_altitud": km_sobre_altitud,
                    "checkpoints_km": checkpoints_validos,
                    "df_segmentos": df_segmentos,
                }
                st.success(
                    f"✅ Carrera cargada como **'{nombre_carrera_guardada}'** — ya está disponible "
                    "en la pestaña 'Métricas Corredor'."
                )

        if st.session_state['carreras_guardadas']:
            with st.expander(f"📚 Carreras cargadas en esta sesión ({len(st.session_state['carreras_guardadas'])})"):
                for nombre in st.session_state['carreras_guardadas']:
                    n_checkpoints = len(st.session_state['carreras_guardadas'][nombre].get('checkpoints_km', []))
                    st.write(f"- {nombre} ({n_checkpoints} checkpoints)")

# ---------------------------------------------
# PESTAÑA 2: Métricas oficiales de un corredor
# ---------------------------------------------
with tab_corredor:
    st.header("🏃 Métricas Oficiales del Corredor")
    st.caption("Extrae tiempos de paso, ritmo y posición directo de la plataforma de cronometraje (UTMB Live).")

    # --- Selector de carrera guardada (analizada en la Pestaña 1) ---
    carreras_disponibles = st.session_state.get('carreras_guardadas', {})
    if not carreras_disponibles:
        st.warning(
            "⚠️ Todavía no guardaste ninguna carrera. Andá a la pestaña "
            "**'🗺️ Análisis Carrera'**, subí el GPX oficial, analizalo y guardalo "
            "con un nombre para poder elegirlo acá."
        )
        carrera_elegida = None
    else:
        carrera_elegida = st.selectbox(
            "¿Qué carrera hizo este corredor?",
            options=list(carreras_disponibles.keys()),
        )
        datos_carrera = carreras_disponibles[carrera_elegida]
        st.caption(
            f"Carrera seleccionada: **{carrera_elegida}** · "
            f"{datos_carrera['total_km']:.1f} km totales"
        )

    st.markdown("---")

    url_corredor = st.text_input(
        "Link del corredor (UTMB Live)",
        placeholder="https://live.utmb.world/aranbyutmb/2026/runners/5",
    )
    boton_cargar = st.button("🔍 Cargar datos del corredor", use_container_width=True)

    if boton_cargar:
        if not url_corredor:
            st.warning("Pegá primero un link válido antes de hacer clic en el botón.")
        else:
            with st.spinner("Conectando con la plataforma de cronometraje..."):
                try:
                    info_general, df_atleta = scrapear_tiempos_paso(url_corredor)
                    error_detalle = None
                except Exception:
                    info_general, df_atleta = None, None
                    error_detalle = traceback.format_exc()

            # --- Ventana de feedback ---
            if error_detalle:
                st.error("❌ Ocurrió un error al intentar obtener los datos del corredor.")
                with st.expander("Ver detalle técnico del error"):
                    st.code(error_detalle, language="python")
            elif df_atleta is None or df_atleta.empty:
                st.warning(
                    "⚠️ No se encontró ninguna tabla de datos en ese enlace. "
                    "Verificá que sea la URL directa del perfil del corredor."
                )
            else:
                st.success("✅ ¡Datos del corredor obtenidos con éxito!")

                # --- Cálculo de índices VPI / DMI / ER ---
                # Requiere que la carrera elegida tenga checkpoints con km
                # cargados en la Pestaña 1 (df_segmentos).
                datos_carrera_actual = carreras_disponibles.get(carrera_elegida, {}) if carrera_elegida else {}
                df_segmentos_carrera = datos_carrera_actual.get("df_segmentos")

                if df_segmentos_carrera is None or df_segmentos_carrera.empty:
                    st.warning(
                        "⚠️ La carrera seleccionada todavía no tiene checkpoints con km cargados. "
                        "Volvé a la pestaña 'Análisis Carrera', cargá los checkpoints de esa carrera "
                        "y guardala de nuevo para poder calcular los índices."
                    )
                else:
                    try:
                        desnivel_total_carrera = calcular_desnivel_positivo_total(datos_carrera_actual["df"])
                        indices, df_cruce = calcular_indices_corredor(
                            df_segmentos_carrera,
                            df_atleta,
                            datos_carrera_actual["total_km"],
                            desnivel_total_carrera,
                        )
                        error_indices = None
                    except Exception as e:
                        indices, df_cruce = None, None
                        error_indices = str(e)

                    st.markdown("### 🎯 Índices de Rendimiento")
                    if error_indices:
                        st.error(f"❌ No se pudieron calcular los índices: {error_indices}")
                    else:
                        i1, i2, i3 = st.columns(3)
                        i1.metric(
                            "🧗 VPI - Escalada Eficiente",
                            f"{indices['VPI']} m/h" if indices["VPI"] is not None else "N/D",
                            help="Vertical Power Index: metros de desnivel positivo por hora en tramos con pendiente ≥12%.",
                        )
                        i2.metric(
                            "📉 DMI - Rompepiernas",
                            f"{indices['DMI']} km/h" if indices["DMI"] is not None else "N/D",
                            help="Descent Mastery Index: velocidad promedio en tramos con pendiente ≤-12%.",
                        )
                        i3.metric(
                            "🏆 ER - Resistencia",
                            f"{indices['ER']}" if indices["ER"] is not None else "N/D",
                            help="Endurance Rating: 100 = ritmo estable, valores menores indican degradación por fatiga.",
                        )
                        if indices["segmentos_sin_match"] > 0:
                            st.caption(
                                f"⚠️ {indices['segmentos_sin_match']} tramo(s) de la carrera no tenían "
                                "un checkpoint equivalente en los datos del corredor y se excluyeron del cálculo."
                            )
                        with st.expander("Ver tramos cruzados (carrera + tiempos del corredor)"):
                            st.dataframe(df_cruce, use_container_width=True)

                        # --- Matriz de degradación por tramo ---
                        st.markdown("---")
                        st.markdown("### 📉 Curva de Degradación por Tramo")
                        st.caption(
                            "VPI y DMI calculados de forma independiente en cada tramo (no acumulados), "
                            "normalizados contra el Tramo 1 de este corredor (Tramo 1 = 100)."
                        )

                        df_tramos_degradacion = calcular_indices_por_tramo(
                            datos_carrera_actual["df"], df_segmentos_carrera, df_atleta
                        )

                        st.dataframe(df_tramos_degradacion, use_container_width=True)

                        fig_degradacion = go.Figure()
                        fig_degradacion.add_trace(go.Scatter(
                            x=df_tramos_degradacion["Km Fin"],
                            y=df_tramos_degradacion["VPI Índice (0-100)"],
                            mode="lines+markers",
                            name="VPI (Escalada)",
                            line=dict(color="#22d3ee", width=3),
                            text=df_tramos_degradacion["Tramo"],
                            hovertemplate="%{text}<br>Km %{x:.0f}<br>VPI Índice: %{y:.1f}<extra></extra>",
                        ))
                        fig_degradacion.add_trace(go.Scatter(
                            x=df_tramos_degradacion["Km Fin"],
                            y=df_tramos_degradacion["DMI Índice (0-100)"],
                            mode="lines+markers",
                            name="DMI (Descenso)",
                            line=dict(color="#ffa500", width=3),
                            text=df_tramos_degradacion["Tramo"],
                            hovertemplate="%{text}<br>Km %{x:.0f}<br>DMI Índice: %{y:.1f}<extra></extra>",
                        ))
                        fig_degradacion.update_layout(
                            template="plotly_dark",
                            xaxis_title="Km Acumulado",
                            yaxis_title="Índice (0-100, Tramo 1 = 100)",
                            height=420,
                            hovermode="x unified",
                        )
                        st.plotly_chart(fig_degradacion, use_container_width=True)

                st.markdown("---")

                # Ficha del corredor
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Corredor", info_general.get("Nombre") or "-")
                c2.metric("Tiempo Final", info_general.get("Tiempo Final") or "-")
                c3.metric("Puesto Scratch", info_general.get("Puesto Scratch") or "-")
                c4.metric("Categoría", info_general.get("Categoría") or "-")

                with st.expander("Ver ficha completa del corredor"):
                    st.json(info_general)

                st.markdown("##### Checkpoints / Tiempos de paso")
                st.dataframe(df_atleta, use_container_width=True)

                # Guardamos para reutilizar en otras pestañas (incluye qué
                # carrera se eligió, para el futuro cálculo de VPI/DMI/ER)
                st.session_state['df_corredor_metricas'] = df_atleta
                st.session_state['info_corredor'] = info_general
                st.session_state['carrera_elegida_para_corredor'] = carrera_elegida

# ---------------------------------------------
# PESTAÑA 3: Documentación de índices y metodología
# ---------------------------------------------
with tab_metodologia:
    st.header("📖 Índices y Metodología de Cálculo")
    st.caption(
        "Definiciones, criterios geométricos y fórmulas de los índices propios de VertLabs. "
        "Estos índices cruzan el relieve oficial del GPX (Pestaña 'Análisis Carrera') con los "
        "tiempos de paso reales del corredor (Pestaña 'Métricas Corredor')."
    )

    st.markdown("### 📐 Índices de Rendimiento")
    for clave_indice in INDEX_CONFIG:
        cfg = INDEX_CONFIG[clave_indice]
        with st.expander(f"{cfg['icon']} {cfg['name']} ({clave_indice}) — {cfg['label_es']}", expanded=False):
            st.markdown(display_metric_documentation(clave_indice))

    st.markdown("---")
    st.markdown("### ⚡ Métricas de Velocidad")
    for clave_metrica, cfg in SPEED_METRICS.items():
        with st.expander(f"{cfg['name']} — {cfg['label_es']}", expanded=False):
            st.markdown(f"""
            * **Descripción:** {cfg['description']}
            * **Fuente de datos:** {cfg['source']}
            * **Fórmula:** `{cfg['formula']}`
            * **Unidad:** {cfg['unit']}
            """)

    st.markdown("---")
    st.info(
        "ℹ️ Estas fórmulas están documentadas y listas, pero el **cálculo automático** de VPI, "
        "DMI y ER todavía no está implementado en las pestañas de Análisis Carrera / Métricas "
        "Corredor — hace falta cruzar los segmentos del GPX con los checkpoints reales del "
        "corredor. Avisame cuando quieras que lo conectemos."
    )