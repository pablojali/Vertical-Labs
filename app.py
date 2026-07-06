import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import gpxpy
import json
import subprocess
import sys
import traceback
from io import StringIO
from playwright.sync_api import sync_playwright


@st.cache_resource
def _asegurar_chromium_instalado():
    """Streamlit Community Cloud no corre 'playwright install' automáticamente
    (solo instala requirements.txt y packages.txt). Lo forzamos nosotros la
    primera vez que arranca la app. @st.cache_resource hace que esto se
    ejecute una sola vez por instancia, no en cada refresco de la página."""
    try:
        subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        st.sidebar.error(f"No se pudo instalar Chromium para Playwright: {e.stderr}")
    return True


_asegurar_chromium_instalado()

# 1. Configuración de la interfaz estilo VertLabs
st.set_page_config(page_title="VertLabs - Trail Analytics", page_icon="🏃‍♂️", layout="wide")

st.title("🏃‍♂️ VertLabs - Motor de Segmentación Geométrica")
st.markdown("---")

# Barra lateral para la carga
st.sidebar.header("⚙️ Entrada de Datos")
archivo_gpx = st.sidebar.file_uploader("1. Subir track de la carrera (.gpx)", type=["gpx"])

# Umbrales fijos (antes eran sliders). Se hardcodean para simplificar la app.
umbral_subida = 15
umbral_bajada = -12

# NUEVO: Sección de Scraping Automatizado para Redes Sociales
st.sidebar.markdown("---")
st.sidebar.header("📊 Datos de Rendimiento (Post-Carrera)")
url_corredor = st.sidebar.text_input("2. Pegar Link del Corredor (UTMB Live / LiveTrail)", 
                                     placeholder="https://livetrail.net/...")
boton_cargar = st.sidebar.button("🔍 Cargar datos del corredor", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption(f"Umbral IEPE (subida): {umbral_subida}% · Umbral TDE (bajada): {umbral_bajada}%")

# 2. Función del Motor Geométrico mejorada
def procesar_gpx_avanzado(file):
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

# NUEVA FUNCIÓN: Web Scraping automático de tablas de cronometraje
# IMPORTANTE: sitios como UTMB Live / LiveTrail son apps React/Next.js.
# El HTML que devuelve el servidor está vacío ("Loading...") y los datos
# se piden después por JavaScript a una API interna en formato JSON.
# Por eso pd.read_html(url) nunca encontraba tablas: no hay <table> en el HTML crudo.
#
# La solución: abrir la página con un navegador headless (Playwright),
# dejar que se ejecute el JS, e interceptar las respuestas JSON que la
# propia app pide para pintar los datos del corredor.

def _buscar_listas_de_diccionarios(obj, encontradas=None):
    """Recorre un JSON (dict/list anidado) y junta todas las listas de
    diccionarios que encuentre, para detectar automáticamente cuál es la
    que contiene los splits/resultados del corredor."""
    if encontradas is None:
        encontradas = []
    if isinstance(obj, list):
        if obj and all(isinstance(item, dict) for item in obj):
            encontradas.append(obj)
        for item in obj:
            _buscar_listas_de_diccionarios(item, encontradas)
    elif isinstance(obj, dict):
        for value in obj.values():
            _buscar_listas_de_diccionarios(value, encontradas)
    return encontradas


def scrapear_tiempos_paso(url):
    respuestas_json = []

    def capturar_respuesta(response):
        try:
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                respuestas_json.append(response.json())
        except Exception:
            pass  # respuestas que no son JSON válido (imágenes, css, etc.)

    html_renderizado = None
    with sync_playwright() as p:
        navegador = p.chromium.launch(headless=True)
        pagina = navegador.new_page()
        pagina.on("response", capturar_respuesta)

        pagina.goto(url, wait_until="networkidle", timeout=30000)
        # Pequeño margen extra para llamadas asíncronas tardías
        pagina.wait_for_timeout(2000)

        html_renderizado = pagina.content()
        navegador.close()

    # 1) Buscar en las respuestas JSON capturadas la lista de diccionarios
    #    más "grande" (normalmente es la tabla de splits/resultados)
    candidatas = []
    for data in respuestas_json:
        candidatas.extend(_buscar_listas_de_diccionarios(data))

    if candidatas:
        mejor_lista = max(candidatas, key=len)
        df = pd.DataFrame(mejor_lista)
        if not df.empty:
            return df

    # 2) Respaldo: si no hubo JSON útil, intentar leer tablas del HTML
    #    ya renderizado (por si la web sí pinta una <table> real en el DOM)
    if html_renderizado:
        try:
            tablas = pd.read_html(StringIO(html_renderizado))
            for df in tablas:
                columnas_str = " ".join(df.columns.astype(str)).lower()
                if "clt" in columnas_str or "pass" in columnas_str or "tps" in columnas_str or "km" in columnas_str:
                    return df
            if tablas:
                return tablas[0]
        except ValueError:
            pass  # no había ninguna tabla ni siquiera en el HTML renderizado

    return None

# 3. Lógica principal
if archivo_gpx is not None:
    df_gpx = procesar_gpx_avanzado(archivo_gpx)
    
    # Clasificación geométrica de cada punto del sendero
    def clasificar_terreno(row):
        if row["Pendiente (%)"] >= umbral_subida:
            return "Muro Subida (IEPE)"
        elif row["Pendiente (%)"] <= umbral_bajada:
            return "Bajada Crítica (TDE)"
        else:
            return "Terreno Mixto / Llano"
            
    df_gpx["Tipo Terreno"] = df_gpx.apply(clasificar_terreno, axis=1)
    
    # Métricas resumen extraídas
    total_km = df_gpx["Distancia (km)"].max()
    km_muro = df_gpx[df_gpx["Tipo Terreno"] == "Muro Subida (IEPE)"].shape[0] * (total_km / len(df_gpx))
    km_bajada = df_gpx[df_gpx["Tipo Terreno"] == "Bajada Crítica (TDE)"].shape[0] * (total_km / len(df_gpx))
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Distancia Total", f"{total_km:.2f} km")
    col2.metric("Total Muros Escaneados (>15%)", f"{km_muro:.2f} km")
    col3.metric("Total Bajadas Críticas (<-12%)", f"{km_bajada:.2f} km")
    
    # 4. Renderizar Gráfica con Clasificación de Colores Multi-Capa
    st.markdown("---")
    st.subheader("📈 Mapa de Esfuerzo Biomecánico del Relieve")
    st.write("El motor ha aislado los tramos genéricos de la carrera basados exclusivamente en la inclinación física:")
    
    fig = go.Figure()
    
    # Capa base: El relieve completo en gris oscuro
    fig.add_trace(go.Scatter(
        x=df_gpx["Distancia (km)"], y=df_gpx["Altitud (m)"],
        mode='lines', name='Perfil Base',
        line=dict(color='#444444', width=1.5)
    ))
    
    # Resaltar en Rojo: Zonas IEPE
    df_muros = df_gpx.copy()
    df_muros.loc[df_muros["Tipo Terreno"] != "Muro Subida (IEPE)", "Altitud (m)"] = None
    fig.add_trace(go.Scatter(
        x=df_muros["Distancia (km)"], y=df_muros["Altitud (m)"],
        mode='lines', name='Zonas de Power Hiking (IEPE)',
        line=dict(color='#ff4b4b', width=3.5)
    ))
    
    # Resaltar en Azul: Zonas TDE
    df_bajadas = df_gpx.copy()
    df_bajadas.loc[df_bajadas["Tipo Terreno"] != "Bajada Crítica (TDE)", "Altitud (m)"] = None
    fig.add_trace(go.Scatter(
        x=df_bajadas["Distancia (km)"], y=df_bajadas["Altitud (m)"],
        mode='lines', name='Zonas Daño Cuádriceps (TDE)',
        line=dict(color='#00bfff', width=3.5)
    ))
    
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Distancia (Kilómetros)",
        yaxis_title="Altitud (Metros)",
        height=450,
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Guardamos el DataFrame enriquecido
    st.session_state['df_gpx_analytics'] = df_gpx

else:
    st.info("👋 El escáner geométrico está listo. Cargá el GPX de tu carrera en la barra lateral para identificar los sectores críticos.")

# NUEVO: Bloque de procesamiento del Atleta vía Scraping.
# Independiente del GPX: se dispara solo al hacer clic en el botón.
if boton_cargar:
    st.markdown("---")
    st.subheader("⏱️ Telemetría Oficial del Corredor Extraída por Scraping")

    if not url_corredor:
        st.warning("Pegá primero un link válido en la barra lateral antes de hacer clic en el botón.")
    else:
        with st.spinner("Conectando con la plataforma de cronometraje (esto puede tardar 10-20 seg)..."):
            try:
                df_atleta = scrapear_tiempos_paso(url_corredor)
                error_detalle = None
            except Exception:
                df_atleta = None
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
            st.dataframe(df_atleta, use_container_width=True)