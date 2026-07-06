import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import gpxpy
import re
import traceback
import requests

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
url_corredor = st.sidebar.text_input("2. Pegar Link del Corredor (UTMB Live)", 
                                     placeholder="https://live.utmb.world/aranbyutmb/2026/runners/5")
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


def scrapear_tiempos_paso(url):
    corredor_id = extraer_id_corredor(url)
    if not corredor_id:
        raise ValueError(
            "No pude encontrar el ID del corredor en esa URL. "
            "Verificá que tenga el formato '.../runners/<numero>' "
            "(ej: https://live.utmb.world/aranbyutmb/2026/runners/5)."
        )

    api_url = f"https://utmblive-api.utmb.world/runners/{corredor_id}?locale=en"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

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

    return info_general, df_passings

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