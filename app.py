import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import gpxpy

# 1. Configuración de la interfaz estilo VertLabs
st.set_page_config(page_title="VertLabs - Trail Analytics", page_icon="🏃‍♂️", layout="wide")

st.title("🏃‍♂️ VertLabs - Motor de Segmentación Geométrica")
st.markdown("---")

# Barra lateral para la carga
st.sidebar.header("⚙️ 1. Entrada Topográfica")
archivo_gpx = st.sidebar.file_uploader("Subir track de la carrera (.gpx)", type=["gpx"])

# Sección de Scraping Automatizado
st.sidebar.markdown("---")
st.sidebar.header("📊 2. Datos de Rendimiento")
url_corredor = st.sidebar.text_input("Pegar Link del Corredor (UTMB Live / LiveTrail)", 
                                     placeholder="https://live.utmb.world/...")

# NUEVO: Botón explícito para accionar la extracción
btn_scraping = st.sidebar.button("Extraer Tiempos 🚀")

# ---------------------------------------------------------
# AJUSTE ESTRATÉGICO: Índices Universales Hardcodeados al 12%
# ---------------------------------------------------------
UMBRAL_SUBIDA = 12.0
UMBRAL_BAJADA = -12.0

# Función del Motor Geométrico
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
                    pendiente = (desnivel / dist) * 100 if dist > 0 else 0
                    
                    puntos_datos.append({
                        "Distancia (km)": distancia_acumulada / 1000.0,
                        "Altitud (m)": punto.elevation,
                        "Pendiente (%)": pendiente
                    })
                punto_previo = punto
    return pd.DataFrame(puntos_datos)

# Función de Web Scraping automático
def scrapear_tiempos_paso(url):
    try:
        # pd.read_html lee todas las tablas de una página web en un solo paso
        tablas = pd.read_html(url)
        if not tablas:
            return None
        
        # Buscamos la tabla que contenga palabras clave típicas de cronometraje
        for df in tablas:
            columnas_str = " ".join(df.columns.astype(str)).lower()
            if "clt" in columnas_str or "pass" in columnas_str or "tps" in columnas_str or "km" in columnas_str:
                return df
        return tablas[0]
    except Exception as e:
        st.sidebar.error(f"Error al scrapear el link: Verifica que sea público. Detalle: {e}")
        return None

# Lógica del botón de Scraping (Guardamos en Session State para no perderlo)
if btn_scraping and url_corredor:
    with st.spinner("Conectando con la plataforma de cronometraje..."):
        df_atleta = scrapear_tiempos_paso(url_corredor)
        if df_atleta is not None:
            st.session_state['df_atleta'] = df_atleta
        else:
            st.sidebar.warning("No pudimos extraer una tabla válida de ese enlace.")

# 3. Lógica principal del Relieve
if archivo_gpx is not None:
    df_gpx = procesar_gpx_avanzado(archivo_gpx)
    
    # Clasificación geométrica estandarizada al 12%
    def clasificar_terreno(row):
        if row["Pendiente (%)"] >= UMBRAL_SUBIDA:
            return "Muro Subida (IEPE)"
        elif row["Pendiente (%)"] <= UMBRAL_BAJADA:
            return "Bajada Crítica (TDE)"
        else:
            return "Terreno Mixto / Llano"
            
    df_gpx["Tipo Terreno"] = df_gpx.apply(clasificar_terreno, axis=1)
    
    # Métricas resumen
    total_km = df_gpx["Distancia (km)"].max()
    km_muro = df_gpx[df_gpx["Tipo Terreno"] == "Muro Subida (IEPE)"].shape[0] * (total_km / len(df_gpx))
    km_bajada = df_gpx[df_gpx["Tipo Terreno"] == "Bajada Crítica (TDE)"].shape[0] * (total_km / len(df_gpx))
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Distancia Total", f"{total_km:.2f} km")
    col2.metric("Total Muros Escaneados (>12%)", f"{km_muro:.2f} km")
    col3.metric("Total Bajadas Críticas (<-12%)", f"{km_bajada:.2f} km")
    
    # Renderizar Gráfica
    st.markdown("---")
    st.subheader("📈 Mapa de Esfuerzo Biomecánico del Relieve")
    
    fig = go.Figure()
    
    # Capa base: Relieve gris
    fig.add_trace(go.Scatter(
        x=df_gpx["Distancia (km)"], y=df_gpx["Altitud (m)"],
        mode='lines', name='Perfil Base',
        line=dict(color='#444444', width=1.5)
    ))
    
    # Capa Roja: Zonas IEPE (>12%)
    df_muros = df_gpx.copy()
    df_muros.loc[df_muros["Tipo Terreno"] != "Muro Subida (IEPE)", "Altitud (m)"] = None
    fig.add_trace(go.Scatter(
        x=df_muros["Distancia (km)"], y=df_muros["Altitud (m)"],
        mode='lines', name='Zonas de Power Hiking (IEPE)',
        line=dict(color='#ff4b4b', width=3.5)
    ))
    
    # Capa Azul: Zonas TDE (<-12%)
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
else:
    st.info("👋 ¡Bienvenido a VertLabs! Cargá el GPX de tu carrera en la barra lateral para iniciar.")

# 4. Mostrar la tabla scrapeada si existe en memoria
if 'df_atleta' in st.session_state:
    st.markdown("---")
    st.subheader("⏱️ Telemetría Oficial del Corredor (Validación en Pantalla)")
    
    df_atleta = st.session_state['df_atleta']
    
    st.success(f"¡Tabla de tiempos de paso obtenida con éxito! Se procesaron {len(df_atleta)} registros.")
    
    # Mostramos las métricas crudas y listado limpio de checkpoints
    col_tabla, col_info = st.columns([2, 1])
    with col_tabla:
        st.dataframe(df_atleta, use_container_width=True)
        
    with col_info:
        st.markdown("💡 **Checkpoints Detectados:**")
        # Toma la primera columna asumiendo que contiene los nombres o pases
        nombres_checkpoints = df_atleta.iloc[:, 0].dropna().unique()
        for cp in nombres_checkpoints:
            st.code(f"📍 {cp}")