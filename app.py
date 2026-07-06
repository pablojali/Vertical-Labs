import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import gpxpy

# 1. Configuración de la interfaz estilo VertLabs
st.set_page_config(page_title="VertLabs - Trail Analytics", page_icon="🏃‍♂️", layout="wide")

st.title("🏃‍♂️ VertLabs - Motor de Segmentación Geométrica")
st.markdown("---")

# Barra lateral para la carga
st.sidebar.header("⚙️ Entrada de Datos")
archivo_gpx = st.sidebar.file_uploader("1. Subir track de la carrera (.gpx)", type=["gpx"])

# NUEVO: Sección de Scraping Automatizado para Redes Sociales
st.sidebar.markdown("---")
st.sidebar.header("📊 Datos de Rendimiento (Post-Carrera)")
url_corredor = st.sidebar.text_input("2. Pegar Link del Corredor (UTMB Live / LiveTrail)", 
                                     placeholder="https://livetrail.net/...")

# Umbrales configurables para hacerlo 100% genérico
st.sidebar.markdown("---")
st.sidebar.markdown("### Ajuste de Índices Universales")
umbral_subida = st.sidebar.slider("Umbral IEPE - Subida Muro (%)", 10, 25, 15)
umbral_bajada = st.sidebar.slider("Umbral TDE - Bajada Crítica (%)", -25, -5, -12)

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
def scrapear_tiempos_paso(url):
    try:
        # pd.read_html lee todas las tablas de una página web usando requests por detrás
        tablas = pd.read_html(url)
        if not tablas:
            return None
        
        # En la mayoría de los lives de LiveTrail/UTMB, la tabla principal es la primera [0] o segunda [1]
        # Buscamos la tabla que contenga palabras clave típicas de cronometraje
        for df in tablas:
            columnas_str = " ".join(df.columns.astype(str)).lower()
            if "clt" in columnas_str or "pass" in columnas_str or "tps" in columnas_str or "km" in columnas_str:
                return df
        return tablas[0]
    except Exception as e:
        st.sidebar.error(f"Error al scrapear el link: {e}")
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

    # NUEVO: Bloque de procesamiento del Atleta via Scraping
    if url_corredor:
        st.markdown("---")
        st.subheader("⏱️ Telemetría Oficial del Corredor Extraída por Scraping")
        
        with st.spinner("Conectando con la plataforma de cronometraje..."):
            df_atleta = scrapear_tiempos_paso(url_corredor)
        
        if df_atleta is not None:
            st.success("¡Tabla de tiempos de paso obtenida con éxito y sin transcribir nada a mano!")
            
            # Mostramos las métricas crudas para validar
            col_tabla, col_info = st.columns([2, 1])
            with col_tabla:
                st.dataframe(df_atleta, use_container_width=True)
            
            with col_info:
                st.markdown("💡 **Siguiente Paso para tus Post de Redes:**")
                st.info(
                    "Tu backend ya tiene el relieve por metro (GPX) y los checkpoints con sus tiempos "
                    "reales. Ahora tu algoritmo puede cruzar los km de esta tabla con los km del GPX "
                    "para aislar matemáticamente los ritmos en los muros (IEPE) y las bajadas (TDE)."
                )
        else:
            st.warning("No pudimos extraer una tabla válida de ese enlace. Verifica que sea la URL directa del perfil del corredor de la carrera.")

else:
    st.info("👋 El escáner geométrico está listo. Cargá el GPX de tu carrera en la barra lateral para identificar los sectores críticos.")