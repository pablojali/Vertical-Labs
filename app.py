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
archivo_gpx = st.sidebar.file_uploader("Subir track de la carrera (.gpx)", type=["gpx"])

# Umbrales configurables para hacerlo 100% genérico
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