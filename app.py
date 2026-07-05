import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import gpxpy

# 1. Configuración de la interfaz estilo VertLabs
st.set_page_config(page_title="VertLabs - Trail Analytics", page_icon="🏃‍♂️", layout="wide")

st.title("🏃‍♂️ VertLabs - Trail Running Analytics")
st.markdown("---")

# 2. Barra lateral para la carga del archivo GPX real
st.sidebar.header("⚙️ Entrada de Datos Genéricos")
archivo_gpx = st.sidebar.file_uploader("Subir track de la carrera (.gpx)", type=["gpx"])

st.sidebar.markdown("---")
st.sidebar.write("Subí el mapa de tu carrera (ej. Saint-Jacques) para inicializar el análisis topográfico.")

# 3. Función del Motor Geométrico para procesar el GPX
def procesar_gpx(file):
    gpx = gpxpy.parse(file)
    puntos_datos = []
    distancia_acumulada = 0.0
    punto_previo = None
    
    for track in gpx.tracks:
        for segment in track.segments:
            for punto in segment.points:
                if punto_previo:
                    # Calcular distancia en metros entre puntos consecutivos
                    dist = punto.distance_2d(punto_previo)
                    distancia_acumulada += dist
                    
                    # Calcular pendiente instantánea (%)
                    desnivel = punto.elevation - punto_previo.elevation
                    pendiente = (desnivel / dist) * 100 if dist > 0 else 0
                    
                    puntos_datos.append({
                        "Distancia (km)": distancia_acumulada / 1000.0,
                        "Altitud (m)": punto.elevation,
                        "Pendiente (%)": pendiente
                    })
                punto_previo = punto
    return pd.DataFrame(puntos_datos)

# 4. Flujo lógico de la aplicación según la carga
if archivo_gpx is not None:
    st.success("¡Archivo GPX procesado correctamente por el motor de VertLabs!")
    
    # Procesar datos reales
    df_gpx = procesar_gpx(archivo_gpx)
    
    # Mostrar resumen básico extraído geométricamente
    total_km = df_gpx["Distancia (km)"].max()
    max_alt = df_gpx["Altitud (m)"].max()
    min_alt = df_gpx["Altitud (m)"].min()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Distancia Geométrica Total", f"{total_km:.2f} km")
    col2.metric("Altitud Máxima", f"{max_alt:.0f} msnm")
    col3.metric("Altitud Mínima", f"{min_alt:.0f} msnm")
    
    # 5. Renderizar perfil de altimetría real e interactivo
    st.markdown("---")
    st.subheader("📈 Perfil Altimétrico Real de la Carrera")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_gpx["Distancia (km)"], 
        y=df_gpx["Altitud (m)"],
        mode='lines',
        name='Relieve',
        line=dict(color='#00ffcc', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 255, 220, 0.1)'
    ))
    
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Distancia Recorrida (Kilómetros)",
        yaxis_title="Altitud sobre el nivel del mar (Metros)",
        height=400,
        hovermode="x"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Guardamos el DataFrame en el estado de la sesión para los siguientes pasos
    st.session_state['df_gpx'] = df_gpx

else:
    st.info("👋 ¡Bienvenido a VertLabs! Por favor, carga un archivo `.gpx` en el menú lateral para ver la altimetría real e iniciar el cálculo de los índices.")