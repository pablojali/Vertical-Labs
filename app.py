import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# 1. Configuración de la interfaz
st.set_page_config(page_title="Trail Vertical Labs Stats", page_icon="🏃‍♂️", layout="wide")

st.title("🏃‍♂️ Trail Running - Vertical Labs (Beta)")
st.markdown("---")

st.sidebar.header("Carga de Datos")
carrera = st.sidebar.selectbox("Selecciona la Carrera", ["Trail du Saint-Jacques", "Val d'Aran by UTMB"])
modo_input = st.sidebar.radio("Método de Análisis", ["Splits Públicos (UTMB Live)", "Archivo de Reloj (.GPX / .FIT)"])

st.sidebar.markdown("---")
st.sidebar.write("Desarrollado para análisis biomecánico avanzado.")

# 2. Simulación de los Índices Genéricos que diseñamos
st.subheader(f"📊 Análisis de Índices Universales: {carrera}")

metricas = {
    "Índice Genérico NextGen": [
        "IEPE (Eficiencia en Pendiente >15%)", 
        "TDE (Tasa de Daño Excéntrico en Bajada)", 
        "ICF (Consistencia del Ritmo GAP)", 
        "IRT (Tracción en Terreno Técnico)"
    ],
    "Tu Rendimiento (Base)": ["74.5% (Progreso Sostenido)", "12.4% (Fatiga Moderada)", "6:15 min/km promedio", "Balanced Cruiser"],
    "Saga Rueda (Referencia Élite)": ["91.2% (Élite Power Hiking)", "0.8% (Resiliencia Perfecta)", "4:55 min/km promedio", "Technical Flyer"]
}

df = pd.DataFrame(metricas)
st.dataframe(df, use_container_width=True)

# 3. Gráfica Interactiva
st.markdown("---")
st.subheader("📈 Curva de Degradación o Separación Teórica")

km_carrera = [0, 10, 25, 40, 60, 80, 100, 120]
desviacion_ritmo = [0, 5, 12, 18, 22, 35, 42, 48] # Simulación de pérdida de ritmo por fatiga

fig = go.Figure()
fig.add_trace(go.Scatter(x=km_carrera, y=desviacion_ritmo, mode='lines+markers', line=dict(color='#ff4b4b', width=3)))
fig.update_layout(template="plotly_dark", xaxis_title="Kilómetros", yaxis_title="Degradación del Ritmo (Segundos/km)")
st.plotly_chart(fig, use_container_width=True)
