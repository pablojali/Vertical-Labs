# =========================================================================
    # BLOQUE DE CONFIRMACIÓN VISUAL: PROCESAMIENTO DEL ATLETA VIA SCRAPING
    # =========================================================================
    if url_corredor:
        st.markdown("---")
        st.subheader("⏱️ Validando Telemetría del Corredor (Live Scraping)")
        
        with st.spinner("Leyendo tablas de tiempos desde la plataforma oficial..."):
            df_atleta = scrapear_tiempos_paso(url_corredor)
        
        if df_atleta is not None:
            # Mensaje de éxito llamativo
            st.success(f"✅ ¡Datos vinculados con éxito! Se detectaron {len(df_atleta)} puntos de control/splits.")
            
            # Dividimos en 2 columnas: la tabla a la izquierda y un resumen rápido de control a la derecha
            col_tabla, col_resumen = st.columns([2, 1])
            
            with col_tabla:
                st.markdown("**Tabla Completa Extraída:**")
                # st.dataframe muestra la tabla nativa de pandas donde puedes ordenar y revisar los datos
                st.dataframe(df_atleta, use_container_width=True)
            
            with col_resumen:
                st.markdown("**🔍 Checkpoints Detectados:**")
                # Intentamos mostrar la columna de nombres de los checkpoints si existe (suele llamarse 'Secteur', 'Emplacement', 'Passage' o similar)
                columna_nombres = [c for c in df_atleta.columns if any(p in c.lower() for p in ['sect', 'empl', 'pass', 'lugar', 'check', 'nom'])]
                
                if columna_nombres:
                    lista_puntos = df_atleta[columna_nombres[0]].dropna().tolist()
                    st.code("\n".join([f"📍 {p}" for p in lista_puntos]), language="text")
                else:
                    st.info("Estructura de columnas variable detectada. Revisa la tabla de la izquierda para confirmar los nombres.")
                
                st.markdown("---")
                st.markdown("💡 **Estado del Motor:** Listo para cruzar las distancias de estos puntos con el relieve de tu GPX.")
        else:
            st.error("❌ Error de lectura: No se encontró una tabla de splits válida en esa URL. Asegúrate de que sea el link directo al perfil de carrera del atleta.")