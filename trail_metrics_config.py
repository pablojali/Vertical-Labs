# ==============================================================================
# BETA TRAIL STATS - INDEX & METRIC CONFIGURATION MODULE
# ==============================================================================
# This module contains the finalized English definitions, geometric criteria,
# and mathematical formulas for the core metrics. Ready to paste into app.py.
# ==============================================================================

INDEX_CONFIG = {
    "VPI": {
        "name": "Vertical Power Index",
        "label_es": "Escalada Eficiente",
        "icon": "🧗",
        "description": "Measures pure uphill climbing efficiency, specific power output, and economy during severe vertical gains.",
        "geometric_criterion": "Slope >= 12%",
        "data_source": "Official Organization GPX (segmented by checkpoints)",
        "formula": (
            "VPI = Sum(Elevation Gain (m) in [CP_i, CP_i+1] where Slope >= 12%) / "
            "Sum(Athlete Time (hours) spent in [CP_i, CP_i+1] segments where Slope >= 12%)"
        ),
        "unit": "VAM (Vertical Meters per Hour)",
        "ui_display": {
            "primary": "VPI Score (Vertical Meters / Hour)",
            "contextual": "Top Speed in Crucial Uphill Section"
        }
    },
    "DMI": {
        "name": "Descent Mastery Index",
        "label_es": "Rompepiernas",
        "icon": "📉",
        "description": "Evaluates technical downhill skill, gravity management, and muscular resilience against heavy eccentric loading.",
        "geometric_criterion": "Slope <= -12%",
        "data_source": "Official Organization GPX (segmented by checkpoints)",
        "formula": (
            "DMI = Sum(Distance (km) in [CP_i, CP_i+1] where Slope <= -12%) / "
            "Sum(Athlete Time (hours) spent in [CP_i, CP_i+1] segments where Slope <= -12%)"
        ),
        "unit": "km/h",
        "ui_display": {
            "primary": "DMI Score (Average Downhill Speed)",
            "contextual": "Top Speed in Crucial Downhill Section"
        }
    },
    "ER": {
        "name": "Endurance Rating",
        "label_es": "Degradación por Fatiga",
        "icon": "🏆",
        "description": "Analyzes physiological pacing degradation, stamina, and strategic execution using terrain-adjusted effort-kilometers.",
        "geometric_criterion": "Split at 50% of Total Effort-Kilometers (Km_E) based on master GPX structure",
        "data_source": "Official Checkpoint Splits (UTMB Live) + Master GPX Profile",
        "formula": (
            "Total_Km_E = Total_Distance_km + (Total_Elevation_Gain_m / 100)\n"
            "Effort_Pace = Elapsed_Time_mins / (Segment_Distance_km + (Segment_Gain_m / 100))\n"
            "Pacing_Decay_% = ((Effort_Pace_2nd_Half / Effort_Pace_1st_Half) - 1) * 100\n"
            "ER = 100 - (Pacing_Decay_% * Distance_Weighting_Coefficient)"
        ),
        "unit": "Score (0 - 100)",
        "ui_display": {
            "primary": "Endurance Rating (Stamina Score)",
            "contextual": "Terrain-Adjusted Performance Stability"
        }
    }
}

SPEED_METRICS = {
    "AVERAGE_SPEED": {
        "name": "Global Average Speed",
        "label_es": "Velocidad Promedio General",
        "description": "Total official race distance divided by total elapsed athlete time.",
        "source": "UTMB Live API",
        "formula": "Total Distance (km) / Total Time (hours)",
        "unit": "km/h"
    },
    "TOP_SPEED_CRUCIAL_SECTIONS": {
        "name": "Top Speed in Crucial Sections",
        "label_es": "Velocidad Punta en Zonas Críticas",
        "description": "Average speed achieved within the geometrically most demanding segment isolated by VPI or DMI.",
        "source": "Calculated via Master GPX segments + UTMB Live checkpoint split times",
        "formula": "Segment Distance (km) / Time in Segment (hours)",
        "unit": "km/h"
    }
}

# Example implementation helper for Streamlit UI
def display_metric_documentation(index_key):
    """
    Helper function to render the structured English definitions inside your app.py
    """
    if index_key in INDEX_CONFIG:
        cfg = INDEX_CONFIG[index_key]
        doc_string = f"""
        ### {cfg['icon']} {cfg['name']} ({index_key})
        * **Description:** {cfg['description']}
        * **Geometric Criterion:** `{cfg['geometric_criterion']}`
        * **Data Source:** {cfg['data_source']}
        * **Mathematical Formula:** `{cfg['formula']}`
        * **Primary Display:** {cfg['ui_display']['primary']}
        """
        return doc_string
    return "Metric not found."