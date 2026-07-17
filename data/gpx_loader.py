"""
gpx_loader.py

Capa de acceso al catálogo de carreras oficiales de VertLabs.
Lee data/races_registry.json y expone funciones para:
  - listar carreras / años / distancias disponibles
  - resolver la ruta del GPX correspondiente
  - obtener checkpoints guardados para esa combinación
  - construir selectores en cascada para Streamlit (Tab 1)

Jerarquía: carrera -> año -> distancia. Las distancias varían de edición
en edición, así que el año se elige ANTES que la distancia (no al revés).

No modifica session_state ni depende de Streamlit directamente,
salvo la función build_cascading_selector(), que sí es un helper de UI.
"""

import json
import os
from functools import lru_cache

# Ruta al registro, relativa a la raíz del repo (funciona igual en local y en Streamlit Cloud)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_REGISTRY_PATH = os.path.join(_BASE_DIR, "races_registry.json")
_REPO_ROOT = os.path.dirname(_BASE_DIR)  # asume que data/ está en la raíz del repo


@lru_cache(maxsize=1)
def load_registry() -> dict:
    """
    Carga el registro completo desde JSON. Cacheado en memoria: el registro
    solo cambia entre deploys, no dentro de una sesión de usuario.
    """
    if not os.path.exists(_REGISTRY_PATH):
        raise FileNotFoundError(
            f"No se encontró races_registry.json en {_REGISTRY_PATH}. "
            "Verifica que el archivo esté commiteado en el repo."
        )
    with open(_REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_carreras() -> list[tuple[str, str]]:
    """
    Devuelve lista de (slug, nombre_visible) para todas las carreras del registro.
    Ej: [("aran", "Aran by UTMB")]
    """
    registry = load_registry()
    return [(slug, info["nombre"]) for slug, info in registry.items()]


def get_anios(carrera_slug: str) -> list[str]:
    """
    Devuelve los años disponibles para una carrera. Ordenados de más
    reciente a más antiguo.
    """
    registry = load_registry()
    if carrera_slug not in registry:
        return []
    anios = list(registry[carrera_slug]["anios"].keys())
    return sorted(anios, reverse=True)


def get_distancias(carrera_slug: str, anio: str) -> list[str]:
    """
    Devuelve las distancias disponibles (en km, como string) para una
    carrera en un año específico, ya que la oferta de distancias cambia
    de edición en edición. Ordenadas de mayor a menor.
    """
    registry = load_registry()
    try:
        distancias = list(registry[carrera_slug]["anios"][anio].keys())
    except KeyError:
        return []
    return sorted(distancias, key=lambda d: float(d), reverse=True)


def get_carrera_info(carrera_slug: str, anio: str, distancia: str) -> dict:
    """
    Devuelve el bloque completo del registro para esa combinación exacta
    (gpx_file, race_slug_api, checkpoints).
    Lanza KeyError con mensaje claro si la combinación no existe.
    """
    registry = load_registry()
    try:
        return registry[carrera_slug]["anios"][anio][distancia]
    except KeyError:
        raise KeyError(
            f"No hay datos registrados para {carrera_slug} / {anio} / {distancia}K. "
            "Revisa data/races_registry.json."
        )


def get_gpx_path(carrera_slug: str, anio: str, distancia: str) -> str:
    """
    Devuelve la ruta absoluta al archivo GPX, lista para pasar a gpxpy.parse().
    """
    info = get_carrera_info(carrera_slug, anio, distancia)
    rel_path = info["gpx_file"]
    abs_path = os.path.join(_REPO_ROOT, rel_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(
            f"El registro apunta a '{rel_path}' pero el archivo no existe en el repo. "
            "¿Falta subir el GPX o hay un typo en el path?"
        )
    return abs_path


def get_race_slug_api(carrera_slug: str, anio: str, distancia: str) -> str:
    """
    Devuelve el slug que debe usarse para construir el header X-Tenant
    al consultar la API de UTMB Live (puede diferir del slug interno).
    """
    info = get_carrera_info(carrera_slug, anio, distancia)
    return info.get("race_slug_api", carrera_slug)


def get_checkpoints(carrera_slug: str, anio: str, distancia: str) -> list:
    """
    Devuelve los checkpoints guardados (lista de dicts con id/nombre/km) para
    esa combinación exacta, tal como quedaron cargados a mano en el registro.
    """
    info = get_carrera_info(carrera_slug, anio, distancia)
    return info.get("checkpoints", [])


# ---------------------------------------------------------------------------
# Helper de UI para Streamlit (selector en cascada: carrera -> año -> distancia)
# ---------------------------------------------------------------------------

def build_cascading_selector(st, key_prefix: str = "race_selector"):
    """
    Renderiza 3 selectboxes encadenados en Streamlit y devuelve
    (carrera_slug, anio, distancia) o (None, None, None) si el registro
    está vacío.

    El año va ANTES que la distancia porque las distancias ofrecidas
    cambian de edición en edición.

    Uso en la app:
        import streamlit as st
        from data.gpx_loader import build_cascading_selector, get_gpx_path

        carrera, anio, distancia = build_cascading_selector(st)
        if carrera:
            gpx_path = get_gpx_path(carrera, anio, distancia)
    """
    carreras = get_carreras()
    if not carreras:
        st.warning("No hay carreras registradas en races_registry.json todavía.")
        return None, None, None

    carrera_labels = {slug: nombre for slug, nombre in carreras}
    carrera_slug = st.selectbox(
        "Carrera",
        options=list(carrera_labels.keys()),
        format_func=lambda s: carrera_labels[s],
        key=f"{key_prefix}_carrera",
    )

    anios = get_anios(carrera_slug)
    if not anios:
        st.warning(f"'{carrera_slug}' no tiene años registrados.")
        return carrera_slug, None, None

    anio = st.selectbox(
        "Año",
        options=anios,
        key=f"{key_prefix}_anio",
    )

    distancias = get_distancias(carrera_slug, anio)
    if not distancias:
        st.warning(f"'{carrera_slug}' {anio} no tiene distancias registradas.")
        return carrera_slug, anio, None

    distancia = st.selectbox(
        "Distancia (km)",
        options=distancias,
        key=f"{key_prefix}_distancia",
    )

    return carrera_slug, anio, distancia
