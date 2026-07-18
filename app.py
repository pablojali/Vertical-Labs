import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import gpxpy
import re
import traceback
import requests
from trail_metrics_config import INDEX_CONFIG, SPEED_METRICS, display_metric_documentation
from data.gpx_loader import (
    build_cascading_selector,
    get_gpx_path,
    get_checkpoints,
    get_carrera_info,
    get_carreras,
)

# 1. Page configuration - VertLabs style
st.set_page_config(page_title="VertLabs - Trail Analytics", page_icon="🏃‍♂️", layout="wide")

st.title("🏃‍♂️ VertLabs - Race Analysis Engine")
st.caption("Analysis backend: GPX geometric segmentation + official runner metrics.")
st.markdown("---")

# Fixed thresholds (hardcoded to keep the app simple).
STRONG_SLOPE_THRESHOLD = 12       # >= 12% strong climb | <= -12% strong descent
MODERATE_SLOPE_MIN = 5            # between 5% and 12% = moderate climb/descent
MODERATE_SLOPE_MAX = 12
ALTITUDE_THRESHOLD = 1800         # meters above sea level

# Consistent color palette shared by the effort map and the bar chart
SLOPE_CATEGORY_COLORS = {
    "Strong Climb (≥12%)": "#ff4b4b",
    "Strong Descent (≤-12%)": "#00bfff",
    "Moderate Climb (5-12%)": "#ffa500",
    "Moderate Descent (-5 to -12%)": "#7dd3fc",
    "Rolling Terrain (-5 to +5%)": "#4ade80",
}
SLOPE_CATEGORY_ORDER = list(SLOPE_CATEGORY_COLORS.keys())


# ============================================================
# 2. ENGINE FUNCTIONS (backend) - no UI logic
# ============================================================

def process_gpx_advanced(file):
    """Parses a GPX and returns a point-by-point DataFrame with
    cumulative distance, elevation, and instantaneous slope."""
    gpx = gpxpy.parse(file)
    points_data = []
    cumulative_distance = 0.0
    previous_point = None

    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                if previous_point:
                    dist = point.distance_2d(previous_point)
                    cumulative_distance += dist

                    elevation_change = point.elevation - previous_point.elevation
                    # Avoid division by zero on identical points
                    slope = (elevation_change / dist) * 100 if dist > 0 else 0

                    points_data.append({
                        "Distance (km)": cumulative_distance / 1000.0,
                        "Elevation (m)": point.elevation,
                        "Slope (%)": slope
                    })
                previous_point = point
    return pd.DataFrame(points_data)


def classify_slope(slope):
    """Mutually exclusive categories covering the whole range with no
    gaps: strong -> moderate -> rolling."""
    if slope >= STRONG_SLOPE_THRESHOLD:
        return "Strong Climb (≥12%)"
    elif slope <= -STRONG_SLOPE_THRESHOLD:
        return "Strong Descent (≤-12%)"
    elif slope > MODERATE_SLOPE_MIN:
        return "Moderate Climb (5-12%)"
    elif slope < -MODERATE_SLOPE_MIN:
        return "Moderate Descent (-5 to -12%)"
    else:
        return "Rolling Terrain (-5 to +5%)"


def analyze_race(gpx_file):
    """Full geometric analysis engine: receives the GPX file and returns
    the enriched DataFrame with slope and altitude classification, ready
    to plot/display."""
    df_gpx = process_gpx_advanced(gpx_file)

    df_gpx["Slope Type"] = df_gpx["Slope (%)"].apply(classify_slope)

    # Classification by ALTITUDE (independent of slope: a segment can be
    # "Strong Climb" AND "Above 1800m" at the same time)
    df_gpx["Altitude Zone"] = df_gpx["Elevation (m)"].apply(
        lambda alt: f"Above {ALTITUDE_THRESHOLD}m" if alt > ALTITUDE_THRESHOLD else f"Below {ALTITUDE_THRESHOLD}m"
    )
    return df_gpx


def resample_for_chart(df_gpx, step_m=200):
    """Downsamples the point-by-point GPX (often 10k+ GPS points) into
    fixed-distance bins (default 200m) purely for plotting. This does NOT
    affect any of the underlying analysis/index calculations, which keep
    using the full-resolution df_gpx - only the chart gets lighter."""
    df = df_gpx.copy()
    df["bin"] = (df["Distance (km)"] * 1000 // step_m).astype(int)

    def _dominant_category(series):
        mode = series.mode()
        return mode.iat[0] if not mode.empty else series.iloc[0]

    resampled = df.groupby("bin").agg(**{
        "Distance (km)": ("Distance (km)", "mean"),
        "Elevation (m)": ("Elevation (m)", "mean"),
        "Slope Type": ("Slope Type", _dominant_category),
    }).reset_index(drop=True)
    return resampled


def match_checkpoints_with_gpx(df_gpx, checkpoints_km):
    """Receives the already-analyzed GPX DataFrame and a list of checkpoints
    [{'point': int, 'km': float}, ...] and returns a DataFrame with one row
    per segment between consecutive checkpoints: real distance, positive/
    negative elevation change, and average slope, based on the official
    GPX terrain in that km range.

    This is later crossed with the runner's real split times (Tab 2) to
    compute VPI, DMI and ER."""
    sorted_checkpoints = sorted(checkpoints_km, key=lambda c: c["km"])
    rows = []

    for i in range(len(sorted_checkpoints) - 1):
        cp_start = sorted_checkpoints[i]
        cp_end = sorted_checkpoints[i + 1]

        segment = df_gpx[
            (df_gpx["Distance (km)"] >= cp_start["km"]) &
            (df_gpx["Distance (km)"] <= cp_end["km"])
        ]

        if segment.empty:
            elevation_gain = None
            elevation_loss = None
            avg_slope = None
        else:
            elevation_diffs = segment["Elevation (m)"].diff().dropna()
            elevation_gain = elevation_diffs[elevation_diffs > 0].sum()
            elevation_loss = elevation_diffs[elevation_diffs < 0].sum()  # stays negative
            avg_slope = segment["Slope (%)"].mean()

        rows.append({
            "Start Point": cp_start["point"],
            "End Point": cp_end["point"],
            "Start Km": cp_start["km"],
            "End Km": cp_end["km"],
            "Segment Distance (km)": round(cp_end["km"] - cp_start["km"], 3),
            "Elevation Gain (m)": round(elevation_gain, 1) if elevation_gain is not None else None,
            "Elevation Loss (m)": round(elevation_loss, 1) if elevation_loss is not None else None,
            "Average Slope (%)": round(avg_slope, 2) if avg_slope is not None else None,
        })

    return pd.DataFrame(rows)


def parse_time_to_hours(time_str):
    """Converts a 'H:MM:SS' (or 'HH:MM:SS') string into decimal hours.
    If empty/None (e.g. the start checkpoint), treated as 0."""
    if pd.isna(time_str) or time_str is None or str(time_str).strip() == "":
        return 0.0
    parts = [int(p) for p in str(time_str).split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    hours, minutes, seconds = parts[-3], parts[-2], parts[-1]
    return hours + minutes / 60 + seconds / 3600


def calculate_total_elevation_gain(full_df_gpx):
    """Total elevation gain of the WHOLE course (not just the segments
    between checkpoints), used for Total_Km_E in the ER index."""
    diffs = full_df_gpx["Elevation (m)"].diff().dropna()
    return diffs[diffs > 0].sum()


def calculate_runner_indices(df_segments, df_runner, total_km, total_elevation_gain,
                              distance_weighting_coef=1.0):
    """Crosses the official race segments (df_segments, computed in Tab 1
    from the checkpoints) with the runner's real split times (df_runner)
    to calculate VPI, DMI and ER.

    The crossing is done by checkpoint number ('Point'), which must match
    between both tables."""

    if "Point" not in df_runner.columns:
        raise ValueError("The runner table doesn't have a 'Point' column to match checkpoints.")

    time_by_point = {
        row["Point"]: parse_time_to_hours(row.get("Cumulative Time"))
        for _, row in df_runner.iterrows()
    }

    crossed_rows = []
    for _, seg in df_segments.iterrows():
        p_start, p_end = seg["Start Point"], seg["End Point"]
        if p_start in time_by_point and p_end in time_by_point:
            runner_time_h = time_by_point[p_end] - time_by_point[p_start]
        else:
            runner_time_h = None
        row = seg.to_dict()
        row["Runner Time (h)"] = runner_time_h
        crossed_rows.append(row)

    df_crossed = pd.DataFrame(crossed_rows)
    unmatched_segments = df_crossed["Runner Time (h)"].isna().sum()

    df_valid = df_crossed.dropna(subset=["Runner Time (h)"]).copy()
    df_valid = df_valid[df_valid["Runner Time (h)"] > 0]

    if df_valid.empty:
        raise ValueError(
            "No checkpoint from the saved race matches the runner's points. "
            "Check that the 'Point' numbers are the same in both tables."
        )

    # --- VPI: Strong Climb (segment average slope >= 12%) ---
    strong_climb_segments = df_valid[df_valid["Average Slope (%)"] >= STRONG_SLOPE_THRESHOLD]
    climb_time_h = strong_climb_segments["Runner Time (h)"].sum()
    climb_gain_m = strong_climb_segments["Elevation Gain (m)"].sum()
    vpi = (climb_gain_m / climb_time_h) if climb_time_h > 0 else None

    # --- DMI: Strong Descent (segment average slope <= -12%) ---
    strong_descent_segments = df_valid[df_valid["Average Slope (%)"] <= -STRONG_SLOPE_THRESHOLD]
    descent_time_h = strong_descent_segments["Runner Time (h)"].sum()
    descent_dist_km = strong_descent_segments["Segment Distance (km)"].sum()
    dmi = (descent_dist_km / descent_time_h) if descent_time_h > 0 else None

    # --- ER: Endurance Rating (pacing decay between 1st and 2nd half) ---
    total_km_e = total_km + (total_elevation_gain / 100)
    df_valid = df_valid.sort_values("Start Km").reset_index(drop=True)
    df_valid["Effort Km Segment"] = df_valid["Segment Distance (km)"] + (
        df_valid["Elevation Gain (m)"].fillna(0) / 100
    )
    df_valid["Effort Km Cumulative"] = df_valid["Effort Km Segment"].cumsum()
    half_effort_km = total_km_e / 2

    first_half = df_valid[df_valid["Effort Km Cumulative"] <= half_effort_km]
    second_half = df_valid[df_valid["Effort Km Cumulative"] > half_effort_km]

    def _effort_pace(segments):
        time_min = segments["Runner Time (h)"].sum() * 60
        effort_km = segments["Effort Km Segment"].sum()
        return (time_min / effort_km) if effort_km > 0 else None

    pace_1 = _effort_pace(first_half)
    pace_2 = _effort_pace(second_half)

    if pace_1 and pace_2 and pace_1 > 0:
        pacing_decay_pct = ((pace_2 / pace_1) - 1) * 100
        er = 100 - (pacing_decay_pct * distance_weighting_coef)
    else:
        pacing_decay_pct = None
        er = None

    result = {
        "VPI": round(vpi, 1) if vpi is not None else None,
        "DMI": round(dmi, 2) if dmi is not None else None,
        "ER": round(er, 1) if er is not None else None,
        "Pacing_Decay_%": round(pacing_decay_pct, 1) if pacing_decay_pct is not None else None,
        "unmatched_segments": int(unmatched_segments),
    }
    return result, df_valid


def calculate_indices_by_segment(full_df_gpx, df_segments, df_runner):
    """Calculates VPI and DMI INDEPENDENTLY for each segment (degradation
    matrix), instead of one global value for the whole race.

    For each segment between 2 checkpoints:
    - Crops the sub-GPX of that segment (by km range).
    - VPI: filters only points with slope >= 12% within the segment and
      sums their real elevation gain, divided by the time the runner spent
      in that specific segment.
    - DMI: filters only points with slope <= -12% within the segment and
      sums the real distance covered in those points, divided by the same
      time.

    Finally normalizes both indices against the runner's Segment 1
    (Segment 1 = 100), to plot the degradation curve on a comparable
    0-100 scale."""

    full_df_gpx = full_df_gpx.sort_values("Distance (km)").reset_index(drop=True)
    # Reconstruct the point-to-point "mini-segment" (distance and elevation
    # change between each consecutive GPX point) from the cumulative
    # columns we already have.
    incremental_dist_m = full_df_gpx["Distance (km)"].diff() * 1000
    incremental_elevation_m = full_df_gpx["Elevation (m)"].diff()

    time_by_point = {
        row["Point"]: parse_time_to_hours(row.get("Cumulative Time"))
        for _, row in df_runner.iterrows()
    }

    sorted_segments = df_segments.sort_values("Start Km").reset_index(drop=True)
    rows = []

    for i, seg in sorted_segments.iterrows():
        p_start, p_end = seg["Start Point"], seg["End Point"]
        km_start, km_end = seg["Start Km"], seg["End Km"]

        if p_start not in time_by_point or p_end not in time_by_point:
            segment_time_h = None
        else:
            segment_time_h = time_by_point[p_end] - time_by_point[p_start]

        segment_mask = (full_df_gpx["Distance (km)"] >= km_start) & (full_df_gpx["Distance (km)"] <= km_end)

        vpi_raw, dmi_raw = None, None
        if segment_time_h and segment_time_h > 0:
            strong_climb_mask = segment_mask & (full_df_gpx["Slope (%)"] >= STRONG_SLOPE_THRESHOLD)
            vpi_gain_m = incremental_elevation_m[strong_climb_mask].sum()
            vpi_raw = vpi_gain_m / segment_time_h if vpi_gain_m > 0 else None

            strong_descent_mask = segment_mask & (full_df_gpx["Slope (%)"] <= -STRONG_SLOPE_THRESHOLD)
            dmi_dist_km = incremental_dist_m[strong_descent_mask].sum() / 1000
            dmi_raw = dmi_dist_km / segment_time_h if dmi_dist_km > 0 else None

        rows.append({
            "Segment": f"P{p_start}→P{p_end}",
            "Start Km": km_start,
            "End Km": km_end,
            "Runner Time (h)": round(segment_time_h, 2) if segment_time_h is not None else None,
            "VPI Raw (m/h)": round(vpi_raw, 1) if vpi_raw is not None else None,
            "DMI Raw (km/h)": round(dmi_raw, 2) if dmi_raw is not None else None,
        })

    df_segments_out = pd.DataFrame(rows)

    # Normalization against the runner's first valid segment (Segment 1 = 100)
    def _normalize(series):
        valid_values = series.dropna()
        if valid_values.empty:
            return pd.Series([None] * len(series), index=series.index)
        baseline = valid_values.iloc[0]
        if not baseline:
            return pd.Series([None] * len(series), index=series.index)
        return (series / baseline) * 100

    df_segments_out["VPI Index (0-100)"] = _normalize(df_segments_out["VPI Raw (m/h)"]).round(1)
    df_segments_out["DMI Index (0-100)"] = _normalize(df_segments_out["DMI Raw (km/h)"]).round(1)

    return df_segments_out


# Automated timing-data scraping.
# live.utmb.world is a Next.js app: the HTML returned by the server is
# empty and the data is fetched afterwards via JS from an internal API.
# By inspecting the browser's Network tab, we found the real endpoint
# used by the site itself:
#   https://utmblive-api.utmb.world/runners/<ID>?locale=en
# A plain requests.get() is enough: no browser, no Playwright/Chromium,
# no packages.txt needed.

def extract_runner_id(url):
    """Extracts the numeric runner ID from a URL like
    https://live.utmb.world/aranbyutmb/2026/runners/5"""
    match = re.search(r"/runners/(\d+)", url)
    return match.group(1) if match else None


def extract_tenant(url):
    """Extracts 'race_year' from a URL like
    https://live.utmb.world/aranbyutmb/2026/runners/5 -> 'aranbyutmb_2026'
    Required by the API as the X-Tenant header to identify which race
    edition the runner belongs to."""
    match = re.search(r"live\.utmb\.world/([a-zA-Z0-9]+)/(\d{4})/runners/", url)
    if match:
        race, year = match.groups()
        return f"{race}_{year}"
    return None


def scrape_runner_splits(url):
    runner_id = extract_runner_id(url)
    if not runner_id:
        raise ValueError(
            "Couldn't find the runner ID in that URL. "
            "Make sure it has the format '.../runners/<number>' "
            "(e.g. https://live.utmb.world/aranbyutmb/2026/runners/5)."
        )

    tenant = extract_tenant(url)
    if not tenant:
        raise ValueError(
            "Couldn't identify the race/year in that URL. "
            "Make sure it has the format "
            "'https://live.utmb.world/<race>/<year>/runners/<number>'."
        )

    api_url = f"https://utmblive-api.utmb.world/runners/{runner_id}?locale=en"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "*/*",
        "Origin": "https://live.utmb.world",
        "Referer": "https://live.utmb.world/",
        "X-Tenant": tenant,
    }

    response = requests.get(api_url, headers=headers, timeout=15)
    response.raise_for_status()
    data = response.json()

    resume = data.get("resume", {}) or {}
    info = resume.get("info", {}) or {}
    ranking = resume.get("ranking", {}) or {}
    country = data.get("country", {}) or {}

    runner_info = {
        "Name": info.get("fullname"),
        "Bib": resume.get("bib"),
        "Age": info.get("age"),
        "Category": info.get("category"),
        "Club": info.get("club"),
        "Country": country.get("name"),
        "Finish Time": resume.get("raceTime"),
        "Overall Rank": ranking.get("scratch"),
        "Gender Rank": ranking.get("sex"),
        "Category Rank": ranking.get("category"),
        "Status": resume.get("status"),
    }

    passings = (data.get("detail", {}) or {}).get("passings", []) or []
    df_passings = pd.DataFrame(passings)

    # Keep only the useful columns (drop raw/redundant ones like
    # timeSeconds, datetimeIn/Out, and live-prediction fields that don't
    # apply to a finished runner) and rename them in English.
    useful_columns = {
        "pointId": "Point",
        "cumulatedTime": "Cumulative Time",
        "time": "Segment Time",
        "speed": "Speed (km/h)",
        "pace": "Pace (min/km)",
        "rank": "Rank",
        "restTime": "Rest",
    }
    present_columns = [c for c in useful_columns if c in df_passings.columns]
    df_passings = df_passings[present_columns].rename(columns=useful_columns)

    return runner_info, df_passings


# ============================================================
# 3. INTERFACE: three independent tabs
# ============================================================

# In-memory "library" of analyzed races, kept for the duration of the session.
# Structure: { "Race Name": {"df": DataFrame, "total_km": float, ...} }
if 'saved_races' not in st.session_state:
    st.session_state['saved_races'] = {}

tab_race, tab_runner, tab_methodology = st.tabs(
    ["🗺️ Race Analysis", "🏃 Runner Metrics", "📖 Indices & Methodology"]
)

# ---------------------------------------------
# TAB 1: Geometric analysis of the official GPX
# ---------------------------------------------
with tab_race:
    st.header("🗺️ Geometric Race Analysis (GPX)")
    st.caption(
        f"Strong slope: ≥{STRONG_SLOPE_THRESHOLD}% climb / ≤-{STRONG_SLOPE_THRESHOLD}% descent · "
        f"Moderate: {MODERATE_SLOPE_MIN}-{MODERATE_SLOPE_MAX}% · "
        f"Altitude: >{ALTITUDE_THRESHOLD}m"
    )

    race_slug, year, distance = build_cascading_selector(st, key_prefix="tab1_selector")

    if not (race_slug and year and distance):
        st.info("👋 Choose race, year and distance above to load the geometric analysis.")
    else:
        try:
            gpx_path = get_gpx_path(race_slug, year, distance)
            gpx_error = None
        except FileNotFoundError as e:
            gpx_path = None
            gpx_error = str(e)

        if gpx_error:
            st.error(f"❌ {gpx_error}")
        else:
            registry_checkpoints = get_checkpoints(race_slug, year, distance)
            registry_info = get_carrera_info(race_slug, year, distance)
            visible_race_name = dict(get_carreras()).get(race_slug, race_slug)

            # --- Confirmation panel (before running the analysis) ---
            with st.container(border=True):
                st.markdown(f"**GPX found:** `{registry_info['gpx_file']}`")
                colA, colB = st.columns(2)
                colA.metric("Checkpoints in registry", len(registry_checkpoints))
                colB.metric("API slug (X-Tenant)", registry_info.get("race_slug_api", race_slug))

                if registry_checkpoints:
                    # Hide the start and finish checkpoints from this preview
                    # table (they're still used for the actual analysis).
                    checkpoints_to_display = registry_checkpoints
                    if len(checkpoints_to_display) > 2:
                        sorted_cps = sorted(checkpoints_to_display, key=lambda c: c["km"])
                        checkpoints_to_display = sorted_cps[1:-1]

                    st.markdown("**Checkpoints:**")
                    st.dataframe(
                        checkpoints_to_display,
                        column_config={
                            "id": "ID",
                            "nombre": "Name",
                            "km": st.column_config.NumberColumn("Km", format="%.2f"),
                        },
                        hide_index=True,
                        use_container_width=True,
                    )
                else:
                    st.warning(
                        "This combination doesn't have checkpoints in the registry yet. "
                        "The geometric analysis still works, but you won't be able to "
                        "calculate VPI/DMI by segment until they're loaded in "
                        "`data/races_registry.json`."
                    )

            use_race = st.button(
                "✅ Use this race for analysis", type="primary", use_container_width=True
            )
            if use_race:
                st.session_state["active_race_tab1"] = (race_slug, year, distance)

            active_race = st.session_state.get("active_race_tab1")

            if not active_race:
                st.info("Click '✅ Use this race for analysis' to run the geometric engine.")
            elif active_race != (race_slug, year, distance):
                st.info(
                    "You selected a different combination than the one currently active. "
                    "Click '✅ Use this race for analysis' to update it."
                )
            else:
                active_race_slug, active_year, active_distance = active_race
                active_gpx_path = get_gpx_path(active_race_slug, active_year, active_distance)
                active_checkpoints = get_checkpoints(active_race_slug, active_year, active_distance)

                with open(active_gpx_path, "r", encoding="utf-8") as f:
                    df_gpx = analyze_race(f)

                # Summary metrics. Each GPX point represents roughly the
                # same average distance (total_km / point count), so we
                # count points per category and convert them to km.
                total_km = df_gpx["Distance (km)"].max()
                km_per_point = total_km / len(df_gpx)

                km_by_category = {
                    category: (df_gpx["Slope Type"] == category).sum() * km_per_point
                    for category in SLOPE_CATEGORY_ORDER
                }
                km_above_altitude = (df_gpx["Altitude Zone"] == f"Above {ALTITUDE_THRESHOLD}m").sum() * km_per_point

                # --- Top row: Total Distance & Above-altitude, side by side ---
                col1, col2 = st.columns(2)
                col1.metric("Total Distance", f"{total_km:.2f} km")
                col2.metric(f"Above {ALTITUDE_THRESHOLD}m", f"{km_above_altitude:.2f} km")

                # --- Horizontal bar chart: km per slope category, same colors as the map ---
                st.markdown("##### Slope Breakdown")
                fig_bars = go.Figure()
                fig_bars.add_trace(go.Bar(
                    x=[km_by_category[c] for c in SLOPE_CATEGORY_ORDER],
                    y=SLOPE_CATEGORY_ORDER,
                    orientation="h",
                    marker_color=[SLOPE_CATEGORY_COLORS[c] for c in SLOPE_CATEGORY_ORDER],
                    text=[f"{km_by_category[c]:.2f} km" for c in SLOPE_CATEGORY_ORDER],
                    textposition="outside",
                    hovertemplate="%{y}: %{x:.2f} km<extra></extra>",
                ))
                fig_bars.update_layout(
                    template="plotly_dark",
                    xaxis_title="Distance (km)",
                    height=280,
                    margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=False,
                )
                st.plotly_chart(fig_bars, use_container_width=True)

                # --- Effort map (elevation profile colored by slope category) ---
                st.markdown("---")
                st.subheader("📈 Biomechanical Effort Map")
                st.write("The engine isolated the race segments by slope and altitude:")

                df_chart = resample_for_chart(df_gpx, step_m=200)

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=df_chart["Distance (km)"], y=df_chart["Elevation (m)"],
                    mode='lines', name='Base Profile',
                    line=dict(color='#444444', width=1.5),
                    hovertemplate="Km %{x:.1f}<br>%{y:.0f} m<extra></extra>",
                ))

                for category in SLOPE_CATEGORY_ORDER:
                    color = SLOPE_CATEGORY_COLORS[category]
                    width = 3.5 if "Strong" in category else (3.0 if "Moderate" in category else 2.5)
                    df_layer = df_chart.copy()
                    df_layer.loc[df_layer["Slope Type"] != category, "Elevation (m)"] = None
                    fig.add_trace(go.Scatter(
                        x=df_layer["Distance (kms)"], y=df_layer["Elevation (mts)"],
                        mode='lines', name=category,
                        line=dict(color=color, width=width),
                        hovertemplate="Km %{x:.1f}<br>%{y:.0f} m<extra></extra>",
                    ))

                fig.add_hline(
                    y=ALTITUDE_THRESHOLD,
                    line_dash="dash",
                    line_color="#a78bfa",
                    annotation_text=f"{ALTITUDE_THRESHOLD}m",
                    annotation_position="top left",
                )

                fig.update_layout(
                    template="plotly_dark",
                    xaxis_title="Distance (kms)",
                    yaxis_title="Elevation (mts)",
                    height=450,
                    hovermode="closest"
                    
                )
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("View full point-by-point table"):
                    st.dataframe(df_gpx, use_container_width=True)

                st.session_state['df_gpx_analytics'] = df_gpx

                # --- Match registry checkpoints against the GPX ---
                st.markdown("---")
                st.subheader("📍 Official Race Checkpoints")

                valid_checkpoints = []
                invalid_ids = []
                for cp in active_checkpoints:
                    try:
                        valid_checkpoints.append({"point": int(cp["id"]), "km": float(cp["km"])})
                    except (ValueError, TypeError):
                        invalid_ids.append(cp.get("id"))

                if invalid_ids:
                    st.warning(
                        f"Checkpoints with id {invalid_ids} aren't numeric and were excluded "
                        "from matching ('id' must match the UTMB Live 'pointId')."
                    )

                df_segments = None
                if len(valid_checkpoints) >= 2:
                    df_segments = match_checkpoints_with_gpx(df_gpx, valid_checkpoints)
                    st.markdown("##### Matching Preview (segment by segment)")
                    st.dataframe(df_segments, use_container_width=True)
                else:
                    st.info(
                        "This race has fewer than 2 valid numeric checkpoints in the registry: "
                        "the general geometric analysis is still available, but VPI/DMI by "
                        "segment can't be calculated on the 'Runner Metrics' tab."
                    )

                # --- Auto-save to the library (Tab 2) ---
                saved_race_name = f"{visible_race_name} {active_year} - {active_distance}K"
                st.session_state['saved_races'][saved_race_name] = {
                    "df": df_gpx,
                    "total_km": total_km,
                    "km_by_category": km_by_category,
                    "km_above_altitude": km_above_altitude,
                    "checkpoints_km": valid_checkpoints,
                    "df_segments": df_segments,
                }
                st.success(
                    f"✅ Race loaded as **'{saved_race_name}'** — now available on the "
                    "'Runner Metrics' tab."
                )

        if st.session_state['saved_races']:
            with st.expander(f"📚 Races loaded this session ({len(st.session_state['saved_races'])})"):
                for name in st.session_state['saved_races']:
                    n_checkpoints = len(st.session_state['saved_races'][name].get('checkpoints_km', []))
                    st.write(f"- {name} ({n_checkpoints} checkpoints)")

# ---------------------------------------------
# TAB 2: Official runner metrics
# ---------------------------------------------
with tab_runner:
    st.header("🏃 Official Runner Metrics")
    st.caption("Pulls splits, pace and rank directly from the timing platform (UTMB Live).")

    # --- Saved race selector (analyzed in Tab 1) ---
    available_races = st.session_state.get('saved_races', {})
    if not available_races:
        st.warning(
            "⚠️ You haven't loaded any race yet. Go to the "
            "**'🗺️ Race Analysis'** tab, select and analyze a race, and it "
            "will show up here automatically."
        )
        selected_race = None
    else:
        selected_race = st.selectbox(
            "Which race did this runner do?",
            options=list(available_races.keys()),
        )
        race_data = available_races[selected_race]
        st.caption(
            f"Selected race: **{selected_race}** · "
            f"{race_data['total_km']:.1f} km total"
        )

    st.markdown("---")

    runner_url = st.text_input(
        "Runner link (UTMB Live)",
        placeholder="https://live.utmb.world/aranbyutmb/2026/runners/5",
    )
    load_button = st.button("🔍 Load runner data", use_container_width=True)

    if load_button:
        if not runner_url:
            st.warning("Paste a valid link before clicking the button.")
        else:
            with st.spinner("Connecting to the timing platform..."):
                try:
                    runner_info, df_runner = scrape_runner_splits(runner_url)
                    error_detail = None
                except Exception:
                    runner_info, df_runner = None, None
                    error_detail = traceback.format_exc()

            # --- Feedback window ---
            if error_detail:
                st.error("❌ An error occurred while trying to fetch the runner data.")
                with st.expander("View technical error detail"):
                    st.code(error_detail, language="python")
            elif df_runner is None or df_runner.empty:
                st.warning(
                    "⚠️ No data table was found at that link. "
                    "Make sure it's the direct URL to the runner's profile."
                )
            else:
                st.success("✅ Runner data fetched successfully!")

                # --- VPI / DMI / ER index calculation ---
                # Requires the selected race to have checkpoints with km
                # loaded on Tab 1 (df_segments).
                current_race_data = available_races.get(selected_race, {}) if selected_race else {}
                race_segments_df = current_race_data.get("df_segments")

                if race_segments_df is None or race_segments_df.empty:
                    st.warning(
                        "⚠️ The selected race doesn't have checkpoints with km loaded yet. "
                        "Go back to the 'Race Analysis' tab, load the checkpoints for that "
                        "race, and reload it here to calculate the indices."
                    )
                else:
                    try:
                        total_race_gain = calculate_total_elevation_gain(current_race_data["df"])
                        indices, df_crossed = calculate_runner_indices(
                            race_segments_df,
                            df_runner,
                            current_race_data["total_km"],
                            total_race_gain,
                        )
                        indices_error = None
                    except Exception as e:
                        indices, df_crossed = None, None
                        indices_error = str(e)

                    st.markdown("### 🎯 Performance Indices")
                    if indices_error:
                        st.error(f"❌ Couldn't calculate the indices: {indices_error}")
                    else:
                        i1, i2, i3 = st.columns(3)
                        i1.metric(
                            "🧗 VPI - Climbing Efficiency",
                            f"{indices['VPI']} m/h" if indices["VPI"] is not None else "N/A",
                            help="Vertical Power Index: meters of elevation gain per hour on segments with slope ≥12%.",
                        )
                        i2.metric(
                            "📉 DMI - Descent Mastery",
                            f"{indices['DMI']} km/h" if indices["DMI"] is not None else "N/A",
                            help="Descent Mastery Index: average speed on segments with slope ≤-12%.",
                        )
                        i3.metric(
                            "🏆 ER - Endurance Rating",
                            f"{indices['ER']}" if indices["ER"] is not None else "N/A",
                            help="Endurance Rating: 100 = stable pace, lower values indicate fatigue-driven degradation.",
                        )
                        if indices["unmatched_segments"] > 0:
                            st.caption(
                                f"⚠️ {indices['unmatched_segments']} race segment(s) had no "
                                "matching checkpoint in the runner's data and were excluded from the calculation."
                            )
                        with st.expander("View crossed segments (race + runner times)"):
                            st.dataframe(df_crossed, use_container_width=True)

                        # --- Degradation matrix by segment ---
                        st.markdown("---")
                        st.markdown("### 📉 Degradation Curve by Segment")
                        st.caption(
                            "VPI and DMI calculated independently for each segment (not cumulative), "
                            "normalized against this runner's Segment 1 (Segment 1 = 100)."
                        )

                        df_segment_degradation = calculate_indices_by_segment(
                            current_race_data["df"], race_segments_df, df_runner
                        )

                        st.dataframe(df_segment_degradation, use_container_width=True)

                        fig_degradation = go.Figure()
                        fig_degradation.add_trace(go.Scatter(
                            x=df_segment_degradation["End Km"],
                            y=df_segment_degradation["VPI Index (0-100)"],
                            mode="lines+markers",
                            name="VPI (Climbing)",
                            line=dict(color="#22d3ee", width=3),
                            text=df_segment_degradation["Segment"],
                            hovertemplate="%{text}<br>Km %{x:.0f}<br>VPI Index: %{y:.1f}<extra></extra>",
                        ))
                        fig_degradation.add_trace(go.Scatter(
                            x=df_segment_degradation["End Km"],
                            y=df_segment_degradation["DMI Index (0-100)"],
                            mode="lines+markers",
                            name="DMI (Descent)",
                            line=dict(color="#ffa500", width=3),
                            text=df_segment_degradation["Segment"],
                            hovertemplate="%{text}<br>Km %{x:.0f}<br>DMI Index: %{y:.1f}<extra></extra>",
                        ))
                        fig_degradation.update_layout(
                            template="plotly_dark",
                            xaxis_title="Cumulative Km",
                            yaxis_title="Index (0-100, Segment 1 = 100)",
                            height=420,
                            hovermode="x unified",
                        )
                        st.plotly_chart(fig_degradation, use_container_width=True)

                st.markdown("---")

                # Runner card
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Runner", runner_info.get("Name") or "-")
                c2.metric("Finish Time", runner_info.get("Finish Time") or "-")
                c3.metric("Overall Rank", runner_info.get("Overall Rank") or "-")
                c4.metric("Category", runner_info.get("Category") or "-")

                with st.expander("View full runner profile"):
                    st.json(runner_info)

                st.markdown("##### Checkpoints / Split Times")
                st.dataframe(df_runner, use_container_width=True)

                # Save for reuse in other tabs (including which race was
                # chosen, for the VPI/DMI/ER calculation)
                st.session_state['runner_metrics_df'] = df_runner
                st.session_state['runner_info'] = runner_info
                st.session_state['race_selected_for_runner'] = selected_race

# ---------------------------------------------
# TAB 3: Indices and methodology documentation
# ---------------------------------------------
with tab_methodology:
    st.header("📖 Indices & Calculation Methodology")
    st.caption(
        "Definitions, geometric criteria and formulas for VertLabs' proprietary indices. "
        "These indices cross the official GPX terrain (the 'Race Analysis' tab) with the "
        "runner's real split times (the 'Runner Metrics' tab)."
    )

    st.markdown("### 📐 Performance Indices")
    for index_key in INDEX_CONFIG:
        cfg = INDEX_CONFIG[index_key]
        with st.expander(f"{cfg['icon']} {cfg['name']} ({index_key})", expanded=False):
            st.markdown(display_metric_documentation(index_key))

    st.markdown("---")
    st.markdown("### ⚡ Speed Metrics")
    for metric_key, cfg in SPEED_METRICS.items():
        with st.expander(cfg['name'], expanded=False):
            st.markdown(f"""
            * **Description:** {cfg['description']}
            * **Data Source:** {cfg['source']}
            * **Formula:** `{cfg['formula']}`
            * **Unit:** {cfg['unit']}
            """)