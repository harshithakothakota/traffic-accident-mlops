"""Streamlit dashboard.   Run:  streamlit run dashboard.py"""
import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from etl import engine

API_URL = os.environ.get("API_URL", "http://localhost:8000")
COLORS = {"Fatal": "#104281", "Serious": "#2a78d6", "Slight": "#86b6ef"}
ORDER = {"severity_label": ["Fatal", "Serious", "Slight"],
         "weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                     "Saturday", "Sunday"]}

st.set_page_config(page_title="UK Traffic Accidents 2023", layout="wide")


@st.cache_data
def load():
    return pd.read_sql("SELECT * FROM mart_accidents", engine())


page = st.sidebar.radio("Page", ["Dashboard", "Predict Severity"])

# ===================================================================== DASHBOARD
if page == "Dashboard":
    df = load()
    st.sidebar.header("Filters")
    severity = st.sidebar.multiselect("Severity", ORDER["severity_label"], ORDER["severity_label"])
    road = st.sidebar.multiselect("Road type", sorted(df.road_type_label.unique()))
    weather = st.sidebar.multiselect("Weather", sorted(df.weather_label.unique()))
    weekday = st.sidebar.multiselect("Weekday", ORDER["weekday"])

    df = df[df.severity_label.isin(severity)]
    if road:
        df = df[df.road_type_label.isin(road)]
    if weather:
        df = df[df.weather_label.isin(weather)]
    if weekday:
        df = df[df.weekday.isin(weekday)]

    st.title("UK Traffic Accident Severity - 2023")
    st.caption("Source: DfT STATS19 road safety data + Open-Meteo historical weather")
    if df.empty:
        st.warning("No accidents match these filters.")
        st.stop()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accidents", f"{len(df):,}")
    k2.metric("Casualties", f"{df.number_of_casualties.sum():,}")
    k3.metric("Fatal accidents", f"{(df.severity == 1).sum():,}")
    k4.metric("Fatal or serious", f"{(df.severity <= 2).mean():.1%}")

    c1, c2 = st.columns(2)
    sev = df.severity_label.value_counts().reset_index()
    c1.plotly_chart(px.pie(sev, names="severity_label", values="count", hole=0.5,
                           color="severity_label", color_discrete_map=COLORS,
                           title="1. Severity distribution"), use_container_width=True)
    hourly = df.groupby(["hour", "severity_label"]).size().reset_index(name="accidents")
    c2.plotly_chart(px.bar(hourly, x="hour", y="accidents", color="severity_label",
                           color_discrete_map=COLORS, category_orders=ORDER,
                           title="2. Accidents by hour of day"), use_container_width=True)

    c1, c2 = st.columns(2)
    wd = df.weekday.value_counts().reindex(ORDER["weekday"]).reset_index()
    c1.plotly_chart(px.bar(wd, x="weekday", y="count", title="3. Accidents by weekday",
                           labels={"count": "accidents"}, color_discrete_sequence=["#2a78d6"]),
                    use_container_width=True)
    w = df.groupby("weather_label").agg(accidents=("severity", "size"),
                                        serious_share=("severity", lambda s: (s <= 2).mean()))
    c2.plotly_chart(px.bar(w.reset_index().sort_values("accidents"), x="accidents",
                           y="weather_label", orientation="h", hover_data=["serious_share"],
                           title="4. Weather conditions", color_discrete_sequence=["#2a78d6"]),
                    use_container_width=True)

    c1, c2 = st.columns(2)
    r = df.groupby("road_type_label").agg(accidents=("severity", "size"),
                                          pct_fatal_or_serious=("severity", lambda s: 100 * (s <= 2).mean()))
    c1.plotly_chart(px.bar(r.reset_index().sort_values("accidents"), x="accidents",
                           y="road_type_label", orientation="h",
                           hover_data=["pct_fatal_or_serious"], title="5. Road type",
                           color_discrete_sequence=["#2a78d6"]), use_container_width=True)
    t = df.groupby(df.temperature_c.round(-1)).size().reset_index(name="accidents")
    c2.plotly_chart(px.bar(t, x="temperature_c", y="accidents",
                           title="6. Observed temperature at the time (deg C, Open-Meteo)",
                           color_discrete_sequence=["#2a78d6"]), use_container_width=True)

    st.subheader("7. Hotspot map (fatal and serious accidents)")
    severe = df[df.severity <= 2]
    st.plotly_chart(px.scatter_map(severe, lat="latitude", lon="longitude",
                                   color="severity_label", color_discrete_map=COLORS,
                                   zoom=4.6, height=650, map_style="carto-positron",
                                   center={"lat": 54.0, "lon": -2.5},
                                   hover_data=["police_force_name", "road_type_label"]),
                    use_container_width=True)

    st.subheader("Top 10 police force areas")
    top = (df.groupby("police_force_name").agg(accidents=("severity", "size"),
                                      casualties=("number_of_casualties", "sum"),
                                      fatal=("severity", lambda s: (s == 1).sum()))
           .sort_values("accidents", ascending=False).head(10))
    st.dataframe(top, use_container_width=True)

# ============================================================ PREDICT SEVERITY
else:
    st.title("Predict Accident Severity")
    st.caption(f"Sends the inputs to the FastAPI model service at {API_URL}")
    with st.form("predict"):
        c1, c2, c3 = st.columns(3)
        c1.markdown("**When**")
        hour = c1.slider("Hour", 0, 23, 17)
        weekday = c1.selectbox("Weekday (1 = Monday)", list(range(1, 8)), index=4)
        month = c1.slider("Month", 1, 12, 7)
        c2.markdown("**Where and road**")
        lat = c2.number_input("Latitude", 49.9, 60.9, 51.5)
        lon = c2.number_input("Longitude", -8.6, 1.9, -0.12)
        area = c2.radio("Area", [1, 2], format_func={1: "Urban", 2: "Rural"}.get, horizontal=True)
        road_type = c2.selectbox("Road type", [6, 3, 1, 2, 7], format_func={
            6: "Single carriageway", 3: "Dual carriageway", 1: "Roundabout",
            2: "One way street", 7: "Slip road"}.get)
        speed = c2.select_slider("Speed limit (mph)", [20, 30, 40, 50, 60, 70], 30)
        c3.markdown("**Conditions**")
        light = c3.selectbox("Light", [1, 4, 6], format_func={
            1: "Daylight", 4: "Dark - street lights on", 6: "Dark - no street lights"}.get)
        weather = c3.selectbox("Weather", [1, 2, 3, 7], format_func={
            1: "Fine", 2: "Raining", 3: "Snowing", 7: "Fog or mist"}.get)
        surface = c3.selectbox("Road surface", [1, 2, 4], format_func={
            1: "Dry", 2: "Wet", 4: "Frost or ice"}.get)
        temp = c3.number_input("Temperature (deg C)", -15.0, 40.0, 15.0)
        rain = c3.number_input("Precipitation (mm/hour)", 0.0, 50.0, 0.0)
        wind = c3.number_input("Wind speed (km/h)", 0.0, 150.0, 12.0)
        vehicles = c3.number_input("Number of vehicles", 1, 20, 1)
        submit = st.form_submit_button("Predict")
    if submit:
        payload = {"hour": hour, "weekday_number": weekday, "month": month,
                   "latitude": lat, "longitude": lon, "urban_or_rural_area": area,
                   "road_type": road_type, "speed_limit": speed,
                   "light_conditions": light, "weather_conditions": weather,
                   "road_surface_conditions": surface, "number_of_vehicles": vehicles,
                   "temperature_c": temp, "precipitation_mm": rain, "wind_speed_kmh": wind}
        try:
            result = requests.post(f"{API_URL}/predict", json=payload, timeout=10).json()
            st.success(f"Predicted severity: **{result['predicted_severity']}**  "
                       f"(model version {result['model_version']})")
            probs = pd.DataFrame({"severity": list(result["probabilities"]),
                                  "probability": list(result["probabilities"].values())})
            fig = px.bar(probs, x="probability", y="severity", orientation="h",
                         color="severity", color_discrete_map=COLORS, text_auto=".0%",
                         range_x=[0, 1], height=260, title="Probability of each class")
            st.plotly_chart(fig.update_layout(showlegend=False), use_container_width=True)
        except Exception as error:
            st.error(f"Could not reach the API - is it running? ({error})")
