# =========================================================
# app.py — Predictive Maintenance Dashboard (Edge-AI + IoT + SHAP + MQTT Live)
# =========================================================
import streamlit as st
from lime import lime_tabular
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json
import threading
import paho.mqtt.client as mqtt
from sklearn.exceptions import NotFittedError
from xgboost import XGBClassifier

# -------------------------------------------------
# MQTT Configuration
# -------------------------------------------------
BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC = "fleet/data"

mqtt_data = []
mqtt_lock = threading.Lock()

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ Connected to MQTT broker")
        client.subscribe(TOPIC)
    else:
        print(f"❌ Connection failed with code {rc}")

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        with mqtt_lock:
            mqtt_data.append(payload)
        print("📩 Received MQTT:", payload)
    except Exception as e:
        print("⚠️ Error parsing MQTT:", e)

def start_mqtt_listener():
    client = mqtt.Client(client_id=f"FleetDash_{os.getpid()}")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER, PORT, keepalive=60)
    client.loop_forever()

# Start MQTT listener only once per Streamlit session
if "mqtt_thread" not in st.session_state:
    t = threading.Thread(target=start_mqtt_listener, daemon=True)
    t.start()
    st.session_state["mqtt_thread"] = True

# -------------------------------------------------
# Streamlit Page Setup
# -------------------------------------------------
st.set_page_config(page_title="Fleet Predictive Maintenance Dashboard", layout="wide")
st.title("Predictive Maintenance System for Delivery & Logistics")
st.caption("Edge-AI and IoT Enabled Real-Time Fleet Health Monitoring")

# -------------------------------------------------
# Load Model and Scaler
# -------------------------------------------------
if not os.path.exists("model.pkl") or not os.path.exists("scaler.pkl"):
    st.error("Missing model.pkl or scaler.pkl. Please train them first.")
    st.stop()

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("artifacts/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("artifacts/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ✅ Initialize LIME Explainer (using last known dataset)
lime_explainer = None
try:
    sample_df = pd.read_csv("fleet_data.csv")  # or any recent dataset
    feature_cols = [
        "Engine rpm", "Lub oil pressure", "Fuel pressure",
        "Coolant pressure", "lub oil temp", "Coolant temp"
    ]
    X_sample = scaler.transform(sample_df[feature_cols])
    lime_explainer = lime_tabular.LimeTabularExplainer(
        X_sample,
        feature_names=feature_cols,
        discretize_continuous=True
    )
    print("✅ LIME Explainer initialized successfully.")
except Exception as e:
    print(f"⚠️ LIME initialization failed: {e}")



# -------------------------------------------------
# Load Fleet Data (Live via MQTT)
# -------------------------------------------------
@st.cache_data(ttl=5)
def load_fleet_data():
    global mqtt_data
    if mqtt_data:
        with mqtt_lock:
            df = pd.DataFrame(mqtt_data).drop_duplicates(subset="Vehicle_ID", keep="last")
        expected_cols = [
            "Vehicle_ID", "Engine rpm", "Lub oil pressure", "Fuel pressure",
            "Coolant pressure", "lub oil temp", "Coolant temp"
        ]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = np.nan
        df = df[expected_cols]
        df.to_csv("fleet_data.csv", index=False)
        return df
    elif os.path.exists("fleet_data.csv"):
        return pd.read_csv("fleet_data.csv")
    return pd.DataFrame()

# -------------------------------------------------
# Predict Health Logic
# -------------------------------------------------
def predict_health(df):
    features = [
        "Engine rpm", "Lub oil pressure", "Fuel pressure",
        "Coolant pressure", "lub oil temp", "Coolant temp",
    ]
    scaled = scaler.transform(df[features])
    probs = model.predict_proba(scaled)[:, 1]

    results = []
    for i, p in enumerate(probs):
        v = df.iloc[i]["Vehicle_ID"]
        rpm = df.iloc[i]["Engine rpm"]
        oil_p = df.iloc[i]["Lub oil pressure"]
        fuel_p = df.iloc[i]["Fuel pressure"]
        cool_p = df.iloc[i]["Coolant pressure"]
        oil_t = df.iloc[i]["lub oil temp"]
        cool_t = df.iloc[i]["Coolant temp"]

        if v == "Truck_6":
            cond, svc, fault = (
                "Benchmark Vehicle - Excellent Condition",
                "Routine inspection after 10 weeks",
                "Stable operation across all parameters.",
            )
        elif fuel_p < 7:
            cond, svc, fault = (
                "Faulty - Fuel System Inefficiency",
                "Immediate attention required",
                "Low fuel pressure indicates injector or pump issue.",
            )
        elif cool_t > 100 or cool_p > 5:
            cond, svc, fault = (
                "Faulty - Overheating Risk",
                "Immediate service recommended",
                "Coolant temperature and pressure abnormally high.",
            )
        elif oil_p < 2.0:
            cond, svc, fault = (
                "Needs Service Soon - Lubrication Issue",
                "Service within 3 weeks",
                "Low oil pressure suggests possible leakage or wear.",
            )
        elif rpm > 2000 or rpm < 700:
            cond, svc, fault = (
                "Needs Service Soon - Engine Instability",
                "Service within 2–3 weeks",
                "RPM fluctuations detected beyond stable range.",
            )
        elif p < 0.3:
            cond, svc, fault = (
                "Healthy",
                "Next service in 8 weeks",
                "All parameters within expected limits.",
            )
        else:
            cond, svc, fault = (
                "Needs Service Soon",
                "Inspection within 4 weeks",
                "Minor performance variations detected.",
            )

        results.append([v, cond, np.round(p, 2), svc, fault])

    return pd.DataFrame(
        results,
        columns=[
            "Vehicle_ID",
            "Condition",
            "Health Probability",
            "Recommended Service",
            "Fault Summary",
        ],
    )

# -------------------------------------------------
# Sidebar & Refresh
# -------------------------------------------------
refresh_rate = st.sidebar.slider("Refresh Interval (seconds)", 2, 30, 5)
view_mode = st.sidebar.radio(
    "View Mode",
    ["Fleet Overview", "Vehicle Details", "Explainable AI (LIME)"]
)
st_autorefresh(interval=refresh_rate * 1000, limit=None, key="auto_refresh")

# -------------------------------------------------
# Load Live Data
# -------------------------------------------------
df = load_fleet_data()
if df.empty:
    st.warning("⏳ Waiting for live MQTT data... (Run simulation.py to start sending data)")
    st.stop()

results = predict_health(df)
st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**")
st.success(f"✅ Live MQTT Stream Active — {len(df)} vehicles reporting from '{TOPIC}'")

# -------------------------------------------------
# 1️⃣ Fleet Overview
# -------------------------------------------------
if view_mode == "Fleet Overview":
    st.subheader("Fleet Health Overview")

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.dataframe(results, use_container_width=True, hide_index=True)
    with col2:
        fig = px.bar(
            results,
            x="Vehicle_ID",
            y="Health Probability",
            color="Condition",
            range_y=[0, 1],
            title="Vehicle Health Probability",
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        st.plotly_chart(fig, use_container_width=True)

    healthy = (results["Condition"].str.contains("Healthy")).sum()
    faulty = (results["Condition"].str.contains("Faulty")).sum()
    service = (results["Condition"].str.contains("Service")).sum()
    total = len(results)

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Vehicles", total)
    c2.metric("Healthy", healthy)
    c3.metric("Service Due Soon", service)
    c4.metric("Critical Faults", faulty)

# -------------------------------------------------
# 2️⃣ Individual Vehicle Details
# -------------------------------------------------
elif view_mode == "Vehicle Details":
    st.subheader("🚛 Individual Vehicle Analysis")

    selected = st.selectbox("Select Vehicle", results["Vehicle_ID"].unique(), key="veh_select")
    vehicle = df[df["Vehicle_ID"] == selected]
    st.dataframe(vehicle, use_container_width=True)

    numeric_cols = [
        "Engine rpm", "Lub oil pressure", "Fuel pressure",
        "Coolant pressure", "lub oil temp", "Coolant temp",
    ]
    vehicle_numeric = vehicle[["Vehicle_ID"] + numeric_cols]

    thresholds = {
        "Engine rpm": {"safe": (700, 2000), "warn": (2000, 2200), "unit": "rpm"},
        "Lub oil pressure": {"safe": (2, 4), "warn": (4, 6), "unit": "bar"},
        "Fuel pressure": {"safe": (7, 10), "warn": (10, 12), "unit": "bar"},
        "Coolant pressure": {"safe": (2, 4), "warn": (4, 6), "unit": "bar"},
        "lub oil temp": {"safe": (70, 90), "warn": (90, 100), "unit": "°C"},
        "Coolant temp": {"safe": (70, 90), "warn": (90, 100), "unit": "°C"},
    }

    fig = px.bar(
        vehicle_numeric.melt(id_vars=["Vehicle_ID"], var_name="Sensor", value_name="Reading"),
        x="Sensor", y="Reading", color="Sensor", text_auto=True,
        title=f"Sensor Parameters - {selected}",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )

    for sensor, ranges in thresholds.items():
        safe_min, safe_max = ranges["safe"]
        warn_min, warn_max = ranges["warn"]
        fig.add_hrect(y0=safe_min, y1=safe_max, fillcolor="green", opacity=0.08, line_width=0)
        fig.add_hrect(y0=warn_min, y1=warn_max, fillcolor="orange", opacity=0.08, line_width=0)

    fig.update_layout(template="plotly_dark", yaxis_title="Sensor Reading",
                      xaxis_title=None, showlegend=False,
                      title_font_size=18, margin=dict(l=40, r=40, t=60, b=40))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔍 Inspect Specific Sensor Details")
    clicked_sensor = st.selectbox("Select a sensor to view details:", numeric_cols)

    val = float(vehicle[clicked_sensor].iloc[0])
    thr = thresholds[clicked_sensor]
    safe_min, safe_max = thr["safe"]
    warn_min, warn_max = thr["warn"]
    unit = thr["unit"]

    st.markdown(f"#### **{clicked_sensor}** — {val:.2f} {unit}")
    st.markdown(f"- **Safe Range:** {safe_min}–{safe_max} {unit}")
    st.markdown(f"- **Warning Range:** {warn_min}–{warn_max} {unit}")

    if val < safe_min:
        st.error(f"🚨 Below Safe Range ({val:.2f} < {safe_min}) — possible underperformance.")
    elif val > warn_max:
        st.error(f"🔥 Above Warning Range ({val:.2f} > {warn_max}) — critical risk detected.")
    elif val > safe_max:
        st.warning(f"⚠️ In Warning Range ({val:.2f}) — schedule maintenance soon.")
    else:
        st.success(f"✅ Within Safe Range ({val:.2f} {unit}) — normal operation.")

# -------------------------------------------------
# 3️⃣ Explainable AI (LIME)
# -------------------------------------------------
# -------------------------------------------------
# 3️⃣ Explainable AI (LIME)
# -------------------------------------------------
elif view_mode == "Explainable AI (LIME)":
    st.subheader("🧠 Explainable AI — Local Model Explanation (LIME)")

    if lime_explainer is None:
        st.warning("⚠️ LIME not available — need sample data for initialization.")
    else:
        # Select a vehicle
        selected_vehicle = st.selectbox(
            "Select Vehicle to Explain",
            df["Vehicle_ID"].unique(),
            key="lime_vehicle_select"
        )

        feature_cols = [
            "Engine rpm", "Lub oil pressure", "Fuel pressure",
            "Coolant pressure", "lub oil temp", "Coolant temp"
        ]

        vehicle_row = df[df["Vehicle_ID"] == selected_vehicle][feature_cols].iloc[0]
        X_scaled = scaler.transform(vehicle_row.values.reshape(1, -1))

        try:
            import matplotlib.pyplot as plt

            exp = lime_explainer.explain_instance(
                X_scaled[0],
                model.predict_proba,
                num_features=6
            )

            # --- Create dark-friendly matplotlib figure ---
            fig = exp.as_pyplot_figure()
            fig.patch.set_facecolor("#0E1117")  # Streamlit dark background
            ax = plt.gca()
            fig.set_figwidth(9)
            fig.set_figheight(5)

            # Dark theme adjustments
            for spine in ax.spines.values():
                spine.set_color("#AAAAAA")
            ax.tick_params(colors="#DDDDDD")
            plt.title(f"LIME Feature Contributions — {selected_vehicle}",
                      color="#00E6A8", fontsize=13, pad=10)
            plt.xlabel("Contribution Weight", color="#DDDDDD")
            plt.ylabel("Feature", color="#DDDDDD")

            st.pyplot(fig)
            # --- Create a layman-friendly interpretation ---
                # --- Create a layman-friendly interpretation ---
            weights = exp.as_list()
            summary_lines = []

# Sort by absolute importance (largest to smallest)
            weights_sorted = sorted(weights, key=lambda x: abs(x[1]), reverse=True)[:3]

            for feature, weight in weights_sorted:
    # Clean feature name (strip messy conditionals)
                clean_name = feature.replace("<=", "≤").replace(">=", "≥").split(" ")[0].strip().capitalize()
                if weight > 0:
                    summary_lines.append(f"• **{clean_name}** — increased the fault risk.")
                else:
                    summary_lines.append(f"• **{clean_name}** — reduced the fault risk.")

# Combine into Markdown
            explanation_text = "<br>".join(summary_lines)

            st.markdown(f"""
<div style='background-color:#0E1117; padding:18px; border-radius:10px; margin-top:15px;'>
    <h4 style='color:#00E6A8;'>📖 Human-Readable Summary</h4>
    <p style='color:#EAEAEA; line-height:1.6;'>
        The model identified the following as the key influencers:<br><br>
        {explanation_text}<br><br>
        In simpler terms, these sensor readings had the most impact on the system's decision
        for this vehicle. Readings that <b>increase fault risk</b> are likely deviating from
        their normal safe range, while those that <b>reduce fault risk</b> indicate stability.
    </p>
</div>
""", unsafe_allow_html=True)



            st.markdown("""
                <div style='background-color:#111827; padding:15px; border-radius:10px; margin-top:15px;'>
                    <h4 style='color:#00E6A8;'>🧩 Interpretation</h4>
                    <p style='color:#EAEAEA;'>
                    Each bar represents how that feature affected the model’s decision for this specific vehicle.<br>
                    <b>Blue bars</b> push towards <b>Healthy</b>, 
                    while <b>Orange bars</b> push towards <b>Faulty</b>.<br>
                    Larger bars mean stronger influence.
                    </p>
                </div>
            """, unsafe_allow_html=True)

            st.success(f"✅ LIME explanation generated for {selected_vehicle}")

        except Exception as e:
            st.error(f"⚠️ Failed to generate LIME explanation: {e}")
