"""
SmartCity Digital Twin AI System
AI-powered infrastructure monitoring and risk prediction platform.
"""

import random
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SmartCity Digital Twin AI System",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
CITY_NAMES = [
    "Neo Arcadia", "Techville", "Aurora Prime", "CyberNexus",
    "Lumina City", "Quantum Bay", "Helios Metro", "Nova Sphere",
    "SkyForge", "Titan Haven",
]

FEATURES = [
    "traffic_density",
    "power_usage",
    "temperature",
    "pollution",
    "infrastructure_load",
    "vibration",
]

N_SAMPLES = 500
RANDOM_SEED = 42

# ─────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────

def get_random_city_name() -> str:
    """Return a random city name from the predefined list."""
    return random.choice(CITY_NAMES)


def get_risk_level(score: float) -> tuple[str, str]:
    """Return (risk_label, css_color) based on city health score."""
    if score >= 70:
        return "Low Risk", "green"
    elif score >= 45:
        return "Medium Risk", "orange"
    else:
        return "High Risk", "red"


def get_city_status(score: float) -> str:
    """Return city status string based on health score."""
    if score >= 70:
        return "🟢 Healthy"
    elif score >= 45:
        return "🟡 Warning"
    else:
        return "🔴 Critical"


# ─────────────────────────────────────────────
# Data generation
# ─────────────────────────────────────────────

@st.cache_data
def generate_city_data(n: int = N_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate simulated smart-city sensor data."""
    rng = np.random.default_rng(seed)

    traffic_density     = rng.uniform(10, 100, n)
    power_usage         = rng.uniform(10, 100, n)
    temperature         = rng.uniform(15,  45, n)
    pollution           = rng.uniform(5,  100, n)
    infrastructure_load = rng.uniform(10, 100, n)
    vibration           = rng.uniform(1,   80, n)

    # Health score formula (each weighted factor reduces health)
    city_health_score = 100 - (
        traffic_density     * 0.2 +
        power_usage         * 0.2 +
        pollution           * 0.2 +
        infrastructure_load * 0.2 +
        vibration           * 0.2
    )
    city_health_score = np.clip(city_health_score, 0, 100)

    return pd.DataFrame({
        "traffic_density":     traffic_density,
        "power_usage":         power_usage,
        "temperature":         temperature,
        "pollution":           pollution,
        "infrastructure_load": infrastructure_load,
        "vibration":           vibration,
        "city_health_score":   city_health_score,
    })


# ─────────────────────────────────────────────
# Model training
# ─────────────────────────────────────────────

@st.cache_resource
def train_model(df: pd.DataFrame):
    """Train a RandomForestRegressor on city sensor data."""
    X = df[FEATURES]
    y = df["city_health_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    return model, accuracy


# ─────────────────────────────────────────────
# Page renderers
# ─────────────────────────────────────────────

def page_overview(df: pd.DataFrame, city_name: str) -> None:
    """Render the Overview dashboard page."""
    st.header("🏙️ City Overview Dashboard")
    st.subheader(f"Currently monitoring: **{city_name}**")

    # Compute latest snapshot (mean of last 20 rows)
    recent = df.tail(20)
    avg_health  = recent["city_health_score"].mean()
    avg_traffic = recent["traffic_density"].mean()
    avg_power   = recent["power_usage"].mean()
    risk_label, _ = get_risk_level(avg_health)
    status = get_city_status(avg_health)

    st.markdown(f"### City Status: {status}")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    # Color-coded delta strings to mimic green/yellow/red
    def health_delta(score: float) -> str:
        if score >= 70:
            return "✅ Healthy"
        elif score >= 45:
            return "⚠️ Medium"
        return "🚨 At Risk"

    col1.metric(
        label="🩺 City Health Score",
        value=f"{avg_health:.1f} / 100",
        delta=health_delta(avg_health),
    )
    col2.metric(
        label="🚗 Traffic Load",
        value=f"{avg_traffic:.1f}%",
        delta="Live",
    )
    col3.metric(
        label="⚡ Power Usage",
        value=f"{avg_power:.1f}%",
        delta="Live",
    )
    col4.metric(
        label="⚠️ Risk Level",
        value=risk_label,
        delta=status,
    )

    st.divider()

    # Mini trend charts
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.line(
            df.tail(100),
            y="city_health_score",
            title="City Health Score – Last 100 Readings",
            labels={"index": "Time Step", "city_health_score": "Health Score"},
            color_discrete_sequence=["#00cc96"],
        )
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig2 = px.line(
            df.tail(100),
            y=["traffic_density", "power_usage", "pollution"],
            title="Key Metrics – Last 100 Readings",
            labels={"value": "Level (%)", "variable": "Metric"},
            color_discrete_sequence=["#636efa", "#ef553b", "#ffa15a"],
        )
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)


def page_live_simulation(df: pd.DataFrame) -> None:
    """Render the Live Simulation page with animated Plotly charts."""
    st.header("📡 Live City Simulation")
    st.info("Showing the latest 200 sensor readings from the simulated city.")

    recent = df.tail(200).reset_index(drop=True)
    recent.index.name = "Time Step"

    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(
            recent,
            y="traffic_density",
            title="🚗 Traffic Density Over Time",
            labels={"index": "Time Step", "traffic_density": "Traffic Density (%)"},
            color_discrete_sequence=["#636efa"],
        )
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.line(
            recent,
            y="power_usage",
            title="⚡ Power Usage Over Time",
            labels={"index": "Time Step", "power_usage": "Power Usage (%)"},
            color_discrete_sequence=["#ef553b"],
        )
        fig2.update_layout(height=320)
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.line(
        recent,
        y="city_health_score",
        title="🩺 City Health Score Over Time",
        labels={"index": "Time Step", "city_health_score": "Health Score"},
        color_discrete_sequence=["#00cc96"],
    )
    fig3.update_layout(height=350)
    st.plotly_chart(fig3, use_container_width=True)

    # Combined view
    st.subheader("All Sensor Metrics")
    fig4 = px.line(
        recent,
        y=FEATURES,
        title="All Sensor Readings Over Time",
        labels={"value": "Sensor Value", "variable": "Sensor"},
    )
    fig4.update_layout(height=380)
    st.plotly_chart(fig4, use_container_width=True)


def page_risk_prediction(model: RandomForestRegressor) -> None:
    """Render the Risk Prediction page with interactive sliders."""
    st.header("🔮 Risk Prediction Engine")
    st.write("Adjust the city sensor values to predict the infrastructure health score.")

    col_sliders, col_result = st.columns([2, 1])

    with col_sliders:
        traffic    = st.slider("🚗 Traffic Density",     min_value=0,  max_value=100, value=50, step=1)
        power      = st.slider("⚡ Power Usage",          min_value=0,  max_value=100, value=50, step=1)
        pollution  = st.slider("💨 Pollution Level",      min_value=0,  max_value=100, value=40, step=1)
        infra      = st.slider("🏗️ Infrastructure Load", min_value=0,  max_value=100, value=50, step=1)
        vibration  = st.slider("📳 Vibration",            min_value=0,  max_value=80,  value=30, step=1)
        temperature = st.slider("🌡️ Temperature (°C)",   min_value=15, max_value=45,  value=25, step=1)

    # Build input vector in the same feature order as training
    input_df = pd.DataFrame([{
        "traffic_density":     traffic,
        "power_usage":         power,
        "temperature":         temperature,
        "pollution":           pollution,
        "infrastructure_load": infra,
        "vibration":           vibration,
    }])

    predicted_score = model.predict(input_df)[0]
    predicted_score = float(np.clip(predicted_score, 0, 100))
    risk_label, risk_color = get_risk_level(predicted_score)
    status = get_city_status(predicted_score)

    with col_result:
        st.markdown("### Prediction Result")
        st.markdown(
            f"""
            <div style="
                background-color: #1e1e2e;
                border-left: 6px solid {risk_color};
                padding: 20px;
                border-radius: 8px;
            ">
                <h2 style="color:{risk_color}; margin:0;">{predicted_score:.1f} / 100</h2>
                <p style="color:white; font-size:18px; margin:8px 0 0 0;">City Health Score</p>
                <hr style="border-color:{risk_color}; margin:12px 0;">
                <p style="color:{risk_color}; font-size:22px; margin:0;"><b>{risk_label}</b></p>
                <p style="color:white; font-size:16px; margin:4px 0 0 0;">Status: {status}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Gauge chart
    st.divider()
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=predicted_score,
        title={"text": "Predicted City Health Score", "font": {"size": 20}},
        delta={"reference": 70, "increasing": {"color": "green"}, "decreasing": {"color": "red"}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": risk_color},
            "steps": [
                {"range": [0, 45],   "color": "#ff4b4b"},
                {"range": [45, 70],  "color": "#ffa500"},
                {"range": [70, 100], "color": "#00cc96"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 4},
                "thickness": 0.75,
                "value": predicted_score,
            },
        },
    ))
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)


def page_analytics(df: pd.DataFrame) -> None:
    """Render the Analytics page with correlation heatmap and distributions."""
    st.header("📊 City Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Correlation Heatmap")
        corr = df.corr(numeric_only=True)
        fig_heat = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            colorscale="RdBu",
            zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            showscale=True,
        ))
        fig_heat.update_layout(title="Feature Correlation Matrix", height=420)
        st.plotly_chart(fig_heat, use_container_width=True)

    with col2:
        st.subheader("Health Score Distribution")
        fig_hist = px.histogram(
            df,
            x="city_health_score",
            nbins=30,
            title="Distribution of City Health Scores",
            labels={"city_health_score": "City Health Score"},
            color_discrete_sequence=["#636efa"],
        )
        fig_hist.update_layout(height=420)
        st.plotly_chart(fig_hist, use_container_width=True)

    # Scatter matrix
    st.subheader("Pairwise Sensor Relationships")
    fig_scatter = px.scatter_matrix(
        df.sample(200, random_state=RANDOM_SEED),
        dimensions=FEATURES + ["city_health_score"],
        color="city_health_score",
        color_continuous_scale="RdYlGn",
        title="Sensor Pair Relationships (200-sample subset)",
    )
    fig_scatter.update_traces(diagonal_visible=False, marker_size=3)
    fig_scatter.update_layout(height=600)
    st.plotly_chart(fig_scatter, use_container_width=True)


def page_model_insights(model: RandomForestRegressor, accuracy: float, df: pd.DataFrame) -> None:
    """Render the Model Insights page."""
    st.header("🤖 Model Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Performance")
        st.metric(label="R² Accuracy Score", value=f"{accuracy:.4f}")
        st.metric(label="Algorithm", value="Random Forest Regressor")
        st.metric(label="Training Samples", value=f"{int(len(df) * 0.8)}")
        st.metric(label="Test Samples", value=f"{int(len(df) * 0.2)}")
        st.metric(label="Number of Trees", value="100")
        st.metric(label="Features Used", value=str(len(FEATURES)))

    with col2:
        st.subheader("Feature Importance")
        importances = model.feature_importances_
        feat_df = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": importances,
        }).sort_values("Importance", ascending=True)

        fig = px.bar(
            feat_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Importance (Random Forest)",
            color="Importance",
            color_continuous_scale="Blues",
            labels={"Importance": "Importance Score", "Feature": "Sensor Feature"},
        )
        fig.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # Predicted vs actual scatter
    st.divider()
    st.subheader("Predicted vs Actual – Test Set")
    X = df[FEATURES]
    y = df["city_health_score"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    y_pred = model.predict(X_test)

    scatter_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
    fig2 = px.scatter(
        scatter_df,
        x="Actual",
        y="Predicted",
        title="Predicted vs Actual City Health Score",
        color_discrete_sequence=["#636efa"],
        opacity=0.6,
    )
    # Perfect-prediction reference line
    min_v, max_v = scatter_df["Actual"].min(), scatter_df["Actual"].max()
    fig2.add_shape(
        type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v,
        line={"color": "#ef553b", "width": 2, "dash": "dash"},
    )
    fig2.update_layout(height=420)
    st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────
# Main application entry point
# ─────────────────────────────────────────────

def main() -> None:
    # ── Sidebar ──────────────────────────────
    st.sidebar.image(
        "https://img.icons8.com/fluency/96/smart-city.png",
        width=80,
    )
    st.sidebar.title("SmartCity AI")
    st.sidebar.markdown("**Digital Twin System**")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigation",
        options=[
            "🏠 Overview",
            "📡 Live Simulation",
            "🔮 Risk Prediction",
            "📊 Analytics",
            "🤖 Model Insights",
        ],
    )

    st.sidebar.divider()

    # City selector
    if "city_name" not in st.session_state:
        st.session_state.city_name = get_random_city_name()

    if st.sidebar.button("🎲 Randomize City"):
        st.session_state.city_name = get_random_city_name()

    st.sidebar.markdown(f"**Active City:** {st.session_state.city_name}")
    st.sidebar.divider()
    st.sidebar.caption("© 2024 SmartCity Digital Twin AI")

    # ── App header ───────────────────────────
    st.title("🏙️ SmartCity Digital Twin AI System")
    st.markdown(
        "_AI-powered infrastructure monitoring and risk prediction platform_"
    )
    st.divider()

    # ── Data & model (cached) ─────────────────
    df = generate_city_data()
    model, accuracy = train_model(df)

    # ── Page routing ─────────────────────────
    if page == "🏠 Overview":
        page_overview(df, st.session_state.city_name)
    elif page == "📡 Live Simulation":
        page_live_simulation(df)
    elif page == "🔮 Risk Prediction":
        page_risk_prediction(model)
    elif page == "📊 Analytics":
        page_analytics(df)
    elif page == "🤖 Model Insights":
        page_model_insights(model, accuracy, df)


if __name__ == "__main__":
    main()
