# 🏙️ SmartCity Digital Twin AI System

> AI-powered infrastructure monitoring and risk prediction platform

## Overview

SmartCity Digital Twin AI System is a production-level **Streamlit** application that simulates a virtual AI Digital Twin of a smart city. It predicts infrastructure health, risk levels, and resource consumption using Machine Learning — all running entirely in the browser with no backend server or database required.

## Features

| Page | Description |
|---|---|
| 🏠 Overview | Live metrics — City Health Score, Traffic Load, Power Usage, Risk Level |
| 📡 Live Simulation | Real-time Plotly charts for traffic, power, and city health |
| 🔮 Risk Prediction | Interactive sliders → ML prediction of health score and risk level |
| 📊 Analytics | Correlation heatmap, health score histogram, pairwise scatter matrix |
| 🤖 Model Insights | Model accuracy, feature importance chart, predicted vs actual scatter |

Additional highlights:

- **Random city name generator** (click *Randomize City* in the sidebar)
- **City status indicator**: 🟢 Healthy / 🟡 Warning / 🔴 Critical
- **Gauge chart** for at-a-glance risk visualization on the prediction page
- Fully cached data generation and model training for sub-5-second load times

## Tech Stack

```
streamlit==1.31.0
pandas==2.1.4
numpy==1.26.4
scikit-learn==1.3.2
plotly==5.18.0
matplotlib==3.8.2
```

No heavy ML frameworks (PyTorch / TensorFlow) — runs smoothly on **Streamlit Cloud**.

## Project Structure

```
smartcity-digital-twin/
│
├── app.py            # Main Streamlit application
├── requirements.txt  # Pinned dependencies
└── README.md         # This file
```

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploying to Streamlit Cloud

1. Fork / push this repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io) and create a new app.
3. Set **Main file path** to `app.py`.
4. Click **Deploy** — no further configuration needed.

## ML Model Details

- **Algorithm**: `RandomForestRegressor` (100 estimators)
- **Target**: `city_health_score` (0 – 100)
- **Features**: `traffic_density`, `power_usage`, `temperature`, `pollution`, `infrastructure_load`, `vibration`
- **Data**: 500 rows of synthetically generated sensor readings
- **Split**: 80 % train / 20 % test

### Health Score Formula

```
city_health_score = 100 - (
    traffic_density     * 0.2 +
    power_usage         * 0.2 +
    pollution           * 0.2 +
    infrastructure_load * 0.2 +
    vibration           * 0.2
)
clipped to [0, 100]
```

### Risk Level Thresholds

| Score | Status |
|---|---|
| ≥ 70 | 🟢 Healthy / Low Risk |
| 45 – 69 | 🟡 Warning / Medium Risk |
| < 45 | 🔴 Critical / High Risk |

## License

MIT
