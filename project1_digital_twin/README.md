# AI Digital Twin — Vehicle Sensor Monitor

> **Mercedes-Benz Digitalisierung · KI Team **

A production-structured AI pipeline that simulates vehicle sensor data,
detects anomalies with unsupervised machine learning, and renders a
real-time monitoring dashboard — the core concept behind predictive
maintenance in a Digital Twin environment.

---

## Demo Output

![Dashboard](assets/digital_twin_dashboard.png)

---

## Project Structure

```
project1_digital_twin/
│
├── main.py                  ← Entry point (run this)
├── requirements.txt
├── config/
│   └── config.ini           ← All parameters in one place
├── src/
│   ├── __init__.py
│   ├── simulator.py         ← Vehicle sensor data generation
│   ├── detector.py          ← Isolation Forest anomaly detection
│   └── dashboard.py         ← Dashboard rendering
└── tests/
    └── test_pipeline.py     ← Unit tests (pytest)
```

---

## Quickstart

```bash
# 1. Clone and enter the project
git clone https://github.com/Esma Dogrusozlu/mercedes-ai-projects.git
cd mercedes-ai-projects/project1_digital_twin

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python main.py
```

**Custom parameters:**
```bash
python main.py --n_points 3000 --anomaly_rate 0.08 --output my_run.png
python main.py --help   # see all options
```

**Run tests:**
```bash
pytest tests/ -v
```

---

## How It Works

### 1 · Sensor Simulation
Five vehicle sensors are modelled with physically realistic behavior:

| Sensor | Model | Normal Range |
|--------|-------|-------------|
| Speed | Sinusoidal driving cycle + noise | 0 – 130 km/h |
| Engine temp | tanh warm-up curve + noise | 20 – 100 °C |
| Battery | Stable nominal + small drift | 11.5 – 13.5 V |
| Vibration | Speed-dependent baseline | 0.0 – 0.3 g |
| Fuel pressure | Stable pump pressure | 3.0 – 4.0 bar |

Faults are injected at random time steps: overheating, voltage drop,
vibration spike, pressure loss.

### 2 · Isolation Forest (Unsupervised)
```
Normal data point  →  hard to isolate  →  deep tree  →  low score  →  NORMAL
Anomalous point    →  easy to isolate  →  shallow tree →  high score →  ANOMALY
```

**Why unsupervised?**
In real production, fault labels are expensive and rare.
Isolation Forest finds outliers without needing any labels.

### 3 · Dashboard
Six-panel dark-theme visualization:
- Speed, temperature, voltage time series with anomaly markers
- AI anomaly score with threshold fill
- Engine temp vs speed scatter (anomaly clusters visible)
- Performance metrics (Precision / Recall / F1)
- Anomaly distribution pie chart

---

## Results

| Metric | Value |
|--------|-------|
| Precision | ~0.76 |
| Recall | ~0.76 |
| F1 Score | ~0.76 |
| Inference time (2000 points) | < 1 s |

---

## Roadmap

- [ ] Replace simulated data with real CAN Bus input
- [ ] Add LSTM layer for temporal pattern detection
- [ ] NVIDIA Omniverse integration for 3D visualization
- [ ] Streamlit web dashboard
- [ ] Docker container for deployment

---


