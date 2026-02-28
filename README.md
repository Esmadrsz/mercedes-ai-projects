# Mercedes-Benz AI Projects

> Two end-to-end machine learning projects built for the
> **Mercedes-Benz Digitalisierung — KI Team** internship application.

---

## Projects

### 🚗 Project 1 — AI Digital Twin: Vehicle Sensor Monitor
`project1_digital_twin/`

Simulates 5 vehicle sensors (speed, engine temperature, battery voltage,
vibration, fuel pressure), injects realistic fault scenarios, and detects
anomalies using **Isolation Forest** (unsupervised ML).

→ Core concept: **predictive maintenance** in a Digital Twin environment.

### 🏭 Project 2 — AI Quality Control: Synthetic Defect Detection
`project2_defect_detection/`

Generates synthetic factory part images across 5 defect classes (normal,
crack, hole, corrosion, scratch), extracts statistical features, and
classifies them with a **Random Forest** — reaching 99.5% accuracy.

→ Core concept: **synthetic data** for AI training (same approach as NVIDIA Omniverse).

---

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/mercedes-ai-projects.git

# Project 1
cd project1_digital_twin
pip install -r requirements.txt
python main.py

# Project 2
cd ../project2_defect_detection
pip install -r requirements.txt
python main.py
```

---

## Tech Stack

`Python` · `NumPy` · `Pandas` · `Scikit-learn` · `Matplotlib`

---

*Built as interview demo projects — Mercedes-Benz Digitalisierung KI Team, 2026*
