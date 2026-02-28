"""
detector.py
─────────────────────────────────────────────────────────────
Anomaly detection using Isolation Forest.

Isolation Forest builds an ensemble of random trees.
A data point that is easy to isolate (few splits needed)
is considered anomalous — it lies far from the dense
region where normal data clusters.

This approach is unsupervised: no fault labels required
during training, which is realistic for production systems
where labeled fault data is scarce and expensive.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass


FEATURE_COLS = [
    "speed_kmh",
    "engine_temp_c",
    "battery_v",
    "vibration_g",
    "fuel_pressure",
]


@dataclass
class DetectionMetrics:
    """Stores evaluation metrics after detection."""
    precision : float
    recall    : float
    f1        : float
    n_detected: int
    n_true    : int


def detect(df: pd.DataFrame,
           contamination: float = 0.05,
           n_estimators: int = 100,
           random_state: int = 42) -> tuple[pd.DataFrame, DetectionMetrics]:
    """
    Run Isolation Forest anomaly detection on sensor data.

    Parameters
    ----------
    df : pd.DataFrame
        Output of simulator.simulate().
    contamination : float
        Expected fraction of anomalies. Acts as a threshold hint.
    n_estimators : int
        Number of isolation trees in the ensemble.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    df : pd.DataFrame
        Input DataFrame with two new columns:
            ai_anomaly        — binary flag (1 = anomaly)
            anomaly_score_norm — normalized suspicion score [0, 1]
    metrics : DetectionMetrics
        Precision, recall, F1 against ground-truth labels.
    """
    X = df[FEATURE_COLS].values

    # Normalize: bring all sensors to the same scale (mean=0, std=1).
    # Without this, speed (0–130) would dominate voltage (11–13).
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    # fit_predict returns +1 (normal) or -1 (anomaly)
    raw = model.fit_predict(X_scaled)
    df = df.copy()
    df["ai_anomaly"] = (raw == -1).astype(int)

    # decision_function: lower score = more anomalous
    scores = model.decision_function(X_scaled)
    s_min, s_max = scores.min(), scores.max()
    df["anomaly_score_norm"] = 1.0 - (scores - s_min) / (s_max - s_min + 1e-8)

    # ── Evaluate against ground truth ──────────────────────────
    n_detected = int(df["ai_anomaly"].sum())
    n_true     = int(df["true_anomaly"].sum())
    tp         = int(((df["ai_anomaly"] == 1) & (df["true_anomaly"] == 1)).sum())
    precision  = tp / n_detected if n_detected > 0 else 0.0
    recall     = tp / n_true     if n_true     > 0 else 0.0
    f1         = (2 * precision * recall / (precision + recall + 1e-8))

    metrics = DetectionMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        n_detected=n_detected,
        n_true=n_true,
    )

    return df, metrics
