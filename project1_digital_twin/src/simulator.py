"""
simulator.py
─────────────────────────────────────────────────────────────
Vehicle sensor data simulator.

Generates a realistic time-series dataset that mimics the kind
of data coming from a vehicle's CAN Bus — the internal network
that connects all electronic control units in a car.

Each sensor is modeled with physically plausible behavior:
    - Speed    : sinusoidal urban driving cycle + noise
    - Engine T : tanh warm-up curve + steady-state noise
    - Battery  : stable nominal voltage + small fluctuations
    - Vibration: speed-dependent baseline + road noise
    - Fuel P   : stable pressure + pump noise

Faults are injected at random time steps to simulate real
failure scenarios: overheating, voltage drops, etc.
"""

import numpy as np
import pandas as pd


FAULT_TYPES = ["overheat", "voltage_drop", "vibration_spike", "pressure_loss"]


def simulate(n_points: int = 2000,
             anomaly_rate: float = 0.05,
             random_seed: int = 42) -> pd.DataFrame:
    """
    Simulate vehicle sensor time-series with injected faults.

    Parameters
    ----------
    n_points : int
        Number of time steps (one per second).
    anomaly_rate : float
        Fraction of time steps that contain an injected fault.
    random_seed : int
        NumPy random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: time_s, speed_kmh, engine_temp_c, battery_v,
                 vibration_g, fuel_pressure, true_anomaly
    """
    np.random.seed(random_seed)

    t = np.arange(n_points, dtype=np.float32)

    # ── Normal signals ──────────────────────────────────────────
    speed = np.clip(
        50 + 30 * np.sin(2 * np.pi * t / 600) + 10 * np.random.randn(n_points),
        0, 130
    ).astype(np.float32)

    engine_temp = np.clip(
        90 * np.tanh(t / 300) + 5 * np.random.randn(n_points),
        20, 120
    ).astype(np.float32)

    battery_v = (12.4 + 0.3 * np.random.randn(n_points)).astype(np.float32)

    vibration = np.clip(
        0.1 + 0.05 * (speed / 100) + 0.02 * np.random.randn(n_points),
        0, 1
    ).astype(np.float32)

    fuel_pressure = (3.5 + 0.2 * np.random.randn(n_points)).astype(np.float32)

    # ── Fault injection ─────────────────────────────────────────
    n_faults = int(n_points * anomaly_rate)
    fault_idx = np.sort(
        np.random.choice(n_points, n_faults, replace=False)
    )

    for idx in fault_idx:
        fault = np.random.choice(FAULT_TYPES)
        if fault == "overheat":
            engine_temp[idx] = np.random.uniform(110, 130)
        elif fault == "voltage_drop":
            battery_v[idx] = np.random.uniform(9.0, 11.0)
        elif fault == "vibration_spike":
            vibration[idx] = np.random.uniform(0.6, 1.0)
        elif fault == "pressure_loss":
            fuel_pressure[idx] = np.random.uniform(0.5, 2.0)

    df = pd.DataFrame({
        "time_s"        : t.astype(int),
        "speed_kmh"     : speed,
        "engine_temp_c" : engine_temp,
        "battery_v"     : battery_v,
        "vibration_g"   : vibration,
        "fuel_pressure" : fuel_pressure,
        "true_anomaly"  : 0,
    })
    df.loc[fault_idx, "true_anomaly"] = 1

    return df
