"""
tests/test_pipeline.py
─────────────────────────────────────────────────────────────
Unit tests for the Digital Twin pipeline.

Run with:
    pytest tests/ -v
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.simulator import simulate
from src.detector  import detect


# ── Simulator tests ──────────────────────────────────────────

class TestSimulator:

    def test_output_shape(self):
        df = simulate(n_points=100, anomaly_rate=0.05)
        assert len(df) == 100, "DataFrame must have exactly n_points rows"

    def test_required_columns(self):
        df = simulate(n_points=100)
        expected = {"time_s", "speed_kmh", "engine_temp_c",
                    "battery_v", "vibration_g", "fuel_pressure", "true_anomaly"}
        assert expected.issubset(df.columns), "Missing expected columns"

    def test_anomaly_count(self):
        n, rate = 1000, 0.05
        df = simulate(n_points=n, anomaly_rate=rate)
        expected = int(n * rate)
        assert df["true_anomaly"].sum() == expected, \
            f"Expected {expected} anomalies, got {df['true_anomaly'].sum()}"

    def test_sensor_ranges(self):
        df = simulate(n_points=500, anomaly_rate=0.0)  # No faults
        assert df["speed_kmh"].between(0, 130).all(), "Speed out of range"
        assert df["engine_temp_c"].between(20, 120).all(), "Engine temp out of range"
        assert df["vibration_g"].between(0, 1).all(), "Vibration out of range"

    def test_reproducibility(self):
        df1 = simulate(n_points=100, random_seed=42)
        df2 = simulate(n_points=100, random_seed=42)
        pd.testing.assert_frame_equal(df1, df2,
            check_names=True, obj="Reproducibility check")

    def test_different_seeds(self):
        df1 = simulate(n_points=100, random_seed=1)
        df2 = simulate(n_points=100, random_seed=2)
        assert not df1["speed_kmh"].equals(df2["speed_kmh"]), \
            "Different seeds should produce different data"


# ── Detector tests ───────────────────────────────────────────

class TestDetector:

    @pytest.fixture
    def sample_df(self):
        return simulate(n_points=500, anomaly_rate=0.05, random_seed=42)

    def test_new_columns(self, sample_df):
        result, _ = detect(sample_df)
        assert "ai_anomaly" in result.columns
        assert "anomaly_score_norm" in result.columns

    def test_binary_predictions(self, sample_df):
        result, _ = detect(sample_df)
        assert set(result["ai_anomaly"].unique()).issubset({0, 1}), \
            "ai_anomaly must be binary (0 or 1)"

    def test_score_range(self, sample_df):
        result, _ = detect(sample_df)
        assert result["anomaly_score_norm"].between(0, 1).all(), \
            "Normalized scores must be in [0, 1]"

    def test_metrics_range(self, sample_df):
        _, metrics = detect(sample_df)
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall    <= 1
        assert 0 <= metrics.f1        <= 1

    def test_original_df_unchanged(self, sample_df):
        original_cols = list(sample_df.columns)
        detect(sample_df)
        assert list(sample_df.columns) == original_cols, \
            "detect() must not modify the input DataFrame in place"

    def test_detects_something(self, sample_df):
        result, metrics = detect(sample_df)
        assert metrics.n_detected > 0, "Model should detect at least some anomalies"
