"""
tests/test_pipeline.py
─────────────────────────────────────────────────────────────
Unit tests for the defect detection pipeline.

Run with:
    pytest tests/ -v
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.generator  import generate_image, build_dataset, CLASS_NAMES
from src.features   import extract_one, extract_all, N_FEATURES
from src.classifier import train


# ── Generator tests ──────────────────────────────────────────

class TestGenerator:

    @pytest.mark.parametrize("defect_type", CLASS_NAMES)
    def test_image_shape(self, defect_type):
        img = generate_image(defect_type, image_size=64)
        assert img.shape == (64, 64), f"Wrong shape for {defect_type}"

    @pytest.mark.parametrize("defect_type", CLASS_NAMES)
    def test_pixel_range(self, defect_type):
        img = generate_image(defect_type, image_size=64)
        assert img.min() >= 0.0 and img.max() <= 1.0, \
            f"Pixel values out of [0, 1] for {defect_type}"

    @pytest.mark.parametrize("defect_type", CLASS_NAMES)
    def test_dtype(self, defect_type):
        img = generate_image(defect_type)
        assert img.dtype == np.float32

    def test_dataset_size(self):
        n = 10
        images, labels = build_dataset(n_per_class=n)
        assert len(images) == n * len(CLASS_NAMES)
        assert len(labels) == len(images)

    def test_label_balance(self):
        images, labels = build_dataset(n_per_class=20)
        for cls_idx in range(len(CLASS_NAMES)):
            count = (labels == cls_idx).sum()
            assert count == 20, f"Class {cls_idx} has {count} samples, expected 20"


# ── Feature extraction tests ─────────────────────────────────

class TestFeatures:

    def test_vector_length(self):
        img = generate_image("normal")
        feat = extract_one(img)
        assert len(feat) == N_FEATURES

    def test_output_dtype(self):
        img = generate_image("crack")
        feat = extract_one(img)
        assert feat.dtype == np.float32

    def test_batch_extraction(self):
        images, _ = build_dataset(n_per_class=5)
        features  = extract_all(images)
        assert features.shape == (len(images), N_FEATURES)

    def test_no_nan(self):
        images, _ = build_dataset(n_per_class=10)
        features  = extract_all(images)
        assert not np.isnan(features).any(), "Features contain NaN values"

    def test_different_classes_differ(self):
        # Normal and crack should produce different feature vectors
        img_normal = generate_image("normal", random_seed := 42)
        img_crack  = generate_image("crack",  random_seed)
        f_normal   = extract_one(img_normal)
        f_crack    = extract_one(img_crack)
        assert not np.allclose(f_normal, f_crack), \
            "Normal and crack features should not be identical"


# ── Classifier tests ─────────────────────────────────────────

class TestClassifier:

    @pytest.fixture(scope="class")
    def trained(self):
        images, labels = build_dataset(n_per_class=30, random_seed=42)
        from src.features import extract_all as ea
        features = ea(images)
        return train(features, labels, CLASS_NAMES, random_state=42)

    def test_accuracy_reasonable(self, trained):
        assert trained.accuracy > 0.60, \
            f"Accuracy {trained.accuracy:.2%} is suspiciously low"

    def test_confusion_matrix_shape(self, trained):
        n = len(CLASS_NAMES)
        assert trained.confusion_matrix.shape == (n, n)

    def test_feature_importances_sum(self, trained):
        total = trained.feature_importances.sum()
        assert abs(total - 1.0) < 1e-5, \
            "Feature importances must sum to 1.0"

    def test_predictions_length(self, trained):
        assert len(trained.y_pred) == len(trained.y_test)

    def test_per_class_counts(self, trained):
        total = sum(trained.per_class_counts)
        assert total == len(trained.y_test)
