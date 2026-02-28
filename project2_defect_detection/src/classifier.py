"""
classifier.py
─────────────────────────────────────────────────────────────
Random Forest classifier for defect detection.

Random Forest builds many independent decision trees, each
trained on a random subset of the data and features.
Final prediction = majority vote across all trees.

Advantages over a single decision tree:
    · Much lower variance (less overfitting)
    · Built-in feature importance scores
    · Handles class imbalance reasonably well
    · No hyperparameter tuning required to get good results

The module also produces a structured ClassificationResult
so the visualization layer has everything it needs.
"""

import numpy as np
from dataclasses import dataclass, field
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler


@dataclass
class ClassificationResult:
    """All outputs from a training run."""
    model             : RandomForestClassifier
    scaler            : StandardScaler
    accuracy          : float
    confusion_matrix  : np.ndarray
    report            : str
    feature_importances: np.ndarray
    y_test            : np.ndarray
    y_pred            : np.ndarray
    class_names       : list[str]
    per_class_accuracy: list[float] = field(default_factory=list)
    per_class_counts  : list[int]   = field(default_factory=list)


def train(features: np.ndarray,
          labels: np.ndarray,
          class_names: list[str],
          test_size: float = 0.20,
          n_estimators: int = 100,
          max_depth: int = 15,
          min_samples_split: int = 5,
          random_state: int = 42) -> ClassificationResult:
    """
    Train a Random Forest classifier and evaluate on a held-out test set.

    Parameters
    ----------
    features : np.ndarray, shape (N, n_features)
    labels   : np.ndarray, shape (N,)
    class_names : list[str]
    test_size   : Fraction of data reserved for testing.
    n_estimators, max_depth, min_samples_split : RF hyperparameters.
    random_state : Seed for reproducibility.

    Returns
    -------
    ClassificationResult
    """
    # ── Train / test split ───────────────────────────────────────
    # stratify=labels: each class is represented proportionally in both splits.
    # This matters especially when classes are imbalanced.
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    # ── Normalization ────────────────────────────────────────────
    # IMPORTANT: fit scaler only on training data to avoid data leakage.
    # Data leakage = using test-set statistics during training → over-optimistic results.
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)   # fit + transform
    X_test_sc  = scaler.transform(X_test)         # transform only

    # ── Model ────────────────────────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train_sc, y_train)

    # ── Evaluation ───────────────────────────────────────────────
    y_pred   = model.predict(X_test_sc)
    accuracy = accuracy_score(y_test, y_pred)
    cm       = confusion_matrix(y_test, y_pred)
    report   = classification_report(y_test, y_pred, target_names=class_names)

    # Per-class accuracy
    per_class_acc    = []
    per_class_counts = []
    for i in range(len(class_names)):
        mask = y_test == i
        per_class_acc.append(float((y_pred[mask] == y_test[mask]).mean()) if mask.sum() > 0 else 0.0)
        per_class_counts.append(int(mask.sum()))

    return ClassificationResult(
        model=model,
        scaler=scaler,
        accuracy=accuracy,
        confusion_matrix=cm,
        report=report,
        feature_importances=model.feature_importances_,
        y_test=y_test,
        y_pred=y_pred,
        class_names=class_names,
        per_class_accuracy=per_class_acc,
        per_class_counts=per_class_counts,
    )
