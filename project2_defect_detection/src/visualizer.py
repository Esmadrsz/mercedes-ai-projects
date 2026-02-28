"""
visualizer.py
─────────────────────────────────────────────────────────────
Dashboard renderer for defect detection results.

Layout (3 rows):
    Row 0 (full width) : Sample images — 3 examples per class
    Row 1 left (×2)    : Normalized confusion matrix
    Row 1 right        : Top-12 feature importance bar chart
    Row 2 (full width) : Per-class accuracy bar chart
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .classifier import ClassificationResult
from .features   import FEATURE_NAMES

_BG      = "#0a0e1a"
_PANEL   = "#111827"
_CARD    = "#1f2937"
_TEXT    = "#f1f5f9"
_SUB     = "#94a3b8"
_GRID    = "#1e293b"
_ACCENT  = "#3b82f6"

# One distinct color per defect class
CLASS_COLORS = ["#10b981", "#ef4444", "#f59e0b", "#8b5cf6", "#06b6d4"]


def render(images: np.ndarray,
           labels: np.ndarray,
           result: ClassificationResult,
           output_path: str = "defect_detection_results.png",
           dpi: int = 150) -> None:
    """
    Build and save the full results dashboard.

    Parameters
    ----------
    images      : Raw image array (N, H, W)
    labels      : Ground-truth label array (N,)
    result      : ClassificationResult from classifier.train()
    output_path : Where to save the PNG.
    dpi         : Output resolution.
    """
    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor(_BG)

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.45, wspace=0.35,
                           left=0.05, right=0.97,
                           top=0.93, bottom=0.06)

    # ── Header ──────────────────────────────────────────────────
    fig.text(0.5, 0.97,
             "Mercedes-Benz  |  AI Quality Control — Synthetic Defect Detection",
             color=_TEXT, fontsize=17, fontweight="bold", ha="center")
    fig.text(0.5, 0.945,
             f"Random Forest  |  5-Class Defect Recognition  |  "
             f"Overall Accuracy: {result.accuracy:.1%}",
             color=_SUB, fontsize=11, ha="center")

    class_names = result.class_names

    # ── Row 0: Sample images ─────────────────────────────────────
    ax_imgs = fig.add_subplot(gs[0, :])
    ax_imgs.set_facecolor(_PANEL)
    ax_imgs.axis("off")
    ax_imgs.set_title("Sample Images by Class  (3 per class)",
                       color=_TEXT, fontsize=13, fontweight="bold", pad=10)

    n_cols    = len(class_names)
    n_samples = 3

    for col, (cls_idx, cls_name) in enumerate(zip(range(n_cols), class_names)):
        cls_imgs  = images[labels == cls_idx]
        chosen    = np.random.choice(len(cls_imgs), n_samples, replace=False)

        for row in range(n_samples):
            left   = 0.03 + col * 0.19
            bottom = 0.03 + (n_samples - 1 - row) * 0.27
            ax_in  = ax_imgs.inset_axes([left, bottom, 0.16, 0.24])
            ax_in.imshow(cls_imgs[chosen[row]], cmap="gray", vmin=0, vmax=1)
            ax_in.axis("off")
            if row == 0:
                ax_in.set_title(cls_name.upper(),
                                color=CLASS_COLORS[cls_idx],
                                fontsize=9, fontweight="bold")

    # ── Row 1 left: Confusion matrix ─────────────────────────────
    ax_cm = fig.add_subplot(gs[1, :2])
    ax_cm.set_facecolor(_PANEL)

    cm_norm = result.confusion_matrix.astype(float)
    cm_norm /= cm_norm.sum(axis=1, keepdims=True)

    im = ax_cm.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    ax_cm.set_xticks(range(len(class_names)))
    ax_cm.set_yticks(range(len(class_names)))
    ax_cm.set_xticklabels([c.upper() for c in class_names],
                           color=_SUB, fontsize=9, rotation=30)
    ax_cm.set_yticklabels([c.upper() for c in class_names],
                           color=_SUB, fontsize=9)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm_norm[i, j] > 0.5 else _TEXT
            ax_cm.text(j, i, f"{cm_norm[i, j]:.2f}",
                       ha="center", va="center",
                       color=color, fontsize=10, fontweight="bold")

    ax_cm.set_xlabel("Predicted", color=_SUB)
    ax_cm.set_ylabel("True", color=_SUB)
    ax_cm.set_title("Confusion Matrix (Normalized)",
                     color=_TEXT, fontsize=12, fontweight="bold", pad=10)
    ax_cm.tick_params(colors=_SUB)

    # ── Row 1 right: Feature importance ──────────────────────────
    ax_fi = fig.add_subplot(gs[1, 2])
    ax_fi.set_facecolor(_PANEL)

    n_top   = 12
    imp     = result.feature_importances
    top_idx = np.argsort(imp)[-n_top:][::-1]
    top_imp = imp[top_idx]
    top_names = [FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feat_{i}"
                 for i in top_idx]

    bar_colors = plt.cm.RdYlGn(top_imp / (top_imp.max() + 1e-8))
    ax_fi.barh(range(n_top), top_imp[::-1],
               color=bar_colors[::-1], alpha=0.9, edgecolor="none")
    ax_fi.set_yticks(range(n_top))
    ax_fi.set_yticklabels(top_names[::-1], color=_SUB, fontsize=8)
    ax_fi.set_xlabel("Importance", color=_SUB, fontsize=9)
    ax_fi.set_title("Top Feature Importance",
                     color=_TEXT, fontsize=11, fontweight="bold", pad=10)
    ax_fi.tick_params(colors=_SUB)
    for sp in ax_fi.spines.values():
        sp.set_color(_GRID)
    ax_fi.grid(True, alpha=0.15, color=_GRID, axis="x")
    ax_fi.set_facecolor(_PANEL)

    # ── Row 2: Per-class accuracy ────────────────────────────────
    ax_cls = fig.add_subplot(gs[2, :])
    ax_cls.set_facecolor(_PANEL)

    x    = np.arange(len(class_names))
    bars = ax_cls.bar(x, result.per_class_accuracy,
                      color=CLASS_COLORS, alpha=0.85,
                      edgecolor="none", width=0.6)

    for bar, acc, cnt in zip(bars, result.per_class_accuracy, result.per_class_counts):
        ax_cls.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{acc:.1%}\n(n={cnt})",
                    ha="center", color=_TEXT, fontsize=10, fontweight="bold")

    ax_cls.axhline(result.accuracy, color="white", ls="--", lw=2, alpha=0.6,
                   label=f"Overall accuracy: {result.accuracy:.1%}")
    ax_cls.set_xticks(x)
    ax_cls.set_xticklabels([n.upper() for n in class_names],
                            color=_TEXT, fontsize=11, fontweight="bold")
    ax_cls.set_ylabel("Accuracy", color=_SUB)
    ax_cls.set_ylim(0, 1.15)
    ax_cls.set_title("Per-Class Accuracy",
                      color=_TEXT, fontsize=12, fontweight="bold", pad=10)
    ax_cls.legend(facecolor=_CARD, labelcolor=_TEXT)
    ax_cls.tick_params(colors=_SUB)
    for sp in ax_cls.spines.values():
        sp.set_color(_GRID)
    ax_cls.grid(True, alpha=0.15, color=_GRID, axis="y")

    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor=_BG, edgecolor="none")
    print(f"  [OK] Results saved → {output_path}")
    plt.close()
