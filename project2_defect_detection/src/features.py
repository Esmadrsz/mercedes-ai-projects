"""
features.py
─────────────────────────────────────────────────────────────
Statistical and texture feature extraction from grayscale images.

Instead of feeding raw pixels (64×64 = 4096 values) directly into
the classifier, we compute 34 meaningful statistics per image.

Benefits:
    · Much lower dimensionality → faster, less prone to overfitting
    · Interpretable: we can explain what each feature captures
    · Strong baseline: often beats deep learning on small datasets

Feature groups
--------------
1. Global statistics (8)
   Mean, std, min, max, median, Q25, Q75, IQR

2. Regional statistics (18)
   The image is divided into a 3×3 grid of sub-regions.
   For each region: mean and std of pixel brightness.
   Captures WHERE in the image the anomaly appears.

3. Edge / gradient strength (4)
   Pixel-to-pixel differences along x and y axes.
   High gradient = sharp edges → cracks, hole rims.

4. Texture descriptors (4)
   Dark pixel ratio  : fraction of pixels below 0.30
   Bright pixel ratio: fraction of pixels above 0.80
   Mid-tone ratio    : fraction of pixels in [0.40, 0.60]
   Entropy           : Shannon entropy of the brightness histogram
"""

import numpy as np

N_FEATURES = 34   # Total number of features per image


def _feature_names() -> list[str]:
    """Return human-readable names for all 34 features (used in plots)."""
    names = ["mean", "std", "min", "max", "median", "q25", "q75", "iqr"]
    for i in range(9):
        names += [f"region_{i}_mean", f"region_{i}_std"]
    names += ["grad_x_mean", "grad_y_mean", "grad_x_std", "grad_y_std"]
    names += ["dark_ratio", "bright_ratio", "mid_ratio", "entropy"]
    return names


FEATURE_NAMES = _feature_names()


def extract_one(img: np.ndarray) -> np.ndarray:
    """
    Extract a feature vector from a single image.

    Parameters
    ----------
    img : np.ndarray, shape (H, W), values in [0, 1]

    Returns
    -------
    np.ndarray, shape (N_FEATURES,), dtype float32
    """
    f = []

    # ── 1. Global statistics ────────────────────────────────────
    f.append(float(np.mean(img)))
    f.append(float(np.std(img)))
    f.append(float(np.min(img)))
    f.append(float(np.max(img)))
    f.append(float(np.median(img)))
    f.append(float(np.percentile(img, 25)))
    f.append(float(np.percentile(img, 75)))
    f.append(float(np.percentile(img, 75) - np.percentile(img, 25)))

    # ── 2. Regional statistics (3×3 grid) ───────────────────────
    H, W = img.shape
    for i in range(3):
        for j in range(3):
            region = img[i * H // 3:(i + 1) * H // 3,
                         j * W // 3:(j + 1) * W // 3]
            f.append(float(np.mean(region)))
            f.append(float(np.std(region)))

    # ── 3. Edge / gradient strength ─────────────────────────────
    # np.diff computes adjacent pixel differences (first-order gradient)
    grad_x = np.diff(img, axis=1)   # horizontal edges
    grad_y = np.diff(img, axis=0)   # vertical edges
    f.append(float(np.mean(np.abs(grad_x))))
    f.append(float(np.mean(np.abs(grad_y))))
    f.append(float(np.std(grad_x)))
    f.append(float(np.std(grad_y)))

    # ── 4. Texture descriptors ───────────────────────────────────
    f.append(float(np.mean(img < 0.30)))                        # dark  (cracks, holes)
    f.append(float(np.mean(img > 0.80)))                        # bright (scratches, rims)
    f.append(float(np.mean((img >= 0.40) & (img <= 0.60))))    # mid-tone (normal surface)

    # Shannon entropy of a 20-bin histogram of brightness
    hist, _ = np.histogram(img, bins=20, range=(0, 1))
    prob     = hist / (hist.sum() + 1e-10)
    entropy  = -float(np.sum(prob * np.log2(prob + 1e-10)))
    f.append(entropy)

    return np.array(f, dtype=np.float32)


def extract_all(images: np.ndarray) -> np.ndarray:
    """
    Extract features from an array of images.

    Parameters
    ----------
    images : np.ndarray, shape (N, H, W)

    Returns
    -------
    np.ndarray, shape (N, N_FEATURES), dtype float32
    """
    features = np.vstack([extract_one(img) for img in images])
    assert features.shape[1] == N_FEATURES
    return features
