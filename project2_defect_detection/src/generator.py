"""
generator.py
─────────────────────────────────────────────────────────────
Synthetic factory part image generator.

Produces grayscale images (pixel values in [0, 1]) that
simulate a camera inspection system on a manufacturing line.

Five classes are supported:
    normal     — clean metal surface with soft lighting gradient
    crack      — thin dark line (horizontal / vertical / diagonal)
    hole       — circular void with bright metallic rim
    corrosion  — irregular dark spots scattered across the surface
    scratch    — thin bright line (light reflecting off a groove)

Why synthetic data?
    Real defect images are scarce and proprietary.
    Synthetic generation allows unlimited labeled training data —
    the same principle used by NVIDIA Omniverse and Unity at scale.

Design note
───────────
Defects are rendered with moderate contrast against a noisy,
variable-brightness background. Random lighting gradients and
per-image brightness offsets ensure the model must learn genuine
structural features — not just simple brightness thresholds.
Target accuracy: ~88% (realistic for real-world inspection systems).
"""

import numpy as np
from typing import Literal

CLASS_NAMES = ["normal", "crack", "hole", "corrosion", "scratch"]
DefectType = Literal["normal", "crack", "hole", "corrosion", "scratch"]


def generate_image(defect_type: DefectType,
                   image_size: int = 64,
                   noise_level: float = 0.10) -> np.ndarray:
    """
    Generate one synthetic grayscale image of a factory part.

    Parameters
    ----------
    defect_type : str
        One of CLASS_NAMES.
    image_size : int
        Width and height in pixels.
    noise_level : float
        Std-dev of Gaussian camera noise added to every image.

    Returns
    -------
    np.ndarray, shape (image_size, image_size), dtype float32, range [0, 1]
    """
    # ── Base surface ──────────────────────────────────────────────
    # Random per-image brightness offset simulates varying lighting
    # conditions across different camera positions on the line.
    base_brightness = np.random.uniform(0.44, 0.56)
    img = np.full((image_size, image_size), base_brightness, dtype=np.float64)

    # Low-frequency surface texture (simulates machining marks)
    q = image_size // 4
    low_freq = np.random.normal(0, 0.03, (q, q))
    low_freq = np.repeat(np.repeat(low_freq, 4, axis=0), 4, axis=1)[:image_size, :image_size]
    img += low_freq

    # Fine grain camera noise
    img += np.random.normal(0, 0.025, img.shape)

    # Random illumination gradient (uneven light source)
    grad_strength = np.random.uniform(0.0, 0.06)
    if np.random.random() < 0.5:
        img += np.linspace(-grad_strength, grad_strength, image_size).reshape(-1, 1)
    else:
        img += np.linspace(-grad_strength, grad_strength, image_size).reshape(1, -1)

    # ── Defect rendering ──────────────────────────────────────────

    if defect_type == "normal":
        # Soft center highlight — overhead lighting gradient
        y, x = np.ogrid[:image_size, :image_size]
        cx = cy = image_size // 2
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        img += 0.06 * np.exp(-dist**2 / (2 * (image_size // 3)**2))

    elif defect_type == "crack":
        # Thin dark line — moderate contrast against surface
        crack_depth = np.random.uniform(0.18, 0.28)
        start  = np.random.randint(10, 30)
        length = np.random.randint(18, 40)
        angle  = np.random.choice(["horizontal", "diagonal", "vertical"])
        for i in range(length):
            if angle == "horizontal":
                r = image_size // 2 + np.random.randint(-2, 3)
                c = start + i
            elif angle == "vertical":
                c = image_size // 2 + np.random.randint(-2, 3)
                r = start + i
            else:
                r = start + i
                c = start + i
            for dr in range(-1, 2):
                nr = r + dr
                if 0 <= nr < image_size and 0 <= c < image_size:
                    img[nr, c] -= crack_depth + np.random.uniform(-0.03, 0.03)

    elif defect_type == "hole":
        # Circular void with bright metallic rim
        cr         = np.random.randint(20, 45)
        cc         = np.random.randint(20, 45)
        radius     = np.random.randint(5, 11)
        hole_depth = np.random.uniform(0.20, 0.35)
        rim_bright = np.random.uniform(0.15, 0.25)
        y, x = np.ogrid[:image_size, :image_size]
        dist = np.sqrt((x - cc)**2 + (y - cr)**2)
        img[dist < radius]                           -= hole_depth
        img[(dist >= radius) & (dist < radius + 2)] += rim_bright

    elif defect_type == "corrosion":
        # Scattered dark spots of varying size
        n_spots = np.random.randint(10, 28)
        for _ in range(n_spots):
            sr         = np.random.randint(5, image_size - 5)
            sc         = np.random.randint(5, image_size - 5)
            size       = np.random.randint(2, 6)
            corr_depth = np.random.uniform(0.10, 0.20)
            y, x = np.ogrid[:image_size, :image_size]
            dist = np.sqrt((x - sc)**2 + (y - sr)**2)
            mask = dist < size
            img[mask] -= corr_depth + np.random.normal(0, 0.02, np.sum(mask))

    elif defect_type == "scratch":
        # Thin bright line — light reflecting off a surface groove
        for _ in range(np.random.randint(1, 3)):
            scratch_bright = np.random.uniform(0.18, 0.28)
            r0     = np.random.randint(5, image_size - 10)
            c0     = np.random.randint(5, image_size - 20)
            length = np.random.randint(18, 45)
            for i in range(length):
                r = r0 + np.random.randint(-1, 2)
                c = c0 + i
                if 0 <= r < image_size and 0 <= c < image_size:
                    img[r, c] += scratch_bright + np.random.uniform(-0.02, 0.02)

    # Final camera noise
    img += np.random.normal(0, noise_level, img.shape)
    return np.clip(img, 0, 1).astype(np.float32)


def build_dataset(n_per_class: int = 200,
                  image_size: int = 64,
                  noise_level: float = 0.10,
                  random_seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a complete labeled dataset for all five defect classes.

    Parameters
    ----------
    n_per_class : int
        Number of images to generate per class.
    image_size : int
        Pixel resolution (square).
    noise_level : float
        Camera noise level.
    random_seed : int
        NumPy seed for reproducibility.

    Returns
    -------
    images : np.ndarray, shape (n_total, image_size, image_size)
    labels : np.ndarray, shape (n_total,), int values 0–4
    """
    np.random.seed(random_seed)
    images, labels = [], []
    for class_idx, name in enumerate(CLASS_NAMES):
        print(f"  Generating class '{name}': {n_per_class} images...")
        for _ in range(n_per_class):
            images.append(generate_image(name, image_size, noise_level))
            labels.append(class_idx)
    return np.array(images), np.array(labels)

