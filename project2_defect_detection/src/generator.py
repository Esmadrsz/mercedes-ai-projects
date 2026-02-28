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
"""

import numpy as np
from dataclasses import dataclass
from typing import Literal

# All supported defect class names (order defines the integer label)
CLASS_NAMES = ["normal", "crack", "hole", "corrosion", "scratch"]

DefectType = Literal["normal", "crack", "hole", "corrosion", "scratch"]


def generate_image(defect_type: DefectType,
                   image_size: int = 64,
                   noise_level: float = 0.05) -> np.ndarray:
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
    # Base: uniform mid-gray = clean metal surface
    img = np.full((image_size, image_size), 0.5, dtype=np.float64)

    # Subtle surface texture: small random brightness variations
    img += np.random.normal(0, 0.03, img.shape)

    if defect_type == "normal":
        # Soft center highlight — simulates overhead lighting
        y, x = np.ogrid[:image_size, :image_size]
        cx   = cy = image_size // 2
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        img += 0.1 * np.exp(-dist ** 2 / (2 * (image_size // 3) ** 2))

    elif defect_type == "crack":
        start   = np.random.randint(10, 30)
        length  = np.random.randint(20, 45)
        angle   = np.random.choice(["horizontal", "diagonal", "vertical"])
        for i in range(length):
            if angle == "horizontal":
                r = image_size // 2 + np.random.randint(-3, 3)
                c = start + i
            elif angle == "vertical":
                c = image_size // 2 + np.random.randint(-3, 3)
                r = start + i
            else:
                r = start + i
                c = start + i
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < image_size and 0 <= nc < image_size:
                        img[nr, nc] = 0.1 + np.random.uniform(0, 0.05)

    elif defect_type == "hole":
        cr     = np.random.randint(20, 45)
        cc     = np.random.randint(20, 45)
        radius = np.random.randint(5, 12)
        y, x   = np.ogrid[:image_size, :image_size]
        dist   = np.sqrt((x - cc) ** 2 + (y - cr) ** 2)
        img[dist < radius]                           = 0.05  # dark void
        img[(dist >= radius) & (dist < radius + 3)]  = 0.90  # bright rim

    elif defect_type == "corrosion":
        for _ in range(np.random.randint(15, 40)):
            sr   = np.random.randint(5, image_size - 5)
            sc   = np.random.randint(5, image_size - 5)
            size = np.random.randint(2, 8)
            y, x = np.ogrid[:image_size, :image_size]
            dist = np.sqrt((x - sc) ** 2 + (y - sr) ** 2)
            img[dist < size] = np.random.uniform(0.2, 0.35)

    elif defect_type == "scratch":
        for _ in range(np.random.randint(1, 4)):
            r0 = np.random.randint(5, image_size // 2)
            c0 = np.random.randint(5, image_size - 30)
            length = np.random.randint(15, 50)
            for i in range(length):
                r = r0 + np.random.randint(-1, 2)
                c = c0 + i
                if 0 <= r < image_size and 0 <= c < image_size:
                    img[r, c] = 0.85

    # Add Gaussian camera noise and clip to valid range
    img += np.random.normal(0, noise_level, img.shape)
    return np.clip(img, 0, 1).astype(np.float32)


def build_dataset(n_per_class: int = 200,
                  image_size: int = 64,
                  noise_level: float = 0.05,
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

    images = []
    labels = []

    for class_idx, name in enumerate(CLASS_NAMES):
        print(f"  Generating class '{name}': {n_per_class} images...")
        for _ in range(n_per_class):
            images.append(generate_image(name, image_size, noise_level))
            labels.append(class_idx)

    return np.array(images), np.array(labels)
