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

Difficulty design note
──────────────────────
Defects are rendered with subtle contrast against a noisy,
variable-brightness background.  Random lighting gradients,
low-frequency surface texture, and per-image brightness offsets
ensure that a simple threshold on dark/bright pixel ratios is
NOT sufficient — the model must learn genuine structural features.
"""

import numpy as np
from typing import Literal

# All supported defect class names (order defines the integer label)
CLASS_NAMES = ["normal", "crack", "hole", "corrosion", "scratch"]

DefectType = Literal["normal", "crack", "hole", "corrosion", "scratch"]


def generate_image(defect_type: DefectType,
                   image_size: int = 64,
                   noise_level: float = 0.08) -> np.ndarray:
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
    base_brightness = np.random.uniform(0.43, 0.57)
    img = np.full((image_size, image_size), base_brightness, dtype=np.float64)

    # Low-frequency surface texture (blocky, like machining marks)
    q = image_size // 4
    low_freq = np.random.normal(0, 0.04, (q, q))
    low_freq = np.repeat(np.repeat(low_freq, 4, axis=0), 4, axis=1)[
        :image_size, :image_size
    ]
    img += low_freq
    img += np.random.normal(0, 0.03, img.shape)  # fine grain noise

    # Random lighting gradient (top-bottom or left-right)
    grad_dir = np.random.choice(["tb", "lr"])
    grad_strength = np.random.uniform(0.0, 0.08)
    if grad_dir == "tb":
        img += np.linspace(-grad_strength, grad_strength, image_size).reshape(-1, 1)
    else:
        img += np.linspace(-grad_strength, grad_strength, image_size).reshape(1, -1)

    # ── Defect rendering ──────────────────────────────────────────

    if defect_type == "normal":
        # Normal surfaces can still have minor blemishes (handling
        # scratches, small marks) — prevents "perfectly clean = normal".
        if np.random.random() < 0.4:
            r0 = np.random.randint(10, image_size - 10)
            c0 = np.random.randint(5, image_size - 20)
            for i in range(np.random.randint(5, 15)):
                r = r0 + np.random.randint(-1, 2)
                c = c0 + i
                if 0 <= r < image_size and 0 <= c < image_size:
                    img[r, c] += np.random.uniform(-0.04, 0.04)

        # Soft center highlight — overhead lighting
        y, x = np.ogrid[:image_size, :image_size]
        cx = cy = image_size // 2
        dist = np.sqrt((x - cx) * 2 + (y - cy) * 2)
        img += np.random.uniform(0.03, 0.08) * np.exp(
            -dist * 2 / (2 * (image_size // 3) * 2)
        )

    elif defect_type == "crack":
        # Dark line — moderate contrast against surface.
        crack_depth = np.random.uniform(0.12, 0.25)
        start   = np.random.randint(10, 30)
        length  = np.random.randint(15, 40)
        width   = np.random.choice([1, 1, 2, 2])  # 1–2 px wide
        angle   = np.random.choice(["horizontal", "diagonal", "vertical"])
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
            for dr in range(-width, width + 1):
                nr = r + dr
                if 0 <= nr < image_size and 0 <= c < image_size:
                    img[nr, c] -= crack_depth + np.random.uniform(-0.03, 0.03)

    elif defect_type == "hole":
        cr         = np.random.randint(20, 45)
        cc         = np.random.randint(20, 45)
        radius     = np.random.randint(4, 10)        # moderate (was 5–12)
        hole_depth = np.random.uniform(0.15, 0.30)   # moderate (was ~0.45)
        rim_bright = np.random.uniform(0.10, 0.22)   # moderate (was ~0.40)
        y, x   = np.ogrid[:image_size, :image_size]
        dist   = np.sqrt((x - cc) * 2 + (y - cr) * 2)
        img[dist < radius] -= hole_depth
        img[(dist >= radius) & (dist < radius + 2)] += rim_bright

    elif defect_type == "corrosion":
        # Fewer, smaller, subtler spots
        n_spots = np.random.randint(8, 25)
        for _ in range(n_spots):
            sr   = np.random.randint(5, image_size - 5)
            sc   = np.random.randint(5, image_size - 5)
            size = np.random.randint(2, 6)
            corr_depth = np.random.uniform(0.08, 0.20)
            y, x = np.ogrid[:image_size, :image_size]
            dist = np.sqrt((x - sc) * 2 + (y - sr) * 2)
            mask = dist < size
            img[mask] -= corr_depth + np.random.normal(0, 0.02, np.sum(mask))

    elif defect_type == "scratch":
        # Thinner, lower-contrast bright lines
        for _ in range(np.random.randint(1, 3)):
            scratch_bright = np.random.uniform(0.12, 0.25)
            r0     = np.random.randint(5, image_size - 10)
            c0     = np.random.randint(5, image_size - 20)
            length = np.random.randint(15, 40)
            for i in range(length):
                r = r0 + np.random.randint(-1, 2)
                c = c0 + i
                if 0 <= r < image_size and 0 <= c < image_size:
                    img[r, c] += scratch_bright + np.random.uniform(-0.02, 0.02)

    # Add Gaussian camera noise and clip to valid range
    img += np.random.normal(0, noise_level, img.shape)
    return np.clip(img, 0, 1).astype(np.float32)


def build_dataset(n_per_class: int = 200,
                  image_size: int = 64,
                  noise_level: float = 0.08,
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
