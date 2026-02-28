"""
main.py — Entry point for the Defect Detection pipeline.

Usage examples
--------------
    python main.py                          # Default (config/config.ini)
    python main.py --n_per_class 300        # More training data
    python main.py --no_show                # Save without opening a window
    python main.py --help                   # Full option list
"""

import argparse
import configparser
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.generator  import build_dataset, CLASS_NAMES
from src.features   import extract_all
from src.classifier import train
from src.visualizer import render

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_config(path: str = "config/config.ini") -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(path)
    return cfg


def parse_args(cfg: configparser.ConfigParser) -> argparse.Namespace:
    ds  = cfg["dataset"]
    mod = cfg["model"]
    out = cfg["output"]

    parser = argparse.ArgumentParser(
        description="Mercedes-Benz AI Quality Control — Synthetic Defect Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n_per_class",       type=int,   default=int(ds["n_per_class"]))
    parser.add_argument("--image_size",        type=int,   default=int(ds["image_size"]))
    parser.add_argument("--noise_level",       type=float, default=float(ds["noise_level"]))
    parser.add_argument("--seed",              type=int,   default=int(ds["random_seed"]))
    parser.add_argument("--n_estimators",      type=int,   default=int(mod["n_estimators"]))
    parser.add_argument("--max_depth",         type=int,   default=int(mod["max_depth"]))
    parser.add_argument("--min_samples_split", type=int,   default=int(mod["min_samples_split"]))
    parser.add_argument("--test_size",         type=float, default=float(mod["test_size"]))
    parser.add_argument("--output",            type=str,   default=out["results_filename"])
    parser.add_argument("--no_show",           action="store_true")
    return parser.parse_args()


def main() -> None:
    t_start = time.perf_counter()

    print("=" * 65)
    print("  Mercedes-Benz AI  |  Quality Control")
    print("  Synthetic Defect Detection — 5-Class Classifier")
    print("=" * 65)

    cfg  = load_config()
    args = parse_args(cfg)

    # ── Step 1: Generate dataset ─────────────────────────────────
    log.info("Generating synthetic part images...")
    images, labels = build_dataset(
        n_per_class=args.n_per_class,
        image_size=args.image_size,
        noise_level=args.noise_level,
        random_seed=args.seed,
    )
    log.info(f"Dataset: {len(images)} images  |  "
             f"{len(CLASS_NAMES)} classes  |  "
             f"{args.image_size}×{args.image_size} px")

    # ── Step 2: Extract features ─────────────────────────────────
    log.info("Extracting statistical features from images...")
    features = extract_all(images)
    log.info(f"Feature matrix: {features.shape}  "
             f"({features.shape[1]} features per image)")

    # ── Step 3: Train and evaluate ───────────────────────────────
    log.info("Training Random Forest classifier...")
    result = train(
        features, labels, CLASS_NAMES,
        test_size=args.test_size,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=args.seed,
    )
    log.info(f"Accuracy: {result.accuracy:.2%}")
    print(f"\n{result.report}")

    # ── Step 4: Visualize ────────────────────────────────────────
    log.info("Rendering results dashboard...")

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")

    render(images, labels, result,
           output_path=args.output,
           dpi=int(cfg["output"]["dpi"]))

    elapsed = time.perf_counter() - t_start
    print(f"\n{'=' * 65}")
    print(f"  DONE in {elapsed:.1f}s  |  Accuracy: {result.accuracy:.1%}")
    print(f"  Output: {args.output}")
    print("=" * 65)


if __name__ == "__main__":
    main()
