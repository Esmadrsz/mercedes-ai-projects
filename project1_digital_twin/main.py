"""
main.py — Entry point for the Digital Twin Dashboard.

Usage examples
--------------
# Default run (uses config/config.ini values):
    python main.py

# Custom parameters via command line:
    python main.py --n_points 3000 --anomaly_rate 0.08 --output my_dashboard.png

# Quiet mode (no plots shown, only saved):
    python main.py --no_show
"""

import argparse
import configparser
import logging
import sys
import time
from pathlib import Path

# ── make src importable ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from src.simulator import simulate
from src.detector  import detect
from src.dashboard import render

# ── logging setup ────────────────────────────────────────────
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
    sim = cfg["simulation"]
    mod = cfg["model"]
    out = cfg["output"]

    parser = argparse.ArgumentParser(
        description="Mercedes-Benz AI Digital Twin — Vehicle Sensor Monitor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n_points",      type=int,   default=int(sim["n_points"]),
                        help="Number of simulated time steps")
    parser.add_argument("--anomaly_rate",  type=float, default=float(sim["anomaly_rate"]),
                        help="Fraction of time steps with an injected fault")
    parser.add_argument("--seed",          type=int,   default=int(sim["random_seed"]),
                        help="Random seed for reproducibility")
    parser.add_argument("--contamination", type=float, default=float(mod["contamination"]),
                        help="Expected anomaly fraction (Isolation Forest hint)")
    parser.add_argument("--n_estimators",  type=int,   default=int(mod["n_estimators"]),
                        help="Number of Isolation Forest trees")
    parser.add_argument("--output",        type=str,   default=out["dashboard_filename"],
                        help="Output PNG filename")
    parser.add_argument("--csv",           type=str,   default=out["csv_filename"],
                        help="Output CSV filename (set to '' to skip)")
    parser.add_argument("--no_show",       action="store_true",
                        help="Save dashboard without displaying it")
    return parser.parse_args()


def main() -> None:
    t_start = time.perf_counter()

    print("=" * 65)
    print("  Mercedes-Benz AI  |  Digital Twin Dashboard")
    print("  Vehicle Sensor Monitor with Unsupervised Anomaly Detection")
    print("=" * 65)

    cfg  = load_config()
    args = parse_args(cfg)

    # ── Step 1: Simulate sensor data ────────────────────────────
    log.info("Simulating vehicle sensor data...")
    df = simulate(
        n_points=args.n_points,
        anomaly_rate=args.anomaly_rate,
        random_seed=args.seed,
    )
    n_faults = df["true_anomaly"].sum()
    log.info(f"Generated {args.n_points} time steps  |  "
             f"{n_faults} faults injected ({args.anomaly_rate:.0%})")

    # ── Step 2: Detect anomalies ─────────────────────────────────
    log.info("Running Isolation Forest anomaly detection...")
    df, metrics = detect(
        df,
        contamination=args.contamination,
        n_estimators=args.n_estimators,
    )
    log.info(
        f"Detected {metrics.n_detected} anomalies  |  "
        f"Precision {metrics.precision:.2%}  "
        f"Recall {metrics.recall:.2%}  "
        f"F1 {metrics.f1:.2%}"
    )

    # ── Step 3: Render dashboard ─────────────────────────────────
    log.info("Rendering dashboard...")

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")   # Non-interactive backend (no window)

    render(df, metrics, output_path=args.output,
           dpi=int(cfg["output"]["dpi"]))

    # ── Step 4: Export CSV ───────────────────────────────────────
    if args.csv:
        df.to_csv(args.csv, index=False)
        log.info(f"Sensor data saved → {args.csv}")

    elapsed = time.perf_counter() - t_start
    print(f"\n{'=' * 65}")
    print(f"  DONE in {elapsed:.1f}s")
    print(f"  Output : {args.output}")
    if args.csv:
        print(f"  Data   : {args.csv}")
    print("=" * 65)


if __name__ == "__main__":
    main()
