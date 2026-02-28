"""
dashboard.py
─────────────────────────────────────────────────────────────
Renders the Digital Twin monitoring dashboard.

Layout (4 rows × 4 columns):
    Rows 0-3, cols 0-2  — Sensor time series (Speed, Temp, Battery, Score)
    Rows 0-1, col 3     — Engine Temp vs Speed scatter (anomaly clusters)
    Row  2,   col 3     — AI performance metrics card
    Row  3,   col 3     — Anomaly distribution pie chart
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from .detector import DetectionMetrics


# Mercedes dark theme palette
_C = {
    "bg"      : "#0a0e1a",
    "panel"   : "#111827",
    "card"    : "#1f2937",
    "normal"  : "#00d4aa",
    "anomaly" : "#ff4757",
    "speed"   : "#00b4d8",
    "temp"    : "#ff6b35",
    "battery" : "#ffd60a",
    "text"    : "#f1f5f9",
    "sub"     : "#94a3b8",
    "grid"    : "#1e293b",
    "accent"  : "#3b82f6",
}


def _style(ax, title: str, ylabel: str = "", xlabel: str = "") -> None:
    """Apply consistent dark-theme styling to an axes panel."""
    ax.set_facecolor(_C["panel"])
    ax.set_title(title, color=_C["text"], fontsize=11, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, color=_C["sub"], fontsize=9)
    ax.set_xlabel(xlabel, color=_C["sub"], fontsize=9)
    ax.tick_params(colors=_C["sub"], labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(_C["grid"])
    ax.grid(True, alpha=0.15, color=_C["grid"], linewidth=0.5)


def render(df: pd.DataFrame,
           metrics: DetectionMetrics,
           output_path: str = "digital_twin_dashboard.png",
           dpi: int = 150) -> None:
    """
    Build and save the full monitoring dashboard.

    Parameters
    ----------
    df          : DataFrame with ai_anomaly and anomaly_score_norm columns.
    metrics     : DetectionMetrics from detector.detect().
    output_path : Where to save the PNG file.
    dpi         : Output resolution.
    """
    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor(_C["bg"])

    gs = gridspec.GridSpec(4, 4, figure=fig,
                           hspace=0.45, wspace=0.30,
                           left=0.06, right=0.97,
                           top=0.92, bottom=0.07)

    # ── Header ──────────────────────────────────────────────────
    fig.text(0.5, 0.960,
             "Mercedes-Benz  |  AI Digital Twin — Vehicle Sensor Monitor",
             color=_C["text"], fontsize=18, fontweight="bold",
             ha="center", fontfamily="monospace")
    fig.text(0.5, 0.935,
             "Real-time Anomaly Detection  |  Isolation Forest (Unsupervised ML)  |  5 Sensors",
             color=_C["sub"], fontsize=11, ha="center")

    mask = df["ai_anomaly"] == 1   # Boolean: True where AI flags an anomaly

    # ── Speed ───────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :3])
    _style(ax1, "Vehicle Speed", "km/h")
    ax1.plot(df["time_s"], df["speed_kmh"],
             color=_C["speed"], lw=0.8, alpha=0.9)
    ax1.scatter(df.loc[mask, "time_s"], df.loc[mask, "speed_kmh"],
                color=_C["anomaly"], s=20, zorder=5, alpha=0.8,
                label="Anomaly flagged")
    ax1.legend(facecolor=_C["card"], labelcolor=_C["sub"], fontsize=8)

    # ── Engine Temperature ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :3])
    _style(ax2, "Engine Temperature", "°C")
    ax2.plot(df["time_s"], df["engine_temp_c"],
             color=_C["temp"], lw=0.8, alpha=0.9)
    ax2.axhline(100, color=_C["anomaly"], ls="--", lw=1, alpha=0.6,
                label="Critical threshold: 100°C")
    ax2.scatter(df.loc[mask, "time_s"], df.loc[mask, "engine_temp_c"],
                color=_C["anomaly"], s=20, zorder=5, alpha=0.8)
    ax2.legend(facecolor=_C["card"], labelcolor=_C["sub"], fontsize=8)

    # ── Battery Voltage ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, :3])
    _style(ax3, "Battery Voltage", "Volt")
    ax3.plot(df["time_s"], df["battery_v"],
             color=_C["battery"], lw=0.8, alpha=0.9)
    ax3.axhline(11.5, color=_C["anomaly"], ls="--", lw=1, alpha=0.6,
                label="Minimum safe: 11.5 V")
    ax3.scatter(df.loc[mask, "time_s"], df.loc[mask, "battery_v"],
                color=_C["anomaly"], s=20, zorder=5, alpha=0.8)
    ax3.legend(facecolor=_C["card"], labelcolor=_C["sub"], fontsize=8)

    # ── AI Anomaly Score ─────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3, :3])
    _style(ax4, "AI Anomaly Score  (0 = Normal  ·  1 = Anomaly)",
           "Score", "Time (seconds)")
    ax4.fill_between(df["time_s"], df["anomaly_score_norm"],
                     where=df["anomaly_score_norm"] < 0.6,
                     color=_C["normal"], alpha=0.4, label="Normal zone")
    ax4.fill_between(df["time_s"], df["anomaly_score_norm"],
                     where=df["anomaly_score_norm"] >= 0.6,
                     color=_C["anomaly"], alpha=0.6, label="Anomaly detected")
    ax4.axhline(0.6, color="white", ls="--", lw=1, alpha=0.4,
                label="Detection threshold")
    ax4.legend(facecolor=_C["card"], labelcolor=_C["sub"], fontsize=8)

    # ── Scatter: Engine Temp vs Speed ────────────────────────────
    ax5 = fig.add_subplot(gs[0:2, 3])
    _style(ax5, "Engine Temp vs Speed\n(Anomaly Clusters)",
           "Temp (°C)", "Speed (km/h)")
    norm_d = df[df["ai_anomaly"] == 0]
    anom_d = df[df["ai_anomaly"] == 1]
    ax5.scatter(norm_d["speed_kmh"], norm_d["engine_temp_c"],
                c=_C["normal"], s=5, alpha=0.4, label="Normal")
    ax5.scatter(anom_d["speed_kmh"], anom_d["engine_temp_c"],
                c=_C["anomaly"], s=30, alpha=0.9, label="Anomaly", zorder=5)
    ax5.legend(facecolor=_C["card"], labelcolor=_C["sub"], fontsize=8)

    # ── Metrics Card ─────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 3])
    ax6.set_facecolor(_C["card"])
    ax6.axis("off")
    lines = [
        ("AI MODEL METRICS",                         _C["accent"],   12, "bold"),
        ("",                                          _C["sub"],       8, "normal"),
        (f"Precision    {metrics.precision:.2%}",     _C["normal"],   11, "bold"),
        (f"Recall       {metrics.recall:.2%}",        _C["normal"],   11, "bold"),
        (f"F1 Score     {metrics.f1:.2%}",             _C["battery"],  12, "bold"),
        ("",                                          _C["sub"],       8, "normal"),
        ("Algorithm : Isolation Forest",              _C["sub"],       9, "normal"),
        ("Mode       : Unsupervised",                 _C["sub"],       9, "normal"),
        (f"Detected   : {metrics.n_detected} faults", _C["sub"],      9, "normal"),
    ]
    y = 0.95
    for txt, col, sz, wt in lines:
        ax6.text(0.08, y, txt, color=col, fontsize=sz, fontweight=wt,
                 transform=ax6.transAxes, fontfamily="monospace")
        y -= 0.11

    # ── Pie Chart ────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[3, 3])
    ax7.set_facecolor(_C["panel"])
    n_norm = (df["ai_anomaly"] == 0).sum()
    n_anom = (df["ai_anomaly"] == 1).sum()
    wedges, texts, autotexts = ax7.pie(
        [n_norm, n_anom],
        labels=["Normal", "Anomaly"],
        colors=[_C["normal"], _C["anomaly"]],
        autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(edgecolor=_C["bg"], linewidth=2),
    )
    for t in texts:
        t.set_color(_C["sub"]); t.set_fontsize(9)
    for t in autotexts:
        t.set_color(_C["bg"]); t.set_fontweight("bold"); t.set_fontsize(9)
    ax7.set_title("Anomaly Distribution",
                  color=_C["text"], fontsize=10, fontweight="bold", pad=5)

    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor=_C["bg"], edgecolor="none")
    print(f"  [OK] Dashboard saved → {output_path}")
    plt.close()
