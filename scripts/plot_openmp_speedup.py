"""Generate OpenMP speedup table figure and Amdahl comparison plot.

Usage:
    python scripts/plot_openmp_speedup.py

Outputs:
    reporte/evidencias/openmp_speedup_table.png
    reporte/evidencias/openmp_amdahl_analysis.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "reporte" / "evidencias"
METRICS_PATH = EVIDENCE_DIR / "openmp_benchmark.json"

THREADS = np.array([1, 2, 4, 8], dtype=float)
TIME_SEC = np.array([27.262, 17.325, 14.155, 13.449], dtype=float)
IMAGE_COUNT = 3312


def speedup(time_sec: np.ndarray) -> np.ndarray:
    return time_sec[0] / time_sec


def estimate_serial_fraction(time_sec: np.ndarray, threads: np.ndarray) -> float:
    """Fit Amdahl serial fraction s from T(p) = s*T1 + (1-s)*T1/p."""
    inv_threads = 1.0 / threads
    design = np.column_stack([np.ones_like(inv_threads), inv_threads])
    coefficients, _, _, _ = np.linalg.lstsq(design, time_sec, rcond=None)
    serial_time, _parallel_time = coefficients
    return float(serial_time / time_sec[0])


def amdahl_speedup(threads: np.ndarray, serial_fraction: float) -> np.ndarray:
    parallel_fraction = 1.0 - serial_fraction
    return 1.0 / (serial_fraction + parallel_fraction / threads)


def save_metrics(serial_fraction: float) -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "image_count": IMAGE_COUNT,
        "threads": THREADS.astype(int).tolist(),
        "time_sec": TIME_SEC.tolist(),
        "speedup": speedup(TIME_SEC).tolist(),
        "efficiency": (speedup(TIME_SEC) / THREADS).tolist(),
        "amdahl_serial_fraction": round(serial_fraction, 4),
        "amdahl_max_speedup": round(1.0 / serial_fraction, 2),
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def plot_speedup_table(serial_fraction: float) -> Path:
    observed = speedup(TIME_SEC)
    output_path = EVIDENCE_DIR / "openmp_speedup_table.png"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax_table = axes[0]
    ax_table.axis("off")
    table_data = [
        ["Threads", "Time (s)", "Speedup", "Efficiency"],
        *[
            [str(int(t)), f"{time:.3f}", f"{sp:.2f}", f"{sp / t:.2f}"]
            for t, time, sp in zip(THREADS, TIME_SEC, observed, strict=True)
        ],
    ]
    table = ax_table.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
        colWidths=[0.22, 0.26, 0.26, 0.26],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.1, 1.8)
    for column in range(4):
        table[(0, column)].set_facecolor("#E8EEF7")
        table[(0, column)].set_text_props(weight="bold")
    ax_table.set_title(
        f"OpenMP preprocessing benchmark ({IMAGE_COUNT} images)",
        fontsize=12,
        pad=12,
    )

    ax_plot = axes[1]
    thread_grid = np.linspace(1, 8, 100)
    ax_plot.plot(
        thread_grid,
        amdahl_speedup(thread_grid, serial_fraction),
        linestyle="--",
        color="#C44E52",
        linewidth=2,
        label=f"Amdahl (s={serial_fraction:.2f})",
    )
    ax_plot.plot(
        THREADS,
        observed,
        marker="o",
        color="#4C72B0",
        linewidth=2,
        markersize=8,
        label="Measured",
    )
    ax_plot.axhline(1.0 / serial_fraction, color="#55A868", linestyle=":", linewidth=1.5,
                    label=f"Max speedup = {1.0 / serial_fraction:.2f}x")
    ax_plot.set_xlabel("Threads")
    ax_plot.set_ylabel("Speedup")
    ax_plot.set_title("Speedup vs. thread count")
    ax_plot.set_xticks(THREADS)
    ax_plot.grid(True, alpha=0.3)
    ax_plot.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_amdahl_analysis(serial_fraction: float) -> Path:
    observed = speedup(TIME_SEC)
    output_path = EVIDENCE_DIR / "openmp_amdahl_analysis.png"
    thread_grid = np.arange(1, 17, dtype=float)
    theoretical = amdahl_speedup(thread_grid, serial_fraction)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax_time = axes[0]
    serial_time = serial_fraction * TIME_SEC[0]
    parallel_time = (1.0 - serial_fraction) * TIME_SEC[0]
    ax_time.bar(
        ["Serial fraction", "Parallel fraction"],
        [serial_time, parallel_time],
        color=["#C44E52", "#4C72B0"],
    )
    ax_time.set_ylabel("Estimated time at 1 thread (s)")
    ax_time.set_title("Amdahl decomposition (1-thread baseline)")
    ax_time.grid(axis="y", alpha=0.3)

    ax_curve = axes[1]
    ax_curve.plot(thread_grid, theoretical, linestyle="--", color="#C44E52", linewidth=2,
                  label="Amdahl model")
    ax_curve.plot(THREADS, observed, marker="o", color="#4C72B0", linewidth=2, markersize=8,
                  label="Measured")
    ax_curve.set_xlabel("Threads")
    ax_curve.set_ylabel("Speedup")
    ax_curve.set_title("Observed vs. theoretical speedup")
    ax_curve.set_xlim(1, 16)
    ax_curve.set_ylim(1, max(2.5, theoretical.max() * 1.05))
    ax_curve.grid(True, alpha=0.3)
    ax_curve.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    serial_fraction = estimate_serial_fraction(TIME_SEC, THREADS)
    save_metrics(serial_fraction)

    table_path = plot_speedup_table(serial_fraction)
    amdahl_path = plot_amdahl_analysis(serial_fraction)

    print(f"Serial fraction (Amdahl fit): {serial_fraction:.4f}")
    print(f"Maximum theoretical speedup: {1.0 / serial_fraction:.2f}x")
    print(f"Saved: {table_path}")
    print(f"Saved: {amdahl_path}")
    print(f"Saved: {METRICS_PATH}")


if __name__ == "__main__":
    main()
