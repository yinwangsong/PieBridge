"""Measured 12-epoch training time under periodic foreground inference.

Data are exported by orin_scheduling/render_period_sweep.py from the verified
final runs. Training time excludes evaluation; each condition was run once.
Run with Python and Matplotlib to regenerate the vector PDF and PNG.
Styling follows Figure 17: DejaVu Sans, gray/red hatched bars, black axes,
horizontal dashed grid lines, and a framed legend.
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent


def main():
    rows = json.loads((OUT / "orin_period_sweep_data.json").read_text())
    periods = [row["period_ms"] for row in rows]
    fifo = [row["fifo_training_seconds"] for row in rows]
    reorder = [row["prefetch_reorder_training_seconds"] for row in rows]
    savings = [row["training_time_saved_percent"] for row in rows]
    style = {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.labelweight": "normal",
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "text.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.linewidth": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "legend.frameon": True,
        "legend.edgecolor": "0.8",
        "legend.framealpha": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "hatch.linewidth": 0.75,
    }
    with mpl.rc_context(style):
        fig, ax = plt.subplots(figsize=(80 / 25.4, 60 / 25.4))
        fig.subplots_adjust(left=0.19, right=0.98, bottom=0.21, top=0.81)
        x = np.arange(len(rows))
        width = 0.34
        baseline_bars = ax.bar(
            x - width / 2, fifo, width, label="FIFO", color="#808080",
            edgecolor="black", linewidth=1.0, hatch="/")
        reorder_bars = ax.bar(
            x + width / 2, reorder, width, label="Reorder + prefetch",
            color="#ff0000", edgecolor="black", linewidth=1.0, hatch="x")
        for index, (baseline, combined) in enumerate(zip(baseline_bars, reorder_bars)):
            baseline.set_gid(f"fifo_{periods[index]}")
            combined.set_gid(f"reorder_{periods[index]}")
            ax.text(
                x[index], max(fifo[index], reorder[index]) + 22,
                f"{savings[index]:.2f}%", ha="center", va="bottom",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2},
                gid=f"saving_{periods[index]}")
        ax.set_xlim(-0.6, len(rows) - 0.4)
        ax.set_ylim(0, 850)
        ax.set_xticks(x, [str(period) for period in periods])
        ax.set_yticks([0, 200, 400, 600, 800])
        ax.set_xlabel("Foreground inference period (ms)", labelpad=4)
        ax.set_ylabel("Training time (s)", labelpad=3)
        ax.tick_params(length=3, width=0.75, pad=3)
        ax.set_axisbelow("line")
        ax.grid(axis="y", color="#b0b0b0", linestyle="--", linewidth=0.5)
        ax.legend(
            loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
            handlelength=2.4, handletextpad=0.4, columnspacing=0.9,
            borderaxespad=0)
        fig.savefig(OUT / "orin_period_sweep.pdf", metadata={
            "Title": "Training time under different foreground inference periods",
            "Subject": "12 training epochs excluding evaluation; one run per condition",
        })
        fig.savefig(OUT / "orin_period_sweep.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
