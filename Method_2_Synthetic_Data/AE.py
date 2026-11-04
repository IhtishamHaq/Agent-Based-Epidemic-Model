from __future__ import annotations

import os
import re
import argparse
from typing import Dict, Optional

import numpy as np

os.environ.pop("MPLBACKEND", None)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def _load_abs_err_by_window(
    metrics_path: str,
    t_final: Optional[int] = None,
) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}

    with np.load(metrics_path, allow_pickle=True) as d:
        for k in d.files:
            if not k.endswith("_abs_err"):
                continue

            m = re.search(r"window_(\d+)_abs_err$", k)
            if m:
                w = int(m.group(1))
            else:
                m2 = re.search(r"(\d+)", k)
                if not m2:
                    continue
                w = int(m2.group(1))

            arr = np.asarray(d[k], float).ravel()

            if t_final is not None:
                time_key = f"window_{w}_times"
                if time_key in d.files:
                    ts = np.asarray(d[time_key], float).ravel()
                    n = min(len(arr), len(ts))
                    arr = arr[:n]
                    ts = ts[:n]
                    arr = arr[ts <= t_final]

            arr = arr[np.isfinite(arr)]
            if arr.size:
                out[w] = arr

    return out


def plot_box_compare(
    metrics_sabl: str,
    metrics_abl: str,
    metrics_sir: str,
    save_fig: bool = True,
    fig_name: str = "compare_SABL_ABL_SIR.pdf",
    show: bool = False,
    t_final: Optional[int] = None,
) -> None:
    abl = _load_abs_err_by_window(metrics_abl, t_final=t_final)
    sabl = _load_abs_err_by_window(metrics_sabl, t_final=t_final)
    sir = _load_abs_err_by_window(metrics_sir, t_final=t_final)

    common_ws = sorted(set(abl) & set(sabl) & set(sir))
    if not common_ws:
        raise RuntimeError(
            "No common filter windows were found in all three metrics files."
        )

    arrays = []
    positions = []
    xticks = []
    xticklabels = []

    gap = 2.2
    offset = 0.52

    for i, w in enumerate(common_ws):
        base = i * gap

        arrays.extend([abl[w], sabl[w], sir[w]])
        positions.extend([base - offset, base, base + offset])

        xticks.append(base)
        xticklabels.append(f"Filter window {w}")

    fig, ax = plt.subplots(figsize=(9, 5))

    bp = ax.boxplot(
        arrays,
        positions=positions,
        widths=0.42,
        patch_artist=True,
        showfliers=False,
    )

    colors = ["#90EE90", "#87CEFA", "#FFB6C1"]
    for j, box in enumerate(bp["boxes"]):
        box.set_facecolor(colors[j % 3])

    for median in bp["medians"]:
        median.set_color("black")

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=10, fontweight="bold")
    ax.set_ylabel("Relative absolute error", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    handles = [
        mpatches.Patch(color="#90EE90", label="PPF method 1"),
        mpatches.Patch(color="#87CEFA", label="PPF method 2"),
        mpatches.Patch(color="#FFB6C1", label="SIR-PF"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False)

    plt.tight_layout()

    if save_fig:
        plt.savefig(fig_name, dpi=600, bbox_inches="tight")
        print("Saved plot to:", fig_name)

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-sabl", required=True)
    ap.add_argument("--metrics-abl", required=True)
    ap.add_argument("--metrics-sir", required=True)
    ap.add_argument("--fig-name", default="compare_SABL_ABL_SIR.pdf")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--t-final", type=int, default=None)
    args = ap.parse_args()

    plot_box_compare(
        metrics_sabl=args.metrics_sabl,
        metrics_abl=args.metrics_abl,
        metrics_sir=args.metrics_sir,
        fig_name=args.fig_name,
        show=args.show,
        t_final=args.t_final,
    )


if __name__ == "__main__":
    main()
