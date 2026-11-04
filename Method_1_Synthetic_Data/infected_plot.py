from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple


def _pf_results_path(bank_path: str) -> str:
    root, _ = os.path.splitext(bank_path)
    return root + "_pf_results.npz"


def _load_path_for_plots(path: str) -> str:
    pf = _pf_results_path(path)
    return pf if os.path.exists(pf) else path


def _bank_key(t: int, name: str) -> str:
    return f"t{int(t)}_{name}"


def _find_times_with_selected_I(bank_path: str) -> List[int]:
    bank_path = _load_path_for_plots(bank_path)
    with np.load(bank_path, allow_pickle=True) as d:
        pat = re.compile(r"^t(\d+)_C_half_I_sim$")
        times: List[int] = []
        for k in d.files:
            m = pat.match(k)
            if m:
                times.append(int(m.group(1)))
        return sorted(times)


def load_selected_infected(bank_path: str, t: int) -> np.ndarray:
    bank_path = _load_path_for_plots(bank_path)
    kI = _bank_key(t, "C_half_I_sim")
    with np.load(bank_path, allow_pickle=True) as d:
        if kI not in d.files:
            return np.array([], dtype=int)
        return np.asarray(d[kI], dtype=int).ravel()


def load_selected_weights(bank_path: str, t: int) -> np.ndarray:
    bank_path = _load_path_for_plots(bank_path)
    kW = _bank_key(t, "C_half_weights")
    with np.load(bank_path, allow_pickle=True) as d:
        if kW not in d.files:
            return np.array([], dtype=float)
        return np.asarray(d[kW], dtype=float).ravel()


def load_selected_infected_over_time(
    bank_path: str,
    t_max: Optional[int] = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    times = _find_times_with_selected_I(bank_path)
    if t_max is not None:
        times = [t for t in times if int(t) <= int(t_max)]
    series = [load_selected_infected(bank_path, t) for t in times]
    return np.asarray(times, dtype=int), series


def load_observed(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    t_obs = df["Time"].to_numpy(dtype=float)
    y_obs = df["Infected"].to_numpy(dtype=float)
    return t_obs, y_obs


def plot_selected_infected_multi_subplots(
    bank_paths: List[str],
    labels: Optional[List[str]] = None,
    observed_csv_path: Optional[str] = None,
    title: str = "Observed and PPF estimated infected cases by window size",
    save_path: str = "figs_multi/infected_windows_subplots.pdf",
    xlim=None,
    ylim=None,
    xtick=None,
    ytick=None,
    sharex: bool = True,
    sharey: bool = True,
    fig_size=(10, 8),
    t_max: Optional[int] = None,
    show: bool = True,
    close: bool = True,
    window_text_pos: Tuple[float, float] = (0.5, 0.92), 
    window_text_size: int = 11,
    window_text_weight: str = "bold",
    legend_ncol: int = 3,
):
    if labels is None:
        labels = [os.path.splitext(os.path.basename(p))[0] for p in bank_paths]
    if len(labels) != len(bank_paths):
        raise ValueError("labels must have same length as bank_paths (or be None).")

  
    t_obs = y_obs = None
    if observed_csv_path:
        t_obs, y_obs = load_observed(observed_csv_path)

        if t_max is not None:
            m = np.asarray(t_obs, float) <= float(t_max)
            t_obs = np.asarray(t_obs, float)[m]
            y_obs = np.asarray(y_obs, float)[m]

        t_obs = np.asarray(t_obs, float)
        y_obs = np.asarray(y_obs, float)

    n = len(bank_paths)
    fig, axes = plt.subplots(n, 1, sharex=sharex, sharey=sharey, figsize=fig_size)
    if n == 1:
        axes = [axes]

    fig.suptitle(title, y=0.93, fontsize=15, fontweight="bold")

    legend_handles = None
    legend_labels = None

    for ax, bank_path, lab in zip(axes, bank_paths, labels):
        bank_path = _load_path_for_plots(bank_path)

        ts, arrs = load_selected_infected_over_time(bank_path, t_max=t_max)
        if ts.size == 0:
            ax.grid(True, alpha=0.3)
            ax.text(
                window_text_pos[0], window_text_pos[1],
                f"{lab} (no data)",
                transform=ax.transAxes,
                fontsize=window_text_size,
                fontweight=window_text_weight,
                va="top"
            )
            continue

        xs: List[float] = []
        ys: List[float] = []
        for t, a in zip(ts.tolist(), arrs):
            a = np.asarray(a, dtype=float).ravel()
            if a.size == 0:
                continue
            xs.extend([float(t)] * int(a.size))
            ys.extend(a.tolist())

        weighted_I: List[float] = []
        for t, a in zip(ts.tolist(), arrs):
            a = np.asarray(a, dtype=float).ravel()
            if a.size == 0:
                weighted_I.append(np.nan)
                continue

            w = load_selected_weights(bank_path, int(t))
            w = np.asarray(w, dtype=float).ravel()
            if w.size != a.size:
                raise RuntimeError(f"Size mismatch at t={t}: weights {w.size}, I_sim {a.size}")

            sw = float(np.sum(w))
            if sw <= 0.0 or not np.isfinite(sw):
                w = np.full_like(w, 1.0 / len(w))
            else:
                w = w / sw

            weighted_I.append(float(np.sum(w * a)))

        weighted_I = np.asarray(weighted_I, dtype=float)

      
       # h1 = ax.scatter(np.asarray(xs, float), np.asarray(ys, float), s=5, color="green", alpha=0.85, label="Selected particles")
        h1 = ax.scatter(xs, ys, s=28, color="green", alpha=0.35, label="Selected particles")

        h2, = ax.plot(ts.astype(float), weighted_I,  marker='*', markersize=1, linewidth=2.5, color="red", label="Estimated infected")

        h3 = ax.scatter(ts.astype(float), weighted_I,  s=5, color="black", zorder=3)

        if t_obs is not None:
            h4, = ax.plot(t_obs, y_obs, marker='*', markersize=1,   markerfacecolor='red', color="orange", linewidth=2.5, label="Observed data")
            ax.scatter(t_obs, y_obs, s=5, color="orange", zorder=5)

       
    
        ax.text(
            window_text_pos[0], window_text_pos[1],
            lab,
            transform=ax.transAxes,
            fontsize=window_text_size,
            fontweight=window_text_weight,
            va="top"
        )

        if legend_handles is None:
            legend_handles = [h1, h2, h4] 
            legend_labels = [h.get_label() for h in legend_handles]

       
        ax.set_ylabel("")
        ax.grid(True, alpha=0.3)

        if ylim is not None and len(ylim) == 2:
            ax.set_ylim(float(ylim[0]), float(ylim[1]))
        if ytick is not None and np.size(ytick):
            ax.set_yticks([float(y) for y in np.asarray(ytick).ravel()])
            
    
    if legend_handles is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=int(legend_ncol),
            frameon=False,
            bbox_to_anchor=(0.5, 0.01),

        )

   
    fig.supxlabel("Time", x=0.55, y=0.08, fontsize=15, fontweight="bold")
    fig.supylabel("Number of infected individuals", y=0.55, fontsize=13, fontweight="bold")

    if xlim is not None and len(xlim) == 2:
        axes[-1].set_xlim(float(xlim[0]), float(xlim[1]))
    if xtick is not None and np.size(xtick):
        axes[-1].set_xticks([float(x) for x in np.asarray(xtick).ravel()])

  

    fig.tight_layout(rect=(0, 0.05, 1, 0.96))

    out_dir = os.path.dirname(save_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(save_path, dpi=600, bbox_inches="tight")

    if show:
        plt.show()
    if close:
        plt.close(fig)

