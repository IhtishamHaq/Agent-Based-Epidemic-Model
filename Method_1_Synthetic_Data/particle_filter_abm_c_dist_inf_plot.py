from __future__ import annotations
import os
os.environ.pop("MPLBACKEND", None)

import matplotlib
matplotlib.use("TkAgg")

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
        times = []
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
    weeks = df["Time"].to_numpy(dtype=float)
    t_obs = weeks 
    cases = df["Infected"].to_numpy(dtype=float)
    return t_obs, cases




def plot_selected_infected_time_series(
    bank_path: str,
    observed_csv: Optional[str] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    close: bool = False,
    ylim: Optional[Tuple[float, float]] = None,
    t_max: Optional[int] = None,
):
    ts, arrs = load_selected_infected_over_time(bank_path, t_max=t_max)
    if ts.size == 0:
        raise RuntimeError("error.")

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
            raise RuntimeError(f"error")

        sw = float(np.sum(w))
        if sw <= 0.0 or not np.isfinite(sw):
            w = np.full_like(w, 1.0 / len(w))
        else:
            w = w / sw

        weighted_I.append(float(np.sum(w * a)))

    weighted_I = np.asarray(weighted_I, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.scatter(np.asarray(xs, float), np.asarray(ys, float), s=28, color="green", alpha=0.85, label="Selected infected")
    ax.plot(ts.astype(float), weighted_I,  linewidth=2.0, color="red", label="Estimated infected")
    ax.scatter(ts.astype(float), weighted_I, s=30, color="red", zorder=3)

    if observed_csv is not None:
        t_obs, y_obs = load_observed(observed_csv)

      
        if t_max is not None:
            m = np.asarray(t_obs, float) <= float(t_max)
            t_obs = np.asarray(t_obs, float)[m]
            y_obs = np.asarray(y_obs, float)[m]

        ax.plot(t_obs, y_obs, color="orange", linewidth=2.5, label="Observed data")
        ax.scatter(t_obs, y_obs, s=5, color="orange", zorder=3)

    ax.set_title(title or "Observed infected and particle filter simulated infected over time")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Infected")

    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))

    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()

    if save_path:
        out_dir = os.path.dirname(save_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    if close:
        plt.close(fig)

    return fig if not show else None


#FOR PRINT 

def _print_estimate_ci_at_time(
    ts: np.ndarray,
    mean_I: np.ndarray,
    lo95: np.ndarray,
    hi95: np.ndarray,
    t_query: int,
):
    ts = np.asarray(ts, int)
    idx = np.where(ts == int(t_query))[0]
    if idx.size == 0:
        raise RuntimeError(f"error")
    i = int(idx[0])
    print(
        f"t={int(ts[i])}: Estimated I = {mean_I[i]:.3f}, "
        f"95% CI = [{lo95[i]:.3f}, {hi95[i]:.3f}]"
    )
    return i


def _weighted_mean_ci(
    x: np.ndarray,
    w: np.ndarray,
    ci: int = 95,   # 50, 75, 90, 95
):
    z_table = {
        50: 0.674,
        75: 1.150,
        90: 1.645,
        95: 1.960,
    }

    if ci not in z_table:
        raise ValueError(f"error")

    z = z_table[ci]

    x = np.asarray(x, float)
    w = np.asarray(w, float)

    sw = np.sum(w)
    if sw <= 0 or not np.isfinite(sw):
        w = np.full_like(w, 1.0 / max(len(w), 1))
    else:
        w = w / sw

    mean = float(np.sum(w * x))
    var  = float(np.sum((x - mean) ** 2))
    se   = np.sqrt(var / max(len(x), 1))

    lo = mean - z * se
    hi = mean + z * se

    return mean, lo, hi



def plot_selected_infected_time_series_CI(
    bank_path: str,
    observed_csv: Optional[str] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    close: bool = False,
    ylim: Optional[Tuple[float, float]] = None,
    t_max: Optional[int] = None,
    t_print: Optional[int] = None,  
):
    ts, arrs = load_selected_infected_over_time(bank_path, t_max=t_max)
    if ts.size == 0:
        raise RuntimeError("error.")

    mean_I = []

    ci_levels = [95, 90, 75, 50]
    ci_lo = {c: [] for c in ci_levels}
    ci_hi = {c: [] for c in ci_levels}

    for t, a in zip(ts.tolist(), arrs):
        a = np.asarray(a, float).ravel()
        if a.size == 0:
            mean_I.append(np.nan)
            for c in ci_levels:
                ci_lo[c].append(np.nan)
                ci_hi[c].append(np.nan)
            continue

        w = load_selected_weights(bank_path, int(t))
        w = np.asarray(w, float).ravel()
        if w.size != a.size:
            raise RuntimeError(f"Size mismatch at t={t}")

        
        m, _, _ = _weighted_mean_ci(a, w, ci=95)
        mean_I.append(m)

        for c in ci_levels:
            _, lo, hi = _weighted_mean_ci(a, w, ci=c)
            ci_lo[c].append(lo)
            ci_hi[c].append(hi)

    mean_I = np.asarray(mean_I)

    for c in ci_levels:
        ci_lo[c] = np.asarray(ci_lo[c])
        ci_hi[c] = np.asarray(ci_hi[c])


    if t_print is not None:
        i = _print_estimate_ci_at_time(ts, mean_I, ci_lo[95], ci_hi[95], t_print)
        
    fig, ax = plt.subplots(figsize=(9, 4.5))


    ax.set_facecolor("lightgray")
    fig.patch.set_facecolor("white")

    ax.fill_between(ts, ci_lo[95], ci_hi[95], color="blue", alpha=0.1, label="95% CI")
    ax.fill_between(ts, ci_lo[90], ci_hi[90], color="blue", alpha=0.3, label="90% CI")
    ax.fill_between(ts, ci_lo[75], ci_hi[75], color="blue", alpha=0.6, label="75% CI")
    ax.fill_between(ts, ci_lo[50], ci_hi[50], color="blue", alpha=0.9, label="50% CI")
    ax.plot(ts, mean_I, color="red", linewidth=2.0, label="Estimated infected")
    ax.grid(True, color="white", alpha=0.8)

    ax.plot(ts, mean_I, color="red", linewidth=2.0, label="Estimated infected")

   
  
    if observed_csv is not None:
        t_obs, y_obs = load_observed(observed_csv)
        t_obs = np.asarray(t_obs, float)
        y_obs = np.asarray(y_obs, float)

        if t_max is not None:
            msk = t_obs <= float(t_max)
            t_obs = t_obs[msk]
            y_obs = y_obs[msk]

        ax.plot(t_obs, y_obs, color="orange", linewidth=2.5, label="Observed")
        ax.scatter(t_obs, y_obs, s=10, color="orange")

    ax.set_title("PF infected with multiple confidence intervals", fontweight="bold", fontsize=15)
    ax.set_xlabel("Time", fontweight="bold", fontsize=12)
    ax.set_ylabel("Infected", fontweight="bold", fontsize=12)

    if ylim is not None:
        ax.set_ylim(*ylim)


    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.grid(True, alpha=1.0)
    ax.legend(frameon=False)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    if close:
        plt.close(fig)

    return fig if not show else None


def plot_selected_infected_at_t(
    bank_path: str,
    t: int,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    close: bool = False,
    ylim: Optional[Tuple[float, float]] = None,
):
    I = load_selected_infected(bank_path, int(t))
    if I.size == 0:
        raise RuntimeError(f"No saved selected infected at t={t}.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(np.arange(I.size), I, s=55, color="green", label="Selected infected data")
    ax.axhline(np.mean(I), color="red", linewidth=2, label=f"Mean = {np.mean(I):.2f}")

    ax.set_title(title or f"Selected infected at t={t}")
    ax.set_xlabel("Particle index")
    ax.set_ylabel("Infected")

    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))

    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    if close:
        plt.close(fig)

    return fig if not show else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bank-path", required=True)
    p.add_argument("--observed-csv", default=None)
    p.add_argument("--mode", choices=("over_time", "at_t"), default="over_time")
    p.add_argument("--t", type=int, default=None)
    p.add_argument("--t-max", type=int, default=None)
    p.add_argument("--save", default=None)
    p.add_argument("--ymin", type=float, default=None)
    p.add_argument("--ymax", type=float, default=None)
    p.add_argument("--t-print", type=int, default=None)
    p.add_argument("--title", default=None)
    args = p.parse_args()

    ylim = None
    if args.ymin is not None and args.ymax is not None:
        ylim = (args.ymin, args.ymax)

    if args.mode == "at_t":
        if args.t is None:
            raise SystemExit("mode=at_t requires --t")
        plot_selected_infected_at_t(
            bank_path=args.bank_path,
            t=args.t,
            save_path=args.save,
            ylim=ylim,
        )
    else:
        plot_selected_infected_time_series_CI(
            bank_path=args.bank_path,
            observed_csv=args.observed_csv,
            save_path=args.save,
            ylim=ylim,
            t_max=args.t_max,
            t_print=args.t_print, 
        )


if __name__ == "__main__":
    main()
