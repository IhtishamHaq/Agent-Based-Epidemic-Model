from __future__ import annotations
import os
os.environ.pop("MPLBACKEND", None)

import matplotlib
matplotlib.use("TkAgg")

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
            return np.array([], dtype=float)
        return np.asarray(d[kI], dtype=float).ravel()


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


def _find_times_all(bank_path: str) -> List[int]:
    bank_path = _load_path_for_plots(bank_path)
    with np.load(bank_path, allow_pickle=True) as d:
        pat = re.compile(r"^t(\d+)_rep_I_all$")
        times = []
        for k in d.files:
            m = pat.match(k)
            if m:
                times.append(int(m.group(1)))
        return sorted(times)


def load_all_particles(bank_path: str, t: int) -> np.ndarray:
    bank_path = _load_path_for_plots(bank_path)
    k = _bank_key(t, "rep_I_all")
    with np.load(bank_path, allow_pickle=True) as d:
        if k not in d.files:
            return np.array([], dtype=float)
        return np.asarray(d[k], dtype=float).ravel()


def load_weights_all(bank_path: str, t: int) -> np.ndarray:
    bank_path = _load_path_for_plots(bank_path)
    k = _bank_key(t, "weights_all")
    with np.load(bank_path, allow_pickle=True) as d:
        if k not in d.files:
            return np.array([], dtype=float)
        return np.asarray(d[k], dtype=float).ravel()


def _find_times(bank_path: str):
    return _find_times_all(bank_path)


def load_series(bank_path, t_max=None):
    times = _find_times(bank_path)
    if t_max is not None:
        times = [t for t in times if t <= t_max]
    arrs = [load_all_particles(bank_path, t) for t in times]
    return np.array(times), arrs


def load_series_all(
    bank_path: str,
    t_max: Optional[int] = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    times = _find_times_all(bank_path)
    if t_max is not None:
        times = [t for t in times if t <= int(t_max)]
    arrs = [load_all_particles(bank_path, t) for t in times]
    return np.asarray(times, dtype=int), arrs



def _get_prev_window_time(d, t_current):
    pattern = re.compile(r"^t(\d+)_weights_all$")
    times = []
    for k in d.files:
        m = pattern.match(k)
        if m:
            times.append(int(m.group(1)))

    if not times:
        return None

# this will previous time for weight because we didnt update weight at the current time t . so mean the weight now is the weight of the previous time.
    times = sorted(times)
    prev_times = [tt for tt in times if tt == t_current]
    if not prev_times:
        return None

    return max(prev_times)



def load_observed(csv_path):
    df = pd.read_csv(csv_path)
    return df["Time"].values, df["kyoto_cases"].values



def _weighted_mean_ci(x, w, ci=95):
    z_table = {50: 0.674, 75: 1.150, 90: 1.645, 95: 1.960}
    z = z_table.get(ci, 1.960)

    x = np.asarray(x, float)
    w = np.asarray(w, float)

    m = np.isfinite(x) & np.isfinite(w)
    x = x[m]
    w = w[m]

    if x.size == 0:
        return float("nan"), float("nan"), float("nan")

    sw = np.sum(w)
    if sw <= 0 or not np.isfinite(sw):
        w = np.full_like(w, 1.0 / len(w))
    else:
        w = w / sw

    mean = float(np.sum(w * x))

    var = float(np.sum((x - mean) ** 2))
    se = np.sqrt(var / max(len(x), 1))

    lo = mean - z * se
    hi = mean + z * se

    return mean, lo, hi

def compute_mape(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true != 0)
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100


def plot_selected_infected_time_series_CI(bank_path, observed_csv=None, save_path=None, t_max=None):
    ts, arrs = load_series(bank_path, t_max)

    mean_I = []
    ci_levels = [95, 90, 75, 50]
    ci_lo = {c: [] for c in ci_levels}
    ci_hi = {c: [] for c in ci_levels}

    bank_path = _load_path_for_plots(bank_path)
    with np.load(bank_path, allow_pickle=True) as d:
        for t, a in zip(ts, arrs):
            a = np.asarray(a, float)

            t_prev = _get_prev_window_time(d, t)
            if t_prev is None:
                w = load_weights_all(bank_path, t)
            else:
                w = load_weights_all(bank_path, t_prev)
                

            m, _, _ = _weighted_mean_ci(a, w, 95)
            mean_I.append(m)

            for c in ci_levels:
                _, lo, hi = _weighted_mean_ci(a, w, c)
                ci_lo[c].append(lo)
                ci_hi[c].append(hi)

    mean_I = np.array(mean_I)
    for c in ci_levels:
        ci_lo[c] = np.array(ci_lo[c])
        ci_hi[c] = np.array(ci_hi[c])

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.fill_between(ts, ci_lo[95], ci_hi[95], color="blue", alpha=0.1, label="95% CI" )
    ax.fill_between(ts, ci_lo[90], ci_hi[90], color="blue", alpha=0.3, label="90% CI")
    ax.fill_between(ts, ci_lo[75], ci_hi[75], color="blue", alpha=0.6, label="75% CI")
    ax.fill_between(ts, ci_lo[50], ci_hi[50], color="blue", alpha=0.9, label="50% CI")

    ax.plot(ts, mean_I, color="red", linewidth=2, label="Estimated infected", zorder=1.5)
    ax.scatter(ts, mean_I, facecolors="none", edgecolors="red", s=50, marker="o", zorder=2)

    if observed_csv:
        t_obs, y_obs = load_observed(observed_csv)
        mean_interp = np.interp(t_obs, ts, mean_I)
        ax.scatter(t_obs, y_obs, color="gold", s=40, marker='o', edgecolors='gold', label="Observed", zorder=1)
        ax.scatter(t_obs, y_obs, s=50, color="gold", zorder=1)

    ax.set_title("Particle filter estimation vs observed data")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Infected")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.tight_layout()
    plt.show()


def plot_selected_infected_time_series_week(bank_path, observed_csv=None, save_path=None, t_max=None):
    ts, arrs = load_series(bank_path, t_max)
    weeks = ts / 7

    mean_I = []
    ci_levels = [95, 90, 75, 50]
    ci_lo = {c: [] for c in ci_levels}
    ci_hi = {c: [] for c in ci_levels}

    bank_path = _load_path_for_plots(bank_path)
    with np.load(bank_path, allow_pickle=True) as d:
        for t, a in zip(ts, arrs):
            a = np.asarray(a, float)

            t_prev = _get_prev_window_time(d, t)
            if t_prev is None:
                w = load_weights_all(bank_path, t)
            else:
                w = load_weights_all(bank_path, t_prev)

            m, _, _ = _weighted_mean_ci(a, w, 95)
            mean_I.append(m)

            for c in ci_levels:
                _, lo, hi = _weighted_mean_ci(a, w, c)
                ci_lo[c].append(lo)
                ci_hi[c].append(hi)

    mean_I = np.array(mean_I)
    for c in ci_levels:
        ci_lo[c] = np.array(ci_lo[c])
        ci_hi[c] = np.array(ci_hi[c])

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.set_xticks(np.arange(0, 547 / 7, 2))
    plt.xticks(rotation=70, fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')

    ax.set_facecolor("lightgray")
    ax.patch.set_alpha(0.5)
    fig.patch.set_facecolor("white")

    ax.fill_between(weeks, ci_lo[95], ci_hi[95], color="blue", alpha=0.55, label="95% CI")
    ax.plot(weeks, mean_I, color="gold", label="Estimated infected")
    ax.scatter(weeks, mean_I, color="gold", s=40)

    if observed_csv:
        t_obs, y_obs = load_observed(observed_csv)
        t_obs = t_obs / 7
        mean_interp = np.interp(t_obs, weeks, mean_I)
        print("MAPE =", compute_mape(y_obs, mean_interp))
        ax.scatter(t_obs, y_obs, color="red", label="Observed")

    ax.set_ylim(bottom=0)

    ax.axvspan(161/7, 217/7, color='pink', alpha=0.7, zorder=0)
    ax.axvspan(490/7, 546/7, color='pink', alpha=0.7, zorder=0)
    ax.axvspan(189/7, 189/7, color='green', linewidth=2.0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.set_title("Particle filter forcast vs observed data", fontweight="bold", fontsize=13)
    ax.set_xlabel("Time (weeks)", fontweight="bold", fontsize=12)
    ax.set_ylabel("Weekly infected cases", fontweight="bold", fontsize=12, labelpad=30)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.tight_layout()
    plt.show()




# Togather 
def plot_weekly_current_and_previous_together(
    bank_path: str,
    observed_csv: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    close: bool = False,
    ylim: Optional[Tuple[float, float]] = None,
    t_max: Optional[int] = None,
):
    ts_cur, arrs_cur = load_selected_infected_over_time(bank_path, t_max=t_max)
    if ts_cur.size == 0:
        raise RuntimeError("No saved selected infected found.")

    weeks_cur = ts_cur / 7.0
    mean_cur, lo95_cur, hi95_cur = [], [], []

    for t, a in zip(ts_cur.tolist(), arrs_cur):
        a = np.asarray(a, float).ravel()
        if a.size == 0:
            mean_cur.append(np.nan)
            lo95_cur.append(np.nan)
            hi95_cur.append(np.nan)
            continue

        w = load_selected_weights(bank_path, int(t))
        w = np.asarray(w, float).ravel()

        if w.size != a.size:
            raise RuntimeError(
                f"Size mismatch at t={t}: current weights={w.size}, selected infected={a.size}"
            )

        m, lo, hi = _weighted_mean_ci(a, w, ci=95)
        mean_cur.append(m)
        lo95_cur.append(lo)
        hi95_cur.append(hi)

    mean_cur = np.asarray(mean_cur, dtype=float)
    lo95_cur = np.asarray(lo95_cur, dtype=float)
    hi95_cur = np.asarray(hi95_cur, dtype=float)

    ts_prev, arrs_prev = load_series_all(bank_path, t_max=t_max)
    if ts_prev.size == 0:
        raise RuntimeError("No saved all-particle infected found.")

    valid_ts_prev = []
    mean_prev, lo95_prev, hi95_prev = [], [], []

    bank_path = _load_path_for_plots(bank_path)
    with np.load(bank_path, allow_pickle=True) as d:
        for t, a in zip(ts_prev.tolist(), arrs_prev):
            a = np.asarray(a, float).ravel()
            if a.size == 0:
                continue

            t_prev = _get_prev_window_time(d, int(t))
            if t_prev is None:
                continue

            w = load_weights_all(bank_path, int(t_prev))
            w = np.asarray(w, float).ravel()

            if w.size != a.size:
                raise RuntimeError(
                    f"Size mismatch at t={t}: prev/current all-weights={w.size}, rep_I_all={a.size}"
                )

            sw = np.sum(w)
            if w.size == 0:
                raise RuntimeError(f"Empty previous weights at t={t}")
            elif sw <= 0 or not np.isfinite(sw):
                w = np.full_like(w, 1.0 / len(w))
            else:
                w = w / sw

            m, lo, hi = _weighted_mean_ci(a, w, ci=95)
            valid_ts_prev.append(t)
            mean_prev.append(m)
            lo95_prev.append(lo)
            hi95_prev.append(hi)

    weeks_prev = np.asarray(valid_ts_prev, dtype=float) / 7.0
    mean_prev = np.asarray(mean_prev, dtype=float)
    lo95_prev = np.asarray(lo95_prev, dtype=float)
    hi95_prev = np.asarray(hi95_prev, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.set_xticks(np.arange(0, 547 / 7, 2))
    plt.xticks(rotation=70, fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')

    ax.set_facecolor("lightgray")
    ax.patch.set_alpha(0.5)
    fig.patch.set_facecolor("white")

    ax.text(40/7, 7030.0, "Start of 2025")

    ax.annotate(
        "",
        xytext=(100/7, 6900),
        xy=(180/7, 6200),
        arrowprops=dict(arrowstyle="->", color="red", lw=2)
    )
    
    
    ax.fill_between(
        weeks_cur, lo95_cur, hi95_cur,
        color="blue", alpha=0.30, label="95% CI of estimated"
    )
    ax.plot(
        weeks_cur, mean_cur,
        color="red", alpha=0.9, linewidth=2.0, zorder=1.5,
        label="Estimated infected"
    )
    ax.scatter(weeks_cur, mean_cur, s=50, color="red", zorder=1.5)

    ax.fill_between(
        weeks_prev, lo95_prev, hi95_prev,
        color="green", alpha=0.20, label="95% CI forecast"
    )
    ax.plot(
        weeks_prev, mean_prev,
        color="darkgreen", alpha=0.9, linewidth=2.0,
        label="Forecast infected"
    )
    ax.scatter(weeks_prev, mean_prev, s=40, color="darkgreen")

    ax.grid(True, color="lightgray", alpha=0.19)

    if observed_csv is not None:
        t_obs, y_obs = load_observed(observed_csv)
        t_obs = np.asarray(t_obs, float) / 7.0
        y_obs = np.asarray(y_obs, float)

        if t_max is not None:
            msk = t_obs <= (t_max / 7.0)
            t_obs = t_obs[msk]
            y_obs = y_obs[msk]

        mean_at_obs_cur = np.interp(t_obs, weeks_cur, mean_cur)
        mape_cur = compute_mape(y_obs, mean_at_obs_cur)
        print(f"MAPE current (%) = {mape_cur:.2f}")

        mean_at_obs_prev = np.interp(t_obs, weeks_prev, mean_prev)
        mape_prev = compute_mape(y_obs, mean_at_obs_prev)
        print(f"MAPE previous (%) = {mape_prev:.2f}")

        ax.scatter(
            t_obs, y_obs,
            color="gold", s=55, marker='o',
            edgecolors='gold', zorder=1.0, label="Observed"
        )
        ax.scatter(t_obs, y_obs, s=50, color="gold", zorder=1.0)

    ax.set_title(
        "Particle filter estimation vs. reported data in Kyoto",
        x=0.5, y=1.0, fontweight="bold", fontsize=16
    )
    ax.set_xlabel("Time (weeks)", fontweight="bold", fontsize=12)
    ax.set_ylabel("Weekly infected cases", fontweight="bold", fontsize=12, labelpad=30)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_ylim(bottom=0)

    ax.axvspan(161/7, 217/7, color='pink', alpha=0.7, zorder=0)
    ax.axvspan(490/7, 546/7, color='pink', alpha=0.7, zorder=0)
    ax.axvspan(189/7, 189/7, color='green', linewidth=2.0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

    ax.legend(frameon=False, loc=(0.5, 0.70), fontsize=9, ncol=1)

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
    p.add_argument("--mode", choices=("over_time", "week", "week_combined"), default="week")
    p.add_argument("--t-max", type=int, default=None)
    p.add_argument("--save", default=None)
    p.add_argument("--ymin", type=float, default=None)
    p.add_argument("--ymax", type=float, default=None)
    args = p.parse_args()

    ylim = None
    if args.ymin is not None and args.ymax is not None:
        ylim = (args.ymin, args.ymax)

    if args.mode == "week_combined":
        plot_weekly_current_and_previous_together(
            bank_path=args.bank_path,
            observed_csv=args.observed_csv,
            save_path=args.save,
            ylim=ylim,
            t_max=args.t_max,
        )
    elif args.mode == "over_time":
        plot_selected_infected_time_series_CI(
            args.bank_path, args.observed_csv, args.save, args.t_max
        )
    else:
        plot_selected_infected_time_series_week(
            args.bank_path, args.observed_csv, args.save, args.t_max
        )



if __name__ == "__main__":
    main()