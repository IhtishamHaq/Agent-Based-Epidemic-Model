from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import gaussian_kde


def _pf_results_path(bank_path: str) -> str:
    root, _ = os.path.splitext(bank_path)
    return root + "_pf_results.npz"


def _load_path_for_plots(path: str) -> str:
    pf = _pf_results_path(path)
    return pf if os.path.exists(pf) else path


def _bank_key(t: int, name: str) -> str:
    return f"t{int(t)}_{name}"


def _find_times_with_mean_betas(bank_path: str) -> List[int]:
    bank_path = _load_path_for_plots(bank_path)
    with np.load(bank_path, allow_pickle=True) as d:
        pat = re.compile(r"^t(\d+)_C_half_mean_beta$")
        return sorted(int(m.group(1)) for k in d.files for m in [pat.match(k)] if m)


def load_mean_betas_from_simulations(bank_path: str, times: Optional[List[int]] = None):
    bank_path = _load_path_for_plots(bank_path)
    if times is None:
        times = _find_times_with_mean_betas(bank_path)

    meanC = []
    with np.load(bank_path, allow_pickle=True) as d:
        for t in times:
            keyC = _bank_key(t, "C_half_mean_beta")
            if keyC in d.files and d[keyC].size:
                c = float(np.asarray(d[keyC]).ravel()[0])
            else:
                c = np.nan
            meanC.append(c)

    return np.asarray(times, int), np.asarray(meanC, float)


def load_ci_from_simulations(bank_path: str, times: np.ndarray):
    bank_path = _load_path_for_plots(bank_path)
    lo, hi = [], []
    with np.load(bank_path, allow_pickle=True) as d:
        for t in times:
            k = _bank_key(int(t), "C_half_mean_beta_ci")
            if k in d.files:
                arr = np.asarray(d[k]).ravel()
                if arr.size >= 2 and np.isfinite(arr[:2]).all():
                    lo.append(float(arr[0]))
                    hi.append(float(arr[1]))
                else:
                    lo.append(np.nan)
                    hi.append(np.nan)
            else:
                lo.append(np.nan)
                hi.append(np.nan)
    return np.asarray(lo, float), np.asarray(hi, float)


def _load_beta_from_csv(csv_path: str):
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    cols = {c.lower(): c for c in df.columns}

    time_candidates = ["time", "t"]
    beta_candidates = ["beta", "beta_mean", "beta_hat"]

    time_col = next((cols[c] for c in time_candidates if c in cols), None)
    beta_col = next((cols[c] for c in beta_candidates if c in cols), None)

    if time_col is None:
        raise ValueError(f"error")
    if beta_col is None:
        raise ValueError(f"error")

    def _to_num(s: pd.Series) -> pd.Series:
        x = s.astype(str).str.strip().str.replace(",", "", regex=False)
        x = x.str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False)
        return pd.to_numeric(x, errors="coerce")

    tnum = _to_num(df[time_col])
    bnum = _to_num(df[beta_col])

    s = pd.DataFrame({"t": tnum, "beta": bnum}).dropna()
    s = s.groupby("t", as_index=False)["beta"].mean().sort_values("t")
    return s["t"].astype(int).to_numpy(), s["beta"].astype(float).to_numpy()


def plot_mean_betas_multi(
    bank_paths: List[str],
    labels: Optional[List[str]] = None,
    observed_csv_path: Optional[str] = None,
    title = r"Estimated $\hat{\boldsymbol{\beta}}_{t}$ using different window sizes",
    save_path: str = "figs/mean_beta_multi.pdf",
    xlim=None,
    ylim=None,
    xtick=None,
    ytick=None,
    show: bool = True,
    close: bool = True,
    fig_size=(9, 4.5),
    line_width: float = 2.2,
    line_colors: Optional[List[str]] = None,
    ci_colors: Optional[List[str]] = None,
    line_alphas: Optional[List[float]] = None,
    ci_alphas: Optional[List[float]] = None,
    observed_marker: str = "o",
    observed_size: float = 1,
    observed_color: str = "black",
    observed_zorder: int = 5,
):
    if labels is None:
        labels = [os.path.splitext(os.path.basename(p))[0] for p in bank_paths]

    if len(labels) != len(bank_paths):
        raise ValueError("error.")

    n = len(bank_paths)
    if line_colors is not None and len(line_colors) != n:
        raise ValueError("error")
    
    if ci_colors is not None and len(ci_colors) != n:
        raise ValueError("error.")
    if line_alphas is not None and len(line_alphas) != n:
        raise ValueError("error.")
    if ci_alphas is not None and len(ci_alphas) != n:
        raise ValueError("error")

    fig, ax = plt.subplots(figsize=fig_size)

    for i, (bank_path, lab) in enumerate(zip(bank_paths, labels)):
        bank_path = _load_path_for_plots(bank_path)

        ts, meanC = load_mean_betas_from_simulations(bank_path)
        if getattr(ts, "size", 0) == 0:
            continue

        lo, hi = load_ci_from_simulations(bank_path, ts)

        tt = ts.astype(float)
        mC = meanC.astype(float)

        lc = None if line_colors is None else line_colors[i]
        la = 1.0 if line_alphas is None else float(line_alphas[i])

        cc = lc if ci_colors is None else ci_colors[i]
        ca = 0.18 if ci_alphas is None else float(ci_alphas[i])

        (line,) = ax.plot(
            tt,
            mC,
            linewidth=float(line_width),
            label=lab,
            color=lc,
            alpha=la,
        )

        if cc is None:
            cc = line.get_color()

        mask = np.isfinite(lo) & np.isfinite(hi)
        if np.any(mask):
            ax.fill_between(tt[mask], lo[mask], hi[mask], alpha=ca, color=cc)

    if observed_csv_path:
        t_obs, b_obs = _load_beta_from_csv(observed_csv_path)
        ax.scatter(
            t_obs.astype(float),
            b_obs.astype(float),
            s=float(observed_size),
            color=observed_color,
            label=r"Observed $\beta_{t}$",
            zorder=int(observed_zorder),
            marker=str(observed_marker),
        )

    ax.set_title(title, fontsize=20, fontweight="bold")
    ax.set_xlabel("Time", fontsize=15, fontweight="bold")
    ax.set_ylabel(r"$\hat{\boldsymbol{\beta}}_{t}$", fontsize=15, fontweight="bold")

    if xlim is not None and len(xlim) == 2:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    if ylim is not None and len(ylim) == 2:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    if xtick is not None and np.size(xtick):
        ax.set_xticks([float(x) for x in np.asarray(xtick).ravel()])
    if ytick is not None and np.size(ytick):
        ax.set_yticks([float(y) for y in np.asarray(ytick).ravel()])

    ax.legend(frameon=False)
    fig.tight_layout()

    out_dir = os.path.dirname(save_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(save_path, dpi=600, bbox_inches="tight")

    if show:
        plt.show()
    if close:
        plt.close(fig)


def plot_posterior_at_time(
    bank_path: str,
    time_point: int,
    observed_csv_path: Optional[str] = None,
    bins: int = 25,
    fig_size=(6, 4),
    save_path: Optional[str] = None,
):

    bank_path = _load_path_for_plots(bank_path)

    key_particles = f"t{int(time_point)}_C_half_betas"
    key_weights   = f"t{int(time_point)}_C_half_weights"
    key_mean      = f"t{int(time_point)}_C_half_mean_beta"
    key_ci        = f"t{int(time_point)}_C_half_mean_beta_ci"

    with np.load(bank_path, allow_pickle=True) as d:

        if key_particles not in d.files:
            raise ValueError(f"No posterior particles found for time {time_point}")

        particles = np.asarray(d[key_particles]).ravel()

        if key_weights in d.files:
            weights = np.asarray(d[key_weights]).ravel().astype(float)
            weights_sum = weights.sum()
            if weights_sum > 0:
                weights = weights / weights_sum
            else:
                weights = None
        else:
            weights = None

        if key_mean in d.files:
            mean_est = float(np.asarray(d[key_mean]).ravel()[0])
        else:
            mean_est = np.sum(particles * weights) if weights is not None else np.mean(particles)

        # Only corrected part: compute CI from the same particles and weights used in the histogram
        if weights is not None:
            idx = np.argsort(particles)
            sorted_p = particles[idx]
            sorted_w = weights[idx]
            cum_w = np.cumsum(sorted_w)
            ci_lo = np.interp(0.025, cum_w, sorted_p)
            ci_hi = np.interp(0.975, cum_w, sorted_p)
        else:
            ci_lo, ci_hi = np.percentile(particles, [2.5, 97.5])

    fig, ax = plt.subplots(figsize=fig_size)

    ax.hist(
        particles,
        bins=bins,
        density=True,
        weights=weights,
        facecolor="#90EE90",
        edgecolor="black",
        alpha=1.0,
        linewidth=1,
    )

    try:
        kde = gaussian_kde(particles, weights=weights) if weights is not None else gaussian_kde(particles)
        x_grid = np.linspace(particles.min(), particles.max(), 1000)
        ax.plot(x_grid, kde(x_grid), color="dodgerblue", linewidth=3)
    except Exception:
        pass

    ax.axvline(mean_est, color="black", linewidth=3, label="Posterior mean")
    ax.axvline(ci_lo, color="dodgerblue", linestyle="--", linewidth=2)
    ax.axvline(ci_hi, color="dodgerblue", linestyle="--", linewidth=2)

    if observed_csv_path:
        t_obs, b_obs = _load_beta_from_csv(observed_csv_path)
        if time_point in t_obs:
            obs_val = b_obs[t_obs.tolist().index(time_point)]
            ax.axvline(obs_val, color="gold", linestyle="-", linewidth=4, label=r"Observed $\beta_{t}$")

    xmin = particles.min() - 0.1
    xmax = particles.max() + 0.1
    ax.set_xlim(xmin, xmax)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xticks(np.arange(xmin, xmax, 0.1))

    ax.set_title(r"Posterior distribution of $\boldsymbol{\beta}_{t}$ (window size = 20)", fontsize=14, fontweight="bold")
    ax.set_xlabel(r"$\boldsymbol{\beta}_{t}$", fontsize=15, fontweight="bold")
    ax.set_ylabel("Density", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)

    import matplotlib.colors as mcolors
    ax.set_facecolor(mcolors.to_rgba("#f2f2f2", alpha=0.5))
    fig.patch.set_facecolor("white")

    ax.legend(frameon=False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")

    plt.show()