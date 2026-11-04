# plot_particles.py
from __future__ import annotations

import os
import re
import argparse
import numpy as np

os.environ.pop("MPLBACKEND", None)
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def _bank_key(t: int, name: str) -> str:
    return f"t{int(t)}_{name}"


def _find_times(bank_path: str) -> list[int]:
    with np.load(bank_path, allow_pickle=True) as d:
        pat = re.compile(r"^t(\d+)_rep_betas_all$")
        ts = sorted(int(m.group(1)) for k in d.files for m in [pat.match(k)] if m)
    return ts


def _load_one_time(d: np.lib.npyio.NpzFile, t: int):
    kb = _bank_key(t, "rep_betas_all")
    kA = _bank_key(t, "rep_A_dists_all")

    betas_all = np.asarray(d[kb], float) if kb in d.files else np.array([], float)
    dA_all    = np.asarray(d[kA], float) if kA in d.files else np.array([], float)

    khb = _bank_key(t, "A_half_betas")
    khA = _bank_key(t, "A_half_dists")
    betas_half = np.asarray(d[khb], float) if khb in d.files else np.array([], float)
    dA_half    = np.asarray(d[khA], float) if khA in d.files else np.array([], float)

    km = _bank_key(t, "A_half_mean_beta")
    if km in d.files and np.asarray(d[km]).size:
        est_beta = float(np.asarray(d[km]).ravel()[0])
    else:
        est_beta = np.nan

    return betas_all, dA_all, betas_half, dA_half, est_beta


def _to_num(x) -> float:
    if x is None:
        return np.nan
    try:
        if isinstance(x, str):
            x = x.strip().replace(",", "")
        return float(x)
    except Exception:
        return np.nan


def _load_observed_beta_by_time(csv_path: str | None) -> dict[int, float]:
    if not csv_path or (not os.path.exists(csv_path)):
        return {}
    try:
        import pandas as pd
    except Exception:
        return {}

    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    cols = {c.lower(): c for c in df.columns}

    time_col = next((cols[c] for c in ("time", "t") if c in cols), None)
    beta_col = next((cols[c] for c in ("beta", "true_beta", "observed_beta") if c in cols), None)
    if time_col is None or beta_col is None:
        return {}

    out: dict[int, list[float]] = {}
    for _, row in df.iterrows():
        tt = _to_num(row.get(time_col))
        bb = _to_num(row.get(beta_col))
        if not np.isfinite(tt) or not np.isfinite(bb):
            continue
        t = int(round(tt))
        out.setdefault(t, []).append(float(bb))

    return {t: float(np.mean(v)) for t, v in out.items() if len(v)}


def plot_grid(
    *,
    bank_path: str,
    times: list[int],
    nrows: int = 4,
    ncols: int = 5,
    figsize: tuple[float, float] | None = None,
    save_path: str = "particles_grid.png",
    show: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    observed_csv: str | None = None,
):
    if figsize is None:
        figsize = (ncols * 3.0, nrows * 2.6)

    obs_beta = _load_observed_beta_by_time(observed_csv)

    with np.load(bank_path, allow_pickle=True) as d:
        loaded = []
        xmin, xmax = +np.inf, -np.inf
        ymin, ymax = +np.inf, -np.inf

        for t in times:
            betas_all, dA_all, betas_half, dA_half, est_beta = _load_one_time(d, t)
            loaded.append((t, betas_all, dA_all, betas_half, dA_half, est_beta))

            for arr in (betas_all, betas_half):
                arr = arr[np.isfinite(arr)]
                if arr.size:
                    xmin = min(xmin, float(arr.min()))
                    xmax = max(xmax, float(arr.max()))

            for arr in (dA_all, dA_half):
                arr = arr[np.isfinite(arr)]
                if arr.size:
                    ymin = min(ymin, float(arr.min()))
                    ymax = max(ymax, float(arr.max()))

        if xlim is None:
            if np.isfinite(xmin) and np.isfinite(xmax) and xmin != xmax:
                pad = 0.05 * (xmax - xmin)
                xlim = (xmin - pad, xmax + pad)
            else:
                xlim = (0.0, 1.0)

        if ylim is None:
            if np.isfinite(ymin) and np.isfinite(ymax) and ymin != ymax:
                pad = 0.08 * (ymax - ymin)
                ylim = (ymin - pad, ymax + pad)
            else:
                ylim = (0.0, 1.0)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        axes = axes.ravel()

        for i, ax in enumerate(axes):
            if i >= len(loaded):
                ax.axis("off")
                continue

            t, betas_all, dA_all, betas_half, dA_half, est_beta = loaded[i]

            ax.add_patch(
                Rectangle(
                    (0, 0), 1, 1,
                    transform=ax.transAxes,
                    facecolor="black",
                    edgecolor="#5AA9FF",
                    linewidth=1.0,
                    zorder=0,
                )
            )

            m_half = np.isfinite(betas_half) & np.isfinite(dA_half)
            if m_half.any():
                ax.scatter(
                    betas_half[m_half], dA_half[m_half],
                    s=30, c="yellow", alpha=1.0,
                    edgecolors="black", linewidths=0.25,
                    zorder=3,
                )

                y_dot = ylim[0] + 0.08 * (ylim[1] - ylim[0])

                if np.isfinite(est_beta):
                    ax.scatter([est_beta], [y_dot], s=55, c="red", zorder=6)

                b_obs = obs_beta.get(int(t), np.nan)
                if np.isfinite(b_obs):
                    ax.scatter([b_obs], [y_dot], s=55, c="white",
                               edgecolors="black", linewidths=0.8, zorder=7)

            ax.set_title(f"t = {t}", fontsize=10, pad=4)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            ax.grid(True, alpha=0.18, zorder=1)
            ax.tick_params(labelsize=8)

            if i % ncols != 0:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            else:
                ax.set_ylabel("Distance", fontsize=9)

            if i < (nrows - 1) * ncols:
                ax.set_xlabel("")
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("β", fontsize=9)

        fig.suptitle("Particles per window (selected) with estimated and observed β", fontsize=12, y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.98])

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[OK] Saved: {save_path}")

        if show:
            plt.show()
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Plot particles as a grid of subplots across windows.")
    p.add_argument("--bank-path", required=True, help="Your .npz bank file saved by window_step/run_simulations.")
    p.add_argument("--observed-csv", default=None, help="CSV containing observed beta.")
    p.add_argument("--t-end", type=int, default=400, help="Final time to plot (default 400).")
    p.add_argument("--window", type=int, default=20, help="Window size (default 20).")
    p.add_argument("--nrows", type=int, default=4, help="Grid rows (default 4).")
    p.add_argument("--ncols", type=int, default=5, help="Grid cols (default 5).")
    p.add_argument("--save", type=str, default="particles_grid.png", help="Output image path.")
    p.add_argument("--show", action="store_true", help="Also show the figure (still saved).")
    args = p.parse_args()

    times = list(range(args.window, args.t_end + 1, args.window))
    present = set(_find_times(args.bank_path))
    times = [t for t in times if t in present]

    if not times:
        raise RuntimeError("No matching times found in bank. Check that t*_rep_betas_all exists.")

    cap = args.nrows * args.ncols
    if len(times) > cap:
        times = times[:cap]

    plot_grid(
        bank_path=args.bank_path,
        times=times,
        nrows=args.nrows,
        ncols=args.ncols,
        save_path=args.save,
        show=args.show,
        observed_csv=args.observed_csv,
    )


if __name__ == "__main__":
    main()
