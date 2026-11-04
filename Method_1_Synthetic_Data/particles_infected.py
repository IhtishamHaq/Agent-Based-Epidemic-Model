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
    kA = _bank_key(t, "rep_C_dists_all")

    betas_all = np.asarray(d[kb], float) if kb in d.files else np.array([], float)
    dA_all = np.asarray(d[kA], float) if kA in d.files else np.array([], float)

    khb = _bank_key(t, "C_half_betas")
    betas_half = np.asarray(d[khb], float) if khb in d.files else np.array([], float)

    ki = _bank_key(t, "C_half_I_sim")
    I_half = np.asarray(d[ki], float) if ki in d.files else np.array([], float)

    km = _bank_key(t, "C_half_mean_beta")
    if km in d.files and np.asarray(d[km]).size:
        est_beta = float(np.asarray(d[km]).ravel()[0])
    else:
        est_beta = np.nan

    return betas_all, dA_all, betas_half, I_half, est_beta


def _to_num(x) -> float:
    if x is None:
        return np.nan
    try:
        if isinstance(x, str):
            x = x.strip().replace(",", "")
        return float(x)
    except Exception:
        return np.nan


def _load_observed_beta_I(csv_path: str | None):
    if not csv_path or (not os.path.exists(csv_path)):
        return {}

    try:
        import pandas as pd
    except Exception:
        return {}

    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    cols = {c.lower(): c for c in df.columns}

    time_col = next((cols[c] for c in ("time", "t", "day", "step", "frame") if c in cols), None)
    if time_col is None:
        return {}

    beta_col = next((cols[c] for c in ("beta", "b", "beta_true", "beta_obs", "true_beta", "observed_beta") if c in cols), None)
    inf_col = next(
        (cols[c] for c in (
            "infected", "i", "infected_count", "cases", "count", "reported_cases",
            "hokkaido_cases", "total_infected"
        ) if c in cols),
        None,
    )

    out = {}
    for _, row in df.iterrows():
        tt = _to_num(row.get(time_col))
        if not np.isfinite(tt):
            continue
        t = int(round(tt))
        b = _to_num(row.get(beta_col)) if beta_col else np.nan
        I = _to_num(row.get(inf_col)) if inf_col else np.nan

        if t not in out:
            out[t] = {"beta": [], "I": []}
        out[t]["beta"].append(b)
        out[t]["I"].append(I)

    obs = {}
    for t, v in out.items():
        bb = np.asarray(v["beta"], float)
        ii = np.asarray(v["I"], float)
        b = float(np.nanmean(bb)) if np.isfinite(bb).any() else np.nan
        I = float(np.nansum(ii)) if np.isfinite(ii).any() else np.nan
        obs[t] = (b, I)

    return obs


def _strip_axes(ax):
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.minorticks_off()
    ax.tick_params(axis="both", which="both", length=0, width=0)
    ax.tick_params(top=False, right=False)


def plot_grid(
    *,
    bank_path: str,
    times: list[int],
    nrows: int = 4,
    ncols: int = 5,
    figsize: tuple[float, float] | None = None,
    save_path: str = "particles_grid_infected.png",
    show: bool = False,
    xlim: tuple[float, float] | None = None,
    observed_csv: str | None = None,
):
    if figsize is None:
        figsize = (8, 5)

    obs = _load_observed_beta_I(observed_csv)

    with np.load(bank_path, allow_pickle=True) as d:
        loaded = []
        xmin, xmax = +np.inf, -np.inf

        for t in times:
            betas_all, dA_all, betas_half, I_half, est_beta = _load_one_time(d, t)
            loaded.append((t, betas_half, I_half, est_beta))

            arr = betas_half[np.isfinite(betas_half)]
            if arr.size:
                xmin = min(xmin, float(arr.min()))
                xmax = max(xmax, float(arr.max()))

        xlim = (0.0, 1.0001) if xlim is None else xlim

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        fig.supylabel("Infected", fontsize=15)
        fig.supxlabel(r"$\beta_{t}$", fontsize=15)

        axes = axes.ravel()

        for i, ax in enumerate(axes):
            _strip_axes(ax)

            if i >= len(loaded):
                ax.axis("off")
                continue

            t, betas_half, I_half, est_beta = loaded[i]

            ax.add_patch(
                Rectangle(
                    (0, 0), 1.0, 1.0,
                    transform=ax.transAxes,
                    facecolor="black",
                    edgecolor="#5AA9FF",
                    linewidth=1.5,
                    zorder=0,
                    clip_on=False,
                )
            )

            m = np.isfinite(betas_half) & np.isfinite(I_half)
            if m.any():
                ax.scatter(
                    betas_half[m],
                    I_half[m],
                    s=30,
                    c="green",
                    edgecolors="green",
                    linewidths=0.25,
                    zorder=2,
                )

                est_I = np.nan
                if np.isfinite(est_beta):
                    j = int(np.nanargmin(np.abs(betas_half[m] - est_beta)))
                    est_I = float(I_half[m][j])

                if np.isfinite(est_beta) and np.isfinite(est_I):
                    ax.scatter([est_beta], [est_I], s=50, c="red", zorder=5)

            if t in obs:
                ob, oI = obs[t]
                if np.isfinite(ob) and np.isfinite(oI):
                    ax.scatter([ob], [oI], s=50, c="white", edgecolors="black", linewidths=1.4, zorder=5)

            ax.set_title(f"t = {t}", fontsize=10, pad=4)
            ax.set_xlim(xlim)

            ymax_local = float(np.nanmax(I_half)) + 500.0
            ymin_local = max(0.0, float(np.nanmin(I_half)) - 300.0)
            ax.set_ylim(ymin_local, ymax_local)

            col = i % ncols
            if col != 0:
                ax.set_ylabel("")
                ax.set_yticklabels([])

            if i < (nrows - 1) * ncols:
                ax.set_xlabel("")
                ax.set_xticklabels([])

        fig.suptitle("Snapshots of particles at different times", fontsize=15, y=0.995, fontweight="bold")
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
    p.add_argument("--observed-csv", default=None, help="CSV containing observed beta and infected.")
    p.add_argument("--t-end", type=int, default=400, help="Final time to plot (default 400).")
    p.add_argument("--window", type=int, default=20, help="Window size (default 20).")
    p.add_argument("--nrows", type=int, default=4, help="Grid rows (default 4).")
    p.add_argument("--ncols", type=int, default=5, help="Grid cols (default 5).")
    p.add_argument("--save", type=str, default="particles_grid_infected.png", help="Output image path.")
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
