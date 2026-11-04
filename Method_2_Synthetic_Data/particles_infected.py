from __future__ import annotations

import os
import re
import json
import argparse
import numpy as np

os.environ.pop("MPLBACKEND", None)
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def _bank_key(t: int, name: str) -> str:
    return f"t{int(t)}_{name}"


def _pf_results_path(bank_path: str) -> str:
    root, _ = os.path.splitext(bank_path)
    return root + "_pf_results.npz"


def _load_path_for_plots(path: str) -> str:
    pf = _pf_results_path(path)
    return pf if os.path.exists(pf) else path


def _find_times(bank_path: str, prefix: str = "A") -> list[int]:
    bank_path = _load_path_for_plots(bank_path)
    with np.load(bank_path, allow_pickle=True) as d:
        pat = re.compile(rf"^t(\d+)_{re.escape(prefix)}_half_betas$")
        ts = sorted(int(m.group(1)) for k in d.files for m in [pat.match(k)] if m)
    return ts


def _to_num(x) -> float:
    if x is None:
        return np.nan
    try:
        if isinstance(x, str):
            x = x.strip().replace(",", "")
        return float(x)
    except Exception:
        return np.nan


def _load_observed_time_beta_infected(csv_path: str | None) -> dict[int, tuple[float, float]]:
    if not csv_path or (not os.path.exists(csv_path)):
        return {}
    try:
        import pandas as pd
    except Exception:
        return {}

    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    cols = {c.lower(): c for c in df.columns}

    time_col = cols.get("time")
    beta_col = cols.get("beta")
    inf_col = cols.get("infected")

    if time_col is None or beta_col is None or inf_col is None:
        return {}

    out: dict[int, tuple[float, float]] = {}
    for _, row in df.iterrows():
        tt = _to_num(row.get(time_col))
        bb = _to_num(row.get(beta_col))
        ii = _to_num(row.get(inf_col))
        if np.isfinite(tt) and np.isfinite(bb) and np.isfinite(ii):
            out[int(round(tt))] = (float(bb), float(ii))
    return out


def _flatten_obj_array_blocks(x):
    out = []
    if x is None:
        return out
    arr = np.asarray(x, dtype=object).ravel()
    for item in arr:
        if item is None:
            continue
        if isinstance(item, dict):
            out.append(item)
        elif isinstance(item, (list, tuple)):
            for z in item:
                if isinstance(z, dict):
                    out.append(z)
    return out


def _infer_infected_code_from_meta(bank_path: str) -> int:
    bank_path = _load_path_for_plots(bank_path)
    try:
        with np.load(bank_path, allow_pickle=True) as d:
            pat = re.compile(r"^t(\d+)_meta$")
            ts = []
            for k in d.files:
                m = pat.match(k)
                if m:
                    ts.append(int(m.group(1)))
            if not ts:
                return 1
            ts = sorted(ts)
            raw = d[_bank_key(ts[-1], "meta")]
            if isinstance(raw, np.ndarray) and raw.dtype == np.uint8:
                meta = json.loads(raw.tobytes().decode("utf-8"))
                return int(meta.get("status_to_code", {}).get("I", 1))
    except Exception:
        pass
    return 1


def load_selected_infected(bank_path: str, t: int) -> np.ndarray:
    bank_path = _load_path_for_plots(bank_path)
    k = _bank_key(t, "half_reps_states")

    with np.load(bank_path, allow_pickle=True) as d:
        if k not in d.files:
            return np.array([], dtype=float)
        infected_code = _infer_infected_code_from_meta(bank_path)
        states_obj = d[k]

    states = _flatten_obj_array_blocks(states_obj)
    I_counts = []
    for st in states:
        S = np.asarray(st.get("S"), int).ravel()
        I_counts.append(int(np.sum(S == infected_code)))
    return np.asarray(I_counts, dtype=float)


def load_selected_weights(bank_path: str, t: int, prefix: str = "A") -> np.ndarray:
    bank_path = _load_path_for_plots(bank_path)
    kd = _bank_key(t, f"{prefix}_half_dists")

    with np.load(bank_path, allow_pickle=True) as d:
        if kd not in d.files:
            return np.array([], dtype=float)
        errs = np.asarray(d[kd], float).ravel()

    if errs.size == 0:
        return np.array([], float)

    med = np.median(errs)
    mad = np.median(np.abs(errs - med))
    emin = 1.01826 * mad if mad > 1e-12 else max(med, 1.0)
    emin = max(emin, 1e-9) * 2.0
    tau = max(emin * 0.3, 1e-12)

    W = np.exp(-(errs - np.min(errs)) / tau)
    s = np.sum(W)
    if s > 0:
        W /= s
    return W


def _load_one_time(d: np.lib.npyio.NpzFile, t: int, prefix: str = "A"):
    khb = _bank_key(t, f"{prefix}_half_betas")
    betas_half = np.asarray(d[khb], float) if khb in d.files else np.array([], float)

    km = _bank_key(t, f"{prefix}_half_mean_beta")
    if km in d.files and np.asarray(d[km]).size:
        est_beta = float(np.asarray(d[km]).ravel()[0])
    else:
        est_beta = np.nan

    return betas_half, est_beta


def _strip_axes(ax):
    ax.grid(False)
    ax.minorticks_off()
    ax.tick_params(labelsize=8)
    ax.tick_params(axis="both", which="both", length=0)
    ax.tick_params(top=False, right=False)


def plot_grid_infected_vs_beta(
    *,
    bank_path: str,
    times: list[int],
    prefix: str = "A",
    nrows: int = 4,
    ncols: int = 5,
    figsize: tuple[float, float] | None = None,
    save_path: str = "infected_vs_beta_grid.png",
    show: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    observed_csv: str | None = None,
):
    if figsize is None:
        figsize = (8, 5)

    bank_path2 = _load_path_for_plots(bank_path)
    obs = _load_observed_time_beta_infected(observed_csv)

    loaded = []
    xmin, xmax = +np.inf, -np.inf
    ymin, ymax = +np.inf, -np.inf

    with np.load(bank_path2, allow_pickle=True) as d:
        for t in times:
            betas_half, est_beta = _load_one_time(d, t, prefix=prefix)
            I_half = load_selected_infected(bank_path2, t)

            mlen = min(len(betas_half), len(I_half))
            if mlen <= 0:
                continue

            betas_half = np.asarray(betas_half[:mlen], float)
            I_half = np.asarray(I_half[:mlen], float)

            mask = np.isfinite(betas_half) & np.isfinite(I_half)
            if not mask.any():
                continue

            b_obs, I_obs = obs.get(int(t), (np.nan, np.nan))
            loaded.append((t, betas_half, I_half, est_beta, b_obs, I_obs))

            bb = betas_half[np.isfinite(betas_half)]
            ii = I_half[np.isfinite(I_half)]
            if bb.size:
                xmin = min(xmin, float(bb.min()))
                xmax = max(xmax, float(bb.max()))
            if ii.size:
                ymin = min(ymin, float(ii.min()))
                ymax = max(ymax, float(ii.max()))

            if np.isfinite(b_obs):
                xmin = min(xmin, float(b_obs))
                xmax = max(xmax, float(b_obs))
            if np.isfinite(I_obs):
                ymin = min(ymin, float(I_obs))
                ymax = max(ymax, float(I_obs))

    if not loaded:
        raise RuntimeError("No particle windows found to plot.")

    if xlim is None:
        if np.isfinite(xmin) and np.isfinite(xmax) and xmin != xmax:
            pad = 0.06 * (xmax - xmin)
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
    fig.supylabel("Infected", fontsize=14)
    fig.supxlabel(r"$\beta_{t}$", fontsize=14)

    for i, ax in enumerate(axes):
        _strip_axes(ax)

        if i >= len(loaded):
            ax.axis("off")
            continue

        t, betas_half, I_half, est_beta, b_obs, I_obs = loaded[i]

        ax.add_patch(
            Rectangle(
                (0, 0), 1, 1,
                transform=ax.transAxes,
                facecolor="black",  # darkblue
                edgecolor="#5AA9FF",   # #5AA9FF
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
                s=20,
                c="blue",
                edgecolors="blue",
                linewidths=0.25,
                zorder=2,
                ) 

        w = load_selected_weights(bank_path2, int(t), prefix=prefix)
        w = np.asarray(w, float).ravel()
        mm = min(len(w), len(I_half))
        w = w[:mm]
        Ih = np.asarray(I_half, float).ravel()[:mm]

        if w.size and np.isfinite(est_beta) and np.isfinite(Ih).any() and np.sum(w) > 0:
            w = w / np.sum(w)
            est_I = float(np.sum(w * Ih))
            ax.scatter([est_beta], [est_I], s=50, c="red", zorder=5)

        if np.isfinite(b_obs) and np.isfinite(I_obs):
            ax.scatter([b_obs], [I_obs], s=50,  c="white", edgecolors="black", linewidths=1.4, zorder=5)

        ax.set_title(f"t = {t}", fontsize=10, pad=4)
        
        ymax_local = float(np.nanmax(I_half)) + 500.0
        ymin_local = max(0.0, float(np.nanmin(I_half)) - 300.0)
        ax.set_ylim(ymin_local, ymax_local)
        ax.set_xlim(xlim)
        

        col = i % ncols
        if col != 0:
            ax.set_yticklabels([])
        if i < (nrows - 1) * ncols:
            ax.set_xticklabels([])

    fig.suptitle("Snapshots of particles at different times", fontsize=15, y=0.995, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Grid plot: Infected vs Beta (selected particles).")
    p.add_argument("--bank-path", required=True, help="Your .npz bank file.")
    p.add_argument("--prefix", default="A", help="Key prefix: A or C etc (default A).")
    p.add_argument("--observed-csv", default=None, help="ONE CSV with columns: Time,Beta,Infected (required for observed point).")
    p.add_argument("--t-end", type=int, default=400)
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--nrows", type=int, default=4)
    p.add_argument("--ncols", type=int, default=5)
    p.add_argument("--save", type=str, default="infected_vs_beta_grid.png")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    present = set(_find_times(args.bank_path, prefix=args.prefix))
    times = list(range(args.window, args.t_end + 1, args.window))
    times = [t for t in times if t in present]

    if not times:
        raise RuntimeError("No matching times found. Expected keys like t{t}_A_half_betas (or your prefix).")

    cap = args.nrows * args.ncols
    times = times[: max(cap * 2, cap)]

    plot_grid_infected_vs_beta(
        bank_path=args.bank_path,
        times=times,
        prefix=args.prefix,
        nrows=args.nrows,
        ncols=args.ncols,
        save_path=args.save,
        show=args.show,
        observed_csv=args.observed_csv,
    )


if __name__ == "__main__":
    main()
