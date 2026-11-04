import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional
import matplotlib.colors as mcolors


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
            c = float(d[keyC][0]) if keyC in d.files and d[keyC].size else np.nan
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

    time_candidates = ["time", "t", "day"]
    beta_candidates = ["beta", "beta_hat", "est_beta"]

    time_col = next((cols[c] for c in time_candidates if c in cols), None)
    beta_col = next((cols[c] for c in beta_candidates if c in cols), None)

    if time_col is None:
        raise ValueError(f"No time column")
    if beta_col is None:
        raise ValueError(f"No beta column")

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
    bank_paths,
    labels=None,
    title=r"Estimated $\mathbf{\hat{\beta}_{t}}$",
    save_path="figs/mean_beta_multi.pdf",
    xlim=None,
    ylim=None,
    xtick=np.arange(0, 365, 14),
    ytick=None,
    show=True,
    close=True,
    fig_size=(10, 5),
    line_width=2.2,
    line_colors=None,
    lines_colors=None,
    lined_colors=None,
    ci_colors=None,
    line_alphas=None,
    ci_alphas=None,
    ncols=3,
):
    if labels is None:
        labels = [os.path.splitext(os.path.basename(p))[0] for p in bank_paths]

    n = len(bank_paths)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=fig_size, sharex=False, sharey=False)
    axes = np.asarray(axes).ravel()

    ymax_global = 0.0

    for i, (bank_path, lab) in enumerate(zip(bank_paths, labels)):
        ax = axes[i]
        bank_path = _load_path_for_plots(bank_path)

        ts, meanC = load_mean_betas_from_simulations(bank_path)
        if getattr(ts, "size", 0) == 0:
            ax.set_visible(False)
            continue

        lo, hi = load_ci_from_simulations(bank_path, ts)

        tt = ts.astype(float) / 7.0
        mC = meanC.astype(float)

        se = (hi - lo) / (2 * 1.96)
        z65 = 0.94
        lo65 = mC - z65 * se
        hi65 = mC + z65 * se

        ymax_global = max(ymax_global, np.nanmax(hi))

        lc = line_colors[i]
        lcs = lines_colors[i]
        cd = lined_colors[i]
        la = 1.0 if line_alphas is None else line_alphas[i]
        cc = lc if ci_colors is None else ci_colors[i]
        ca = 0.18 if ci_alphas is None else ci_alphas[i]

        (line,) = ax.plot(tt, mC, linewidth=2.0, color=lcs, alpha=la)

        if isinstance(xlim, list):
            ax.set_xlim(np.array(xlim[i]) / 7.0)
        elif xlim is not None:
            ax.set_xlim(np.array(xlim) / 7.0)

        ax.set_xticks(xtick / 7.0)

        if isinstance(ylim, list):
            ax.set_ylim(ylim[i])
        elif ylim is not None:
            ax.set_ylim(ylim)

        if isinstance(ytick, list):
            ax.set_yticks(ytick[i])
        elif ytick is not None:
            ax.set_yticks(ytick)
            

        if cc is None:
            cc = line.get_color()

        mask = np.isfinite(lo) & np.isfinite(hi)
        if np.any(mask):
            ax.fill_between(tt[mask], lo[mask], hi[mask], color=cc, alpha=ca)

        mask65 = np.isfinite(lo65) & np.isfinite(hi65)
        if np.any(mask65):
            ax.fill_between(tt[mask65], lo65[mask65], hi65[mask65], color=cd, alpha=ca * 1.8)

        ax.set_facecolor(mcolors.to_rgba('lightgray', alpha=0.3))

        ax.set_title(lab, fontsize=12, fontweight="bold", x=0.6, y=0.7)
       
        ax.grid(False)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=15, fontweight="bold", x=0.5, y=0.9)
    fig.text(0.5, 0.04, "Time (weeks)", ha="center", fontsize=14, fontweight="bold")
    fig.text(0.04, 0.5, r"$\mathbf{\hat{\beta}_{t}}$", va="center",
             fontsize=14, fontweight="bold")

    fig.tight_layout(rect=[0.05, 0.05, 1, 0.95])

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    if close:
        plt.close(fig)