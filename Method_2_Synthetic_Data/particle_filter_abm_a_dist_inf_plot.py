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


from Beta_Distance_Dostributions import get_infected_distances_from_csv


def _bank_key(t: int, name: str) -> str:
    return f"t{int(t)}_{name}"


def _find_times_with_half_states(bank_path: str) -> List[int]: 
    with np.load(bank_path, allow_pickle=True) as d:
        pat = re.compile(r"^t(\d+)_half_reps_states$")
        times = []
        for k in d.files:
            m = pat.match(k)
            if m:
                times.append(int(m.group(1)))
        return sorted(times)


def _infer_infected_code_from_meta(bank_path: str) -> int:
    try:
        with np.load(bank_path, allow_pickle=True) as d:
            pat = re.compile(r"^t(\d+)_meta$")
            ts = []
            for k in d.files:
                m = pat.match(k)
                if m:
                    ts.append(int(m.group(1)))
            ts = sorted(ts)
            if not ts:
                return 1
            kmeta = _bank_key(ts[-1], "meta")
            raw = d[kmeta]
            if isinstance(raw, np.ndarray) and raw.dtype == np.uint8:
                import json
                meta = json.loads(raw.tobytes().decode("utf-8"))
                stc = meta.get("status_to_code", {})
                return int(stc.get("I", 1))
    except Exception:
        pass
    return 1


def observed_infected_count_at_t(observed_csv: str, t: int, center=(0, 0)) -> int:
    if get_infected_distances_from_csv is not None:
        dists = get_infected_distances_from_csv(observed_csv, center=center, t_obs=int(t))
        return int(len(dists))

    df = pd.read_csv(observed_csv, dtype=str, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    cols = {c.lower(): c for c in df.columns}

    time_col = None
    for key in ("time", "t", "day", "step", "frame"):
        if key in cols:
            time_col = cols[key]
            break
    if time_col is None:
        raise ValueError("error")
    s = pd.to_numeric(df[time_col].astype(str).str.strip(), errors="coerce")
    return int(np.isclose(s.to_numpy(float), float(t), atol=1e-9).sum())


def observed_infected_series(observed_csv: str, times: np.ndarray, center=(0, 0)) -> np.ndarray:
    return np.asarray([observed_infected_count_at_t(observed_csv, int(t), center=center) for t in times], float)


def _flatten_obj_array_blocks(x) -> List[dict]:
    out: List[dict] = []
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


def selected_infected_counts_from_bank(
    bank_path: str, t: int, infected_code: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    kb = _bank_key(t, "A_half_betas")
    ks = _bank_key(t, "half_reps_states")

    with np.load(bank_path, allow_pickle=True) as d:
        if kb not in d.files:
            raise KeyError(f"Missing {kb} in bank.")
        if ks not in d.files:
            raise KeyError(f"Missing {ks} in bank.")

        betas = np.asarray(d[kb], float).ravel()
        states_obj = d[ks]

    states = _flatten_obj_array_blocks(states_obj)
    if len(states) != betas.size:
        raise RuntimeError(f"Size mismatch at t={t}: betas={betas.size}, states={len(states)}")

    I_counts = []
    for st in states:
        S = np.asarray(st.get("S"), int).ravel()
        I_counts.append(int(np.sum(S == int(infected_code))))
    return betas, np.asarray(I_counts, float)


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

def _weights_from_A_half(bank_path: str, t: int, scale: float = 2.0, temp: float = 0.3) -> np.ndarray:
    kb = _bank_key(t, "A_half_betas")
    kd = _bank_key(t, "A_half_dists")

    with np.load(bank_path, allow_pickle=True) as d:
        if kb not in d.files or kd not in d.files:
            return np.array([], float)
        betas = np.asarray(d[kb], float).ravel()
        errs = np.asarray(d[kd], float).ravel()

    m = np.isfinite(betas) & np.isfinite(errs)
    if not m.any():
        return np.full_like(errs, 1.0 / max(len(errs), 1), dtype=float) if errs.size else np.array([], float)

    e = errs[m]

    med = np.median(e)
    mad = np.median(np.abs(e - med))
    emin = 1.01826 * mad if mad > 1e-12 else max(med, 1.0)
    emin = max(emin, 1e-9) * float(scale)
    tau = max(emin * float(temp), 1e-12)

    W = np.exp(-(e - np.min(e)) / tau)
    s = float(np.sum(W))
    if s <= 0.0 or not np.isfinite(s):
        w = np.full_like(W, 1.0 / len(W), dtype=float)
    else:
        w = W / s

    w_full = np.zeros_like(errs, dtype=float)
    w_full[m] = w
    return w_full


def plot_observed_vs_selected_infected_over_time(
    bank_path: str,
    observed_csv: str,
    center=(0, 0),
    t_max: Optional[int] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    close: bool = False,
):
    times = _find_times_with_half_states(bank_path)
    if t_max is not None:
        times = [t for t in times if int(t) <= int(t_max)]
    if not times:
        raise RuntimeError("error.")

    infected_code = _infer_infected_code_from_meta(bank_path)

    xs: List[float] = []
    ys: List[float] = []

    I_weighted = []
    I_selected_mean = []

    for t in times:
        _betas, I_sel = selected_infected_counts_from_bank(bank_path, int(t), infected_code=infected_code)

    
        xs.extend([float(t)] * int(I_sel.size))
        ys.extend(I_sel.tolist())
        w = _weights_from_A_half(bank_path, int(t), scale=2.0, temp=0.3)
        if w.size == I_sel.size and I_sel.size > 0:
            I_weighted.append(float(np.sum(w * I_sel)))
        else:
            I_weighted.append(float(np.nan))

        I_selected_mean.append(float(np.mean(I_sel)) if I_sel.size else float(np.nan))

    ts = np.asarray(times, float)
    I_weighted = np.asarray(I_weighted, float)
    I_selected_mean = np.asarray(I_selected_mean, float)

    I_obs = observed_infected_series(observed_csv, ts, center=center)

    fig, ax = plt.subplots(figsize=(9, 4.8))

    ax.scatter(np.asarray(xs, float),np.asarray(ys, float),s=22,alpha=0.75,label="Simulated infected",)

    ax.plot(ts, I_weighted, linewidth=2.2, label="Estimated infected")
    ax.scatter(ts, I_weighted, color="black", s=28, zorder=3)

    ax.plot(ts, I_obs, linewidth=2.2, label="Observed infected")
    ax.scatter(ts, I_obs, s=32, zorder=3)

    ax.set_title(title or "Observed infected and PF estimated infected")
    ax.set_xlabel("Time")
    ax.set_ylabel("Infected")
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
        raise ValueError(f"Unsupported CI level: {ci}")

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
    center=(0, 0),
    ylim: Optional[Tuple[float, float]] = None,
    t_max: Optional[int] = None,
    t_print: Optional[int] = None,
):
    times = _find_times_with_half_states(bank_path)
    if t_max is not None:
        times = [t for t in times if int(t) <= int(t_max)]
    if not times:
        raise RuntimeError("error.")

    infected_code = _infer_infected_code_from_meta(bank_path)

    mean_I: List[float] = []

    ci_levels = [95, 90, 75, 50]
    ci_lo = {c: [] for c in ci_levels}
    ci_hi = {c: [] for c in ci_levels}

    xs: List[float] = []
    ys: List[float] = []

    for t in times:
        _betas, I_sel = selected_infected_counts_from_bank(bank_path, int(t), infected_code=infected_code)

        xs.extend([float(t)] * int(I_sel.size))
        ys.extend(I_sel.tolist())

        w = _weights_from_A_half(bank_path, int(t), scale=2.0, temp=0.3)

        if w.size != I_sel.size or I_sel.size == 0:
            mean_I.append(np.nan)
            for c in ci_levels:
                ci_lo[c].append(np.nan)
                ci_hi[c].append(np.nan)
            continue

        m, _, _ = _weighted_mean_ci(I_sel, w, ci=95)
        mean_I.append(m)

        for c in ci_levels:
            _, lo, hi = _weighted_mean_ci(I_sel, w, ci=c)
            ci_lo[c].append(lo)
            ci_hi[c].append(hi)

    ts = np.asarray(times, dtype=int)
    mean_I = np.asarray(mean_I, dtype=float)
    for c in ci_levels:
        ci_lo[c] = np.asarray(ci_lo[c], dtype=float)
        ci_hi[c] = np.asarray(ci_hi[c], dtype=float)

    if t_print is not None:
        _print_estimate_ci_at_time(ts, mean_I, ci_lo[95], ci_hi[95], t_print)

    I_obs = None
    if observed_csv is not None:
        I_obs = observed_infected_series(observed_csv, ts.astype(float), center=center)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_facecolor("lightgray")
    fig.patch.set_facecolor("white")

    for spine in ax.spines.values():
        spine.set_visible(False)

    m95 = np.isfinite(ci_lo[95]) & np.isfinite(ci_hi[95])
    m90 = np.isfinite(ci_lo[90]) & np.isfinite(ci_hi[90])
    m75 = np.isfinite(ci_lo[75]) & np.isfinite(ci_hi[75])
    m50 = np.isfinite(ci_lo[50]) & np.isfinite(ci_hi[50])

    if np.any(m95):
        ax.fill_between(ts[m95].astype(float), ci_lo[95][m95], ci_hi[95][m95], color="blue", alpha=0.10, label="95% CI")
    if np.any(m90):
        ax.fill_between(ts[m90].astype(float), ci_lo[90][m90], ci_hi[90][m90], color="blue", alpha=0.30, label="90% CI")
    if np.any(m75):
        ax.fill_between(ts[m75].astype(float), ci_lo[75][m75], ci_hi[75][m75], color="blue", alpha=0.60, label="75% CI")
    if np.any(m50):
        ax.fill_between(ts[m50].astype(float), ci_lo[50][m50], ci_hi[50][m50], color="blue", alpha=0.90, label="50% CI")

    ax.scatter(np.asarray(xs, float), np.asarray(ys, float), s=22, alpha=0.75, label="Simulated infected")
    ax.plot(ts.astype(float), mean_I, color="red", linewidth=2.0, label="Estimated infected")

    if I_obs is not None:
        ax.plot(ts.astype(float), I_obs, color="orange", linewidth=2.5, label="Observed")
        ax.scatter(ts.astype(float), I_obs, s=10, color="orange")

    ax.set_title(title or "PF infected with multiple confidence intervals", fontweight="bold", fontsize=15)
    ax.set_xlabel("Time", fontweight="bold", fontsize=12)
    ax.set_ylabel("Infected", fontweight="bold", fontsize=12)

    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))

    ax.grid(True, color="white", alpha=0.8)
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
    p.add_argument("--observed-csv", required=True)
    p.add_argument("--mode", choices=("over_time", "at_t"), default="over_time")
    p.add_argument("--t", type=int, default=None)
    p.add_argument("--t-max", type=int, default=None)
    p.add_argument("--center-x", type=int, default=0)
    p.add_argument("--center-y", type=int, default=0)
    p.add_argument("--t-print", type=int, default=None)
    p.add_argument("--save", default=None)
    p.add_argument("--title", default=None)
    p.add_argument("--no-show", action="store_true", help="Do not call plt.show() (useful in notebooks)")
    args = p.parse_args()

    center = (args.center_x, args.center_y)

    if args.mode == "at_t":
        if args.t is None:
            raise SystemExit("mode=at_t requires --t")
        raise SystemExit("error.")
    else:
        plot_selected_infected_time_series_CI(
            bank_path=args.bank_path,
            observed_csv=args.observed_csv,
            center=center,
            t_max=args.t_max,
            title=args.title,
            save_path=args.save,
            show=(not args.no_show),
            t_print=args.t_print, 
            close=False,
        )



if __name__ == "__main__":
    main()
