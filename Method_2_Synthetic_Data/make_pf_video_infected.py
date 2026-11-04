import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter, writers

from particle_filter_abm_a_di import (
    _find_times_with_mean_betas,
    _bank_key,
)


def _pf_results_path(bank_path: str) -> str:
    root, _ = os.path.splitext(bank_path)
    return root + "_pf_results.npz"


def _load_path_for_plots(path: str) -> str:
    pf = _pf_results_path(path)
    return pf if os.path.exists(pf) else path



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



def _flatten_states(x):
    out = []
    arr = np.asarray(x, dtype=object).ravel()
    for item in arr:
        if isinstance(item, dict):
            out.append(item)
        elif isinstance(item, (list, tuple)):
            for z in item:
                if isinstance(z, dict):
                    out.append(z)
    return out


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

    import pandas as pd

    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    cols = {c.lower(): c for c in df.columns}

    time_col = next((cols[c] for c in ("time", "t") if c in cols), None)
    beta_col = next((cols[c] for c in ("beta", "b", "beta_obs") if c in cols), None)
    inf_col  = next((cols[c] for c in ("infected") if c in cols), None)

    if time_col is None:
        return {}

    tmp = {}
    for _, row in df.iterrows():
        tt = _to_num(row.get(time_col))
        if not np.isfinite(tt):
            continue
        t = int(round(tt))

        bb = _to_num(row.get(beta_col)) if beta_col else np.nan
        ii = _to_num(row.get(inf_col)) if inf_col else np.nan

        tmp.setdefault(t, {"b": [], "i": []})
        if np.isfinite(bb):
            tmp[t]["b"].append(bb)
        if np.isfinite(ii):
            tmp[t]["i"].append(ii)

    out = {}
    for t, v in tmp.items():
        b = float(np.mean(v["b"])) if len(v["b"]) else np.nan
        I = float(np.sum(v["i"])) if len(v["i"]) else np.nan
        out[t] = (b, I)
    return out


def make_pf_video_beta_vs_infected(
    bank_path: str,
    observed_csv: str | None = None,
    out_path: str = "pf_beta_vs_infected.mp4",
    fps: int = 3,
):

    bank_path = _load_path_for_plots(bank_path)
    times = _find_times_with_mean_betas(bank_path)
    obs = _load_observed_beta_I(observed_csv)

    fig, ax = plt.subplots(figsize=(8, 5))

    

    
    scat_sel = ax.scatter([], [], s=80, alpha=1.0, color="gold", label="Selected Particles")
    est_dot = ax.scatter([], [], s=120, color="red", zorder=6, label=r"Estimated $(\hat{\beta}_{t}, \hat{I}_{t_k})$")
    obs_dot = ax.scatter([], [], s=120, color="white", edgecolors="black", linewidths=1.0, zorder=7, label=r"Observed $(\beta_{t}, I_{t_k})$")

    ax.set_xlabel(r"$\beta_{t}$", fontsize=12)
    ax.set_ylabel("Infected individuals", fontsize=12)
    ax.grid(True, alpha=0.05)
    ax.legend(fontsize=10, frameon=False)

    all_betas = []
    all_I = []

    def _load_frame(t):

        with np.load(bank_path, allow_pickle=True) as d:
            keyB = _bank_key(t, "A_half_betas")
            keyS = _bank_key(t, "half_reps_states")
            keyM = _bank_key(t, "A_half_mean_beta")

            if keyB not in d.files or keyS not in d.files:
                return np.array([]), np.array([]), np.nan, np.nan

            betas = np.asarray(d[keyB], float).ravel()
            states = _flatten_states(d[keyS])
            est_beta = float(np.asarray(d[keyM]).ravel()[0]) if keyM in d.files else np.nan

        I_sel = []
        for st in states:
            S = np.asarray(st.get("S"), int).ravel()
            I_sel.append(np.sum(S == 1))

        I_sel = np.asarray(I_sel, float)

        m = np.isfinite(betas) & np.isfinite(I_sel)
        betas = betas[m]
        I_sel = I_sel[m]

        w = _weights_from_A_half(bank_path, t)
        est_I = np.nan
        if w.size == I_sel.size and I_sel.size > 0:
            est_I = float(np.sum(w * I_sel))

        return betas, I_sel, est_beta, est_I

    
    for t in times:
        b, I, _, _ = _load_frame(t)
        all_betas.extend(b)
        all_I.extend(I)

    all_betas = np.asarray(all_betas)
    all_I = np.asarray(all_I)

    xmin, xmax = np.nanmin(all_betas), np.nanmax(all_betas)
    ymin, ymax = np.nanmin(all_I), np.nanmax(all_I)

    def update(i):
        t = int(times[i])
        betas, I_sel, est_beta, est_I = _load_frame(t)

        scat_sel.set_offsets(np.c_[betas, I_sel] if betas.size else np.empty((0, 2)))

  
        ax.set_xlim(0, 2.5)
        ax.set_ylim(0, ymax + 10)

        if np.isfinite(est_beta) and np.isfinite(est_I):
            est_dot.set_offsets([[est_beta, est_I]])
        else:
            est_dot.set_offsets(np.empty((0, 2)))

        if t in obs:
            b, I = obs[t]
            if np.isfinite(b) and np.isfinite(I):
                obs_dot.set_offsets([[b, I]])
            else:
                obs_dot.set_offsets(np.empty((0, 2)))

        ax.set_title(fr"Selected Particles at Time $t = {t}$ (method 2, window 5, synthetic data)",fontsize=12)
        return scat_sel, est_dot, obs_dot

    anim = FuncAnimation(fig, update, frames=len(times), interval=1000 // fps)

    if writers.is_available("ffmpeg"):
        anim.save(out_path, writer=FFMpegWriter(fps=fps))
    else:
        anim.save(out_path.replace(".mp4", ".gif"), writer=PillowWriter(fps=fps))

    plt.close(fig)
    return out_path