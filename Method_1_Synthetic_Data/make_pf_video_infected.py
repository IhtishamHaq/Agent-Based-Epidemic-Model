import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter, writers

from particle_filter_abm_c_dist import (
    _load_path_for_plots,
    _find_times_with_mean_betas,
    _bank_key,
)

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

    time_col = next((cols[c] for c in ("time", "t") if c in cols), None)
    beta_col = next((cols[c] for c in ("beta", "b", "observed_beta") if c in cols), None)
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
        I = float(np.sum(v["i"]))  if len(v["i"]) else np.nan
        out[t] = (b, I)
    return out


def make_pf_video_beta_vs_infected(
    bank_path: str,
    observed_csv: str | None = None,
    out_path: str = "pf_beta_vs_infected.mp4",
    fps: int = 3,
    xlim=None,
    ylim=None,
    show_est_on_curve: bool = True,
    show_observed_dot: bool = True,
):
    bank_path = _load_path_for_plots(bank_path)
    times = _find_times_with_mean_betas(bank_path)
    if not times:
        raise RuntimeError("error.")

    obs = _load_observed_beta_I(observed_csv)

    fig, ax = plt.subplots(figsize=(8, 5))


 
    scat_sel = ax.scatter([], [], s=80, alpha=1.0, color="green", label="Selected Particles")
    est_dot = ax.scatter([], [], s=120, color="red", zorder=6, label=r"Estimated $(\hat{\beta}_{t}, \hat{I}_{t_k})$")
    obs_dot = ax.scatter([], [], s=120, color="white", edgecolors="black", linewidths=1.0, zorder=7, label=r"Observed $(\beta_{t}, I_{t_k})$")

    ax.set_xlabel(r"$\beta_{t}$", fontsize=12)
    ax.set_ylabel("Infected individuals", fontsize=12)
    ax.grid(True, alpha=0.05)
    ax.legend(fontsize=10, frameon=False)

    def _load_frame(t: int):
        with np.load(bank_path, allow_pickle=True) as d:
            keyB = _bank_key(t, "C_half_betas")
            keyI = _bank_key(t, "C_half_I_sim")
            keyM = _bank_key(t, "C_half_mean_beta")

            betas = np.asarray(d[keyB], dtype=float) if keyB in d.files else np.array([], float)
            I_sel = np.asarray(d[keyI], dtype=float) if keyI in d.files else np.array([], float)
            est_beta = float(np.asarray(d[keyM]).ravel()[0]) if (keyM in d.files and np.asarray(d[keyM]).size) else np.nan

        n = min(betas.size, I_sel.size)
        if n == 0:
            return np.array([], float), np.array([], float), np.nan, np.nan

        betas = betas[:n]
        I_sel = I_sel[:n]
        m = np.isfinite(betas) & np.isfinite(I_sel)
        betas = betas[m]
        I_sel = I_sel[m]

        est_I = np.nan
        if np.isfinite(est_beta) and betas.size:
            j = int(np.nanargmin(np.abs(betas - est_beta)))
            est_I = float(I_sel[j])

        return betas, I_sel, est_beta, est_I

    def _set_local_limits(betas: np.ndarray, I_sel: np.ndarray):
        
        ax.set_xlim(0,2.5)

        if ylim is None:
            if np.isfinite(I_sel).any():
                ymin_local = 0.0
                ymax_local = float(np.nanmax(I_sel)) + 20.0
                if ymin_local < ymax_local:
                    ax.set_ylim(ymin_local, ymax_local)
        else:
            ax.set_ylim(ylim)

    def update(frame_i: int):
        t = int(times[frame_i])
        betas, I_sel, est_beta, est_I = _load_frame(t)

        scat_sel.set_offsets(np.c_[betas, I_sel] if betas.size else np.empty((0, 2)))
        _set_local_limits(betas, I_sel)

        if np.isfinite(est_beta) and np.isfinite(est_I) and show_est_on_curve:
            est_dot.set_offsets([[est_beta, est_I]])
        else:
            est_dot.set_offsets(np.empty((0, 2)))

        if show_observed_dot and (t in obs):
            ob, oI = obs[t]
            if np.isfinite(ob) and np.isfinite(oI):
                obs_dot.set_offsets([[ob, oI]])
            else:
                obs_dot.set_offsets(np.empty((0, 2)))
        else:
            obs_dot.set_offsets(np.empty((0, 2)))

        ax.set_title(fr"Selected Particles at Time $t = {t}$ (method 1, window 5, synthetic data)",fontsize=12)
        return scat_sel, est_dot, obs_dot

    anim = FuncAnimation(fig, update, frames=len(times), interval=int(1000 / fps), blit=False)

    ext = os.path.splitext(out_path)[1].lower()
    ffmpeg_ok = writers.is_available("ffmpeg")

    if ext == ".mp4":
        if ffmpeg_ok:
            anim.save(out_path, writer=FFMpegWriter(fps=fps))
        else:
            out_path = os.path.splitext(out_path)[0] + ".gif"
            anim.save(out_path, writer=PillowWriter(fps=fps))
            print("error", out_path)
    elif ext == ".gif":
        anim.save(out_path, writer=PillowWriter(fps=fps))
    else:
        raise ValueError(f"error")

    plt.close(fig)
    return out_path