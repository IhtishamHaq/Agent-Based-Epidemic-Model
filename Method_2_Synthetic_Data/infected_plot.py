from __future__ import annotations

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple


def _pf_results_path(bank_path: str) -> str:
    root, _ = os.path.splitext(bank_path)
    return root + "_pf_results.npz"


def _load_path_for_plots(path: str) -> str:
    pf = _pf_results_path(path)
    return pf if os.path.exists(pf) else path


def _bank_key(t: int, name: str) -> str:
    return f"t{int(t)}_{name}"



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



#def _find_times_with_selected_I(bank_path: str) -> List[int]:
#    bank_path = _load_path_for_plots(bank_path)
#    with np.load(bank_path, allow_pickle=True) as d:
#        pat = re.compile(r"^t(\d+)_half_reps_states$")
#        times = []
#        for k in d.files:
#            m = pat.match(k)
#            if m:
#                times.append(int(m.group(1)))
#        return sorted(times)

def _find_times_with_selected_I(bank_path: str) -> List[int]:
    bank_path = _load_path_for_plots(bank_path)

    with np.load(bank_path, allow_pickle=True) as d:
        times = set()

        pat_states = re.compile(r"^t(\d+)_half_reps_states$")
        pat_I = re.compile(r"^t(\d+)_A_half_I_sim$")

        for k in d.files:
            m1 = pat_states.match(k)
            if m1:
                times.add(int(m1.group(1)))

            m2 = pat_I.match(k)
            if m2:
                times.add(int(m2.group(1)))

        return sorted(times)


#def load_selected_infected(bank_path: str, t: int) -> np.ndarray:
#    bank_path = _load_path_for_plots(bank_path)
#    k = _bank_key(t, "half_reps_states")
#
#    with np.load(bank_path, allow_pickle=True) as d:
#        if k not in d.files:
#            return np.array([], dtype=float)
#
#        infected_code = _infer_infected_code_from_meta(bank_path)
#        states_obj = d[k]
#
#    states = _flatten_obj_array_blocks(states_obj)
#
#    I_counts = []
#    for st in states:
#        S = np.asarray(st.get("S"), int).ravel()
#        I_counts.append(int(np.sum(S == infected_code)))
#
#    return np.asarray(I_counts, dtype=float)



def load_selected_infected(bank_path: str, t: int) -> np.ndarray:
    bank_path = _load_path_for_plots(bank_path)

    k_I = _bank_key(t, "A_half_I_sim")
    k_states = _bank_key(t, "half_reps_states")

    with np.load(bank_path, allow_pickle=True) as d:

        # First choice: use saved infected counts directly
        # This works for the first estimate, e.g. t15_A_half_I_sim
        if k_I in d.files:
            return np.asarray(d[k_I], dtype=float).ravel()

        # Second choice: use saved states for later windows
        if k_states not in d.files:
            return np.array([], dtype=float)

        infected_code = _infer_infected_code_from_meta(bank_path)
        states_obj = d[k_states]

    states = _flatten_obj_array_blocks(states_obj)

    I_counts = []
    for st in states:
        S = np.asarray(st.get("S"), int).ravel()
        I_counts.append(int(np.sum(S == infected_code)))

    return np.asarray(I_counts, dtype=float)

def load_selected_weights(bank_path: str, t: int) -> np.ndarray:
    bank_path = _load_path_for_plots(bank_path)

    kb = _bank_key(t, "A_half_betas")
    kd = _bank_key(t, "A_half_dists")

    with np.load(bank_path, allow_pickle=True) as d:
        if kb not in d.files or kd not in d.files:
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
    W /= np.sum(W)

    return W




def load_selected_infected_over_time(
    bank_path: str,
    t_max: Optional[int] = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:

    times = _find_times_with_selected_I(bank_path)
    if t_max is not None:
        times = [t for t in times if int(t) <= int(t_max)]

    series = [load_selected_infected(bank_path, t) for t in times]
    return np.asarray(times, dtype=int), series




def load_observed_A_agent_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:

    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    cols = {c.lower(): c for c in df.columns}

    time_candidates = ["time", "t"]
    status_candidates = ["status", "state"]
    code_candidates = ["s", "status_code", "code"]

    time_col = next((cols[c] for c in time_candidates if c in cols), None)
    if time_col is None:
        raise ValueError("No time column found in agent_data.csv")

    status_col = next((cols[c] for c in status_candidates if c in cols), None)
    code_col = next((cols[c] for c in code_candidates if c in cols), None)

    t = pd.to_numeric(df[time_col], errors="coerce")
    df = df.assign(__t=t).dropna(subset=["__t"])

    if status_col is not None:
        s = df[status_col].astype(str).str.upper()
        infected = (s == "I") | (s == "INFECTED")
    elif code_col is not None:
        code = pd.to_numeric(df[code_col], errors="coerce")
        infected = (code == 1)
    else:
        raise ValueError("No status column found in agent_data.csv")

    df = df.assign(__infected=infected.astype(int))
    g = df.groupby("__t", as_index=False)["__infected"].sum().sort_values("__t")

    return g["__t"].to_numpy(float), g["__infected"].to_numpy(float)



def plot_selected_infected_multi_subplots(
    bank_paths: List[str],
    labels: Optional[List[str]] = None,
    observed_csv_path: Optional[str] = None,
    title: str = "Observed and PPF estimated infected cases by window size",
    save_path: str = "figs_multi/infected_windows_subplots.pdf",
    t_max: Optional[int] = None,
    show: bool = True,
    close: bool = True,
    xlim=None,
    xtick=None,
    legend_ncol: int = 3,
) -> None:
    
    if labels is None:
        labels = [os.path.splitext(os.path.basename(p))[0] for p in bank_paths]
    if len(labels) != len(bank_paths):
        raise ValueError("error")

    t_obs = y_obs = None
    if observed_csv_path:
        t_obs, y_obs = load_observed_A_agent_csv(observed_csv_path)

        t_obs = np.asarray(t_obs, float)
        y_obs = np.asarray(y_obs, float)

        if t_max is not None:
            m = t_obs <= float(t_max)
            t_obs = t_obs[m]
            y_obs = y_obs[m]

    n = len(bank_paths)
    fig, axes = plt.subplots(n, 1, sharex=True, sharey=True, figsize=(10, 8))
    if n == 1:
        axes = [axes]

    for ax, bank_path, lab in zip(axes, bank_paths, labels):
        ts, arrs = load_selected_infected_over_time(bank_path, t_max=t_max)

    
        xs, ys = [], []
        for t, a in zip(ts, arrs):
            xs.extend([t] * len(a))
            ys.extend(a)

        weighted_I = []
        for t, a in zip(ts, arrs):
            a = np.asarray(a, dtype=float).ravel()
            if a.size == 0:
                weighted_I.append(np.nan)
                continue
            w = load_selected_weights(bank_path, int(t))
            w = np.asarray(w, dtype=float).ravel()
            s = np.sum(w)
            if s <= 0 or not np.isfinite(s):
                weighted_I.append(np.nan)
                continue
            w = w / s
            weighted_I.append(np.sum(w * a))

        weighted_I = np.asarray(weighted_I, dtype=float)


        ax.scatter(xs, ys, s=28, color="blue", alpha=0.35, label="Selected particles")

        ax.plot(ts, weighted_I, linewidth=2.5, marker='*', markersize=2, color="red", label="Estimated infected")
        ax.scatter(ts, weighted_I, s=50, color="black", label="_nolegend_")

        if t_obs is not None:
            ax.plot(t_obs, y_obs, marker='*', markersize=2,   markerfacecolor='red',  color="orange", alpha=0.8, linewidth=2.5, label="Observed data")
            ax.scatter(t_obs, y_obs, s=8, color="orange", alpha=0.8, label="_nolegend_")

        ax.text(
            0.5, 0.92, lab,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
        )

        ax.grid(True, alpha=0.3)

 
    fig.suptitle(title, y=0.995, fontsize=13)

    handles, labels_all = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels_all.extend(l)

    seen = set()
    uniq_handles, uniq_labels = [], []
    for h, l in zip(handles, labels_all):
        if not l or l == "_nolegend_":
            continue
        if l not in seen:
            uniq_handles.append(h)
            uniq_labels.append(l)
            seen.add(l)


    if uniq_handles:
        fig.legend(
            uniq_handles,
            uniq_labels,
            loc="lower center",
            ncol=int(legend_ncol),
            frameon=False,
            bbox_to_anchor=(0.5, 0.0),
        )

    fig.supxlabel("Time", x=0.55, y=0.05, fontsize=15, fontweight="bold")
    fig.supylabel("Number of infected individuals", y=0.55, fontsize=13, fontweight="bold")
    fig.suptitle(title, y=0.96, fontsize=15, fontweight="bold")
    if xlim is not None and len(xlim) == 2:
        axes[-1].set_xlim(float(xlim[0]), float(xlim[1]))
    if xtick is not None and np.size(xtick):
        axes[-1].set_xticks([float(x) for x in np.asarray(xtick).ravel()])
    fig.tight_layout()
    fig.subplots_adjust(top=0.93, bottom=0.12)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print("Saved:", save_path)

    if show:
        plt.show()
    if close:
        plt.close(fig)
