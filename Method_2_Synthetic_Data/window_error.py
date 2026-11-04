from __future__ import annotations

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
os.environ.pop("MPLBACKEND", None)
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict


def _bank_key(t: int, name: str) -> str:
    return f"t{int(t)}_{name}"


def _pf_results_path(bank_path: str) -> str:
    root, _ = os.path.splitext(bank_path)
    return root + "_pf_results.npz"


def _load_path_for_plots(path: str) -> str:
    pf = _pf_results_path(path)
    return pf if os.path.exists(pf) else path


def _flatten_obj_array_blocks(x):
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


def _infer_infected_code_from_meta(bank_path: str) -> int:
    try:
        with np.load(bank_path, allow_pickle=True) as d:
            meta_keys = [k for k in d.files if k.endswith("_meta")]
            if not meta_keys:
                return 1
            raw = d[meta_keys[-1]]
            if isinstance(raw, np.ndarray) and raw.dtype == np.uint8:
                meta = json.loads(raw.tobytes().decode("utf-8"))
                return int(meta.get("status_to_code", {}).get("I", 1))
    except Exception:
        pass
    return 1


def _find_times(bank_path: str, t_max: Optional[int] = None) -> List[int]:
    bank_path = _load_path_for_plots(bank_path)
    with np.load(bank_path, allow_pickle=True) as d:
        pat = re.compile(r"^t(\d+)_half_reps_states$")
        times = sorted(int(pat.match(k).group(1)) for k in d.files if pat.match(k))

    if t_max is not None:
        times = [t for t in times if t <= t_max]

    return times


def load_selected_infected(bank_path: str, t: int) -> np.ndarray:

    bank_path = _load_path_for_plots(bank_path)
    key = _bank_key(t, "half_reps_states")

    with np.load(bank_path, allow_pickle=True) as d:
        if key not in d.files:
            return np.array([])

        infected_code = _infer_infected_code_from_meta(bank_path)
        states_obj = d[key]

    states = _flatten_obj_array_blocks(states_obj)

    I_counts = []
    for st in states:
        S = np.asarray(st.get("S"), int).ravel()
        I_counts.append(np.sum(S == infected_code))

    return np.asarray(I_counts, float)


def load_selected_weights(bank_path: str, t: int) -> np.ndarray:

    bank_path = _load_path_for_plots(bank_path)
    key = _bank_key(t, "A_half_dists")

    with np.load(bank_path, allow_pickle=True) as d:
        if key not in d.files:
            return np.array([])
        errs = np.asarray(d[key], float).ravel()

    if errs.size == 0:
        return np.array([])

    # robust adaptive likelihood
    med = np.median(errs)
    mad = np.median(np.abs(errs - med))
    tau = max((1.01826 * mad) * 0.3, 1e-12)

    W = np.exp(-(errs - np.min(errs)) / tau)
    W /= np.sum(W)
    return W


def estimated_infected_weighted_series(
    bank_path: str,
    t_max: Optional[int] = None,
):

    ts = _find_times(bank_path, t_max=t_max)
    weighted_I = []

    for t in ts:
        a = load_selected_infected(bank_path, t)
        if a.size == 0:
            weighted_I.append(np.nan)
            continue

        w = load_selected_weights(bank_path, t)
        if w.size != a.size:
            weighted_I.append(np.nan)
            continue

        weighted_I.append(np.sum((w / np.sum(w)) * a))

    return np.asarray(ts), np.asarray(weighted_I)


def load_observed_agent_csv(csv_path: str):

    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    cols = {c.lower(): c for c in df.columns}

    time_col = next((cols[c] for c in ["time"] if c in cols), None)
    state_col = next((cols[c] for c in ["state", "status"] if c in cols), None)

    if time_col is None or state_col is None:
        raise ValueError("Agent CSV must contain time and state columns")

    df["__t"] = pd.to_numeric(df[time_col], errors="coerce")
    df = df.dropna(subset=["__t"])

    s = df[state_col].astype(str).str.upper()
    infected = (s == "I") | (s == "INFECTED")

    df["__infected"] = infected.astype(int)

    g = df.groupby("__t", as_index=False)["__infected"].sum().sort_values("__t")

    return g["__t"].to_numpy(float), g["__infected"].to_numpy(float)


def compute_window_rmse(
    bank_path: str,
    observed_csv: str,
    t_max: Optional[int] = None,
):

    ts_pf, I_hat = estimated_infected_weighted_series(
        bank_path,
        t_max=t_max,
    )
    t_obs, I_obs = load_observed_agent_csv(observed_csv)

    obs_map = {float(t): float(y) for t, y in zip(t_obs, I_obs)}
    I_obs_aligned = np.array([obs_map.get(float(t), np.nan) for t in ts_pf])

    diff = I_hat - I_obs_aligned
    den = np.abs(I_obs_aligned)
    den[den == 0] = np.nan
    abs_err = np.abs(diff) / den
    rmse = np.sqrt(np.nanmean(diff ** 2))

    return ts_pf, abs_err, rmse


def sweep(
    banks: Dict[int, str],
    observed_csv: str,
    metrics_path: str,
    t_max: Optional[int] = None,
):

    payload = {}

    for w, path in banks.items():

        ts, abs_err, rmse = compute_window_rmse(
            path,
            observed_csv,
            t_max=t_max,
        )

        payload[f"window_{w}_times"] = ts
        payload[f"window_{w}_abs_err"] = abs_err
        payload[f"window_{w}_RMSE"] = np.array([rmse])

        print(f"Window {w} RMSE = {rmse:.4f}")

    np.savez(metrics_path, **payload)
    print("Saved metrics to:", metrics_path)


def plot_box(
    metrics_path: str,
    save_fig: bool = False,
    fig_name: Optional[str] = None,
    show: bool = True,
) -> None:

    with np.load(metrics_path, allow_pickle=True) as d:
        ws: List[int] = []
        arrays: List[np.ndarray] = []

        for k in d.files:
            if not k.endswith("_abs_err"):
                continue

            m = re.search(r"(\d+)", k)
            if not m:
                continue

            w = int(m.group(1))
            err = np.asarray(d[k], dtype=float).ravel()
            err = err[np.isfinite(err)]
            if err.size:
                ws.append(w)
                arrays.append(err)

    if not ws:
        raise RuntimeError("error.")

    order = np.argsort(ws)
    ws = [ws[i] for i in order]
    arrays = [arrays[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(ws)) * 1.4

    bp = ax.boxplot(
        arrays,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
    )

    plt.setp(bp["boxes"], facecolor="lightgreen")
    plt.setp(bp["medians"], color="black")

    ax.set_xticks(positions)
    ax.set_xticklabels([f"Filter window {w}" for w in ws])

    # ax.set_xlabel("Filter window")
    ax.set_ylabel("Relative absolute error")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_fig:
        out = fig_name or "window_boxplot.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print("Saved plot to:", out)

    if show:
        plt.show()
    else:
        plt.close(fig)


def _parse_banks(s: str) -> Dict[int, str]:

    out: Dict[int, str] = {}
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            raise ValueError("banks must look like: 5=bank5.npz,10=bank10.npz,...")
        k, v = p.split("=", 1)
        out[int(k.strip())] = v.strip()
    return out


def parse_banks(s: str):
    out = {}
    for p in s.split(","):
        k, v = p.split("=")
        out[int(k.strip())] = v.strip()
    return out


def main():

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("sweep")
    s1.add_argument("--banks", required=True)
    s1.add_argument("--observed-csv", required=True)
    s1.add_argument("--metrics-path", required=True)
    s1.add_argument("--t-max", type=int, default=None)

    s2 = sub.add_parser("box")
    s2.add_argument("--metrics-path", required=True)
    s2.add_argument("--save-fig", action="store_true")

    args = ap.parse_args()

    if args.cmd == "sweep":
        banks = parse_banks(args.banks)
        sweep(
            banks,
            args.observed_csv,
            args.metrics_path,
            t_max=args.t_max,
        )

    elif args.cmd == "box":
        plot_box(args.metrics_path, args.save_fig)


if __name__ == "__main__":
    main()