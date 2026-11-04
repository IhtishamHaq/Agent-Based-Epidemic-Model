from __future__ import annotations

import os
import re
import argparse
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
os.environ.pop("MPLBACKEND", None)
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def _pf_results_path(bank_path: str) -> str:
    root, _ = os.path.splitext(bank_path)
    return root + "_pf_results.npz"


def _load_path_for_plots(path: str) -> str:
    pf = _pf_results_path(path)
    return pf if os.path.exists(pf) else path


def _bank_key(t: int, name: str) -> str:
    return f"t{int(t)}_{name}"


def _find_times_with_selected_I(bank_path: str) -> List[int]:
    bank_path = _load_path_for_plots(bank_path)
    with np.load(bank_path, allow_pickle=True) as d:
        pat = re.compile(r"^t(\d+)_C_half_I_sim$")
        out: List[int] = []
        for k in d.files:
            m = pat.match(k)
            if m:
                out.append(int(m.group(1)))
        return sorted(out)


def load_selected_infected(bank_path: str, t: int) -> np.ndarray:
    bank_path = _load_path_for_plots(bank_path)
    kI = _bank_key(t, "C_half_I_sim")
    with np.load(bank_path, allow_pickle=True) as d:
        if kI not in d.files:
            return np.array([], dtype=float)
        return np.asarray(d[kI], dtype=float).ravel()


def load_selected_weights(bank_path: str, t: int) -> np.ndarray:
    bank_path = _load_path_for_plots(bank_path)
    kW = _bank_key(t, "C_half_weights")
    with np.load(bank_path, allow_pickle=True) as d:
        if kW not in d.files:
            return np.array([], dtype=float)
        return np.asarray(d[kW], dtype=float).ravel()


def load_selected_infected_over_time(
    bank_path: str,
    t_max: Optional[int] = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    times = _find_times_with_selected_I(bank_path)
    if t_max is not None:
        times = [t for t in times if int(t) <= int(t_max)]
    series = [load_selected_infected(bank_path, t) for t in times]
    return np.asarray(times, dtype=int), series


def load_observed(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)

    if "Time" not in df.columns or "Infected" not in df.columns:
        raise ValueError(
            f"{csv_path} must contain columns 'Time' and 'Infected'. "
            f"Found columns: {list(df.columns)}"
        )

    t_obs = df["Time"].to_numpy(dtype=float)
    y_obs = df["Infected"].to_numpy(dtype=float)
    return t_obs, y_obs


def estimated_infected_weighted_series(
    bank_path: str,
    t_max: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    ts, arrs = load_selected_infected_over_time(bank_path, t_max=t_max)
    if ts.size == 0:
        return ts, np.array([], dtype=float)

    weighted_I: List[float] = []
    for t, a in zip(ts.tolist(), arrs):
        a = np.asarray(a, dtype=float).ravel()
        if a.size == 0:
            weighted_I.append(np.nan)
            continue

        w = load_selected_weights(bank_path, int(t))
        w = np.asarray(w, dtype=float).ravel()

        if w.size != a.size:
            raise RuntimeError(
                f"error"
            )

        sw = float(np.sum(w))
        if sw <= 0.0 or not np.isfinite(sw):
            w = np.full_like(w, 1.0 / max(len(w), 1))
        else:
            w = w / sw

        weighted_I.append(float(np.sum(w * a)))

    return ts.astype(int), np.asarray(weighted_I, dtype=float)


def _npz_to_dict(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def _save_npz(path: str, payload: dict) -> None:
    np.savez(path, **payload)



def compute_window_rmse_from_bank(
    bank_path: str,
    observed_csv: str,
    t_min: Optional[int] = None,
    t_max: Optional[int] = None,
) -> Dict[str, Any]:
    ts_pf, I_hat_pf = estimated_infected_weighted_series(bank_path, t_max=t_max)
    if ts_pf.size == 0:
        raise RuntimeError(f"error")

    t_obs, y_obs = load_observed(observed_csv)

    tw = ts_pf.copy()
    if t_min is not None:
        tw = tw[tw >= int(t_min)]
    if t_max is not None:
        tw = tw[tw <= int(t_max)]
    if tw.size == 0:
        raise RuntimeError(f"No PF times after trimming for bank={bank_path}")

    hat_map = {int(t): float(v) for t, v in zip(ts_pf.tolist(), I_hat_pf.tolist())}
    I_hat = np.asarray([hat_map.get(int(t), np.nan) for t in tw], dtype=float)

    
    obs_map = {float(t): float(y) for t, y in zip(np.asarray(t_obs, float), np.asarray(y_obs, float))}
    I_obs = np.asarray([obs_map.get(float(t), np.nan) for t in tw.astype(float)], dtype=float)

    if np.isnan(I_obs).any():
        tobs = np.asarray(t_obs, float)
        yobs = np.asarray(y_obs, float)
        fixed = I_obs.copy()
        for i, t in enumerate(tw.astype(float)):
            if not np.isfinite(fixed[i]):
                j = int(np.argmin(np.abs(tobs - t)))
                if abs(float(tobs[j]) - float(t)) <= 1e-6:
                    fixed[i] = float(yobs[j])
        I_obs = fixed


    diff = I_hat - I_obs
    den = np.abs(I_obs)
    den[den == 0] = np.nan  
    abs_err = np.abs(diff) / den


    rmse = float(np.sqrt(np.nanmean(diff ** 2))) if np.isfinite(diff).any() else float("nan")
    return dict(times=tw.astype(int), I_hat=I_hat, I_obs=I_obs, abs_err=abs_err, RMSE=rmse)


def save_window_metrics(metrics_path: str, window_name: str, res: Dict[str, Any]) -> None:
    payload = _npz_to_dict(metrics_path)

    payload[f"{window_name}_times"] = np.asarray(res["times"], dtype=int)
    payload[f"{window_name}_I_hat"] = np.asarray(res["I_hat"], dtype=float)
    payload[f"{window_name}_I_obs"] = np.asarray(res["I_obs"], dtype=float)
    payload[f"{window_name}_abs_err"] = np.asarray(res["abs_err"], dtype=float)
    payload[f"{window_name}_RMSE"] = np.asarray([float(res["RMSE"])], dtype=float)

    _save_npz(metrics_path, payload)



def sweep_files(
    banks: Dict[int, str],
    observed_csv: str,
    metrics_path: str,
    t_min: Optional[int] = None,
    t_max: Optional[int] = None,
) -> None:
    for w in sorted(banks.keys()):
        bank_path = banks[w]
        name = f"window_{w}"
        res = compute_window_rmse_from_bank(
            bank_path=bank_path,
            observed_csv=observed_csv,
            t_min=t_min,
            t_max=t_max,
        )
        save_window_metrics(metrics_path, name, res)
        print(f"error")
    print("Metrics file:", metrics_path)



def plot_saved_windows_boxplot(
    metrics_path: str,
    prefix: str = "Window size",
   # title: str = "Absolute error distribution by window",
    save_fig: bool = False,
    fig_name: Optional[str] = None,
    show: bool = True,
) -> None:
    with np.load(metrics_path, allow_pickle=True) as d:
        pat = re.compile(rf"^{re.escape(prefix)}_(\d+)_abs_err$")
        ws: List[int] = []
        arrays: List[np.ndarray] = []

        for k in d.files:
            m = pat.match(k)
            if not m:
                continue
            w = int(m.group(1))
            err = np.asarray(d[k], dtype=float).ravel()
            err = err[np.isfinite(err)]
            if err.size:
                ws.append(w)
                arrays.append(err)

    if not ws:
        raise RuntimeError("error")

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

    
    plt.setp(bp["boxes"], facecolor="lightblue")
    plt.setp(bp["medians"], color="black")

    ax.set_xticks(positions)
    ax.set_xticklabels([f"Filter window {w}" for w in ws])

    #ax.set_xlabel("Filter window")
    ax.set_ylabel(r"Absolute error")
    #ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_fig:
        out = fig_name or (os.path.splitext(metrics_path)[0] + "_abs_error_boxplot.png")
        out_dir = os.path.dirname(out) or "."
        os.makedirs(out_dir, exist_ok=True)
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


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_sw = sub.add_parser("sweep", help="error.")
    ap_sw.add_argument(
        "--banks",
        required=True,
        help='Example: "5=abm_bank_all_5.npz,10=abm_bank_all_10.npz,15=abm_bank_all_15.npz,20=abm_bank_all_20.npz"',
    )
    ap_sw.add_argument("--observed-csv", required=True)
    ap_sw.add_argument("--metrics-path", required=True)
    ap_sw.add_argument("--t-min", type=int, default=None)
    ap_sw.add_argument("--t-max", type=int, default=None)

    ap_plot = sub.add_parser("box", help="Boxplot from saved metrics file.")
    ap_plot.add_argument("--metrics-path", required=True)
    ap_plot.add_argument("--prefix", default="window")
   # ap_plot.add_argument("--title", default="Absolute error distribution by window")
    ap_plot.add_argument("--save-fig", action="store_true")
    ap_plot.add_argument("--fig-name", default=None)
    ap_plot.add_argument("--no-show", action="store_true", help="Do not open a window (useful on servers).")

    args = ap.parse_args()

    if args.cmd == "sweep":
        banks = _parse_banks(args.banks)
        sweep_files(
            banks=banks,
            observed_csv=args.observed_csv,
            metrics_path=args.metrics_path,
            t_min=args.t_min,
            t_max=args.t_max,
        )

    elif args.cmd == "box":
        plot_saved_windows_boxplot(
            metrics_path=args.metrics_path,
            prefix=args.prefix,
           # title=args.title,
            save_fig=bool(args.save_fig),
            fig_name=args.fig_name,
            show=(not bool(args.no_show)),
        )


if __name__ == "__main__":
    main()
