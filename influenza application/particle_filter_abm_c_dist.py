#particle_filter_abm_c_dist.py
from __future__ import annotations

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional

from packages.abm.abm import ABM
from Beta_estimators import estimate_beta_distances

BETA_MIN = 0.0
BETA_MAX = 200.0

fitness = 0.95          
 
sigma_min= 0.15  
sigma_mix = 2.0

def _to_pct_errors(abs_errors: np.ndarray, y_obs: float) -> np.ndarray:
    denom = max(float(y_obs), 1.0)
    return np.exp(-(np.asarray(abs_errors, dtype=float) / denom))



def _clip_beta(b: float, lo: float = BETA_MIN, hi: float = BETA_MAX) -> float:
    b = float(b)
    if b < lo:
        return float(lo)
    if b > hi:
        return float(hi)
    return b


def _pf_results_path(bank_path: str) -> str:
    root, ext = os.path.splitext(bank_path)
    return root + "_pf_results.npz"


def _append_npz(path: str, new_items: dict, overwrite: bool = True) -> None:
    data = {}
    if os.path.exists(path):
        with np.load(path, allow_pickle=True) as d:
            for k in d.files:
                data[k] = d[k]
    data.update(new_items)
    if overwrite and os.path.exists(path):
        os.remove(path)
    np.savez_compressed(path, **data)


def _load_path_for_plots(path: str) -> str:
    pf = _pf_results_path(path)
    return pf if os.path.exists(pf) else path





def uniform_eps(clip=(-0.15, 0.15)) -> float:
    a, b = clip
    return float(np.random.uniform(a, b))


def _no_observed_at_time(observed_csv: str, t: int) -> None:
    try:
        df = pd.read_csv(observed_csv, dtype=str, low_memory=False)
    except Exception as e:
        print(f"Can't read {observed_csv}: {e}")
        return

    cols = {c.lower(): c for c in df.columns}
    for key in ("time", "t", "day", "step", "frame"):
        if key in cols:
            s = pd.to_numeric(df[cols[key]].str.strip(), errors="coerce")
            if not np.isclose(s, float(t), atol=1e-9).any():
                print(f"error.")
            return
    print("No time column.")


def _simulations_num_steps(path: str, expected: int) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"error")
    with np.load(path, allow_pickle=True) as d:
        files = set(d.files)
        if "meta" in files:
            try:
                meta = json.loads(d["meta"].tobytes().decode("utf-8"))
                got = int(meta.get("num_steps", -1))
                if got != int(expected):
                    raise ValueError(f"error.")
                return
            except Exception as e:
                print(
                    f"error "
                    f"error"
                )
        if "I" not in files:
            print(f"error")
            return
        I = d["I"]
        if I.ndim not in (2, 3):
            print(f"error")
            return
        T = int(I.shape[-1])
        if expected not in (T, T - 1):
            raise ValueError(
                f"error."
            )


def detect_runs(bank_path: str) -> int:
    if not os.path.exists(bank_path):
        raise FileNotFoundError(f"error")
    with np.load(bank_path, allow_pickle=True) as d:
        if "S" in d.files:
            S_all = d["S"]
        elif "final_status" in d.files:
            S_all = d["final_status"]
        else:
            raise KeyError("missing status.")
    return S_all.shape[0] if S_all.ndim == 3 else 1


def _bank_key(t: int, name: str) -> str:
    return f"t{int(t)}_{name}"


def _infected_count_from_state(S: np.ndarray, infected_code: int = 1) -> int:
    return int(np.sum(S.astype(int) == int(infected_code)))


def _find_beta_index(betas: np.ndarray, beta: float, atol: float = 1e-9) -> int:
    idxs = np.where(np.isclose(betas.astype(float), float(beta), atol=atol))[0]
    if idxs.size == 0:
        raise ValueError(f"β={beta} not found in bank betas.")
    return int(idxs[0])


def _read_initial_state(bank_path: str, run_idx: int, beta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(bank_path, allow_pickle=True) as d:
        betas = d["betas"].astype(float)
        b_idx = _find_beta_index(betas, beta)
        fx = d["final_x"] if "final_x" in d.files else d["X"]
        fy = d["final_y"] if "final_y" in d.files else d["Y"]
        fs = d["final_status"] if "final_status" in d.files else d["S"]
        if fx.ndim == 3:
            X = fx[int(run_idx), b_idx].astype(int)
            Y = fy[int(run_idx), b_idx].astype(int)
            S = fs[int(run_idx), b_idx].astype(int)
        else:
            X = fx[b_idx].astype(int)
            Y = fy[b_idx].astype(int)
            S = fs[b_idx].astype(int)
    return X, Y, S


def _continue_from_arrays(
    *,
    beta: float,
    gamma: int,
    X: np.ndarray,
    Y: np.ndarray,
    S: np.ndarray,
    A: Optional[np.ndarray],
    steps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ages = None if A is None else np.asarray(A, dtype=int).ravel()
    model = ABM.from_state(
        beta=float(beta),
        gamma=int(gamma),
        X=X,
        Y=Y,
        status_codes=S,
        infected_ages=ages,
    )
    I_series: List[int] = []
    for _ in range(int(steps)):
        model.step()
        I_series.append(sum(1 for ag in model.agents.values() if ag.status == "I"))
    Xn, Yn, Sn, An = model.to_arrays()
    return Xn, Yn, Sn, An, np.asarray(I_series, dtype=int)


def get_observed_infected_count(observed_csv: str, t_obs: int) -> int:
    df = pd.read_csv(observed_csv, dtype=str, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    cols = {c.lower(): c for c in df.columns}

    time_col = next((cols[c] for c in ("time", "t") if c in cols), None)
    if time_col is None:
        raise ValueError("No time column found in observed CSV")

    tt = pd.to_numeric(df[time_col].astype(str).str.strip(), errors="coerce")
    at_t = df[tt == float(t_obs)]
    if at_t.empty:
        return 0

    infected_count_col = next(
        (
            cols[c]
            for c in (
                "okinawa_cases",
            )
            if c in cols
        ),
        None,
    )
    if infected_count_col is not None:
        vals = pd.to_numeric(at_t[infected_count_col].astype(str).str.strip(), errors="coerce").dropna()
        if vals.empty:
            return 0
        return int(np.round(vals.sum()))

    status_col = next((cols[c] for c in ("status", "state") if c in cols), None)
    infected_col = next((cols[c] for c in ("is_infected",) if c in cols), None)

    if status_col is not None:
        s = at_t[status_col].astype(str).str.strip().str.upper()
        return int(((s == "I") | (s == "1")).sum())

    if infected_col is not None:
        v = pd.to_numeric(at_t[infected_col].astype(str).str.strip(), errors="coerce")
        return int((v.fillna(0) > 0).sum())

    return int(len(at_t))



def get_run_distances(
    *,
    observed_csv: str,
    bank_path: str,
    run_idx: int,
    num_infected: int = 1,
    num_steps: int = 20,
    beta_start: float = 0.40,
    beta_step: float = 0.01,
    beta_count: int = 20,
    center=(0, 0),
    start_ring: int = 1,
    strict: bool = True,
):
    observed_I_t = int(get_observed_infected_count(observed_csv, t_obs=num_steps))

    with np.load(bank_path, allow_pickle=True) as d:
        betas_bank = np.asarray(d["betas"], dtype=float)
        I_mat = d["I"]

    I_run = I_mat[int(run_idx)] if I_mat.ndim == 3 else I_mat
    T = I_run.shape[1]
    idx_t = min(max(int(num_steps), 0), T - 1)

    if strict:
        I_sim_at_t = I_run[:, idx_t].astype(float)
        dC = np.abs(observed_I_t - I_sim_at_t)
        betas_C = betas_bank.astype(float)
        return betas_C, dC.astype(float)

    betas_grid = np.round(beta_start + np.arange(beta_count, dtype=float) * beta_step, 6)
    betas_bank_r = np.round(betas_bank, 6)

    kept_betas = []
    dC_vals = []
    for b in betas_grid:
        matches = np.where(np.isclose(betas_bank_r, b, atol=0.0, rtol=0.0))[0]
        if matches.size:
            bi = int(matches[0])
            I_b_t = float(I_run[bi, idx_t])
            kept_betas.append(float(b))
            dC_vals.append(abs(observed_I_t - I_b_t))

    if not kept_betas:
        return np.array([], float), np.array([], float)

    betas_C = np.asarray(kept_betas, dtype=float)
    dC = np.asarray(dC_vals, dtype=float)
    return betas_C, dC


def _weighted_beta_from_half(betas, errs, scale=2.0, temp=0.3):
    betas = np.asarray(betas, float)
    errs  = np.asarray(errs, float)

    m = np.isfinite(betas) & np.isfinite(errs)
    x = betas[m]
    e = errs[m]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan"), np.array([], float)

  
    med = np.median(e)
    mad = np.median(np.abs(e - med))
    emin = 1.01826 * mad if mad > 1e-12 else max(med, 1.0)
    emin = max(emin, 1e-9) * scale
    tau  = max(emin * temp, 1e-12)

    
    W = np.exp(-(e - np.min(e)) / tau)
    s = np.sum(W)
    if s <= 0 or not np.isfinite(s):
        w = np.full_like(W, 1.0 / len(W))
    else:
        w = W / s

    
    beta_hat = float(np.sum(w * x))
    
    var_w = float(np.sum( (x - beta_hat) ** 2))
    se = np.sqrt(var_w / max(len(x), 1))
    z = 1.96
    lo, hi = beta_hat - z * se, beta_hat + z * se

    return beta_hat, lo, hi, w




def _select_half_by_median(arr_beta: np.ndarray, arr_err: np.ndarray) -> np.ndarray:
    arr_beta = np.asarray(arr_beta, float)
    arr_err = np.asarray(arr_err, float)

    m = np.isfinite(arr_beta) & np.isfinite(arr_err)
    if not m.any():
        return np.array([], dtype=int)

    beta = arr_beta[m]
    err = arr_err[m]
    idx_map = np.nonzero(m)[0]

    N = err.size
    k = int(round(N * 0.5))
    if k == 0:
        return np.array([], dtype=int)

    order = np.argsort(err, kind="stable")
    err_sorted = err[order]
    med = np.median(err_sorted)

    lt_mask = err_sorted < med
    eq_mask = np.isclose(err_sorted, med, rtol=0.0, atol=1e-12)
    gt_mask = err_sorted > med

    keep_rel = []

    idx_lt = order[lt_mask]
    if idx_lt.size >= k:
        keep_rel = idx_lt[:k].tolist()
    else:
        keep_rel = idx_lt.tolist()
        remaining = k - len(keep_rel)

        mu = float(np.mean(beta[keep_rel])) if keep_rel else float(np.nanmean(beta))
        idx_eq = order[eq_mask]
        if idx_eq.size:
            eq_order = np.argsort(np.abs(beta[idx_eq] - mu), kind="stable")
            chosen_eq = idx_eq[eq_order[:remaining]]
            keep_rel.extend(chosen_eq.tolist())
            remaining = k - len(keep_rel)

        if remaining > 0:
            idx_gt = order[gt_mask]
            keep_rel.extend(idx_gt[:remaining].tolist())

    keep_idx = idx_map[np.asarray(keep_rel, dtype=int)]
    return keep_idx


def select_lower_half(
    *,
    observed_csv: str,
    bank_path_tobs: str,
    num_infected: int = 1,
    t_obs: int = 20,
    beta_start: float = 0.4,
    beta_step: float = 0.01,
    beta_count: int = 20,
    center=(0, 0),
    start_ring: int = 1,
    strict: bool = True,
):
    _simulations_num_steps(bank_path_tobs, t_obs)
    _no_observed_at_time(observed_csv, t_obs)

    n_runs = detect_runs(bank_path_tobs)

    pairs_C: List[tuple[int, float]] = []
    errs_C: List[float] = []

    for run_idx in range(n_runs):
        betas_C, dC = get_run_distances(
            observed_csv=observed_csv,
            bank_path=bank_path_tobs,
            run_idx=run_idx,
            num_infected=num_infected,
            num_steps=t_obs,
            beta_start=beta_start,
            beta_step=beta_step,
            beta_count=beta_count,
            center=center,
            start_ring=start_ring,
            strict=strict,
        )
        if betas_C.size:
            m = np.isfinite(dC) & np.isfinite(betas_C)
            for b, e in zip(betas_C[m], dC[m]):
                pairs_C.append((int(run_idx), float(b)))
                errs_C.append(float(e))

    arr_err_c = np.asarray(errs_C, dtype=float)
    arr_beta_c = np.asarray([p[1] for p in pairs_C], dtype=float)

    keep_idx_c = _select_half_by_median(arr_beta_c, arr_err_c)

    selection_c = [(int(pairs_C[i][0]), float(pairs_C[i][1]), float(arr_err_c[i])) for i in keep_idx_c]
    betas_C_keep_c = arr_beta_c[keep_idx_c]
    dC_keep_c = arr_err_c[keep_idx_c]

    mean_beta_C, C_ci_lo, C_ci_hi, _ = _weighted_beta_from_half(betas_C_keep_c, dC_keep_c, scale=2.0, temp=0.3)
    return selection_c, betas_C_keep_c, dC_keep_c, mean_beta_C



def second_select_lower_half(
    *,
    observed_csv: str,
    bank_path_tobs: str,
    num_infected: int = 1,
    t_obs: int = 20,
    beta_start: float = 0.4,
    beta_step: float = 0.01,
    beta_count: int = 20,
    center=(0, 0),
    start_ring: int = 1,
    strict: bool = True,
    selection_c: List[Tuple[int, float, float]],
):
    second_arr_beta_c = np.asarray([b for (_, b, _) in selection_c], dtype=float)
    second_arr_err_c = np.asarray([e for (_, _, e) in selection_c], dtype=float)

    arr_beta = np.asarray(second_arr_beta_c, dtype=float)
    arr_err = np.asarray(second_arr_err_c, dtype=float)

    m = np.isfinite(arr_beta) & np.isfinite(arr_err)
    if not m.any():
        return [], np.array([], float), np.array([], float), float("nan"), np.array([], int)

    beta = arr_beta[m]
    err = arr_err[m]
    idx_map = np.nonzero(m)[0]

    N = err.size
    k = N // 2
    if k == 0:
        return [], np.array([], float), np.array([], float), float("nan"), np.array([], int)

    order = np.argsort(err, kind="stable")
    err_sorted = err[order]
    med = np.median(err_sorted)

    lt_mask = err_sorted < med
    eq_mask = np.isclose(err_sorted, med, rtol=0.0, atol=1e-12)
    gt_mask = err_sorted > med

    keep_rel = []

    idx_lt = order[lt_mask]
    if idx_lt.size >= k:
        keep_rel = idx_lt[:k].tolist()
    else:
        keep_rel = idx_lt.tolist()
        remaining = k - len(keep_rel)

        mu = float(np.mean(beta[keep_rel])) if keep_rel else float(np.nanmean(beta))
        idx_eq = order[eq_mask]
        if idx_eq.size:
            eq_order = np.argsort(np.abs(beta[idx_eq] - mu), kind="stable")
            chosen_eq = idx_eq[eq_order[:remaining]]
            keep_rel.extend(chosen_eq.tolist())
            remaining = k - len(keep_rel)

        if remaining > 0:
            idx_gt = order[gt_mask]
            keep_rel.extend(idx_gt[:remaining].tolist())

    second_keep_idx_c = idx_map[np.asarray(keep_rel, dtype=int)]

    second_selection_c = [(int(selection_c[i][0]), float(selection_c[i][1])) for i in second_keep_idx_c]
    second_betas_C_keep_c = second_arr_beta_c[second_keep_idx_c]
    second_dC_keep_c = second_arr_err_c[second_keep_idx_c]
    second_mean_beta_C = float(np.mean(second_betas_C_keep_c)) if second_betas_C_keep_c.size else np.nan




    return (
        second_selection_c,
        second_betas_C_keep_c,
        second_dC_keep_c,
        second_mean_beta_C,
        second_keep_idx_c,
    )


#new window with weight reset please distance

def window_step(
    *,
    observed_csv: str,
    bank_path: str,
    t_prev: int,
    t_next: int,
    num_infected: int = 1,
    beta_start: float = 0.4,
    beta_step: float = 0.01,
    beta_count: int = 20,
    center=(0, 0),
    start_ring: int = 1,
    gamma: int = 14,
    run_replicates: int = 2,
    overwrite: bool = True,
    append_back_to_bank: bool = True,
    out_path: Optional[str] = None,
    pool_pairs_in: Optional[List[Tuple[int, float]]] = None,
    pool_states_in: Optional[List[Dict[str, np.ndarray]]] = None,
) -> dict:

    def _err_from_state(st: Dict[str, np.ndarray], I_obs: int) -> Tuple[float, int]:
        code_I = getattr(ABM, "STATUS_TO_CODE", {"I": 1}).get("I", 1)
        I_sim = _infected_count_from_state(st["S"], infected_code=code_I)
        return float(abs(I_obs - I_sim)), int(I_sim)

    def _alt_signs(n: int) -> np.ndarray:
        s = np.ones(int(n), dtype=int)
        s[1::2] = -1
        return s

    if t_next < t_prev:
        raise ValueError("t_next must be => t_prev")
    _no_observed_at_time(observed_csv, t_prev)
    _no_observed_at_time(observed_csv, t_next)

    steps_to_go = int(t_next) - int(t_prev)
    I_obs_prev = get_observed_infected_count(observed_csv, t_obs=t_prev)
    I_obs_next = get_observed_infected_count(observed_csv, t_obs=t_next)

    pairs_prev: List[Tuple[int, float]] = []
    states_prev: List[Dict[str, np.ndarray]] = []

    have_pool = (pool_pairs_in is not None) and (pool_states_in is not None)

    carry_idx: set[int] = set()
    selection_half_errors: np.ndarray = np.array([], dtype=float)
    I_sim_half_prev_saved = np.array([], dtype=int)

    if not have_pool:
        selection_c, Cprev_b, Cprev_d, betaCprev = select_lower_half(
            observed_csv=observed_csv,
            bank_path_tobs=bank_path,
            num_infected=num_infected,
            t_obs=t_prev,
            beta_start=beta_start,
            beta_step=beta_step,
            beta_count=beta_count,
            center=center,
            start_ring=start_ring,
            strict=True,
        )
        if not selection_c:
            raise RuntimeError(f"No (run,β) were selected at t={t_prev}.")

        selection_half_errors = np.asarray(Cprev_d, dtype=float)

        I_sim_sel = []
        with np.load(bank_path, allow_pickle=True) as d:
            betas_bank = np.asarray(d["betas"], dtype=float)
            I_mat = d["I"]
            for (r, b, _) in selection_c:
                run_idx = int(r)
                beta_val = float(b)
                b_idx = _find_beta_index(betas_bank, beta_val)
                I_run = I_mat[run_idx] if I_mat.ndim == 3 else I_mat
                T = int(I_run.shape[1])
                idx_t = min(max(int(t_prev), 0), T - 1)
                I_sim_sel.append(int(I_run[b_idx, idx_t]))
        I_sim_half_prev_saved = np.asarray(I_sim_sel, dtype=int)

        pf_path = _pf_results_path(bank_path)
        _, lo0, hi0, w_prev = _weighted_beta_from_half(
            np.asarray(Cprev_b, float),
            np.asarray(Cprev_d, float),
            scale=2.0,
        )
        _append_npz(
            pf_path,
            {
                _bank_key(t_prev, "C_half_betas"): np.asarray(Cprev_b, dtype=float),
                _bank_key(t_prev, "C_half_dists"): np.asarray(Cprev_d, dtype=float),
                _bank_key(t_prev, "C_half_I_sim"): np.asarray(I_sim_half_prev_saved, dtype=int),
                _bank_key(t_prev, "C_half_mean_beta"): np.asarray([betaCprev], dtype=float),
                _bank_key(t_prev, "C_half_mean_beta_ci"): np.asarray([lo0, hi0], dtype=float),
                _bank_key(t_prev, "C_half_weights"): np.asarray(w_prev, dtype=float),
                _bank_key(t_prev, "first_half_triplets"): np.asarray(selection_c, dtype=float),
            },
            overwrite=True,
        )

        _, _, _, _, second_keep_idx_c = second_select_lower_half(
            observed_csv=observed_csv,
            bank_path_tobs=bank_path,
            num_infected=num_infected,
            t_obs=t_prev,
            beta_start=beta_start,
            beta_step=beta_step,
            beta_count=beta_count,
            center=center,
            start_ring=start_ring,
            strict=True,
            selection_c=selection_c,
        )

        pairs_prev = [(int(r), float(b)) for (r, b, _) in selection_c]
        for (run_idx, b) in pairs_prev:
            X0, Y0, S0 = _read_initial_state(bank_path, run_idx=int(run_idx), beta=float(b))
            states_prev.append({"X": X0, "Y": Y0, "S": S0, "A": np.zeros_like(S0, dtype=int)})

        carry_idx = set(map(int, np.asarray(second_keep_idx_c).ravel()))

    else:
        pairs_prev = [(int(r), float(b)) for (r, b) in pool_pairs_in]
        states_prev = [st for st in pool_states_in]

        betas_prev = np.asarray([b for (_, b) in pairs_prev], float)
        errs_prev_list = []
        I_sim_prev_list = []
        for st in states_prev:
            e, isv = _err_from_state(st, int(I_obs_prev))
            errs_prev_list.append(float(e))
            I_sim_prev_list.append(int(isv))
        errs_prev = np.asarray(errs_prev_list, float)
        I_sim_prev = np.asarray(I_sim_prev_list, dtype=int)

        idx_first = _select_half_by_median(betas_prev, errs_prev)

        b_first = betas_prev[idx_first]
        d_first = errs_prev[idx_first]
        I_sim_half_prev_saved = I_sim_prev[idx_first]
        selection_half_errors = np.asarray(d_first, dtype=float)

        mean_b, lo, hi, w_prev = _weighted_beta_from_half(b_first, d_first, scale=2.0)

        pf_path = _pf_results_path(bank_path)
        _append_npz(
            pf_path,
            {
                _bank_key(t_prev, "C_half_betas"): np.asarray(b_first, dtype=float),
                _bank_key(t_prev, "C_half_dists"): np.asarray(d_first, dtype=float),
                _bank_key(t_prev, "C_half_I_sim"): np.asarray(I_sim_half_prev_saved, dtype=int),
                _bank_key(t_prev, "C_half_mean_beta"): np.asarray([mean_b], dtype=float),
                _bank_key(t_prev, "C_half_mean_beta_ci"): np.asarray([lo, hi], dtype=float),
                _bank_key(t_prev, "C_half_weights"): np.asarray(w_prev, dtype=float),
                _bank_key(t_prev, "first_half_triplets"): np.asarray(
                    [(pairs_prev[i][0], pairs_prev[i][1], d_first[j]) for j, i in enumerate(idx_first)],
                    dtype=float,
                ),
            },
            overwrite=True,
        )

        pairs_prev = [pairs_prev[i] for i in idx_first]
        states_prev = [states_prev[i] for i in idx_first]

        sel_triplets = [(pairs_prev[i][0], pairs_prev[i][1], float(d_first[i])) for i in range(len(pairs_prev))]

        _, _, _, _, second_keep_idx_c = second_select_lower_half(
            observed_csv=observed_csv,
            bank_path_tobs=bank_path,
            num_infected=num_infected,
            t_obs=t_prev,
            beta_start=beta_start,
            beta_step=beta_step,
            beta_count=beta_count,
            center=center,
            start_ring=start_ring,
            strict=True,
            selection_c=sel_triplets,
        )

        carry_idx = set(map(int, np.asarray(second_keep_idx_c).ravel()))

    mode_any_ge_trigger = False
    mode_all_lt_trigger = False

    if selection_half_errors.size > 0:
        pct_sel = _to_pct_errors(np.asarray(selection_half_errors, float), y_obs=float(I_obs_prev))
        finite = np.isfinite(pct_sel)
        if np.any(finite):
            ok = finite & (pct_sel >= fitness)
            ok_ratio = np.sum(ok) / np.sum(finite)
            mode_any_ge_trigger = ok_ratio >= 0.15
            mode_all_lt_trigger = ok_ratio < 0.15
        else:
            mode_any_ge_trigger = False
            mode_all_lt_trigger = False
    else:
        mode_any_ge_trigger = False
        mode_all_lt_trigger = False

    if mode_all_lt_trigger:
        carry_idx = set()

    R = int(run_replicates)

    children_pairs: List[Tuple[int, float]] = []
    children_states: List[Dict[str, np.ndarray]] = []
    pair_runs: List[int] = []
    pair_betas: List[float] = []
    pair_C_dist: List[float] = []
    pair_I_sim: List[int] = []
    children_weights: List[float] = []

    with np.load(_pf_results_path(bank_path), allow_pickle=True) as d:
        key_w_prev = _bank_key(t_prev, "C_half_weights")
        if key_w_prev in d.files:
            weights_prev = np.asarray(d[key_w_prev], dtype=float)
        else:
            raise KeyError(f"Missing previous weights at t={t_prev}")

    for i, ((r, b), st) in enumerate(zip(pairs_prev, states_prev)):
        w_parent = weights_prev[i]

        if mode_any_ge_trigger:
            if i in carry_idx:
                target_betas = [float(b)] * R
            else:
                signs = _alt_signs(R)
                target_betas = []
                for s in signs:
                    eps = uniform_eps(clip=(-sigma_min, sigma_min))
                    b_child = _clip_beta(float(b) + float(s) * float(eps))
                    target_betas.append(b_child)

        elif mode_all_lt_trigger:
            signs = _alt_signs(R)
            target_betas = []
            for s in signs:
                eps = uniform_eps(clip=(-sigma_mix, sigma_mix))
                b_child = _clip_beta(float(b) + float(s) * float(eps))
                target_betas.append(b_child)

        else:
            if i in carry_idx:
                target_betas = [float(b)] * R
            else:
                signs = _alt_signs(R)
                target_betas = []
                for s in signs:
                    eps = uniform_eps(clip=(-sigma_mix, sigma_mix))
                    b_child = _clip_beta(float(b) + float(s) * float(eps))
                    target_betas.append(b_child)

        for b_child in target_betas:
            XN, YN, SN, AN, Iseg = _continue_from_arrays(
                beta=float(b_child),
                gamma=int(gamma),
                X=st["X"],
                Y=st["Y"],
                S=st["S"],
                A=st.get("A", None),
                steps=steps_to_go,
            )
            C_val, I_sim_val = _err_from_state({"X": XN, "Y": YN, "S": SN, "A": AN}, int(I_obs_next))

            children_pairs.append((int(r), float(b_child)))
            children_states.append({"X": XN, "Y": YN, "S": SN, "A": AN})
            pair_runs.append(int(r))
            pair_betas.append(float(b_child))
            pair_C_dist.append(float(C_val))
            pair_I_sim.append(int(I_sim_val))
            children_weights.append(float(w_parent))

    children_weights = np.asarray(children_weights, dtype=float)
    betas_all_next = np.asarray([b for (_, b) in children_pairs], float)
    dC_all_next = np.asarray(pair_C_dist, dtype=float)

    keep_idx = _select_half_by_median(betas_all_next, dC_all_next)
    b_half_next = betas_all_next[keep_idx]
    d_half_next = dC_all_next[keep_idx]
    half_pairs_next = [children_pairs[i] for i in keep_idx]
    half_states_next = [children_states[i] for i in keep_idx]
    w_half_next = children_weights[keep_idx]

    I_sim_all_next = np.asarray(pair_I_sim, dtype=int)
    I_sim_half_next = I_sim_all_next[keep_idx]

    mean_b_half_next, lo_next, hi_next, _w = _weighted_beta_from_half(
        b_half_next, d_half_next, scale=2.0
    )

    reset_triggered = False
    if d_half_next.size > 0:
        pct_half = _to_pct_errors(np.asarray(d_half_next, float), y_obs=float(I_obs_next))
        finite_half = np.isfinite(pct_half)

        if np.any(finite_half):
            ok_half = finite_half & (pct_half >= fitness)
            ok_ratio_half = np.sum(ok_half) / np.sum(finite_half)

            if ok_ratio_half < 0.15:
                reset_triggered = True
                new_pairs: List[Tuple[int, float]] = []
                new_states: List[Dict[str, np.ndarray]] = []

                for j in range(len(half_pairs_next)):
                    rj, bj = half_pairs_next[j]
                    stj = half_states_next[j]
                    eps = float(np.random.uniform(0.0, sigma_mix))
                    b_plus = _clip_beta(float(bj) + float(eps))
                    b_minus = _clip_beta(float(bj) - float(eps))

                    new_pairs.append((int(rj), b_plus))
                    new_states.append(stj)

                    new_pairs.append((int(rj), b_minus))
                    new_states.append(stj)

               
                pool_pairs_out = new_pairs
                pool_states_out = new_states
            else:
                pool_pairs_out = children_pairs
                pool_states_out = children_states
        else:
            pool_pairs_out = children_pairs
            pool_states_out = children_states
    else:
        pool_pairs_out = children_pairs
        pool_states_out = children_states

    meta_json = {
        "num_steps": int(t_next),
        "continued_from_t": int(t_prev),
        "replicates": int(run_replicates),
        "first_half_dup_if_all_half_err_lt_pct_trigger": bool(reset_triggered),
        "pct_trigger": float(fitness),
        "beta_min": float(BETA_MIN),
        "beta_max": float(BETA_MAX),
        "status_to_code": getattr(ABM, "STATUS_TO_CODE", {"S": 0, "I": 1, "R": 2}),
        "code_to_status": getattr(ABM, "CODE_TO_STATUS", {0: "S", 1: "I", 2: "R"}),
        "saving_note": "THIS RUN SAVES ONLY beta/error outputs. Particle states are not written to disk.",
        "pct_error_reference_for_C": "I_obs (infected count)",
        "pct_error_formula": "exp(-(abs_error/max(I_obs,1))*PCT_SCALE)",
    }

    if out_path:
        small = dict(
            C_half_betas=np.asarray(b_half_next, dtype=float),
            C_half_dists=np.asarray(d_half_next, dtype=float),
            C_half_I_sim=np.asarray(I_sim_half_next, dtype=int),
            C_half_mean_beta=np.asarray([mean_b_half_next], dtype=float),
            C_half_mean_beta_ci=np.asarray([lo_next, hi_next], dtype=float),
            pair_runs=np.asarray(pair_runs, dtype=int),
            pair_betas=np.asarray(pair_betas, dtype=float),
            pair_C_dist=np.asarray(pair_C_dist, dtype=float),
            rep_betas_all=np.asarray(betas_all_next, dtype=float),
            rep_C_dists_all=np.asarray(dC_all_next, dtype=float),
            half_pairs=np.asarray(half_pairs_next, dtype=float),
            C_half_weights=np.asarray(_w, dtype=float),
            meta=np.frombuffer(json.dumps(meta_json).encode("utf-8"), dtype=np.uint8),
        )
        if os.path.exists(out_path) and overwrite:
            os.remove(out_path)
        np.savez_compressed(out_path, **small)

    pf_path = _pf_results_path(bank_path)
    _append_npz(
        pf_path,
        {
            _bank_key(t_next, "pair_runs"): np.asarray(pair_runs, dtype=int),
            _bank_key(t_next, "pair_betas"): np.asarray(pair_betas, dtype=float),
            _bank_key(t_next, "pair_C_dist"): np.asarray(pair_C_dist, dtype=float),
            _bank_key(t_next, "rep_betas_all"): np.asarray(betas_all_next, dtype=float),
            _bank_key(t_next, "rep_C_dists_all"): np.asarray(dC_all_next, dtype=float),
            _bank_key(t_next, "C_half_betas"): np.asarray(b_half_next, dtype=float),
            _bank_key(t_next, "C_half_dists"): np.asarray(d_half_next, dtype=float),
            _bank_key(t_next, "C_half_I_sim"): np.asarray(I_sim_half_next, dtype=int),
            _bank_key(t_next, "C_half_mean_beta"): np.asarray([mean_b_half_next], dtype=float),
            _bank_key(t_next, "C_half_mean_beta_ci"): np.asarray([lo_next, hi_next], dtype=float),
            _bank_key(t_next, "C_half_weights"): np.asarray(_w, dtype=float),
            _bank_key(t_next, "rep_I_all"): np.asarray(I_sim_all_next, dtype=int),
            _bank_key(t_next, "weights_all"): np.asarray(children_weights, dtype=float),
            _bank_key(t_next, "half_pairs"): np.asarray(half_pairs_next, dtype=float),
            _bank_key(t_next, "meta"): np.frombuffer(json.dumps(meta_json).encode("utf-8"), dtype=np.uint8),
        },
        overwrite=True,
    )

    return dict(
        t_prev=t_prev,
        t_next=t_next,
        children_count=len(children_pairs),
        all_pairs=dict(betas_C=betas_all_next, dC=dC_all_next),
        half_at_t_next=dict(
            betas=b_half_next,
            d=d_half_next,
            mean_beta=mean_b_half_next,
            ci=(lo_next, hi_next),
        ),
        pool_pairs_out=children_pairs,
        pool_states_out=children_states,
    )


def run_simulations(
    *,
    observed_csv: str,
    bank_path: str,
    t_start: int = 20,
    t_final: int = 80,
    window: int = 20,
    num_infected: int = 1,
    beta_start: float = 0.0,
    beta_step: float = 0.01,
    beta_count: int = 20,
    center=(0, 0),
    start_ring: int = 1,
    gamma: int = 14,
    run_replicates: int = 2,
    append_back_to_bank: bool = True,
) -> dict:
    if t_final < t_start:
        raise ValueError("t_final must be => t_start")
    if window <= 0:
        raise ValueError("window must be positive")

    current = int(t_start)
    last_summary = {}

    pool_pairs = None
    pool_states = None

    while current < int(t_final):
        nxt = min(current + int(window), int(t_final))
        last_summary = window_step(
            observed_csv=observed_csv,
            bank_path=bank_path,
            t_prev=current,
            t_next=nxt,
            num_infected=num_infected,
            beta_start=beta_start,
            beta_step=beta_step,
            beta_count=beta_count,
            center=center,
            start_ring=start_ring,
            gamma=gamma,
            run_replicates=run_replicates,
            overwrite=True,
            append_back_to_bank=append_back_to_bank,
            out_path=None,
            pool_pairs_in=pool_pairs,
            pool_states_in=pool_states,
        )
        pool_pairs = last_summary["pool_pairs_out"]
        pool_states = last_summary["pool_states_out"]
        current = nxt

    return last_summary


def _save_fig(fig, save_path: Optional[str]):
    if not save_path:
        return
    save_dir = os.path.dirname(os.path.abspath(save_path))
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_half_simulations(
    *,
    betas_C,
    dC,
    mean_beta_C: float | None,
    title: str,
    xlim=None,
    xtick=None,
    save_path: Optional[str] = None,
    show: bool = True,
    close: bool = False,
):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(betas_C, dC, alpha=0.85, label="Count-Difference", color="orange")
    if mean_beta_C is not None and np.isfinite(mean_beta_C):
        ax.axvline(mean_beta_C, linestyle="--", linewidth=1.6, label=f"β Count-Diff {mean_beta_C:.3f}")
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    if xtick is not None:
        ax.set_xticks(xtick if not np.isscalar(xtick) else [float(xtick)])
    ax.set_xlabel("β")
    ax.set_ylabel("Count difference")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, frameon=True)
    plt.tight_layout()
    _save_fig(fig, save_path)
    ret = None
    if show:
        plt.show()
    else:
        ret = fig
    if close:
        plt.close(fig)
    return ret


def plot_half(
    bank_path: str,
    t: int,
    title: Optional[str] = None,
    xlim=None,
    xtick=None,
    save_path: Optional[str] = None,
    show: bool = True,
    close: bool = False,
):
    bank_path = _load_path_for_plots(bank_path)

    keyB = _bank_key(t, "C_half_betas")
    keyD = _bank_key(t, "C_half_dists")
    keyM = _bank_key(t, "C_half_mean_beta")
    with np.load(bank_path, allow_pickle=True) as d:
        betas_C = d[keyB] if keyB in d.files else np.array([], float)
        dC = d[keyD] if keyD in d.files else np.array([], float)
        meanC = float(d[keyM][0]) if keyM in d.files and d[keyM].size else np.nan
    return plot_half_simulations(
        betas_C=betas_C,
        dC=dC,
        mean_beta_C=meanC,
        title=title or f"Lower-half at t={t}",
        xlim=xlim,
        xtick=xtick,
        save_path=save_path,
        show=show,
        close=close,
    )


def get_all_simulations(bank_path: str, t: int, level: str = "rep"):
    bank_path = _load_path_for_plots(bank_path)
    with np.load(bank_path, allow_pickle=True) as d:
        if level == "rep":
            kb = _bank_key(t, "rep_betas_all")
            kC = _bank_key(t, "rep_C_dists_all")
            if kb not in d.files:
                raise KeyError(f"error")
            betas_all = d[kb]
            dC_all = d[kC] if kC in d.files else np.array([], float)
            return betas_all, dC_all
        if level == "pair":
            kb = _bank_key(t, "pair_betas")
            kC = _bank_key(t, "pair_C_dist")
            if kb not in d.files:
                raise KeyError(f"error")
            betas_all = d[kb]
            dC_all = d[kC] if kC in d.files else np.array([], float)
            return betas_all, dC_all
    raise ValueError("error")



def plot_all_runs(
    *,
    betas_C,
    dC,
    title: str,
    xlim=(0, 0),
    xtick=None,
    save_path: Optional[str] = None,
    show: bool = True,
    close: bool = False,
):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(betas_C, dC, alpha=0.85, s=50, label="Count-Diff", color="orange")
    ax.set_title(title)

    if xtick is not None:
        if np.isscalar(xtick):
            ax.set_xticks([float(xtick)])
        elif isinstance(xtick, tuple) and len(xtick) == 3:
            ax.set_xticks(np.arange(*xtick))
        else:
            ax.set_xticks(list(xtick))
    if xlim is not None:
        ax.set_xlim(xlim)

    ax.set_xlabel("β")
    ax.set_ylabel("Count difference")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, frameon=True)
    plt.tight_layout()
    _save_fig(fig, save_path)
    ret = None
    if show:
        plt.show()
    else:
        ret = fig
    if close:
        plt.close(fig)
    return ret




def plot_all_vs_half(
    bank_path: str,
    t: int,
    title: Optional[str] = None,
    xlim=None,
    xtick=None,
    save_path: Optional[str] = None,
    observed_csv: Optional[str] = None,
    show: bool = True,
    close: bool = False,
):
    bank_path = _load_path_for_plots(bank_path)

    fig, ax = plt.subplots(figsize=(8, 5))

    
    betas_C_all, dC_all = get_all_simulations(bank_path, t, level="rep")
    ax.scatter(betas_C_all, dC_all, alpha=0.85, s=50, color="gold", label="Unselected runs")

    
    keyC  = _bank_key(t, "C_half_betas")
    keyCd = _bank_key(t, "C_half_dists")
    keyW  = _bank_key(t, "C_half_weights")

    with np.load(bank_path, allow_pickle=True) as d:
        betas_C_half = np.asarray(d[keyC], dtype=float) if keyC in d.files else np.array([], float)
        dC_half      = np.asarray(d[keyCd], dtype=float) if keyCd in d.files else np.array([], float)
        w_half       = np.asarray(d[keyW], dtype=float).ravel() if keyW in d.files else None

    ax.scatter(betas_C_half, dC_half, color="green", alpha=0.85, s=50, label="Selected runs")

    
    est_beta = np.nan
    y_val = np.nan

    if betas_C_half.size and dC_half.size and (w_half is not None):
        if w_half.size == betas_C_half.size:
            sw = float(np.sum(w_half))
            if sw > 0.0 and np.isfinite(sw):
                w = w_half / sw
                est_beta = float(np.sum(w * betas_C_half))

                j = int(np.nanargmin(np.abs(betas_C_half - est_beta)))
                y_val = float(dC_half[j])
                ax.scatter(
                    est_beta, y_val,
                    color="blue", s=120, zorder=6,
                    label=rf"Estimated $\beta^* = {est_beta:.3f}$"
                )
        else:
            print(f"error")

    ax.set_title(title or f"error")
    if xlim is not None:
        ax.set_xlim(xlim)
    if xtick is not None:
        ax.set_xticks(xtick if not np.isscalar(xtick) else [float(xtick)])

    ax.set_xlabel("β")
    ax.set_ylabel("Distance")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, frameon=True)
    plt.tight_layout()
    _save_fig(fig, save_path)

    ret = None
    if show:
        plt.show()
    else:
        ret = fig
    if close:
        plt.close(fig)
    return ret


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


def _load_beta_from_csv(csv_path: str):
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    cols = {c.lower(): c for c in df.columns}

    time_candidates = ["time", "t"]
    beta_candidates = ["beta", "beta_mean", "beta_hat", "est_beta"]

    time_col = next((cols[c] for c in time_candidates if c in cols), None)
    beta_col = next((cols[c] for c in beta_candidates if c in cols), None)

    def _to_num(s: pd.Series) -> pd.Series:
        x = s.astype(str).str.strip().str.replace(",", "", regex=False)
        x = x.str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False)
        return pd.to_numeric(x, errors="coerce")

    tnum = _to_num(df[time_col])
    bnum = _to_num(df[beta_col])

    s = pd.DataFrame({"t": tnum, "beta": bnum}).dropna()
    s = s.groupby("t", as_index=False)["beta"].mean().sort_values("t")
    return s["t"].astype(int).to_numpy(), s["beta"].astype(float).to_numpy()


def plot_mean_betas(
    bank_path,
    title=r"Estimated $\beta$ by method 1",
    overlay_csv_path=None,
    save_path="figs/mean_beta.pdf",
    xlim=None,
    ylim=None,
    xtick=None,
    ytick=None,
    xticks=None,
    yticks=None,
    show=False,
    close=True,
    fig_size=(8, 4),
    spine_sides=("left", "bottom"),
    spine_width=1.2,
    spine_out=0,
    figure_border=False,
    figure_border_width=1.0,
):
    bank_path = _load_path_for_plots(bank_path)

    if xtick is None and xticks is not None:
        xtick = xticks
    if ytick is None and yticks is not None:
        ytick = yticks

    ts, meanC = load_mean_betas_from_simulations(bank_path)
    if getattr(ts, "size", 0) == 0:
        raise RuntimeError("error")

    C_lo, C_hi = [], []
    with np.load(bank_path, allow_pickle=True) as d:
        for t in ts:
            kCci = _bank_key(int(t), "C_half_mean_beta_ci")
            if kCci in d.files:
                arr = np.asarray(d[kCci]).ravel()
                if arr.size >= 2 and np.isfinite(arr[:2]).all():
                    loC, hiC = float(arr[0]), float(arr[1])
                else:
                    loC, hiC = np.nan, np.nan
            else:
                loC, hiC = np.nan, np.nan
            C_lo.append(loC)
            C_hi.append(hiC)

    df = pd.DataFrame(
        {
            "time": np.asarray(ts, float),
            "meanC": np.asarray(meanC, float),
            "C_lo": np.asarray(C_lo, float),
            "C_hi": np.asarray(C_hi, float),
        }
    )

    ov = None
    if overlay_csv_path:
        try:
            csv_t, csv_b = _load_beta_from_csv(overlay_csv_path)
            ov = pd.DataFrame({"time": np.asarray(csv_t, float), "overlay": np.asarray(csv_b, float)})
        except Exception:
            try:
                tmp = pd.read_csv(overlay_csv_path)
                time_col = next(c for c in tmp.columns if c.lower() in ("time", "t"))
                beta_col = next(c for c in tmp.columns if c.lower() in ("beta"))
                ov = tmp[[time_col, beta_col]].rename(columns={time_col: "time", beta_col: "overlay"})
                ov["time"] = ov["time"].astype(float)
                ov["overlay"] = ov["overlay"].astype(float)
            except Exception:
                ov = None

    fig, ax = plt.subplots(figsize=fig_size)

    tt = df["time"].to_numpy(float)
    mC = df["meanC"].to_numpy(float)
    lo = df["C_lo"].to_numpy(float)
    hi = df["C_hi"].to_numpy(float)

    mask = np.isfinite(lo) & np.isfinite(hi)
    if np.any(mask):
        ax.fill_between(tt[mask], lo[mask], hi[mask], color="#D4F4FF", label="95% CI")

    ax.plot(tt, mC, color="#23ABBD", linewidth=2.5, label=r"Estimated $\beta(t)$")
    ax.scatter(tt, mC, color="blue", s=10, zorder=3)

    if ov is not None and len(ov):
        ax.plot(
            ov["time"].to_numpy(float),
            ov["overlay"].to_numpy(float),
            linestyle="--",
            linewidth=1.5,
            color="red",
            label=r"Actual $\beta(t)$",
        )

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\beta(t)$")

    wanted = set(spine_sides or ())
    for side, spine in ax.spines.items():
        vis = side in wanted
        spine.set_visible(vis)
        if vis:
            spine.set_linewidth(float(spine_width))
            if spine_out:
                spine.set_position(("outward", float(spine_out)))
    ax.xaxis.set_ticks_position("bottom" if "bottom" in wanted else "none")
    ax.yaxis.set_ticks_position("left" if "left" in wanted else "none")

    if figure_border:
        from matplotlib.patches import Rectangle

        fig.add_artist(
            Rectangle((0, 0), 1, 1, fill=False, transform=fig.transFigure, linewidth=float(figure_border_width))
        )

    if xlim is not None and len(xlim) == 2:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    if ylim is not None and len(ylim) == 2:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    if xtick is not None and np.size(xtick):
        ax.set_xticks([float(x) for x in np.asarray(xtick).ravel()])
    if ytick is not None and np.size(ytick):
        ax.set_yticks([float(x) for x in np.asarray(ytick).ravel()])

    ax.legend(loc="upper right", frameon=False)

    out_dir = os.path.dirname(save_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=600)
    if show:
        plt.show()
    if close:
        plt.close(fig)


# not want to used the save ddistance but want to reestimate from the numbe rof infected that saved 
def plot_all_vs_half_recomputed_distance(
    bank_path: str,
    observed_csv: str,
    t: int,
    title: Optional[str] = None,
    xlim=None,
    xtick=None,
    save_path: Optional[str] = None,
    show: bool = True,
    close: bool = False,
):
  
    bank_path = _load_path_for_plots(bank_path)

    I_obs_t = int(get_observed_infected_count(observed_csv, t_obs=t))

    key_all_b = _bank_key(t, "rep_betas_all")
    key_all_I = _bank_key(t, "rep_I_all")

    key_half_b = _bank_key(t, "C_half_betas")
    key_half_I = _bank_key(t, "C_half_I_sim")
    keyW = _bank_key(t, "C_half_weights")

    with np.load(bank_path, allow_pickle=True) as d:
        if key_all_b not in d.files:
            raise KeyError(f"Missing {key_all_b}")
        if key_all_I not in d.files:
            raise KeyError(f"error")
        if key_half_b not in d.files:
            raise KeyError(f"error")
        if key_half_I not in d.files:
            raise KeyError(f"error")

        betas_all = np.asarray(d[key_all_b], dtype=float)
        I_all = np.asarray(d[key_all_I], dtype=float)

        betas_half = np.asarray(d[key_half_b], dtype=float)
        I_half = np.asarray(d[key_half_I], dtype=float)

        w_half = np.asarray(d[keyW], dtype=float).ravel() if keyW in d.files else None

    
    d_all_recomputed = (float(I_obs_t) - I_all)
    d_half_recomputed = (float(I_obs_t) - I_half)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(betas_all, d_all_recomputed, alpha=0.85, s=50, color="gold", label="Unselected particles")

    ax.scatter(betas_half,d_half_recomputed, alpha=0.85, s=50, color="green", label="Selected particles" )

   
    est_beta = np.nan
    y_val = np.nan

    if betas_half.size and d_half_recomputed.size and (w_half is not None):
        if w_half.size == betas_half.size:
            sw = float(np.sum(w_half))
            if sw > 0.0 and np.isfinite(sw):
                w = w_half / sw
                est_beta = float(np.sum(w * betas_half))

                #y_val = int(np.sum(w * d_half_recomputed))
                y_val = int(np.mean(d_half_recomputed))

                ax.scatter(
                    est_beta,
                    y_val,
                    color="blue",
                    s=120,
                    zorder=6,
                    label=rf"Estimated $\hat{{\beta}} = {est_beta:.3f}$"
                )
        else:
            print(
                f"[plot_all_vs_half_recomputed_distance] "
                f"weight length mismatch: w={w_half.size}, betas={betas_half.size}"
            )

    #ax.set_title(f"Selected and unselected particles at week {int(t/7)}", fontsize=19, fontweight="bold")
    ax.set_title(f"Particles at week {int(t/7)}", fontsize=19, fontweight="bold")
    if xlim is not None:
        ax.set_xlim(xlim)
    if xtick is not None:
        ax.set_xticks(xtick if not np.isscalar(xtick) else [float(xtick)])

    #ax.set_xlabel(r"$\boldsymbol{\beta}_{t}$", fontsize=25, fontweight="bold")
    ax.set_ylabel(r"$y_{t_{k}} - x_{t_{k}} $", rotation=90, labelpad=15, fontsize=25, fontweight="bold")
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1.5)
    #ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, frameon=True)
    plt.tight_layout()

    _save_fig(fig, save_path)

    ret = None
    if show:
        plt.show()
    else:
        ret = fig

    if close:
        plt.close(fig)

    return ret

def export_mean_betas_to_csv(bank_path: str, out_csv: str) -> None:
    bank_path = _load_path_for_plots(bank_path)

    ts, meanC = load_mean_betas_from_simulations(bank_path)
    if getattr(ts, "size", 0) == 0:
        raise RuntimeError("No mean-beta keys found in bank.")

    C_lo, C_hi = [], []
    with np.load(bank_path, allow_pickle=True) as d:
        for t in ts:
            kCci = _bank_key(int(t), "C_half_mean_beta_ci")
            if kCci in d.files:
                arr = np.asarray(d[kCci]).ravel()
                if arr.size >= 2 and np.isfinite(arr[:2]).all():
                    loC, hiC = float(arr[0]), float(arr[1])
                else:
                    loC, hiC = np.nan, np.nan
            else:
                loC, hiC = np.nan, np.nan
            C_lo.append(loC)
            C_hi.append(hiC)

    df = pd.DataFrame(
        {
            "Time": np.asarray(ts, float),
            "Estimated_Beta": np.asarray(meanC, float),
            "beta_ci_lo": np.asarray(C_lo, float),
            "beta_ci_hi": np.asarray(C_hi, float),
        }
    )
    df.to_csv(out_csv, index=False)


def main():
    import argparse

    p = argparse.ArgumentParser(description="Windowed continuation driver (β grid; I metrics).")
    p.add_argument("--observed-csv", required=True)
    p.add_argument("--bank-path", required=True)
    p.add_argument("--t-start", type=int, default=20)
    p.add_argument("--t-final", type=int, default=80)
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--num-infected", type=int, default=1)
    p.add_argument("--beta-start", type=float, default=0.40)
    p.add_argument("--beta-step", type=float, default=0.01)
    p.add_argument("--beta-count", type=int, default=21)
    p.add_argument("--center-x", type=int, default=0)
    p.add_argument("--center-y", type=int, default=0)
    p.add_argument("--start-ring", type=int, default=1)
    p.add_argument("--gamma", type=int, default=14)
    p.add_argument("--run-replicates", type=int, default=2)
    args = p.parse_args()

    run_simulations(
        observed_csv=args.observed_csv,
        bank_path=args.bank_path,
        t_start=args.t_start,
        t_final=args.t_final,
        window=args.window,
        num_infected=args.num_infected,
        beta_start=args.beta_start,
        beta_step=args.beta_step,
        beta_count=args.beta_count,
        center=(args.center_x, args.center_y),
        start_ring=args.start_ring,
        gamma=args.gamma,
        run_replicates=args.run_replicates,
        append_back_to_bank=True,
    )


if __name__ == "__main__":
    main()
