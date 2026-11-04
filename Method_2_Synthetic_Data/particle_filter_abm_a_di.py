# particle_filter_abm_a_dist.py
from __future__ import annotations
import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional

from packages.abm.abm import ABM

from Beta_Distance_Dostributions import (
    get_infected_distances_from_csv,
    run_simulation_and_get_infected_distances,
    estimate_beta_distances_using_precomputed,
)


BETA_MIN = 0.0
BETA_MAX = 10.0


fitness = 0.95

sigma_min = 0.15
sigma_mix = 0.3


def _to_pct_errors(abs_errors: np.ndarray, y_obs: float) -> np.ndarray:
    denom = max(float(y_obs), 1.0)
    return np.exp(-(np.asarray(abs_errors, dtype=float) / denom) )


def _clip_beta(b: float, lo: float = BETA_MIN, hi: float = BETA_MAX) -> float:
    b = float(b)
    if b < lo:
        return float(lo)
    if b > hi:
        return float(hi)
    return b


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

def _infected_count_from_state(S: np.ndarray, infected_code: int = 1) -> int:
    return int(np.sum(S.astype(int) == int(infected_code)))

def uniform_eps(clip=(-0.15, 0.15)) -> float:
    a, b = clip
    return float(np.random.uniform(a, b))

def _pf_results_path(bank_path: str) -> str:
    root, ext = os.path.splitext(bank_path)
    return root + "_pf_results.npz"


def _load_path_for_plots(path: str) -> str:
    pf = _pf_results_path(path)
    return pf if os.path.exists(pf) else path

def _alt_signs(n: int) -> np.ndarray:
    s = np.ones(int(n), dtype=int)
    s[1::2] = -1
    return s


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
                print(f"No rows at t={t} in '{cols[key]}'.")
            return
    print("No time column.")


def _simulations_num_steps(path: str, expected: int) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Bank file not found: {path}")
    with np.load(path, allow_pickle=True) as d:
        files = set(d.files)
        if "meta" in files:
            try:
                meta = json.loads(d["meta"].tobytes().decode("utf-8"))
                got = int(meta.get("num_steps", -1))
                if got != int(expected):
                    raise ValueError(f"error")
                return
            except Exception as e:
                print(f"Could not")
        if "I" not in files:
            print(f"error")
            return
        I = d["I"]
        if I.ndim not in (2, 3):
            print(f"Unexpected")
            return
        T = int(I.shape[-1])
        if expected not in (T, T - 1):
            raise ValueError(
                f"error"
            )


def detect_runs(bank_path: str) -> int:
    if not os.path.exists(bank_path):
        raise FileNotFoundError(f"Bank file not found: {bank_path}")
    with np.load(bank_path, allow_pickle=True) as d:
        if "S" in d.files:
            S_all = d["S"]
        elif "final_status" in d.files:
            S_all = d["final_status"]
        else:
            raise KeyError("Bank missing s")
        return S_all.shape[0] if S_all.ndim == 3 else 1


def _bank_key(t: int, name: str) -> str:
    return f"t{int(t)}_{name}"


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


def _infected_distances(
    X: np.ndarray, Y: np.ndarray, S: np.ndarray, center=(0, 0), infected_code: int = 1
) -> List[int]:
    cx, cy = int(center[0]), int(center[1])
    mask = (S.astype(int) == int(infected_code))
    if not np.any(mask):
        return []
    dx = np.abs(X[mask] - cx)
    dy = np.abs(Y[mask] - cy)
    return (dx + dy).astype(int).tolist()


def _continue_from_arrays(
    *, beta: float, gamma: int,
    X: np.ndarray, Y: np.ndarray, S: np.ndarray, A: Optional[np.ndarray],
    steps: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ages = None if A is None else np.asarray(A, dtype=int).ravel()
    model = ABM.from_state(beta=float(beta), gamma=int(gamma),
                           X=X, Y=Y, status_codes=S, infected_ages=ages)
    I_series: List[int] = []
    for _ in range(int(steps)):
        model.step()
        I_series.append(sum(1 for ag in model.agents.values() if ag.status == "I"))
    Xn, Yn, Sn, An = model.to_arrays()
    return Xn, Yn, Sn, An, np.asarray(I_series, dtype=int)


def _read_sim_distances(
    *, beta: float, num_steps: int, center, bank_path: str, run_idx: int, retries: int = 2
) -> list[int] | None:
    last_err = None
    for _ in range(max(1, retries)):
        try:
            dists, _ = run_simulation_and_get_infected_distances(
                beta=float(beta),
                num_steps=num_steps,
                num_infected=None,
                center=center,
                bank_path=bank_path,
                run_idx=run_idx,
            )
            arr = np.asarray(dists).ravel()
            return [int(x) for x in arr.tolist()]
        except Exception as e:
            last_err = e
            continue
    return None

def get_run_distances(
    *, observed_csv: str, bank_path: str, run_idx: int,
    num_infected: int = 1, num_steps: int = 20,
    beta_start: float = 0.40, beta_step: float = 0.01, beta_count: int = 20,
    center=(0, 0), start_ring: int = 1, strict: bool = True,
):
    betas_grid = np.round(beta_start + np.arange(beta_count, dtype=float) * beta_step, 6)
    observed_dists = get_infected_distances_from_csv(observed_csv, center=center, t_obs=num_steps)

    with np.load(bank_path, allow_pickle=True) as _d:
        bank_betas = _d["betas"].astype(float)

    betas_eval = bank_betas if strict else betas_grid

    sim_distances_by_beta = {}
    for b in betas_eval:
        dlist = _read_sim_distances(
            beta=float(b),
            num_steps=num_steps,
            center=center,
            bank_path=bank_path,
            run_idx=run_idx,
            retries=2,
        )
        sim_distances_by_beta[float(b)] = dlist if dlist is not None else []

    betas_A = np.array([], float)
    dA = np.array([], float)

    if sim_distances_by_beta:
        results_A, _ = estimate_beta_distances_using_precomputed(
            observed_distances=observed_dists,
            sim_distances_by_beta=sim_distances_by_beta,
            start_ring=start_ring,
        )

        a_map = {float(b): float(d) for b, d in results_A}
        betas_A = np.asarray(list(betas_eval), dtype=float)
        dA = np.asarray([a_map.get(float(b), np.nan) for b in betas_A], dtype=float)

    return betas_A, dA

def get_all_simulations(bank_path: str, t: int, level: str = "rep"):
    bank_path = _load_path_for_plots(bank_path)

    with np.load(bank_path, allow_pickle=True) as d:
        if level == "rep":
            kb = _bank_key(t, "rep_betas_all")
            kA = _bank_key(t, "rep_A_dists_all")

            if kb not in d.files:
                raise KeyError(f"error")

            betas_all = d[kb]
            dA_all = d[kA] if kA in d.files else np.array([], float)
            return betas_all, dA_all

        elif level == "pair":
            kb = _bank_key(t, "pair_betas")
            kA = _bank_key(t, "pair_A_dist")

            if kb not in d.files:
                raise KeyError(f"error")

            betas_all = d[kb]
            dA_all = d[kA] if kA in d.files else np.array([], float)
            return betas_all, dA_all

        else:
            raise ValueError("pair")
        
        
def _select_half_by_median(arr_beta: np.ndarray, arr_err: np.ndarray) -> np.ndarray:
    arr_beta = np.asarray(arr_beta, float)
    arr_err  = np.asarray(arr_err,  float)

    m = np.isfinite(arr_beta) & np.isfinite(arr_err)
    if not m.any():
        return np.array([], dtype=int)

    beta = arr_beta[m]
    err  = arr_err[m]
    idx_map = np.nonzero(m)[0]

    N = err.size
    k = int(round(N * 0.5))
    if k == 0:
        return np.array([], dtype=int)

    order = np.argsort(err, kind="stable")
    err_sorted  = err[order]

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

    estimated_beta = float(np.sum(w * x))

    var_w = float(np.sum((x - estimated_beta) ** 2))
    se = np.sqrt(var_w / max(len(x), 1))
    z = 1.96
    lo, hi = estimated_beta - z * se, estimated_beta + z * se

    return estimated_beta, lo, hi, w


def select_lower_half(
    *, observed_csv: str, bank_path_tobs: str, num_infected: int = 1, t_obs: int = 20,
    beta_start: float = 0.4, beta_step: float = 0.01, beta_count: int = 20,
    center=(0, 0), start_ring: int = 1, strict: bool = True,
):
    _simulations_num_steps(bank_path_tobs, t_obs)
    _no_observed_at_time(observed_csv, t_obs)

    n_runs = detect_runs(bank_path_tobs)

    pairs_A: List[tuple[int, float]] = []
    errs_A:  List[float] = []

    for run_idx in range(n_runs):
        betas_A, dA = get_run_distances(
            observed_csv=observed_csv, bank_path=bank_path_tobs, run_idx=run_idx,
            num_infected=num_infected, num_steps=t_obs,
            beta_start=beta_start, beta_step=beta_step, beta_count=beta_count,
            center=center, start_ring=start_ring, strict=strict,
        )
        if betas_A.size:
            m = np.isfinite(dA) & np.isfinite(betas_A)
            for b, e in zip(betas_A[m], dA[m]):
                pairs_A.append((int(run_idx), float(b)))
                errs_A.append(float(e))

    arr_err_a  = np.asarray(errs_A, dtype=float)
    arr_beta_a = np.asarray([p[1] for p in pairs_A], dtype=float)

    keep_idx_a = _select_half_by_median(arr_beta_a, arr_err_a)

    selection_a    = [(int(pairs_A[i][0]), float(pairs_A[i][1]), float(arr_err_a[i])) for i in keep_idx_a]
    betas_A_keep_a = arr_beta_a[keep_idx_a]
    dA_keep_a      = arr_err_a[keep_idx_a]

    mean_beta_A, A_ci_lo, A_ci_hi, _ = _weighted_beta_from_half(betas_A_keep_a, dA_keep_a, scale=2.0, temp=0.3)

    return selection_a, betas_A_keep_a, dA_keep_a, mean_beta_A


def second_select_lower_half(
    *, observed_csv: str, bank_path_tobs: str, num_infected: int = 1, t_obs: int = 20,
    beta_start: float = 0.4, beta_step: float = 0.01, beta_count: int = 20,
    center=(0, 0), start_ring: int = 1, strict: bool = True,
    selection_a: List[Tuple[int, float, float]],
):
    second_arr_beta_a = np.asarray([b for (_, b, _) in selection_a], dtype=float)
    second_arr_err_a  = np.asarray([e for (_, _, e) in selection_a], dtype=float)

    arr_beta = np.asarray(second_arr_beta_a, dtype=float)
    arr_err  = np.asarray(second_arr_err_a,  dtype=float)

    m = np.isfinite(arr_beta) & np.isfinite(arr_err)
    if not m.any():
        return [], np.array([], float), np.array([], float), float("nan"), np.array([], int)

    beta = arr_beta[m]
    err  = arr_err[m]
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

    second_keep_idx_a = idx_map[np.asarray(keep_rel, dtype=int)]

    second_selection_a    = [(int(selection_a[i][0]), float(selection_a[i][1])) for i in second_keep_idx_a]
    second_betas_A_keep_a = second_arr_beta_a[second_keep_idx_a]
    second_dA_keep_a      = second_arr_err_a[second_keep_idx_a]
    second_mean_beta_A    = float(np.mean(second_betas_A_keep_a)) if second_betas_A_keep_a.size else np.nan

    
    return (second_selection_a,
            second_betas_A_keep_a,
            second_dA_keep_a,
            second_mean_beta_A,
            second_keep_idx_a)


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

    def _err_from_state(
        st: Dict[str, np.ndarray],
        beta_val: float,
        obs_dists: List[int],
    ) -> Tuple[float, int]:
        code_I = getattr(ABM, "STATUS_TO_CODE", {"I": 1}).get("I", 1)

        I_sim = _infected_count_from_state(
            st["S"],
            infected_code=code_I,
        )

        dlist = _infected_distances(
            st["X"],
            st["Y"],
            st["S"],
            center=center,
            infected_code=code_I,
        )

        resA, _ = estimate_beta_distances_using_precomputed(
            observed_distances=obs_dists,
            sim_distances_by_beta={float(beta_val): dlist},
            start_ring=start_ring,
        )

        A_val = float(resA[0][1]) if resA else np.nan

        return float(A_val), int(I_sim)

    def _alt_signs(n: int) -> np.ndarray:
        s = np.ones(int(n), dtype=int)
        s[1::2] = -1
        return s

    if t_next < t_prev:
        raise ValueError("t_next must be => t_prev")

    _no_observed_at_time(observed_csv, t_prev)
    _no_observed_at_time(observed_csv, t_next)

    steps_to_go = int(t_next) - int(t_prev)

    obs_dists_prev = get_infected_distances_from_csv(
        observed_csv,
        center=center,
        t_obs=t_prev,
    )

    obs_dists_next = get_infected_distances_from_csv(
        observed_csv,
        center=center,
        t_obs=t_next,
    )

    y_obs_prev = float(max(len(obs_dists_prev), 1))
    y_obs_next = float(max(len(obs_dists_next), 1))

    pairs_prev: List[Tuple[int, float]] = []
    states_prev: List[Dict[str, np.ndarray]] = []

    have_pool = (pool_pairs_in is not None) and (pool_states_in is not None)

    carry_idx: set[int] = set()
    selection_half_errors: np.ndarray = np.array([], dtype=float)
    I_sim_half_prev_saved = np.array([], dtype=int)

    if not have_pool:
        selection_a, Aprev_b, Aprev_d, betaAprev = select_lower_half(
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

        if not selection_a:
            raise RuntimeError(f"No (run,β) were selected at t={t_prev}.")

        selection_half_errors = np.asarray(Aprev_d, dtype=float)

        I_sim_sel = []

        with np.load(bank_path, allow_pickle=True) as d:
            betas_bank = np.asarray(d["betas"], dtype=float)
            I_mat = d["I"]

            for (r, b, _) in selection_a:
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
            np.asarray(Aprev_b, float),
            np.asarray(Aprev_d, float),
            scale=2.0,
        )

        _append_npz(
            pf_path,
            {
                _bank_key(t_prev, "A_half_betas"): np.asarray(Aprev_b, dtype=float),
                _bank_key(t_prev, "A_half_dists"): np.asarray(Aprev_d, dtype=float),
                _bank_key(t_prev, "A_half_I_sim"): np.asarray(I_sim_half_prev_saved, dtype=int),
                _bank_key(t_prev, "A_half_mean_beta"): np.asarray([betaAprev], dtype=float),
                _bank_key(t_prev, "A_half_mean_beta_ci"): np.asarray([lo0, hi0], dtype=float),
                _bank_key(t_prev, "A_half_weights"): np.asarray(w_prev, dtype=float),
                _bank_key(t_prev, "first_half_triplets"): np.asarray(selection_a, dtype=float),
            },
            overwrite=True,
        )

        _, _, _, _, second_keep_idx_a = second_select_lower_half(
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
            selection_a=selection_a,
        )

        pairs_prev = [(int(r), float(b)) for (r, b, _) in selection_a]

        for (run_idx, b) in pairs_prev:
            X0, Y0, S0 = _read_initial_state(
                bank_path,
                run_idx=int(run_idx),
                beta=float(b),
            )

            states_prev.append(
                {
                    "X": X0,
                    "Y": Y0,
                    "S": S0,
                    "A": np.zeros_like(S0, dtype=int),
                }
            )

        carry_idx = set(map(int, np.asarray(second_keep_idx_a).ravel()))

    else:
        pairs_prev = [(int(r), float(b)) for (r, b) in pool_pairs_in]
        states_prev = [st for st in pool_states_in]

        betas_prev = np.asarray([b for (_, b) in pairs_prev], float)

        errs_prev_list = []
        I_sim_prev_list = []

        for i, st in enumerate(states_prev):
            e, isv = _err_from_state(
                st,
                betas_prev[i],
                obs_dists_prev,
            )

            errs_prev_list.append(float(e))
            I_sim_prev_list.append(int(isv))

        errs_prev = np.asarray(errs_prev_list, float)
        I_sim_prev = np.asarray(I_sim_prev_list, dtype=int)

        idx_first = _select_half_by_median(betas_prev, errs_prev)

        b_first = betas_prev[idx_first]
        d_first = errs_prev[idx_first]
        I_sim_half_prev_saved = I_sim_prev[idx_first]

        selection_half_errors = np.asarray(d_first, dtype=float)

        mean_b, lo, hi, w_prev = _weighted_beta_from_half(
            b_first,
            d_first,
            scale=2.0,
        )

        pf_path = _pf_results_path(bank_path)

        _append_npz(
            pf_path,
            {
                _bank_key(t_prev, "A_half_betas"): np.asarray(b_first, dtype=float),
                _bank_key(t_prev, "A_half_dists"): np.asarray(d_first, dtype=float),
                _bank_key(t_prev, "A_half_I_sim"): np.asarray(I_sim_half_prev_saved, dtype=int),
                _bank_key(t_prev, "A_half_mean_beta"): np.asarray([mean_b], dtype=float),
                _bank_key(t_prev, "A_half_mean_beta_ci"): np.asarray([lo, hi], dtype=float),
                _bank_key(t_prev, "A_half_weights"): np.asarray(w_prev, dtype=float),
                _bank_key(t_prev, "first_half_triplets"): np.asarray(
                    [
                        (pairs_prev[i][0], pairs_prev[i][1], d_first[j])
                        for j, i in enumerate(idx_first)
                    ],
                    dtype=float,
                ),
            },
            overwrite=True,
        )

        pairs_prev = [pairs_prev[i] for i in idx_first]
        states_prev = [states_prev[i] for i in idx_first]

        sel_triplets = [
            (pairs_prev[i][0], pairs_prev[i][1], float(d_first[i]))
            for i in range(len(pairs_prev))
        ]

        _, _, _, _, second_keep_idx_a = second_select_lower_half(
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
            selection_a=sel_triplets,
        )

        carry_idx = set(map(int, np.asarray(second_keep_idx_a).ravel()))

    mode_any_ge_trigger = False
    mode_all_lt_trigger = False

    if selection_half_errors.size > 0:
        pct_sel = _to_pct_errors(
            np.asarray(selection_half_errors, float),
            y_obs=y_obs_prev,
        )

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
    children_Iseries: List[np.ndarray] = []

    pair_runs: List[int] = []
    pair_betas: List[float] = []
    pair_A_dist: List[float] = []
    pair_I_sim: List[int] = []

    children_weights: List[float] = []

    with np.load(_pf_results_path(bank_path), allow_pickle=True) as d:
        key_w_prev = _bank_key(t_prev, "A_half_weights")

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

            A_val, I_sim_val = _err_from_state(
                {
                    "X": XN,
                    "Y": YN,
                    "S": SN,
                    "A": AN,
                },
                float(b_child),
                obs_dists_next,
            )

            children_pairs.append((int(r), float(b_child)))
            children_states.append(
                {
                    "X": XN,
                    "Y": YN,
                    "S": SN,
                    "A": AN,
                }
            )
            children_Iseries.append(np.asarray(Iseg, dtype=int))

            pair_runs.append(int(r))
            pair_betas.append(float(b_child))
            pair_A_dist.append(float(A_val))
            pair_I_sim.append(int(I_sim_val))
            children_weights.append(float(w_parent))

    children_weights = np.asarray(children_weights, dtype=float)

    betas_all_next = np.asarray(
        [b for (_, b) in children_pairs],
        float,
    )

    dA_all_next = np.asarray(pair_A_dist, dtype=float)

    keep_idx = _select_half_by_median(
        betas_all_next,
        dA_all_next,
    )

    b_half_next = betas_all_next[keep_idx]
    d_half_next = dA_all_next[keep_idx]

    half_pairs_next = [children_pairs[i] for i in keep_idx]
    half_states_next = [children_states[i] for i in keep_idx]
    half_Iseg_next = [children_Iseries[i] for i in keep_idx]

    w_half_next = children_weights[keep_idx]

    I_sim_all_next = np.asarray(pair_I_sim, dtype=int)
    I_sim_half_next = I_sim_all_next[keep_idx]

    mean_b_half_next, lo_next, hi_next, _w = _weighted_beta_from_half(
        b_half_next,
        d_half_next,
        scale=2.0,
    )

    reset_triggered = False

    if d_half_next.size > 0:
        pct_half = _to_pct_errors(
            np.asarray(d_half_next, float),
            y_obs=y_obs_next,
        )

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
        "saving_note": "THIS RUN SAVES beta/error outputs and old state keys for plotting.",
        "pct_error_reference_for_A": "observed infected-distance count",
        "pct_error_formula": "exp(-(distance_error/max(number_of_observed_distances,1)))",
    }

    if out_path:
        small = dict(
            A_half_betas=np.asarray(b_half_next, dtype=float),
            A_half_dists=np.asarray(d_half_next, dtype=float),
            A_half_I_sim=np.asarray(I_sim_half_next, dtype=int),
            A_half_mean_beta=np.asarray([mean_b_half_next], dtype=float),
            A_half_mean_beta_ci=np.asarray([lo_next, hi_next], dtype=float),
            pair_runs=np.asarray(pair_runs, dtype=int),
            pair_betas=np.asarray(pair_betas, dtype=float),
            pair_A_dist=np.asarray(pair_A_dist, dtype=float),
            rep_betas_all=np.asarray(betas_all_next, dtype=float),
            rep_A_dists_all=np.asarray(dA_all_next, dtype=float),
            half_pairs=np.asarray(half_pairs_next, dtype=float),
            half_reps_states=np.asarray(half_states_next, dtype=object),
            half_reps_Iseries=np.asarray(half_Iseg_next, dtype=object),
            A_half_weights=np.asarray(_w, dtype=float),
            meta=np.frombuffer(json.dumps(meta_json).encode("utf-8"), dtype=np.uint8),
        )

        if os.path.exists(out_path) and overwrite:
            os.remove(out_path)

        np.savez_compressed(out_path, **small)

    pf_path = _pf_results_path(bank_path)

    _append_npz(
        pf_path,
        {
            _bank_key(t_next, "pairs"): np.asarray(children_pairs, dtype=float),
            _bank_key(t_next, "reps_states"): np.asarray([[st] for st in children_states], dtype=object),
            _bank_key(t_next, "reps_Iseries"): np.asarray([[np.asarray(x, dtype=int)] for x in children_Iseries], dtype=object),

            _bank_key(t_next, "pair_runs"): np.asarray(pair_runs, dtype=int),
            _bank_key(t_next, "pair_betas"): np.asarray(pair_betas, dtype=float),
            _bank_key(t_next, "pair_A_dist"): np.asarray(pair_A_dist, dtype=float),

            _bank_key(t_next, "rep_pairs_flat"): np.asarray(children_pairs, dtype=float),
            _bank_key(t_next, "rep_betas_all"): np.asarray(betas_all_next, dtype=float),
            _bank_key(t_next, "rep_A_dists_all"): np.asarray(dA_all_next, dtype=float),
            _bank_key(t_next, "rep_I_all"): np.asarray(I_sim_all_next, dtype=int),

            _bank_key(t_next, "A_half_betas"): np.asarray(b_half_next, dtype=float),
            _bank_key(t_next, "A_half_dists"): np.asarray(d_half_next, dtype=float),
            _bank_key(t_next, "A_half_I_sim"): np.asarray(I_sim_half_next, dtype=int),
            _bank_key(t_next, "A_half_mean_beta"): np.asarray([mean_b_half_next], dtype=float),
            _bank_key(t_next, "A_half_mean_beta_ci"): np.asarray([lo_next, hi_next], dtype=float),
            _bank_key(t_next, "A_half_weights"): np.asarray(_w, dtype=float),

            _bank_key(t_next, "weights_all"): np.asarray(children_weights, dtype=float),
            _bank_key(t_next, "half_pairs"): np.asarray(half_pairs_next, dtype=float),
            _bank_key(t_next, "half_reps_states"): np.asarray(half_states_next, dtype=object),
            _bank_key(t_next, "half_reps_Iseries"): np.asarray(half_Iseg_next, dtype=object),

            _bank_key(t_next, "meta"): np.frombuffer(json.dumps(meta_json).encode("utf-8"), dtype=np.uint8),
        },
        overwrite=True,
    )

    if append_back_to_bank:
        with np.load(bank_path, allow_pickle=True) as d:
            bank = {k: d[k] for k in d.files}

        bank[_bank_key(t_next, "pairs")] = np.asarray(children_pairs, dtype=float)
        bank[_bank_key(t_next, "reps_states")] = np.asarray([[st] for st in children_states], dtype=object)
        bank[_bank_key(t_next, "reps_Iseries")] = np.asarray([[np.asarray(x, dtype=int)] for x in children_Iseries], dtype=object)

        bank[_bank_key(t_next, "pair_runs")] = np.asarray(pair_runs, dtype=int)
        bank[_bank_key(t_next, "pair_betas")] = np.asarray(pair_betas, dtype=float)
        bank[_bank_key(t_next, "pair_A_dist")] = np.asarray(pair_A_dist, dtype=float)

        bank[_bank_key(t_next, "rep_pairs_flat")] = np.asarray(children_pairs, dtype=float)
        bank[_bank_key(t_next, "rep_betas_all")] = np.asarray(betas_all_next, dtype=float)
        bank[_bank_key(t_next, "rep_A_dists_all")] = np.asarray(dA_all_next, dtype=float)
        bank[_bank_key(t_next, "rep_I_all")] = np.asarray(I_sim_all_next, dtype=int)

        bank[_bank_key(t_next, "A_half_betas")] = np.asarray(b_half_next, dtype=float)
        bank[_bank_key(t_next, "A_half_dists")] = np.asarray(d_half_next, dtype=float)
        bank[_bank_key(t_next, "A_half_I_sim")] = np.asarray(I_sim_half_next, dtype=int)
        bank[_bank_key(t_next, "A_half_mean_beta")] = np.asarray([mean_b_half_next], dtype=float)
        bank[_bank_key(t_next, "A_half_mean_beta_ci")] = np.asarray([lo_next, hi_next], dtype=float)
        bank[_bank_key(t_next, "A_half_weights")] = np.asarray(_w, dtype=float)

        bank[_bank_key(t_next, "weights_all")] = np.asarray(children_weights, dtype=float)
        bank[_bank_key(t_next, "half_pairs")] = np.asarray(half_pairs_next, dtype=float)
        bank[_bank_key(t_next, "half_reps_states")] = np.asarray(half_states_next, dtype=object)
        bank[_bank_key(t_next, "half_reps_Iseries")] = np.asarray(half_Iseg_next, dtype=object)

        bank[_bank_key(t_next, "meta")] = np.frombuffer(
            json.dumps(meta_json).encode("utf-8"),
            dtype=np.uint8,
        )

        if os.path.exists(bank_path) and overwrite:
            os.remove(bank_path)

        np.savez(bank_path, **bank)

    return dict(
        t_prev=t_prev,
        t_next=t_next,
        children_count=len(children_pairs),
        all_pairs=dict(
            betas_A=betas_all_next,
            dA=dA_all_next,
        ),
        half_at_t_next=dict(
            betas=b_half_next,
            d=d_half_next,
            mean_beta=mean_b_half_next,
            ci=(lo_next, hi_next),
        ),
        pool_pairs_out=pool_pairs_out,
        pool_states_out=pool_states_out,
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
        print(f"Current simulation: t={current} → t={nxt}      ", end="\r", flush=True)

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
    print
    return last_summary


def _save_fig(fig, save_path: Optional[str]):
    if not save_path:
        return
    save_dir = os.path.dirname(os.path.abspath(save_path))
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")



def plot_half_simulations(
    *, betas_A, dA, mean_beta_A: float | None,
    title: str, xlim=None, xtick=None,
    save_path: Optional[str] = None, show: bool = True, close: bool = False,
):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(betas_A, dA, alpha=0.85, label='Distance distribution from center ≤ median')
    if mean_beta_A is not None and np.isfinite(mean_beta_A):
        ax.axvline(mean_beta_A, linestyle='--', linewidth=1.6, label=f'β Dist-Diff {mean_beta_A:.3f}')
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    if xtick is not None:
        ax.set_xticks(xtick if not np.isscalar(xtick) else [float(xtick)])
    ax.set_xlabel('β'); ax.set_ylabel('Distance')
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


def plot_half(bank_path: str, t: int, title: Optional[str] = None, xlim=None, xtick=None,
              save_path: Optional[str] = None, show: bool = True, close: bool = False):
    keyA = _bank_key(t, "A_half_betas")
    keyAd = _bank_key(t, "A_half_dists")
    keyAm = _bank_key(t, "A_half_mean_beta")
    bank_path = _load_path_for_plots(bank_path)
    with np.load(bank_path, allow_pickle=True) as d:
        betas_A = d[keyA] if keyA in d.files else np.array([], float)
        dA      = d[keyAd] if keyAd in d.files else np.array([], float)
        meanA   = float(d[keyAm][0]) if keyAm in d.files and d[keyAm].size else np.nan
    return plot_half_simulations(
        betas_A=betas_A, dA=dA, mean_beta_A=meanA,
        title=title or f"Lower-half at t={t}", xlim=xlim, xtick=xtick,
        save_path=save_path, show=show, close=close
    )




def plot_all_runs(
    *, betas_A, dA, title: str, xlim=(0, 0), xtick=None,
    save_path: Optional[str] = None, show: bool = True, close: bool = False
):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(betas_A, dA, alpha=0.85, s=50, label="Dist-Diff")
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

    ax.set_xlabel('β'); ax.set_ylabel('Distance')
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


def run_t20_return_distances(
    *, observed_csv: str, bank_pattern: str, num_infected: int = 1, num_steps: int = 20,
    beta_start: float = 0.4, beta_step: float = 0.01, beta_count: int = 20,
    center=(0, 0), start_ring: int = 1, strict: bool = True,
):
    _simulations_num_steps(bank_pattern, num_steps)
    _no_observed_at_time(observed_csv, num_steps)

    betas = np.round(beta_start + np.arange(beta_count, dtype=float) * beta_step, 6)
    bank_path = bank_pattern
    if not os.path.exists(bank_path):
        raise FileNotFoundError(f"Bank file not found: {bank_path}")

    with np.load(bank_path, allow_pickle=True) as d:
        S_all = d["S"] if "S" in d.files else d["final_status"]
        n_runs = S_all.shape[0] if S_all.ndim == 3 else 1

    observed_dists = get_infected_distances_from_csv(observed_csv, center=center, t_obs=num_steps)

    dhist_rows = []
    for run_idx in range(n_runs):
        sim_distances_by_beta = {}
        for b in betas:
            dlist = _read_sim_distances(
                beta=float(b), num_steps=num_steps, center=center,
                bank_path=bank_path, run_idx=run_idx, retries=2
            )
            if dlist is not None:
                sim_distances_by_beta[float(b)] = dlist
        if sim_distances_by_beta:
            results_A, _ = estimate_beta_distances_using_precomputed(
                observed_distances=observed_dists,
                sim_distances_by_beta=sim_distances_by_beta,
                start_ring=start_ring,
            )
            mp = {float(bb): float(dd) for bb, dd in results_A}
            dhist_rows.append([mp.get(float(b), np.nan) for b in betas])

    if not dhist_rows:
        raise RuntimeError("error.")

    dhist_mean = np.nanmean(np.array(dhist_rows, float), axis=0)
    dhist = list(zip(betas.tolist(), dhist_mean.tolist()))
    return dhist


def t20_distance(
    *, observed_csv: str, bank_path: str, num_infected: int = 1, num_steps: int = 20,
    beta_start: float = 0.4, beta_step: float = 0.01, beta_count: int = 20,
    center=(0, 0), start_ring: int = 1, strict: bool = True,
):
    dhist = run_t20_return_distances(
        observed_csv=observed_csv, bank_pattern=bank_path,
        num_infected=num_infected, num_steps=num_steps,
        beta_start=beta_start, beta_step=beta_step, beta_count=beta_count,
        center=center, start_ring=start_ring, strict=strict,
    )
    betas_A = np.array([b for b, _ in dhist], dtype=float)
    dA      = np.array([d for _, d in dhist], dtype=float)
    return betas_A, dA


def _find_times_with_mean_betas(bank_path: str) -> List[int]:
    bank_path = _load_path_for_plots(bank_path)
    with np.load(bank_path, allow_pickle=True) as d:
        pat = re.compile(r"^t(\d+)_A_half_mean_beta$")
        return sorted(int(m.group(1)) for k in d.files for m in [pat.match(k)] if m)


def load_mean_betas_from_simulations(bank_path: str, times: Optional[List[int]] = None):
    bank_path = _load_path_for_plots(bank_path)
    if times is None:
        times = _find_times_with_mean_betas(bank_path)
    meanA = []
    with np.load(bank_path, allow_pickle=True) as d:
        for t in times:
            keyA = _bank_key(t, "A_half_mean_beta")
            a = float(d[keyA][0]) if keyA in d.files and d[keyA].size else np.nan
            meanA.append(a)
    return np.asarray(times, int), np.asarray(meanA, float)


def _load_beta_from_csv(csv_path: str):
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    cols = {c.lower(): c for c in df.columns}

    time_candidates = ["time", "t""]
    beta_candidates = ["beta", "b", "beta_mean", "beta_hat", "est_beta"]

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


def _interp_nans(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(y)
    if m.sum() == 0:
        return np.full_like(y, np.nan, dtype=float)
    if m.sum() == 1:
        return np.full_like(y, y[m][0], dtype=float)
    yi = y.copy()
    yi[~m] = np.interp(x[~m], x[m], y[m])
    return yi


def plot_all_and_half(
    bank_path: str, t: int,
    title: Optional[str] = None, xlim=None, xtick=None,
    save_path: Optional[str] = None, show: bool = True, close: bool = False
):
    return plot_all_vs_half(
        bank_path=bank_path,
        t=t,
        title=title,
        xlim=xlim,
        xtick=xtick,
        save_path=save_path,
        show=show,
        close=close,
    )


def plot_all_vs_half(
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

    fig, ax = plt.subplots(figsize=(8, 5))
    

    betas_A_all, dA_all = get_all_simulations(bank_path, t, level="rep")
    ax.scatter(betas_A_all, dA_all, alpha=0.85, s=50, color='gold', label="Unselected runs")

    keyA  = _bank_key(t, "A_half_betas")
    keyAd = _bank_key(t, "A_half_dists")

    with np.load(bank_path, allow_pickle=True) as d:
        betas_A_half = d[keyA]  if keyA  in d.files else np.array([], float)
        dA_half      = d[keyAd] if keyAd in d.files else np.array([], float)

    est_beta = np.nan
    if betas_A_half.size and dA_half.size:
        est_beta, lo, hi, w = _weighted_beta_from_half(
            np.asarray(betas_A_half, float),
            np.asarray(dA_half, float),
            scale=2.0,
            temp=0.3,
        )
    else:
        w = None

    ax.scatter(betas_A_half, dA_half, color='green', alpha=0.85, s=50, label='Selected runs')

    if np.isfinite(est_beta):
        ax.axvline(
            est_beta,
            linestyle='--',
            linewidth=1.6,
            color='red',
            label=rf'Estimated $\beta^* = {est_beta:.3f}$'
        )

    ax.set_title(title or f"All runs vs Lower-half at t={t}")
    if xlim is not None:
        ax.set_xlim(xlim)
    if xtick is not None:
        ax.set_xticks(xtick if not np.isscalar(xtick) else [float(xtick)])
    ax.set_xlabel('β'); ax.set_ylabel('Distance')
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


def plot_mean_betas(
    bank_path,
    title="Estimated β",
    overlay_csv_path=None,
    save_path="figs/mean_beta.pdf",
    xlim=None, ylim=None, xtick=None, ytick=None,
    fig_size=(8, 4),
    show=True,
):
    spine_sides=None
    spine_width=1.2
    spine_out=0
    bank_path = _load_path_for_plots(bank_path)
    ts, meanA = load_mean_betas_from_simulations(bank_path)
    if ts.size == 0:
        raise RuntimeError("No mean-beta keys found in bank.")

    A_lo, A_hi = [], []
    with np.load(bank_path, allow_pickle=True) as d:
        for t in ts:
            kAci = _bank_key(int(t), "A_half_mean_beta_ci")
            if kAci in d.files and d[kAci].size == 2:
                loA, hiA = float(d[kAci][0]), float(d[kAci][1])
            else:
                loA, hiA = np.nan, np.nan
            A_lo.append(loA)
            A_hi.append(hiA)

    df = pd.DataFrame({
        "time": ts.astype(float),
        "meanA": meanA.astype(float),
        "A_lo": np.asarray(A_lo, float),
        "A_hi": np.asarray(A_hi, float),
    })

    overlay_df = None
    if overlay_csv_path:
        try:
            csv_t, csv_b = _load_beta_from_csv(overlay_csv_path)
            overlay_df = pd.DataFrame({"time": csv_t, "overlay": csv_b})
        except Exception:
            overlay_df = None

    fig, ax = plt.subplots(figsize=fig_size)
    ax.fill_between(df["time"], df["A_lo"], df["A_hi"], color="#21FF3E", alpha=0.5, label='95% CI')

    ax.plot(df["time"], df["meanA"], color="#06D422", linewidth=2.5, label=r'Estimated $\beta$')
    ax.scatter(df["time"], df["meanA"], color="#2C1D46", s=10, zorder=3)

    if overlay_df is not None:
        ax.plot(overlay_df["time"], overlay_df["overlay"],
                color='red', linewidth=1.0, linestyle='--', label=r'Actual $\beta$')

    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\beta$")
    ax.set_title(title, pad=12)

    ax.legend(loc="upper right", frameon=False)

    wanted = {"left", "bottom"}
    for side, spine in ax.spines.items():
        vis = side in wanted
        spine.set_visible(vis)
        if vis:
            spine.set_linewidth(float(spine_width))
            if spine_out:
                spine.set_position(("outward", float(spine_out)))

    ax.xaxis.set_ticks_position("bottom" if "bottom" in wanted else "none")
    ax.yaxis.set_ticks_position("left"   if "left"   in wanted else "none")

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xtick is not None:
        ax.set_xticks(xtick)
    if ytick is not None:
        ax.set_yticks(ytick)

    ax.legend(loc='upper right', frameon=False)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[PY] Saved: {save_path}")

    if show:
        plt.show()
    plt.close(fig)


def export_mean_betas_to_csv(bank_path: str, out_csv: str) -> None:
    bank_path = _load_path_for_plots(bank_path)
    ts, meanA = load_mean_betas_from_simulations(bank_path)
    if getattr(ts, "size", 0) == 0:
        raise RuntimeError("No mean-beta keys found in bank.")

    A_lo, A_hi = [], []
    with np.load(bank_path, allow_pickle=True) as d:
        for t in ts:
            kAci = _bank_key(int(t), "A_half_mean_beta_ci")
            if kAci in d.files:
                arr = np.asarray(d[kAci]).ravel()
                if arr.size >= 2 and np.isfinite(arr[:2]).all():
                    loA, hiA = float(arr[0]), float(arr[1])
                else:
                    loA, hiA = np.nan, np.nan
            else:
                loA, hiA = np.nan, np.nan
            A_lo.append(loA)
            A_hi.append(hiA)

    df = pd.DataFrame({
        "time":       np.asarray(ts, float),
        "beta_mean":  np.asarray(meanA, float),
        "beta_ci_lo": np.asarray(A_lo, float),
        "beta_ci_hi": np.asarray(A_hi, float),
    })
    df.to_csv(out_csv, index=False)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Windowed continuation driver (β grid; A metric, C-style flow).")
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
        t_start=args.t_start, t_final=args.t_final, window=args.window,
        num_infected=args.num_infected,
        beta_start=args.beta_start, beta_step=args.beta_step, beta_count=args.beta_count,
        center=(args.center_x, args.center_y), start_ring=args.start_ring,
        gamma=args.gamma, run_replicates=args.run_replicates,
        append_back_to_bank=True,
    )


if __name__ == "__main__":
    main()
