# Beta_Distance_Dostributions.py
import os
import json
import numpy as np
import pandas as pd
from collections import Counter


def get_infected_distances_from_csv(csv_path, center=(0, 0), t_obs: int | None = None, tol=1e-9):
    df = pd.read_csv(csv_path, low_memory=False)

    
    cols = {c.lower().strip(): c for c in df.columns}
    def col(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    
    status_col = col("status")
    if status_col:
        df[status_col] = df[status_col].astype(str).str.strip().str.upper()

    
    time_col = col("time", "t")
    if t_obs is not None and time_col is not None:
        t_raw = df[time_col].astype(str).str.strip().str.replace(",", "", regex=False)
        t_num = pd.to_numeric(t_raw, errors="coerce")
        m = np.isfinite(t_num) & (
            (np.abs(t_num - float(t_obs)) <= tol) |
            (np.isclose(np.rint(t_num), float(int(t_obs)), rtol=0.0, atol=0.0))
        )
        df = df[m]
        if df.empty:
            raise ValueError(f"error")
    elif t_obs is not None and time_col is None:
        raise KeyError("error.")

  
    x_col = col("x"); y_col = col("y")
    if x_col is None or y_col is None:
        raise KeyError("error")

    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m].astype(int).to_numpy()
    y = y[m].astype(int).to_numpy()

   
    if status_col:
        mI = df.loc[m, status_col].eq("I").to_numpy()
        x, y = x[mI], y[mI]

    d = np.abs(x - int(center[0])) + np.abs(y - int(center[1]))
    return d.astype(int).tolist()


def _load_bank(data_path="abm_data_all.npz"):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"error.")
    bank = np.load(data_path, allow_pickle=False)
    meta = json.loads(bank["meta"].tobytes().decode("utf-8")) if "meta" in bank.files else {}
    return bank, meta


def _infected_code_from_meta(meta):
    if "status_to_code" in meta and isinstance(meta["status_to_code"], dict):
        return int(meta["status_to_code"].get("I", 1))
    if "code_to_status" in meta and isinstance(meta["code_to_status"], dict):
        c2s = {int(k): v for k, v in meta["code_to_status"].items()}
        for k, v in c2s.items():
            if v == "I":
                return int(k)
    return 1


def _sanity_check_bank(bank):
    fx, fy, fstat = bank["final_x"], bank["final_y"], bank["final_status"]
    I = bank["I"]
    if fx.ndim not in (2, 3):
        raise AssertionError(f"final_* must be 2D or 3D, got ndim={fx.ndim}")
    if not (fx.shape == fy.shape == fstat.shape):
        raise AssertionError("final_x/final_y/final_status shapes must match.")

 
    if I.ndim not in (2, 3):
        raise AssertionError(f"error")

  
    betas = bank["betas"]
    if fx.ndim == 3:
        if not (I.shape[0] == fx.shape[0] and I.shape[1] == fx.shape[1] and betas.shape[0] == fx.shape[1]):
            raise AssertionError("error.")
    else:
        if not (I.shape[0] == fx.shape[0] and betas.shape[0] == fx.shape[0]):
            raise AssertionError("error.")


def run_simulation_and_get_infected_distances(
    beta,
    num_steps,          
    num_infected,      
    center=(0, 0),
    bank_path="abm_data_all.npz",
    run_idx: int = 0,   
):
  
    import glob

    
    if isinstance(bank_path, str) and ("*" in bank_path or "?" in bank_path):
        paths = sorted(glob.glob(bank_path))
    elif isinstance(bank_path, (list, tuple)):
        paths = list(bank_path)
    else:
        paths = [bank_path]

    if not paths:
        raise FileNotFoundError(f"error")

    last_error = None
    for path in paths:
        try:
            bank, meta = _load_bank(path)
            _sanity_check_bank(bank)

            betas = bank["betas"].astype(float)
            idxs = np.where(np.isclose(betas, float(beta), atol=1e-9))[0]
            if len(idxs) == 0:
                continue
            b_idx = int(idxs[0])

            fx, fy, fstat = bank["final_x"], bank["final_y"], bank["final_status"]
            infected_code = _infected_code_from_meta(meta)

            if fx.ndim == 3:
                n_runs = fx.shape[0]
                if not (0 <= run_idx < n_runs):
                    raise IndexError(f"error")
                X = fx[run_idx, b_idx]
                Y = fy[run_idx, b_idx]
                S = fstat[run_idx, b_idx]
            else:
                X = fx[b_idx]
                Y = fy[b_idx]
                S = fstat[b_idx]

            mask = (S.astype(int) == int(infected_code))
            if not np.any(mask):
                return [], None

            cx, cy = int(center[0]), int(center[1])
            dists = (np.abs(X[mask] - cx) + np.abs(Y[mask] - cy)).astype(int).tolist()
            return dists, None  

        except Exception as e:
            last_error = e
            continue

    if last_error is not None:
        raise RuntimeError(f"error")
    raise ValueError(f"error")


def estimate_beta_distances_using_precomputed(
    observed_distances,
    sim_distances_by_beta,
    start_ring=1
):
    obs = np.asarray(observed_distances, dtype=int).ravel().tolist()

    observed_counter = Counter(obs)
    results = []
    history_by_beta = {}

    for beta, sim_distances in sim_distances_by_beta.items():
        sim = np.asarray(sim_distances, dtype=int).ravel().tolist()
        sim_counter = Counter(sim)

        if (len(obs) > 0) or (len(sim) > 0):
            max_ring = max(
                (max(obs) if len(obs) > 0 else 0),
                (max(sim) if len(sim) > 0 else 0)
            )
            rings = range(int(start_ring), int(max_ring) + 1)
            rows = []
            distance_sum = 0
            for k in rings:
                a = observed_counter.get(k, 0)
                b = sim_counter.get(k, 0)
                d = abs(a - b)
                rows.append((k, a, b, d))
                distance_sum += d
        else:
            rows = []
            distance_sum = 0

        results.append((float(beta), float(distance_sum)))
        history_by_beta[float(beta)] = rows

    results.sort(key=lambda x: x[0])
    return results, history_by_beta
