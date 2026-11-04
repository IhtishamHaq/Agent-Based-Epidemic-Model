import os
import json
import numpy as np

def distance_at_time(observed_I20, simulated_I20):
    return abs(observed_I20 - simulated_I20)

def estimate_beta_distances(
    observed_I20,
    #world_size,
    #num_agents,
    num_infected=None,
    num_steps=20,
    t_obs=20,
    bank_path="abm_bank_all.npz",
    strict=True,
    run_idx=None,
):
    def _from_single_file(path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"error."
            )
        bank = np.load(path, allow_pickle=False)
        betas = bank["betas"]
        I_mat = bank["I"]
        meta  = json.loads(bank["meta"].tobytes().decode("utf-8"))

        def _check(key, expected):
            if key in meta and meta[key] != expected:
                msg = f"error"
                if strict:
                    raise ValueError(msg)
                else:
                    print("⚠️", msg)

      
        if num_infected is not None:
            _check("num_infected", num_infected)
        _check("num_steps", num_steps)

   
        if I_mat.ndim == 3:
            if run_idx is None:
                I_mat = np.nanmean(I_mat, axis=0)
            else:
                I_mat = I_mat[int(run_idx)]

        T = I_mat.shape[1]
        idx_t = int(t_obs)
        if idx_t < 0:
            idx_t = 0
        if idx_t >= T:
            if strict:
                raise ValueError(f"error")
            else:
                print(f"error")
                idx_t = T - 1

        distances = []
        for beta, I_series in zip(betas, I_mat):
            I_sim_t = float(I_series[idx_t])
            if np.isnan(I_sim_t):
                continue
            dist = distance_at_time(observed_I20, I_sim_t)
            distances.append((float(beta), dist))
        return distances

   
    return _from_single_file(bank_path)
