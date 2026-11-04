import os
import json
import numpy as np
import pandas as pd

def estimate_beta_distances_by_position_match(
    num_infected,
    num_steps,
    data_path="abm_data_1.npz",
    observed_csv="agent_data.csv",
    strict=True,
):
   
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"error."
        )

    bank = np.load(data_path, allow_pickle=False)
    betas = bank["betas"]           
    fx    = bank["final_x"]         
    fy    = bank["final_y"]          
    fstat = bank["final_status"]     
    meta  = json.loads(bank["meta"].tobytes().decode("utf-8"))

  
    def _check(key, expected):
        if key in meta and meta[key] != expected:
            msg = f"error"
            if strict:
                raise ValueError(msg)
            else:
                print("error", msg)

  
    _check("num_infected", num_infected)
    _check("num_steps", num_steps)

   
    infected_code = meta.get("status_to_code", {}).get("I", 1)

  
    observed_data = pd.read_csv(observed_csv)
    observed_data.columns = observed_data.columns.str.strip().str.lower()
    observed_data["status"] = observed_data["status"].str.strip().str.upper()

    observed_I_positions = set(zip(
        observed_data.loc[observed_data["status"] == "I", "x"].astype(int),
        observed_data.loc[observed_data["status"] == "I", "y"].astype(int),
    ))

    distances = []
    for beta, x_row, y_row, s_row in zip(betas, fx, fy, fstat):
        sim_I_positions = set(
            (int(x), int(y))
            for x, y, s in zip(x_row, y_row, s_row)
            if int(s) == int(infected_code)
        )
        mismatch = len(sim_I_positions - observed_I_positions) + len(observed_I_positions - sim_I_positions)
        distances.append((float(beta), mismatch))

    return distances
