import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def _bank_key(t, name):
    return f"t{int(t)}_{name}"


def _load_path_for_plots(path):
    root, _ = os.path.splitext(path)
    pf = root + "_pf_results.npz"
    return pf if os.path.exists(pf) else path

#weight is coming from the previous time and not update yet this why used current time

def _get_prev_window_time(d, t_current):
    pattern = re.compile(r"^t(\d+)_C_half_weights$")
    times = []

    for k in d.files:
        m = pattern.match(k)
        if m:
            times.append(int(m.group(1)))

    if not times:
        return None

    times = sorted(times)
    prev_times = [tt for tt in times if tt == t_current]

    if not prev_times:
        return None

    return max(prev_times)




def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


def _load_beta_from_csv(csv_path: str):
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    cols = {c.lower(): c for c in df.columns}

    time_candidates = ["time"]
    infected_candidates = ["okinawa_cases"]

    time_col = next((cols[c] for c in time_candidates if c in cols), None)
    infected_col = next((cols[c] for c in infected_candidates if c in cols), None)

    tnum = _to_num(df[time_col])
    inum = _to_num(df[infected_col])

    s = pd.DataFrame({"t": tnum, "infected": inum}).dropna()
    s = s.groupby("t", as_index=False)["infected"].mean().sort_values("t")

    return s["t"].astype(int).to_numpy(), s["infected"].astype(float).to_numpy()


def _weighted_stats(I, w):
    idx = np.argsort(I)
    I_sorted = I[idx]
    w_sorted = w[idx]

    cum_w = np.cumsum(w_sorted)
    total_w = cum_w[-1]

    mean_est = float(np.sum(w * I))
    ci_lo = float(np.interp(0.025 * total_w, cum_w, I_sorted))
    ci_hi = float(np.interp(0.975 * total_w, cum_w, I_sorted))

    return mean_est, ci_lo, ci_hi


def plot_infected_posterior_at_time(bank_path, observed_csv_path, save_path, time_point, bins=25):
    bank_path = _load_path_for_plots(bank_path)

    kI = _bank_key(time_point, "C_half_I_sim")
    kW = _bank_key(time_point, "C_half_weights")

    with np.load(bank_path, allow_pickle=True) as d:

        I = np.asarray(d[kI]).ravel()

        if kW in d.files:
            w = np.asarray(d[kW]).ravel().astype(float)

            print("Current weights sum:", np.sum(w))
            mean_est, ci_lo, ci_hi = _weighted_stats(I, w)
        else:
            ci_lo, ci_hi = np.percentile(I, [2.5, 97.5])

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(I, bins=bins, density=True, weights=w,
            color="tab:red", alpha=0.5, edgecolor="black")

    try:
        kde = gaussian_kde(I, weights=w) if w is not None else gaussian_kde(I)
        xg = np.linspace(I.min(), I.max(), 1000)
        ax.plot(xg, kde(xg), color="blue", linewidth=2.5)
    except:
        pass

    if observed_csv_path:
        t_obs, infected_obs = _load_beta_from_csv(observed_csv_path)
        t_obs_list = t_obs.tolist() * 7
        if time_point in t_obs_list:
            obs_val = infected_obs[t_obs_list.index(time_point)]
            ax.axvline(
                obs_val,
                color="gold",
                linestyle="-",
                linewidth=4,
                label=r"Observed data"
            )

    ax.axvline(mean_est, color="black", linewidth=3, label=f"Mean = {mean_est:.0f}")
    ax.axvline(ci_lo, linestyle="--", color="blue", label="95% CI")
    ax.axvline(ci_hi, linestyle="--", color="blue")
    ax.set_title(rf"Posterior distribution from the particle filter at week $\mathbf{{{time_point/7:g}}}$", fontweight="bold")
    ax.set_xlabel("Infected", fontsize= 14, fontweight="bold")
    ax.set_ylabel("Density", fontsize= 14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()


def plot_infected_with_previous_weights(bank_path, observed_csv_path, save_path, time_point, bins=25):
    bank_path = _load_path_for_plots(bank_path)

    with np.load(bank_path, allow_pickle=True) as d:
        t_prev = _get_prev_window_time(d, time_point)
        t_prevs = _prev_window_time(d, time_point)

        if t_prev is None:
            raise ValueError("No previous weights found")

        kI = _bank_key(time_point, "rep_I_all")
        kW_prev = _bank_key(t_prev, "weights_all")

        I = np.asarray(d[kI]).ravel()
        w = np.asarray(d[kW_prev]).ravel().astype(float)

        w_sum = np.sum(w)
        if w_sum <= 0 or not np.isfinite(w_sum):
            w = np.ones_like(w, dtype=float) / len(w)
        else:
            w = w / w_sum

        print("Previous weights sum:", np.sum(w))
        print("Previous time:", t_prev)

      
        mean_est = float(np.sum(w * I))

        idx = np.argsort(I)
        I_sorted = I[idx]
        w_sorted = w[idx]

        cum_w = np.cumsum(w_sorted)
        cum_w = cum_w / cum_w[-1]

        ci_lo = float(np.interp(0.025, cum_w, I_sorted))
        ci_hi = float(np.interp(0.975, cum_w, I_sorted))
      

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        I,
        bins=bins,
        density=True,
        weights=w,
        color="tab:green",
        alpha=0.5,
        edgecolor="black"
    )

    try:
        kde = gaussian_kde(I, weights=w)
        xg = np.linspace(I.min(), I.max(), 1000)
        ax.plot(xg, kde(xg), color="blue", linewidth=2.5)
    except Exception:
        pass

    if observed_csv_path:
        t_obs, infected_obs = _load_beta_from_csv(observed_csv_path)
        t_obs_list = t_obs.tolist() * 7
        if time_point in t_obs_list:
            obs_val = infected_obs[t_obs_list.index(time_point)]
            ax.axvline(
                obs_val,
                color="gold",
                linestyle="-",
                linewidth=4,
                label=r"Observed data"
            )

    ax.axvline(mean_est, color="black", linewidth=3, label=f"Mean = {mean_est:.0f}")
    ax.axvline(ci_lo, linestyle="--", color="blue", label="95% CI")
    ax.axvline(ci_hi, linestyle="--", color="blue")

    ax.set_title(
        rf"Model forecast distribution for week $\mathbf{{{time_point/7:g}}}$ using weights from week $\mathbf{{{t_prevs/7:g}}}$",
        fontweight="bold"
    )
    ax.set_xlabel("Infected", fontsize= 14, fontweight="bold")
    ax.set_ylabel("Density", fontsize= 14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()