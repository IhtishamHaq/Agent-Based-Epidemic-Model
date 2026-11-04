# betaplot.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_beta(
    csv_file,
    obs_csv="influenza_prefecture_daily_2024_2025.csv",
    obs_col="fukui_cases",
    title="Kyoto",
    save_path="figs_multi/estimated_transmission_kyoto.pdf"
):


    df = pd.read_csv(csv_file)
    obs = pd.read_csv(obs_csv)

    def beta_tilde(x):
        return 4 * (1 - np.exp(-x / 4))

    df["beta_tilde"] = beta_tilde(df["Estimated_Beta"])
    #df["beta_tilde_lo"] = beta_tilde(df["beta_ci_lo"])
    df["beta_tilde_lo"] = np.maximum(-1.0, df["beta_ci_lo"])
    df["beta_tilde_hi"] = beta_tilde(df["beta_ci_hi"])


    df["Week"] = df["Time"] / 7
    obs["Week"] = obs["Time"] / 7

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(obs["Week"], obs[obs_col], marker="*", markersize=16, linestyle="-", label="Observed cases", color="gold")

    ax1.set_xlabel("Time (weeks)", fontsize=16, fontweight="bold")
   # ax1.set_ylabel("Observed data", fontsize=16, fontweight="bold")
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()

    ax2.plot( df["Week"], df["beta_tilde"], marker="o", markersize=7, linestyle="-", label=r"Estimated $\tilde{\beta}_{t}$")

    ax2.fill_between( df["Week"], df["beta_tilde_lo"], df["beta_tilde_hi"], alpha=0.3)

    ax2.set_ylabel(r"Transmission rate $\tilde{\boldsymbol{\beta}}_{t}$",fontsize=16,fontweight="bold")
    
    #ax2.set_ylim(-4, 4)
    #ax2.tick_params(axis="y")
    ax2.set_ylim(-4, 4)
    ax2.set_yticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
    ax2.set_yticklabels(["", "", "", "", "0", "1", "2", "3", "4"])
    ax2.tick_params(axis="y")

    plt.title(title, fontsize=18, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend( lines1 + lines2,labels1 + labels2,loc="upper left")

    ax1.grid(True)

    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    print("Mean beta_tilde =", np.mean(df["beta_tilde"]))

    plt.show()