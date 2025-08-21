import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Get the directory of the current file
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Arial"
plt.style.use("ggplot")
plt.rcParams.update({'font.size': 11})

# -----------------------------------------------------------------------------
# `est_param`: results table for the **fixed landscape** experiments
# Each row = one (trajectory × parameter) estimate.
# Columns:
#   - n_traj (int): index of the simulated trajectory (0..num_traj-1).
#   - X0 (float): initial condition used for that trajectory.
#   - param (str): which parameter this row refers to ('beta' or 'sigma').
#   - beta (float): true β used for the simulation of this row.
#   - sigma (float): true σ (noise level) used for the simulation of this row.
#   - true_value (float): ground-truth value of the parameter named in `param`.
#   - est_value (float): estimator output (the parameter estimate) for that row.
est_param = pd.read_pickle("../est_param/param_sim_fixed_landscape.pkl")

param_name = ['beta','sigma']
plt.rcParams["font.family"] = "Arial"
plt.style.use("ggplot")

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(11,5), constrained_layout=True)
# Choose two more "neutral" colors:
true_color = "gray"    #"#0072BD"    
est_color  = "#D95319"  
# Desired ticks for each subplot
desired_ticks = [
    [-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8],  # for beta
    [0.2, 0.5, 1.0]                                   # for sigma
]

for p_idx, param in enumerate(param_name):
    param_est = est_param[est_param.param == param].copy()
    param_est['true_str'] = param_est['true_value'].astype(str)

    # --- 1) PLOT TRUE VALUES FIRST (in gray) ---
    unique_param = sorted(param_est['true_value'].unique())
    # Map each numeric true_value to x-position in the categorical plot
    tick_positions = {str(val): i for i, val in enumerate(unique_param)}
    
    x_connect = []
    y_connect = []
    for val in unique_param:
        cat_str = str(val)
        x_connect.append(tick_positions[cat_str])
        y_connect.append(val)
    
    # Plot the true values as a dashed line in gray:
    axs[p_idx].plot(
        x_connect, y_connect,
        marker='o',
        linestyle='dashed',
        color=true_color,    # Gray for true values
        zorder=2
    )

    # --- 2) OVERLAY ESTIMATED VALUES (in blue) ---
    # Remove 'hue' so we don't color by beta or sigma
    # Use color='blue' for all estimated points
    sns.pointplot(
        data=param_est,
        x='true_str', y='est_value',
        color= est_color,
        join=False,
        errorbar="sd",
        ax=axs[p_idx],
        legend=False
    )

    # Set y-ticks
    axs[p_idx].set_yticks(desired_ticks[p_idx])

    # Optionally compute correlation
    mean_value = param_est.groupby("true_value")["est_value"].mean()
    mean_df = mean_value.reset_index()
    true_df = mean_df["true_value"].to_numpy()
    mean_est = mean_df["est_value"].to_numpy()
    corr_coef, p_value = pearsonr(true_df, mean_est)

    # Axis labels/titles
    axs[p_idx].set_xlabel('true value')
    if p_idx == 0:
        axs[p_idx].set_ylabel('estimated value')
    else:
        axs[p_idx].set_ylabel(' ')

axs[0].set_title(r'$\beta$ estimation')
axs[1].set_title(r'$\sigma$ estimation')

plt.savefig('../figures/fig5_fixed_landscape.png',dpi = 300)