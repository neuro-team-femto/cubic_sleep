import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np

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
# `est_param`: long-format results table for the **time-varying landscape**
# Each row = one (trajectory × parameter) estimate.
# Columns:
#   - n_traj (int): index of the simulated trajectory (0..num_traj-1).
#   - param (str): parameter estimated in this row ('alpha', 't0', or 'sigma').
#   - alpha (float): α used in the simulation (β(t) = tanh(α·(t - t0))).
#   - t0 (float): t0 used in the simulation (β(t) = tanh(α·(t - t0))).
#   - sigma (float): σ (noise level) used for the simulation.
#   - true_value (float): ground-truth value for the parameter named in `param`.
#   - est_value (float): the estimator’s value for that parameter and trajectory.
# -----------------------------------------------------------------------------
est_param = pd.read_pickle("../est_param/param_sim_tv_landscape.pkl")

# Choose two more "neutral" colors:
true_color = "gray"    #"#0072BD"    
est_color  = "#D95319"    



fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12,7), constrained_layout=True)

##############################################################################
# 1) t0 ESTIMATION
##############################################################################
t0_est = est_param[est_param.param == 't0'].copy()
t0_est['true_str'] = t0_est['true_value'].astype(str)

unique_t0 = sorted(t0_est['true_value'].unique())
tick_positions_t0 = {str(val): i for i, val in enumerate(unique_t0)}
x_connect_t0 = []
y_connect_t0 = []
for val in unique_t0:
    cat_str = str(val)
    x_connect_t0.append(tick_positions_t0[cat_str])
    y_connect_t0.append(val)

# Plot the "true" values (line+markers) in darkblue
axs[0,0].plot(
    x_connect_t0, y_connect_t0,
    marker='o',
    linestyle='dashed',
    color=true_color,
    label='True (t0)',
    zorder=2
)

# Overlay the estimated values in slategray
sns.pointplot(
    data=t0_est,
    x='true_str', y='est_value',
    color=est_color,
    join=False,
    ax=axs[0,0],
    legend=False
)

axs[0,0].set_title(r"t$_0$ estimation")
axs[0,0].set_xlabel('true value')
axs[0,0].set_ylabel('estimated value')

##############################################################################
# 2) SIGMA ESTIMATION
##############################################################################
sigma_est = est_param[est_param.param == 'sigma'].copy()
sigma_est['true_str'] = sigma_est['true_value'].astype(str)

unique_sigma = sorted(sigma_est['true_value'].unique())
tick_positions_sigma = {str(val): i for i, val in enumerate(unique_sigma)}
x_connect_sigma = []
y_connect_sigma = []
for val in unique_sigma:
    cat_str = str(val)
    x_connect_sigma.append(tick_positions_sigma[cat_str])
    y_connect_sigma.append(val)

# Plot "true" values in darkblue
axs[0,1].plot(
    x_connect_sigma, y_connect_sigma,
    marker='o',
    linestyle='dashed',
    color=true_color,
    label='True (sigma)',
    zorder=2
)

# Estimated in slategray
sns.pointplot(
    data=sigma_est,
    x='true_str', y='est_value',
    color=est_color,
    join=False,
    ax=axs[0,1],
    legend=False
)

axs[0,1].set_title(r"$\sigma$ estimation")
axs[0,1].set_xlabel('true value')
axs[0,1].set_ylabel('')

# axs[0,1].set_ylabel('estimated value')

##############################################################################
# 3) ALPHA ESTIMATION
##############################################################################
alpha_est = est_param[est_param.param == 'alpha'].copy()
alpha_est['true_str'] = alpha_est['true_value'].astype(str)

unique_alpha = sorted(alpha_est['true_value'].unique())
tick_positions_alpha = {str(val): i for i, val in enumerate(unique_alpha)}
x_connect_alpha = []
y_connect_alpha = []
for val in unique_alpha:
    cat_str = str(val)
    x_connect_alpha.append(tick_positions_alpha[cat_str])
    y_connect_alpha.append(val)

# True in darkblue
axs[1,0].plot(
    x_connect_alpha, y_connect_alpha,
    marker='o',
    linestyle='dashed',
    color=true_color,
    label='True (alpha)',
    zorder=2
)

# Estimates in slategray
sns.pointplot(
    data=alpha_est,
    x='true_str', y='est_value',
    color=est_color,
    join=False,
    ax=axs[1,0],
    legend=False
)

axs[1,0].set_title(r'$\alpha$ estimation')
axs[1,0].set_ylim(0.05, 0.8)  # Force the requested y-limits

# Compute Spearman correlation on the average estimates
mean_alpha = alpha_est.groupby("true_value")["est_value"].mean()
mean_alpha_df = mean_alpha.reset_index()
true_alpha = mean_alpha_df["true_value"].values
mean_est_alpha = mean_alpha_df["est_value"].values

r_res = spearmanr(mean_est_alpha, true_alpha)
r, p = r_res.statistic, r_res.pvalue
# axs[1,0].text(
#     0.05, 0.95,
#     f'Spearman r = {r:.3f}',
#     transform=axs[1,0].transAxes,
#     fontsize=12,
#     verticalalignment='top',
# )

axs[1,0].set_xlabel('true value')
axs[1,0].set_ylabel('estimated value')

##############################################################################
# 4) BETA SHAPE  (KEEP MULTI-COLORS)
##############################################################################
axs[1,1].set_title(r'$\beta$ shape')
N = 10000
dt = 0.01
t = np.linspace(0, N, N) * dt
t0_val = 50
tspan = t - t0_val

alpha_vals = sorted(alpha_est['alpha'].unique())
palette = sns.color_palette('deep', n_colors=len(alpha_vals))

# Plot each alpha in a different color, labeling only with the numeric alpha
for a_idx, alpha_val in enumerate(alpha_vals):
    beta = np.tanh(tspan * alpha_val)
    axs[1,1].plot(tspan, beta, color=palette[a_idx], label=f"{alpha_val}")

# Construct a concise legend (just alpha values) to the right of the beta plot
handles, labels = axs[1,1].get_legend_handles_labels()
axs[1,1].legend(
    handles, labels,
    title='alpha',
    loc='center left',
    bbox_to_anchor=(1.02, 0.5),  # Slightly right of the axes
    borderaxespad=0,
)
axs[1,1].set_xlabel('time')
axs[1,1].set_ylabel(r'$\beta(t)$')

plt.savefig('../figures/fig6_tv_landscape.png',dpi = 300)