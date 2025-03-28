import os
import warnings
import sdeint
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import pytensor.tensor as pt
import arviz as az
from pymc.distributions.continuous import Normal
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
import scipy.io as sio
import tvEM
import pytensor


# Define the folder name
folder_name = "compile"

# Check if the folder exists in the current directory; if not, create it.
if not os.path.isdir(folder_name):
    os.makedirs(folder_name)
    print(f"The folder '{folder_name}' was created.")
else:
    print(f"The folder '{folder_name}' already exists.")
plt.style.use("ggplot")
# Ignore UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_SEED = 7000
np.random.seed(RANDOM_SEED)
dt = 0.01



folder_path = '../embedding'  # Replace with the path to your folder
files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

def model_func(t, alpha,t0):
    return  -1*np.tanh(alpha * (t - t0))

# Define the SDE function for the coupled system
def cubic_sde(x,t,t0,alpha,sigma):
 
    # Drift terms
    beta = pt.tanh(alpha * (t-t0))
    dx_drift = -((x + 1) * (x - beta) * (x - 1))

    # Diffusion terms
    dx_diffusion = sigma# + sigma*pt.abs(x)
    return dx_drift, dx_diffusion


# Define the SDE function for the coupled system
def cubic_sde(x,t,t0,alpha,sigma):
 
    # Drift terms
    beta = pt.tanh(alpha * (t-t0))
    dx_drift = -((x + 1) * (x - beta) * (x - 1))

    # Diffusion terms
    dx_diffusion = sigma# + sigma*pt.abs(x)
    return dx_drift, dx_diffusion
for f_idx,filename in enumerate(files):
    f_idx = 0
    filename = files[0]
    file_path = os.path.join(folder_path,filename)
    mu = sio.loadmat(file_path)
    mu = mu['ds_mu']
    print(len(mu))
    N = len(mu)
    t = np.linspace(0,N-1,N)*dt
    popt, pcov = curve_fit(model_func, t, mu[:,0], p0=None)
    alpha_fit, t0_fit = map(lambda x: round(x, 1), popt)
    print(f"prior_alpha = {alpha_fit} prior_t0 = {t0_fit}")
    with pm.Model() as model:
        alpha = pm.HalfNormal("alpha", sigma = 1)
        sigma = pm.HalfNormal("sigma", sigma = 2)
        t0 = pm.Normal("t0", mu = t0_fit, sigma=1)
        tspan = pt.as_tensor_variable(t) 
        state = tvEM.tv_EulerMaruyama(
            "state",
            dt=dt,
            sde_fn=cubic_sde,
            sde_pars=(t0,alpha,sigma),
            shape = N, # I don't understand the Initial conditon here put values across all time span
            initval = mu[:,0],
            tspan = tspan     # init_dist  = init_dist
        )
        observed_x_var = pm.Normal("observed_x", mu = state, sigma = 0.5, observed=mu[:,0])
        with model:
            trace = pm.sample(nuts_sampler="nutpie", random_seed=RANDOM_SEED, target_accept=0.99)
            summary_alpha = az.summary(trace, var_names=["alpha"], hdi_prob=0.94)  # Adjust hdi_prob as needed
            mean_alpha = summary_alpha.loc["alpha", "mean"]
            # Extract the posterior samples of alpha
            alpha_samples = az.extract(trace.posterior)["alpha"].values.flatten()
            
            # Perform KDE on the samples
            kde = gaussian_kde(alpha_samples)
            
            # Generate a grid of alpha values over which to evaluate the KDE
            alpha_grid = np.linspace(alpha_samples.min(), alpha_samples.max(), 1000)
            kde_values = kde(alpha_grid)
            
            # Find the alpha value corresponding to the maximum of the KDE
            mode_alpha = alpha_grid[np.argmax(kde_values)]

            # Extract mean_t0
            summary_t0 = az.summary(trace, var_names=["t0"], hdi_prob=0.94)  # Adjust hdi_prob as needed
            mean_t0 = summary_t0.loc["t0", "mean"]

            # Extract mean_sigma
            summary_sigma = az.summary(trace, var_names=["sigma"], hdi_prob=0.94)  # Adjust hdi_prob as needed
            mean_sigma = summary_sigma.loc["sigma", "mean"]
    # with pm.Model() as model: