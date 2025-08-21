import os
import warnings
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import pytensor.tensor as pt
import arviz as az
from scipy.stats import gaussian_kde
import scipy.io as sio
import tvEM

# Create folder storing estimated parameters
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
est_param_dir = os.path.join(parent_dir, 'est_param')
os.makedirs(est_param_dir, exist_ok=True)

# Ignore UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_SEED = 7000
np.random.seed(RANDOM_SEED)

dt = 0.01
param_names = ('alpha','t0','sigma')
folder_path = os.path.join(parent_dir, 'embedding')
files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
param_vals = np.zeros((len(param_names),len(files)))

def cubic_sde(x,t,t0,alpha,sigma,epislon):
    # Drift terms
    beta = pt.tanh(alpha*epislon*(t-t0))
    dx_drift = -((x + 1) * (x - beta) * (x - 1))*epislon
    dx_diffusion = sigma*pt.sqrt(epislon)#+ sigma*pt.abs(x)
    return dx_drift, dx_diffusion

for f_idx,filename in enumerate(files):
    file_path = os.path.join(folder_path,filename)
    mu = sio.loadmat(file_path)
    mu = mu['ds_mu']
    print(len(mu))
    N = len(mu)
    t = np.linspace(0,N-1,N)*dt
    with pm.Model() as model:
        
        # Priors of parameters
        alpha = pm.HalfNormal("alpha", sigma = 1)
        sigma = pm.HalfNormal("sigma", sigma = 2)
        obs_noise = pm.Uniform("obs_noise", lower = 0.1, upper = 0.4)
        t0 = pm.Normal("t0", mu = t[-1]/2, sigma = t[-1]/4)
        epislon = pm.Uniform("epislon",lower = 0.1,upper = 5)
        
        tspan = pt.as_tensor_variable(t) 
        state = tvEM.tv_EulerMaruyama(
            "state",
            dt=dt,
            sde_fn=cubic_sde,
            sde_pars=(t0,alpha,sigma,epislon),
            shape = N, 
            initval = mu[:,0],
            tspan = tspan     # init_dist  = init_dist
        )
        observed_x_var = pm.Normal("observed_x", mu = state, sigma = obs_noise, observed=mu[:,0])
        with model:
            trace = pm.sample(nuts_sampler="nutpie", random_seed=RANDOM_SEED, target_accept=0.99)   
            for param_idx,param in enumerate(param_names):
                samples =  az.extract(trace.posterior)[param].values.flatten()
                grid = np.linspace(samples.min(), samples.max(), 1000)
                kde = gaussian_kde(samples)
                kde_values = kde(grid)
                param_vals[param_idx,f_idx] = grid[np.argmax(kde_values)]
                
param_file = os.path.join(est_param_dir,'param_real.npy')
np.save(param_file,param_vals)
