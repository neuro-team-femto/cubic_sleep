# TODO list (in general)
# 1. Normalize [t_start,tend] to [0,1]
# 2. Loop different params (random generate), not just ground truth.
# 3. Analyze Param est percentage to the number of runs and number of trajectories.
# define a search grid (alpha=0...1, sigma=0..1, t0=0..1)
# note: normalize t0 so it's between 0 and 1 (100%) of the file duration

# training
import scipy.io as sio
import sdeint
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
import warnings
import time
import pickle

warnings.filterwarnings('ignore')

# Load the time course of V2
V = sio.loadmat('..\\201_V.mat')
V = V['V']
V2 = -1*V[:,1]


rho = 1
gamma = -1
# Initial condition
X0 = np.array([1,0])
dt = 0.06
tend = dt * len(V2)
tspan = np.arange(0,tend, dt)

# Setting parameter grid
t0_vals = np.linspace(0.3,0.7,5)
alpha_vals = np.linspace(0.1,0.8,8)
sigma_vals = np.linspace(0.1,0.4,7)
num_trajs  = np.array([4,10,100,500])
num_trajs.astype(int)
num_runs = 10


# Set HMM Model dicts
HMM_Models = [[[[[0 for _ in range(len(sigma_vals))] for _ in range(len(alpha_vals))] for _ in range(len(t0_vals))] for _ in range(len(num_trajs))] for _ in range(num_runs)]

# Define stochastic differential equation
def f_cubic(x, t):
    idx = np.where(tspan1==t)[0][0]
    return np.array([x[1], -1*(((x[0]+1)*(x[0]-beta[idx])*(x[0]-1)) + rho*x[0]**3) + gamma*x[1]])

def G_brown(x, t):
    return np.diag([sigma, sigma]) # d


for traj_idx,num_traj in enumerate(num_trajs): #10
    print(f'traj_idx={traj_idx}')
    num_traj = int(num_traj)
    
    for t0_idx,t0 in enumerate(t0_vals):
        print(f't0_idx={t0_idx}')
        tspan1 = tspan - t0*tend
        for a_idx,alpha in enumerate(alpha_vals):
            beta = np.tanh(alpha*tspan1)
            for s_idx,sigma in enumerate(sigma_vals): # first value of sigma
                # print(f'traj_idx ={traj_idx},t0_idx = {t0_idx} alpha_idx = {a_idx} sigma_idx = {s_idx}')
                x0 = np.empty((0,1))
                lengths = np.empty((0,1))
                for nseq in range(num_traj):
                    obs = sdeint.itoint(f_cubic, G_brown, X0, tspan1)
                    data = obs[:,0][:,None]
                    x0 = np.concatenate((x0,data))
                    lengths = np.append(lengths,len(data))
                lengths = lengths.astype(int)
                for r_idx in range(num_runs): #00100011
                    model = hmm.GaussianHMM(n_components=2,covariance_type="diag",n_iter = 1000)
                    model.fit(x0,lengths)
                    HMM_Models[r_idx][traj_idx][t0_idx][a_idx][s_idx] = model   


with open("HMM_Model.pkl", "wb") as file:
    pickle.dump(HMM_Models,file)