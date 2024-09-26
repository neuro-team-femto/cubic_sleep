import scipy.io as sio
import sdeint
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
import warnings
import pickle

warnings.filterwarnings('ignore')

# The number of points is equal to the length of V2
num_points = 299

# Setting parameter grid
t0_vals = np.linspace(0.3,0.7,5)
alpha_vals = np.linspace(0.1,0.8,8)
sigma_vals = np.linspace(0.1,0.4,7)
num_trajs  = np.array([4,10,100,500])
num_trajs.astype(int)
num_runs = 10

# Fixed model parameter
rho = 1
gamma = -1
dt = 0.06

# Initial condition
X0 = np.array([1,0])

tend = dt * num_points
tspan = np.arange(0,tend, dt)

shape = (len(t0_vals),len(alpha_vals),len(sigma_vals),len(num_trajs),len(t0_vals),len(alpha_vals),len(sigma_vals),num_runs) 
lh = np.empty(shape)

def f_cubic(x, t):
    idx = np.where(tspan1==t)[0][0]
    return np.array([x[1], -1*(((x[0]+1)*(x[0]-beta[idx])*(x[0]-1)) + rho*x[0]**3) + gamma*x[1]])

def G_brown(x, t):
    return np.diag([sigma, sigma]) # d
    

with open("HMM_Model.pkl", "rb") as file:
    HMM_Models = pickle.load(file)

for t0_idx,t0 in enumerate(t0_vals):
    tspan1 = tspan - t0*tend
    for a_idx,alpha in enumerate(alpha_vals):
        beta = np.tanh(alpha*tspan1)
        for s_idx,sigma in enumerate(sigma_vals):
            # Everytime we get here, we are considering one test sample from one ground truth
            obs = sdeint.itoint(f_cubic, G_brown, X0, tspan1)
            for traj_idx1 in range(len(num_trajs)):
                for t0_idx1,t0 in enumerate(t0_vals):
                    for a_idx1,alpha in enumerate(alpha_vals):
                        for s_idx1,sigma in enumerate(sigma_vals):
                                                # HMM_Models[r_idx][traj_idx][t0_idx][a_idx][s_idx] = model   
                            model_runs = [HMM_Models[i][traj_idx1][t0_idx1][a_idx1][s_idx1] for i in range(len(HMM_Models))]
                            for r_idx in range(len(model_runs)):
                                model = model_runs[r_idx]
                                lh[t0_idx,a_idx,s_idx,traj_idx1,t0_idx1,a_idx1,s_idx1,r_idx] = model.score(obs[:,0][:,None]) # log(P(x_test|HMM)) [-infinity, 0] 
            
sio.savemat('./results/lh.mat',{'lh',lh})
             


