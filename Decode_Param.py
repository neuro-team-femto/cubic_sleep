import pandas as pd
import seaborn as sns
import scipy.io as sio
import numpy as np


t0_vals = np.linspace(0.3,0.7,5)
alpha_vals = np.linspace(0.1,0.8,8)
sigma_vals = np.linspace(0.1,0.4,7)
num_trajs  = np.array([4,10,100,500])
num_trajs.astype(int)
num_runs = 10

lh = sio.loadmat('lh_HMM2.mat')
lh = lh['lh']

true_shape = (3,len(t0_vals),len(alpha_vals),len(sigma_vals))
true_param = np.empty(true_shape)
est_shape = (len(num_trajs),num_runs,3,len(t0_vals),len(alpha_vals),len(sigma_vals))
est_param = np.empty(est_shape)

for t_idx in range(len(t0_vals)):
    for a_idx in range(len(alpha_vals)):
        for s_idx in range(len(sigma_vals)):
            true_param[:,t_idx,a_idx,s_idx] = np.array([t0_vals[t_idx],alpha_vals[a_idx],sigma_vals[s_idx]])
            lh_test = lh[t_idx,a_idx,s_idx,:,:,:,:,:]
            for traj_idx in range(len(num_trajs)):
                for run_idx in range(num_runs):
                    lh_sample = lh_test[traj_idx,:,:,:,run_idx]    
                    max_idx = np.unravel_index(np.argmax(lh_sample),lh_sample.shape)
                    est_param[traj_idx,run_idx,:,t_idx,a_idx,s_idx] = np.array([t0_vals[max_idx[0]],alpha_vals[max_idx[1]],sigma_vals[max_idx[2]]])


# Convert true_param and est_param to the same size [num_traj,num_run,num_param,num_truemodels]
true_param = true_param.reshape(true_param.shape[0], -1)
true_param =  np.broadcast_to(true_param,(len(num_trajs),num_runs,true_param.shape[0],true_param.shape[1]))
est_shape = est_param.shape
est_param = est_param.reshape(est_shape[0],est_shape[1],est_shape[2],-1)

np.save('./results/true_param.npy',true_param)
np.save('./results/est_param.npy',est_param)