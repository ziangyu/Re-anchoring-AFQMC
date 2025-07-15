import numpy as np

sigma_x = np.array([[0.,1.],[1.,0.]])
sigma_y = np.array([[0.,-1j],[1j,0.]])
sigma_z = np.array([[1.,0.],[0.,-1.]])
identity = np.eye(2)

def disordered_state(num_spins):
    mf = np.ones([2, num_spins]) / np.sqrt(2)
    phi_trial_init = []
    for i in range(num_spins):
        phi_trial_init.append(mf[:,i].reshape([1,2,1]))

    return phi_trial_init