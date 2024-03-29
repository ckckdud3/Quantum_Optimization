import torch
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

# WIP
predefined_hamiltonian = {0:[qml.PauliZ(0)], 1:[], 2:[], 3:[]}

# WIP
data_label = {1:['CFI', 'theta_RZ(rad)', 'theta_RX(rad)'], 2:[], 3:[], 4:[]}

def dephase_factor(tau: float):
    """ 
    Calculate the dephasing factor for a given dephasing time tau.

    Args:
        tau (torch.Tensor): Dephasing time.

    Returns:
        torch.Tensor: Dephasing factor.
    """  
    return 1 - np.exp(-2 * tau)


def sweep_range(t_obs: float, num_points: int):
    """
    Returns the sweep range for optimization.

    Args:
        start (float): starting time point.
        end (float): ending time point.
        num_points(int): number of time points to be optimized.
    """
    return torch.tensor(np.linspace(0.0, t_obs, num_points))


def plot_data(data, num_qubit, k, freq):

    timespace = data[:,0]

    reference = (1 + k**2)*np.exp(-2*timespace*k*freq)
    plt.figure(figsize=(12,4))

    plt.subplot(num_qubit, 3, 1)
    plt.title('CFI')
    plt.plot(timespace, data[:,1], label='CFI')
    plt.plot(timespace, reference, label= 'Reference', linestyle='dotted')
    plt.ylim(0,2*num_qubit**2)
    plt.legend()

    for i in range(num_qubit):
        plt.subplot(num_qubit, 3, 3*i+2)
        plt.title(data_label[num_qubit][i*2+1])
        plt.plot(timespace, data[:,2*i+2])

        plt.subplot(num_qubit, 3, 3*i+3)
        plt.title(data_label[num_qubit][i*2+2])
        plt.plot(timespace, data[:,2*i+3])
        plt.ylim(0,2*np.pi)
        
    plt.show()

