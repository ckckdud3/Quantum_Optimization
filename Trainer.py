import time

import torch
import pennylane as qml
import pennylane.numpy as np

from utils.utils import *
from utils.arguments import optarguments
from OptimizationCircuit import OptimizationCircuit

timestamp = None

def select_optimizer(method, parameters_in):
    """
    Select and configure an optimizer based on the specified method.

    Args:
        method (str): The optimization method to use ('LBFGS' or 'Adam').
        parameters_in (torch.Tensor): The parameters to be optimized.

    Returns:
        torch.optim.Optimizer: Configured optimizer object.

    Raises:
        ValueError: If an invalid optimization method is specified.

    This function initializes and returns a PyTorch optimizer based on the provided method.
    For 'LBFGS', it sets up an LBFGS optimizer with specified learning rate, maximum iterations,
    tolerance levels, and history size. For 'Adam', it sets up an Adam optimizer with specified
    learning rate, beta values, epsilon for numerical stability, and weight decay options.
    If a method other than 'LBFGS' or 'Adam' is provided, the function raises a ValueError.
    """
    
    if method == 'LBFGS':
        opt = torch.optim.LBFGS(
                [parameters_in], 
                lr=5e-3,              # Learning rate
                max_iter=80,          # Maximum number of iterations per optimization step
                max_eval=None,        # Maximum number of function evaluations per optimization step
                tolerance_grad=1e-12,  # Termination tolerance on the gradient norm
                tolerance_change=1e-12,# Termination tolerance on the function value/parameter changes
                history_size=200,      # Update history size
                line_search_fn='strong_wolfe'
        )
        return opt
    
    elif method == 'Adam':
        opt = torch.optim.Adam(
            [parameters_in],
            lr=0.002,                # Learning rate (default: 0.001)
            betas=(0.9, 0.999),      # Coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
            eps=1e-08,               # Term added to the denominator to improve numerical stability (default: 1e-8)
            weight_decay=0,          # Weight decay (L2 penalty) (default: 0)
            amsgrad=False            # Whether to use the AMSGrad variant of this algorithm (default: False)
        )
        return opt
    
    else:
        raise ValueError("Invalid optimizer choice.")
    

def cost_function(circuit: OptimizationCircuit, t: torch.Tensor, params:torch.Tensor):
    """ 
    Compute the cost using classical Fisher information for the given parameters.

    Args:
        circuit (OptimizationCircuit): Circuit to be optimized.
        t (float) : Time point.
        params (torch.Tensor) : parameters of the circuit

    Returns:
        torch.Tensor: Computed cost.
    """
    cfi = qml.qinfo.classical_fisher(circuit.circuit)(torch.cat((t, params)))[1:,1:]
    return -torch.trace(cfi)


def fit(circuit: OptimizationCircuit, args: optarguments):
    """ 
    Compute the cost using classical Fisher information for the given parameters.

    Args:
        circuit (OptimizationCircuit): Circuit to be optimized.
        args (optarguments) : Optimization arguments.

    Returns:
        data (torch.Tensor) : Optimization data.
    """

    global timestamp

    data = torch.zeros((circuit.num_points, len(circuit.params)+2))

    data[:,0] = circuit.sweep_list.clone().detach()

    opt = select_optimizer(args.opt, circuit.params)

    def closure():
        opt.zero_grad()
        loss = cost_function(circuit, timestamp, circuit.params)
        loss.backward()
        return loss
    

    steps_per_timepoint = args.steps_per_point
    threshold = args.threshold
    max_patience = args.patience

    f_logs = []

    start_time = time.time()
    for idx, timepoint in enumerate(circuit.sweep_list):

        patience = 0
        lval = None
        timestamp = torch.tensor([timepoint])

        opt = select_optimizer(args.opt, circuit.params)
        f_logs = []

        for i in range(steps_per_timepoint):
            
            lval = opt.step(closure).item()
            f_logs.append(lval)

            if i:
                if np.abs(f_logs[i] - f_logs[i-1]) < threshold:
                    patience += 1
                else: patience = 0

            if patience > max_patience : break

        elapsed_time = time.time() - start_time
        estimated_total_time = (elapsed_time / (idx+1))*circuit.num_points
        remaining_time = estimated_total_time - elapsed_time

        param_format = [f"{x:.6f}" for x in circuit.params.detach().numpy()]

        print(f'CFI = {-lval:.6f} at timepoint {timepoint.item():.8f}, Param = {param_format}, Iter left = {circuit.num_points - idx - 1}, Remaining time = {remaining_time:.2f} sec')

        data[idx,1] = -lval
        data[idx,2:] = circuit.params

    return data

