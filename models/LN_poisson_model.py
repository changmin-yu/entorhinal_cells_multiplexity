from typing import List
import numpy as np
from scipy.linalg import block_diag


def LN_poisson_model(
    param: np.ndarray, 
    X: np.ndarray, 
    spike_train: np.ndarray, 
    model_type: List[int], 
    param_num_configs: List[int], 
):
    """
    Linear-Nonlinear Poisson model
    
    Params
    -------
    
    param: np.ndarray
        (D, ) GLM parameters
    X: np.ndarray
        (N, D) covariate matrix, N is the number of timepoints
    spike_train: np.ndarray
        (N, ) binned spikes matrix for a single neuron
    model_type: List[int]
        binary list indicating which covariates to use in GLM
    param_num_configs: List[int]
        number of bins (parameters) associated with each covariate
    """
    u = np.dot(X, param) # (N, )
    rate = np.exp(u) # firing rate (N, )
    
    pos_reg = 8.0
    hd_reg = 50.0
    speed_reg = 50.0
    theta_reg = 50.0
    
    # compute the grad and Hessian (with respect to the params)
    rX = rate[:, None] * X
    hessian = np.dot(rX.T, X)
    
    # compute roughness (smoothness) penalties for different parameters
    n_pos_params, n_hd_params, n_speed_params, n_theta_params = param_num_configs
    n_pos_params = n_pos_params ** 2
    pos_param, hd_param, speed_param, theta_param = retrieve_params(
        param, model_type, n_pos_params, n_hd_params, n_speed_params, n_theta_params
    )
    
    J_pos, J_pos_grad, J_pos_hessian = 0.0, [], []
    J_hd, J_hd_grad, J_hd_hessian = 0.0, [], []
    J_speed, J_speed_grad, J_speed_hessian = 0.0, [], []
    J_theta, J_theta_grad, J_theta_hessian = 0.0, [], []
    
    if pos_param is not None:
        J_pos, J_pos_grad, J_pos_hessian = smoothness_penalty_2d(pos_param, beta=pos_reg)
    
    if hd_param is not None:
        J_hd, J_hd_grad, J_hd_hessian = smoothness_penalty_1d_circular(hd_param, beta=hd_reg)
    
    if speed_param is not None:
        J_speed, J_speed_grad, J_speed_hessian = smoothness_penalty_1d(speed_param, beta=speed_reg)
        
    if theta_param is not None:
        J_theta, J_theta_grad, J_theta_hessian = smoothness_penalty_1d_circular(theta_param, beta=theta_reg)
    
    objective = np.sum(rate - spike_train * u) + J_pos + J_hd + J_speed + J_theta
    objective_grad = np.real(np.dot(X.T, rate - spike_train)) + np.concatenate([J_pos_grad, J_hd_grad, J_speed_grad, J_theta_grad], axis=0) # TODO: verify this!
    active_hessians = [h for h in [J_pos_hessian, J_hd_hessian, J_speed_hessian, J_theta_hessian] if len(h) > 0]
    objective_hessian = hessian + block_diag(*active_hessians) # TODO: verify this!
    
    return objective, objective_grad, objective_hessian
    

def smoothness_penalty_2d(param, beta):
    n = len(param)
    n_sqrt = int(np.sqrt(n))
    
    D1 = np.zeros((n_sqrt-1, n_sqrt))
    D1[np.arange(n_sqrt-1), np.arange(n_sqrt-1)] = -1
    D1[np.arange(n_sqrt-1), np.arange(n_sqrt-1)+1] = 1
    DD1 = np.dot(D1.T, D1)
    
    M1 = np.kron(np.eye(n_sqrt), DD1)
    M2 = np.kron(DD1, np.eye(n_sqrt))
    M = M1 + M2
    
    J = 0.5 * beta * np.dot(param, np.dot(M, param))
    J_grad = beta * np.dot(M, param)
    J_hessian = beta * M
    
    return J, J_grad, J_hessian


def smoothness_penalty_1d_circular(param, beta):
    n = len(param)
    
    D1 = np.zeros((n - 1, n))
    D1[np.arange(n-1), np.arange(n-1)] = -1
    D1[np.arange(n-1), np.arange(n-1)+1] = 1
    DD1 = np.dot(D1.T, D1)
    
    DD1[0, :] = np.roll(DD1[1, :], -1)
    DD1[-1, :] = np.roll(DD1[-2, :], 1)
    
    J = 0.5 * beta * np.dot(param, np.dot(DD1, param))
    J_grad = beta * np.dot(DD1, param)
    J_hessian = beta * DD1
    return J, J_grad, J_hessian


def smoothness_penalty_1d(param, beta):
    n = len(param)
    D1 = np.zeros((n-1, n))
    D1[np.arange(n-1), np.arange(n-1)] = -1
    D1[np.arange(n-1), np.arange(n-1)+1] = 1
    DD1 = np.dot(D1.T, D1)
    
    J = 0.5 * beta * np.dot(param, np.dot(DD1, param))
    J_grad = beta * np.dot(DD1, param)
    J_hessian = beta * DD1
    return J, J_grad, J_hessian
    

def retrieve_params(
    param: np.ndarray, 
    model_type: List[int], 
    n_pos_params: int, 
    n_hd_params: int, 
    n_speed_params: int, 
    n_theta_params: int, 
):
    pos_param = None
    hd_param = None
    speed_param = None
    theta_param = None
    
    if np.all(model_type == [1, 0, 0, 0]):
        pos_param = param
    elif np.all(model_type == [0, 1, 0, 0]):
        hd_param = param
    elif np.all(model_type == [0, 0, 1, 0]):
        speed_param = param
    elif np.all(model_type == [0, 0, 0, 1]):
        theta_param = param
    
    elif np.all(model_type == [1, 1, 0, 0]):
        pos_param = param[:n_pos_params]
        hd_param = param[n_pos_params:]
    elif np.all(model_type == [1, 0, 1, 0]):
        pos_param = param[:n_pos_params]
        speed_param = param[n_pos_params:]
    elif np.all(model_type == [1, 0, 0, 1]):
        pos_param = param[:n_pos_params]
        theta_param = param[n_pos_params:]
    elif np.all(model_type == [0, 1, 1, 0]):
        hd_param = param[:n_hd_params]
        speed_param = param[n_hd_params:]
    elif np.all(model_type == [0, 1, 0, 1]):
        hd_param = param[:n_hd_params]
        theta_param = param[n_hd_params:]
    elif np.all(model_type == [0, 0, 1, 1]):
        speed_param = param[:n_speed_params]
        theta_param = param[n_speed_params:]
    
    elif np.all(model_type == [1, 1, 1, 0]):
        pos_param = param[:n_pos_params]
        hd_param = param[n_pos_params:(n_pos_params + n_hd_params)]
        speed_param = param[(n_pos_params + n_hd_params):]
    elif np.all(model_type == [1, 1, 0, 1]):
        pos_param = param[:n_pos_params]
        hd_param = param[n_pos_params:(n_pos_params + n_hd_params)]
        theta_param = param[(n_pos_params + n_hd_params):]
    elif np.all(model_type == [1, 0, 1, 1]):
        pos_param = param[:n_pos_params]
        speed_param = param[n_pos_params:(n_pos_params + n_speed_params)]
        theta_param = param[(n_pos_params + n_speed_params):]
    elif np.all(model_type == [0, 1, 1, 1]):
        hd_param = param[:n_hd_params]
        speed_param = param[n_hd_params:(n_hd_params + n_speed_params)]
        theta_param = param[(n_hd_params + n_speed_params):]
    
    elif np.all(model_type == [1, 1, 1, 1]):
        pos_param = param[:n_pos_params]
        hd_param = param[n_pos_params:(n_pos_params + n_hd_params)]
        speed_param = param[(n_pos_params + n_hd_params):(n_pos_params + n_hd_params + n_speed_params)]
        theta_param = param[(n_pos_params + n_hd_params + n_speed_params):]
    
    return pos_param, hd_param, speed_param, theta_param
