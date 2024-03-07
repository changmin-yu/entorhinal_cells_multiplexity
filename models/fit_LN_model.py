from typing import List
import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial

import time

from models.LN_poisson_model import LN_poisson_model
from utils.construct_maps import position_map, hd_map, speed_map, theta_map
from utils.general_utils import gaussian_filter_normalised


def fit_LN_model(
    X: np.ndarray, 
    dt: float, 
    spike_train: np.ndarray, 
    filter: np.ndarray, 
    model_type: List[int], 
    num_cv_folds: int, 
    param_num_configs: List[int], 
):
    D = X.shape[1]
    N = spike_train.shape[0]
    
    sections = num_cv_folds * 5
    
    edges = np.round(np.linspace(0, N, sections+1, endpoint=True), 0).astype(np.int32)
    
    test_fit = np.zeros((num_cv_folds, 6)) # variance explained, correlation, log-likelihood increase, MSE, number of spikes, length of test data
    train_fit = np.zeros((num_cv_folds, 6))
    param_mat = np.zeros((num_cv_folds, D))
    
    for i in range(num_cv_folds):
        print(f"Cross validation hold {i} of {num_cv_folds}")
        t0 = time.time()
        
        test_ind = np.concatenate([
            np.arange(edges[i + k * num_cv_folds], edges[i + k * num_cv_folds + 1])
        for k in range(5)])
        
        test_spikes = spike_train[test_ind]
        test_spikes_smooth = np.convolve(test_spikes, filter, mode="same") # TODO: convolve within each subsequecne?
        test_firing_rate_smooth = test_spikes_smooth / dt
        test_X  = X[test_ind]
        
        train_inds = np.setdiff1d(np.arange(N), test_ind)
        train_spikes = spike_train[train_inds]
        train_spikes_smooth = np.convolve(train_spikes, filter, mode="same") 
        train_firing_rate_smooth = train_spikes_smooth / dt
        train_X = X[train_inds]
        
        if i == 0:
            init_param = 1e-3 * np.random.randn(D)
        
        objective_fn = lambda w: LN_poisson_model(w, train_X, train_spikes, model_type, param_num_configs)[0]
        objective_grad_fn = lambda w: LN_poisson_model(w, train_X, train_spikes, model_type, param_num_configs)[1]
        objective_hessian_fn = lambda w: LN_poisson_model(w, train_X, train_spikes, model_type, param_num_configs)[2]
        
        # objective, objective_grad, objective_hessian = LN_poisson_model(init_param, train_X, train_spikes, model_type, param_num_configs)
        
        res = minimize(objective_fn, init_param, jac=objective_grad_fn, hess=objective_hessian_fn, method="Newton-CG")
        
        fitted_param = res.x
        
        # validating fitted params on test data
        
        test_firing_rate_hat = np.exp(np.dot(test_X, fitted_param)) / dt
        test_firing_rate_smooth_hat = np.convolve(test_firing_rate_hat, filter, mode="same")
        
        # explained variance
        sse = np.sum(np.square(test_firing_rate_smooth_hat - test_firing_rate_smooth))
        sst = np.sum(np.square(test_firing_rate_smooth - np.mean(test_firing_rate_smooth)))
        test_variance_explained = 1 - sse / sst
        
        # correlation
        test_correlation = np.corrcoef(test_firing_rate_smooth, test_firing_rate_smooth_hat)[0, 1]
        
        # log-likelihood increase from "mean firing rate model" (no smoothing)
        rate_hat = np.exp(np.dot(test_X, fitted_param))
        test_mean_firing_rate = np.nanmean(test_spikes)
        
        log_like_model_test = np.nansum(rate_hat - test_spikes * np.log(rate_hat) + np.log(factorial(test_spikes))) / np.sum(test_spikes)
        log_like_mean_test = np.nansum(test_mean_firing_rate - test_spikes * np.log(test_mean_firing_rate) + np.log(factorial(test_spikes))) / np.sum(test_spikes)
        
        test_delta_ll = (-log_like_model_test + log_like_mean_test) / np.log(2) # in bits
        
        # MSE
        test_mse = np.nanmean(np.square(test_firing_rate_smooth_hat - test_firing_rate_smooth))
        
        # collate
        test_fit[i, :] = np.array([test_variance_explained, test_correlation, test_delta_ll, test_mse, np.sum(test_spikes), len(test_ind)])
        
        # validating fitted params on train data
        train_firing_rate_hat = np.exp(np.dot(train_X, fitted_param)) / dt
        train_firing_rate_smooth_hat = np.convolve(train_firing_rate_hat, filter, mode="same")
        
        # explained variance
        sse = np.sum(np.square(train_firing_rate_smooth_hat - train_firing_rate_smooth))
        sst = np.sum(np.square(train_firing_rate_smooth - np.mean(train_firing_rate_smooth)))
        train_variance_explained = 1 - sse / sst
        
        # correlation
        train_correlation = np.corrcoef(train_firing_rate_smooth, train_firing_rate_smooth_hat)[0, 1]
        
        # log-likelihood increase
        rate_hat = np.exp(np.dot(train_X, fitted_param))
        train_mean_firing_rate = np.nanmean(train_spikes)
        
        log_like_model_train = np.nansum(rate_hat - train_spikes * np.log(rate_hat) + np.log(factorial(train_spikes))) / np.sum(train_spikes)
        log_like_mean_train = np.nansum(train_mean_firing_rate - train_spikes * np.log(train_mean_firing_rate) + np.log(factorial(train_spikes))) / np.sum(train_spikes)
        train_delta_ll = (-log_like_model_train + log_like_mean_train) / np.log(2)
        
        # MSE
        train_mse = np.nanmean(np.square(train_firing_rate_smooth - train_firing_rate_smooth_hat))
        
        # collate
        train_fit[i, :] = np.array([train_variance_explained, train_correlation, train_delta_ll, train_mse, np.sum(train_spikes), len(train_inds)])
        
        param_mat[i, :] = fitted_param
        
        init_param = fitted_param
        
        print(f"time taken: {time.time() - t0:.2f}s")
        
    mean_param = np.nanmean(param_mat, axis=0)
    
    return test_fit, train_fit, param_mat, mean_param


def fit_all_LN_models(
    pos_middle: np.ndarray, 
    pos_left: np.ndarray, 
    pos_right: np.ndarray, 
    filtered_lfp: np.ndarray, 
    time: np.ndarray, 
    spike_train: np.ndarray, 
    num_cv_folds: int, 
    param_num_configs: List[int], 
    box_size: float, 
    sample_rate: float = 50, 
    max_speed: float = 50, 
):
    n_pos_bins, n_hd_bins, n_speed_bins, n_theta_bins = param_num_configs
    
    pos_grid, pos_bin_locs = position_map(pos_middle, n_pos_bins, box_size=box_size)
    hd_grid, hd_bin_locs, hd_rad = hd_map(pos_left, pos_right, n_hd_bins)
    speed_grid, speed_bin_locs, speed = speed_map(pos_middle, n_speed_bins, sample_rate=sample_rate, max_speed=max_speed)
    theta_grid, theta_bin_locs, theta_rad = theta_map(filtered_lfp, time, sample_rate, n_theta_bins)
    
    valid_speed_inds = np.where(speed < 50)[0]
    pos_grid = pos_grid[valid_speed_inds]
    hd_grid = hd_grid[valid_speed_inds]
    speed_grid = speed_grid[valid_speed_inds]
    theta_grid = theta_grid[valid_speed_inds]
    spike_train = spike_train[valid_speed_inds]
    
    num_models = 15
    d = {}
    d["test_fit"] = {}
    d["train_fit"] = {}
    d["fitted_params"] = {}
    d["model_type"] = {}
    
    # all variables
    d["model_type"][0] = [1, 1, 1, 1]
    # three variables
    d["model_type"][1] = [1, 1, 1, 0]
    d["model_type"][2] = [1, 1, 0, 1]
    d["model_type"][3] = [1, 0, 1, 1]
    d["model_type"][4] = [0, 1, 1, 1]
    # two variables
    d["model_type"][5] = [1, 1, 0, 0]
    d["model_type"][6] = [1, 0, 1, 0]
    d["model_type"][7] = [1, 0, 0, 1]
    d["model_type"][8] = [0, 1, 1, 0]
    d["model_type"][9] = [0, 1, 0, 1]
    d["model_type"][10] = [0, 0, 1, 1]
    # one variable
    d["model_type"][11] = [1, 0, 0, 0]
    d["model_type"][12] = [0, 1, 0, 0]
    d["model_type"][13] = [0, 0, 1, 0]
    d["model_type"][14] = [0, 0, 0, 1]
    
    filter = gaussian_filter_normalised(-4.0, 4.0, 1.0, mean=0.0, sigma=2.0)
    
    dt = time[2] - time[1]
    firing_rate = spike_train / dt
    firing_rate_smooth = np.convolve(firing_rate, filter, mode="same")
    
    for i in range(num_models):
        print(f"Fitting model {i} ({d['model_type'][i]}) of {num_models}")
        
        test_fit, train_fit, param_mat, mean_param = fit_LN_model(
            d["covariates"][i], dt, spike_train, filter, d["model_type"][i], num_cv_folds, param_num_configs
        )
        
        d["test_fit"][i] = test_fit
        d["train_fit"][i] = train_fit
        d["fitted_params"][i] = mean_param
        
    return d, hd_rad, speed, theta_rad, firing_rate_smooth