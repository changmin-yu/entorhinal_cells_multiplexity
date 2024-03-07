from typing import Any, Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_tuning_curves(
    pos_tuning_curve: np.ndarray, 
    hd_tuning_curve: np.ndarray, 
    speed_tuning_curve: np.ndarray, 
    theta_tuning_curve: np.ndarray, 
    n_hd_bins: int, 
    n_speed_bins: int, 
    max_speed: float = 50.0, 
    ax: Optional[Any] = None
):
    hd_bin_size = 2 * np.pi / n_hd_bins
    speed_bin_size = max_speed / n_speed_bins
    
    hd_locs = np.linspace(hd_bin_size / 2, 2*np.pi - hd_bin_size/2, n_hd_bins, endpoint=True)
    theta_locs = hd_locs
    speed_locs = np.linspace(speed_bin_size / 2, 50 - speed_bin_size / 2, n_speed_bins, endpoint=True)
    
    if ax is None:
        fig, ax = plt.subplots(1, 4, figsize=(15, 5))
    
    assert len(ax) == 4
    
    ax[0].imshow(pos_tuning_curve, cmap="jet")
    ax[0].colorbar()
    ax[0].axis("off")
    ax[0].set_title("Position")
    
    ax[1].plot(hd_locs, hd_tuning_curve, color="k", linewidth=3)
    ax[1].set_xlim(0, 2*np.pi)
    ax[1].set_xlabel("direction angle")
    ax[1].set_title("Head direction")
    
    ax[2].plot(speed_locs, speed_tuning_curve, color="k", linewidth=3)
    ax[2].set_xlim(0, 50.0)
    ax[2].set_title("Speed")
    
    ax[3].plot(theta_locs, theta_tuning_curve, color="k", linewidth=3)
    ax[3].set_xlim(0, 2*np.pi)
    ax[3].set_title("Theta phase")


def plot_model_response(
    params: np.ndarray, 
    n_pos_params: int, 
    n_hd_params: int, 
    n_speed_params: int, 
    ax: Optional[Any] = None,
    max_speed: float = 50.0,  
):
    hd_bin_size = 2 * np.pi / n_hd_params
    speed_bin_size = max_speed / n_speed_params
    
    hd_locs = np.linspace(hd_bin_size / 2, 2*np.pi - hd_bin_size/2, n_hd_params, endpoint=True)
    theta_locs = hd_locs
    speed_locs = np.linspace(speed_bin_size / 2, 50 - speed_bin_size / 2, n_speed_params, endpoint=True)
    
    pos_params = params[:n_pos_params]
    hd_params = params[n_pos_params:(n_pos_params+n_hd_params)]
    speed_params = params[(n_pos_params+n_hd_params):(n_pos_params+n_hd_params+n_speed_params)]
    theta_params = params[(n_pos_params+n_hd_params+n_speed_params)]
    
    pos_scale_factor = np.mean(np.exp(hd_params)) * np.mean(np.exp(speed_params)) * np.mean(np.exp(theta_params)) * 50
    hd_scale_factor = np.mean(np.exp(pos_params)) * np.mean(np.exp(speed_params)) * np.mean(np.exp(theta_params)) * 50
    speed_scale_factor = np.mean(np.exp(pos_params)) * np.mean(np.exp(hd_params)) * np.mean(np.exp(theta_params)) * 50
    theta_scale_factor = np.mean(np.exp(pos_params)) * np.mean(np.exp(hd_params)) * np.mean(np.exp(speed_params)) * 50
    
    pos_response = pos_scale_factor * np.exp(pos_params)
    hd_response = hd_scale_factor * np.exp(hd_params)
    speed_response = speed_scale_factor * np.exp(speed_params)
    theta_response = theta_scale_factor * np.exp(theta_params)
    
    if ax is None:
        fig, ax = plt.subplots(1, 4, figsize=(15, 5))
    
    assert len(ax) == 4
    
    pos_response = np.reshape(pos_response, (int(np.sqrt(n_pos_params)), int(np.sqrt(n_pos_params))))
    
    ax[0].imshow(pos_response, cmap="jet")
    ax[0].colorbar()
    ax[0].axis("off")
    ax[0].set_title("Position")
    
    ax[1].plot(hd_locs, hd_response, color="k", linewidth=3)
    ax[1].set_xlim(0, 2*np.pi)
    ax[1].set_xlabel("direction angle")
    ax[1].set_title("Head direction")
    
    ax[2].plot(speed_locs, speed_response, color="k", linewidth=3)
    ax[2].set_xlim(0, 50.0)
    ax[2].set_title("Speed")
    
    ax[3].plot(theta_locs, theta_response, color="k", linewidth=3)
    ax[3].set_xlim(0, 2*np.pi)
    ax[3].set_title("Theta phase")
    

def plot_model_performance(
    log_like: np.ndarray, 
    selected_model: int, 
):
    num_cv_folds = log_like.shape[0]
    
    log_like_mean = np.mean(log_like, axis=-1)
    log_like_std = np.std(log_like, axis=-1) / np.sqrt(num_cv_folds)
    
    plt.plot(np.arange(num_cv_folds), log_like_mean, "o-", color="k", linewidth=3, label="Model performance")
    plt.fill_between(np.arange(num_cv_folds), log_like_mean-log_like_std, log_like_mean+log_like_std, color="gray", alpha=0.2)
    
    plt.scatter(selected_model, log_like_mean[selected_model], marker="*", s=20, color="red", label="Selected model")
    plt.axhline(0.0, color="blue", linestyle="--", label="Baseline")
    
    plt.xticks(np.arange(num_cv_folds), labels=[
        "PHST", "PHS", "PHT", "PST", "HST", "PH", "PS", "PT", "HS", "HT", "ST", "P", "H", "S", "T, "
    ])
    
    plt.legend()
