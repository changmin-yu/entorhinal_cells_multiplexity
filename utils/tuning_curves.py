import numpy as np
from scipy.ndimage import gaussian_filter


def compute_1D_tuning_curve(
    x: np.ndarray, 
    firing_rate: np.ndarray, 
    num_bins: int, 
    min_val: float, 
    max_val: float, 
):
    edges = np.linspace(min_val, max_val, num=num_bins+1, endpoint=True)
    tuning_curve = np.zeros((num_bins, ))
    
    for i in range(num_bins):
        tuning_curve[i] = np.mean(firing_rate[(x >= edges[i]) * (x < edges[i+1])])
        
        if i == (num_bins - 1):
            tuning_curve[i] = np.mean(firing_rate[(x >= edges[i]) * (x <= edges[i+1])])
            
    return tuning_curve


def compute_2D_tuning_curve(
    x: np.ndarray, 
    firing_rate: np.ndarray, 
    num_bins: int, 
    min_val: float, 
    max_val: float, 
    gaussian_filter_sigma: float = 0.5, 
    gaussian_filter_truncate: float = 4.0, 
):
    assert x.shape[-1] == 2
    x1, x2 = x[:, 0], x[:, 1]
    
    x_axis = np.linspace(min_val, max_val, num=num_bins+1, endpoint=True)
    y_axis = np.linspace(min_val, max_val, num=num_bins+1, endpoint=True)
    
    tuning_curve = np.zeros((num_bins, num_bins))
    
    for i in range(num_bins):
        if i == (num_bins - 1):
            x_ind = np.where((x1 >= x_axis[i]) * (x1 <= x_axis[i+1]))[0]
        else:
            x_ind = np.where((x1 >= x_axis[i]) * (x1 < x_axis[i+1]))[0]
        
        for j in range(num_bins):
            if j == (num_bins - 1):
                y_ind = np.where((x2 >= y_axis[j]) * (x2 <= y_axis[j+1]))[0]
            else:
                y_ind = np.where((x2 >= y_axis[j]) * (x2 < y_axis[j+1]))[0]
            
            ind = np.intersect1d(x_ind, y_ind)
            
            tuning_curve[num_bins - 1 - j, i] = np.mean(firing_rate[ind])
    
    # smooth tuning curve
    nan_ind = np.where(np.isnan(tuning_curve))
    
    if len(nan_ind[0]) > 0:
        
        for i in range(len(nan_ind)):
            ind_i = nan_ind[1][i]
            ind_j = nan_ind[0][i]
            
            right = tuning_curve[ind_j, min(ind_i+1, num_bins-1)]
            left = tuning_curve[ind_j, max(ind_i-1, 0)]
            down = tuning_curve[min(ind_j+1, num_bins-1), ind_i]
            up = tuning_curve[max(ind_j-1, 0), ind_i]
            
            ru = tuning_curve[max(ind_j-1, 0), min(ind_i+1, num_bins-1)]
            lu = tuning_curve[max(ind_j-1, 0), max(ind_i-1, 0)]
            ld = tuning_curve[min(ind_j+1, num_bins-1), max(ind_i-1, 0)]
            rd = tuning_curve[min(ind_j+1, num_bins-1), min(ind_i+1, num_bins-1)]
            
            tuning_curve[ind_j, ind_i] = np.nanmean(np.array([right, left, up, down, ru, lu, ld, rd]))
    
    tuning_curve = gaussian_filter(tuning_curve, sigma=gaussian_filter_sigma, truncate=gaussian_filter_truncate)
    
    return tuning_curve


def compute_all_tuning_curves(
    pos: np.ndarray, 
    hd: np.ndarray, 
    speed: np.ndarray, 
    theta: np.ndarray, 
    firing_rate_smooth: np.ndarray, 
    n_pos_bins: int, 
    n_hd_bins: int, 
    n_speed_bins: int, 
    n_theta_bins: int, 
    box_size: float, 
    max_speed: float = 50.0, 
    gaussian_filter_sigma: float = 0.5, 
    gaussian_filter_truncate: float = 4.0, 
):
    valid_speed_inds = np.where(speed < max_speed)
    
    pos = pos[valid_speed_inds]
    hd = hd[valid_speed_inds]
    speed = speed[valid_speed_inds]
    theta = theta[valid_speed_inds]
    
    pos_tuning_curve = compute_2D_tuning_curve(
        pos, firing_rate_smooth, n_pos_bins, min_val=0.0, max_val=box_size, gaussian_filter_sigma=gaussian_filter_sigma, gaussian_filter_truncate=gaussian_filter_truncate, 
    )
    hd_tuning_curve = compute_1D_tuning_curve(
        hd, firing_rate_smooth, n_hd_bins, min_val=0.0, max_val=2*np.pi, 
    )
    speed_tuning_curve = compute_1D_tuning_curve(
        speed, firing_rate_smooth, n_speed_bins, min_val=0.0, max_val=50.0, 
    )
    theta_tuning_curve = compute_1D_tuning_curve(
        theta, firing_rate_smooth, n_theta_bins, min_val=0.0, max_val=2*np.pi, 
    )
    
    return pos_tuning_curve, hd_tuning_curve, speed_tuning_curve, theta_tuning_curve
