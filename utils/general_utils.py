import numpy as np


def gaussian_filter_normalised(start_x: float, end_x: float, step_size: float, mean: float, sigma: float):
    x = np.arange(start_x, end_x + step_size, step_size)
    filter = np.exp(-np.square(x - mean) / (2 * sigma ** 2))
    filter = filter / np.sum(filter)
    
    return filter