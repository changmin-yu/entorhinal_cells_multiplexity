import numpy as np
from scipy.signal import hilbert


def position_map(pos: np.ndarray, num_bins: int, box_size: int):
    bin_size = box_size / num_bins
    bin_locs = np.arange(bin_size / 2, box_size - bin_size / 2 + 1e-8, bin_size)
    
    pos_grid = np.zeros((len(pos), num_bins, num_bins))
    
    for i in range(len(pos)):
        
        x_coor = np.argmin(np.abs(pos[i, 0] - bin_locs))
        y_coor = np.argmin(np.abs(pos[i, 1] - bin_locs))
        
        pos_grid[i, x_coor, y_coor] = 1
    
    pos_grid = pos_grid.reshape(len(pos_grid), -1)
    
    return pos_grid, bin_locs


def hd_map(pos_left: np.ndarray, pos_right: np.ndarray, num_bins: int):
    direction = np.arctan2(pos_right[:, 1] - pos_left[:, 1], pos_right[:, 0] - pos_left[:, 0]) + np.pi / 2
    direction[direction < 0] += 2 * np.pi
    
    hd_grid = np.zeros((len(pos_left), num_bins))
    bin_size = 2 * np.pi / num_bins
    
    bin_locs = np.arange(bin_size / 2, 2 * np.pi - bin_size / 2 + 1e-8, bin_size)
    
    for i in range(len(pos_left)):
        ind = np.argmin(np.abs(direction[i] - bin_locs))
        hd_grid[i, ind] = 1
    
    return hd_grid, bin_locs, direction


def speed_map(
    pos: np.ndarray, 
    num_bins: int, 
    sample_rate: float = 50.0, 
    max_speed: float = 50.0, 
):
    displacement = np.diff(pos, axis=0)
    speed = np.sqrt(np.sum(np.square(displacement), axis=-1)) * sample_rate
    speed = np.concatenate([np.array([0.0]), speed])
    
    speed[speed > max_speed] = max_speed
    
    bin_size = max_speed / num_bins
    bin_locs = np.arange(bin_size / 2, max_speed - bin_size / 2 + 1e-8, bin_size)
    
    speed_grid = np.zeros((len(pos), num_bins))
    
    for i in range(len(pos)):
        ind = np.argmin(np.abs(speed[i] - bin_locs))
        speed_grid[i, ind] = 1
    
    return speed_grid, bin_locs, speed


def theta_map(
    filtered_eeg: np.ndarray, 
    time: np.ndarray, 
    sample_rate: float, 
    num_bins: int, 
):
    eeg_hilbert = hilbert(filtered_eeg)
    phase = np.angle(eeg_hilbert)
    phase[phase < 0] += 2 * np.pi
    
    phase_ind = np.round(time * sample_rate).astype(np.int32)
    
    phase_ind = phase_ind[phase_ind < len(filtered_eeg)]
    phase_time = phase[phase_ind]
    
    theta_grid = np.zeros((len(time), num_bins))
    
    bin_size = 2 * np.pi / num_bins
    bin_locs = np.arange(bin_size / 2, 2 * np.pi - bin_size / 2 + 1e-8, bin_size)
    
    for i in range(len(time)):
        try:
            ind = np.argmin(np.abs(phase_time[i] - bin_locs))
            theta_grid[i, ind] = 1
        except Exception:
            pass
    
    return theta_grid, bin_locs, phase_time
