import numpy as np
from scipy.io import loadmat


def load_data(data_dir: str):
    df = loadmat(data_dir)
    
    pos_left = np.concatenate([df["posx"], df["posy"]], axis=-1)
    pos_right = np.concatenate([df["posx2"], df["posy2"]], axis=-1)
    pos_middle = np.concatenate([df["posx_c"], df["posy_c"]], axis=-1)
    
    filtered_lfp = df["filt_eeg"][:, 0]
    
    time = df["post"][:, 0]
    
    spike_train = df["spiketrain"][:, 0]
    
    sample_rate = df["sampleRate"][0, 0]
    box_size = df["boxSize"][0, 0]
    lfp_sample_rate = df["eeg_sample_rate"][0, 0]
    
    return pos_left, pos_right, pos_middle, filtered_lfp, time, spike_train, sample_rate, box_size, lfp_sample_rate
