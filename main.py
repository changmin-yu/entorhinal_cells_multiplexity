import pickle

from utils.data_loader import load_data
from models.fit_LN_model import fit_all_LN_models
from models.model_selection import model_selection
from utils.tuning_curves import compute_all_tuning_curves


DATADIR = "data/data_for_cell77.mat"



def main():
    pos_left, pos_right, pos_middle, filtered_lfp, time, spike_train, sample_rate, box_size, lfp_sample_rate = load_data(DATADIR)
    
    n_pos_bins = 20
    n_hd_bins = 18
    n_speed_bins = 10
    n_theta_bins = 18
    
    max_speed = 50.0
    
    d_fitting, hd_rad, speed, theta_rad, firing_rate_smooth = fit_all_LN_models(
        pos_middle=pos_middle, 
        pos_left=pos_left, 
        pos_right=pos_right, 
        filtered_lfp=filtered_lfp, 
        time=time, 
        spike_train=spike_train, 
        num_cv_folds=10, 
        param_num_configs=[n_pos_bins, n_hd_bins, n_speed_bins, n_theta_bins], 
        box_size=box_size, 
        sample_rate=sample_rate, 
        max_speed=max_speed, 
    )
    
    with open("data/fitted_params.pkl", "wb") as f:
        pickle.dump((d_fitting, hd_rad, speed, theta_rad, firing_rate_smooth), f)
    f.close()
    
    # fitting all 16 models is time-consuming, so we cache the fitted params!
    
    # with open("data/fitted_params.pkl", "rb") as f:
    #     d_fitting = pickle.load(f)
    # f.close()
    
    selected_model = model_selection(d_fitting)
    
    pos_tuning_curve, hd_tuning_curve, speed_tuning_curve, theta_tuning_curve = compute_all_tuning_curves(
        pos=pos_middle, 
        hd=hd_rad, 
        speed=speed, 
        theta=theta_rad, 
        firing_rate_smooth=firing_rate_smooth, 
        n_pos_bins=n_pos_bins, 
        n_hd_bins=n_hd_bins, 
        n_speed_bins=n_speed_bins, 
        n_theta_bins=n_theta_bins, 
        box_size=box_size, 
        max_speed=max_speed, 
    )
    
    return d_fitting, selected_model, pos_tuning_curve, hd_tuning_curve, speed_tuning_curve, theta_tuning_curve
    

if __name__=="__main__":
    d_fitting, selected_model, pos_tuning_curve, hd_tuning_curve, speed_tuning_curve, theta_tuning_curve = main()