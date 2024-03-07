from typing import Dict, Any
import numpy as np
from scipy.stats import wilcoxon


def model_selection(d: Dict[str, Any]):
    test_fit = np.array([v for _, v in d["test_fit"].items()])
    log_likelihood = test_fit[:, :, 2] # (num_models, num_fold)
    
    single_model_inds = np.array([11, 12, 13, 14])
    top_single_model = single_model_inds[np.argmax(np.nanmean(log_likelihood[single_model_inds], axis=-1))]
    
    if top_single_model == 11: # P -> PH, PS, PT
        two_models_inds = np.array([5, 6, 7])
    elif top_single_model == 12:
        two_models_inds = np.array([5, 8, 9])
    elif top_single_model == 13:
        two_models_inds = np.array([6, 8, 10])
    elif top_single_model == 14:
        two_models_inds = np.array([7, 9, 10])
    top_two_models = two_models_inds[np.argmax(np.nanmean(log_likelihood[two_models_inds], axis=-1))]
    
    if top_two_models == 5:
        three_models_inds = np.array([1, 2])
    elif top_two_models == 6:
        three_models_inds = np.array([1, 3])
    elif top_two_models == 7:
        three_models_inds = np.array([2, 3])
    elif top_two_models == 8:
        three_models_inds = np.array([1, 4])
    elif top_two_models == 9:
        three_models_inds = np.array([2, 4])
    elif top_two_models == 10:
        three_models_inds = np.array([3, 4])
    top_three_models = three_models_inds[np.argmax(np.nanmean(log_likelihood[three_models_inds], axis=-1))]

    top_four_models = 0
    
    log_like_one_model = log_likelihood[top_single_model]
    log_like_two_models = log_likelihood[top_two_models]
    log_like_three_models = log_likelihood[top_three_models]
    log_like_four_models = log_likelihood[top_four_models]
    
    res_12 = wilcoxon(log_like_two_models, log_like_one_model, alternative="greater")
    res_23 = wilcoxon(log_like_three_models, log_like_two_models, alternative="greater")
    res_34 = wilcoxon(log_like_four_models, log_like_three_models, alternative="greater")
    
    p12 = res_12.pvalue
    p23 = res_23.pvalue
    p34 = res_34.pvalue
    
    if p12 < 0.05:
        if p23 < 0.05:
            if p34 < 0.05:
                selected_model = top_four_models
            else:
                selected_model = top_three_models
        else:
            selected_model = top_two_models
    else:
        selected_model = top_single_model
    
    res_baseline = wilcoxon(log_likelihood[selected_model], alternative="greater")
    pval_baseline = res_baseline.pvalue
    
    if pval_baseline > 0.05:
        selected_model = np.NAN
    
    return selected_model
