import numpy as np
from src.utils_calculation import cal_kl_btw_wsi


def make_cdal_dist(embedding) : 
    n = len(embedding) 
    dist = np.zeros((n,n))

    for i, wsi in enumerate(embedding[:-1]) : 
        weight_N, weight_D, weight_M = wsi 

        for j, n_wsi in enumerate(embedding[i+1:]) : 
            sum = 0 
            new_N, new_D, new_M = n_wsi 
            sum += cal_kl_btw_wsi(weight_N, new_N)
            sum += cal_kl_btw_wsi(weight_D, new_D)
            sum += cal_kl_btw_wsi(weight_M, new_M)
            dist[i,i+j+1] = sum 
            dist[i+j+1,i] = sum 
            
    return dist 