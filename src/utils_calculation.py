import torch
import numpy as np 
from scipy.optimize import minimize
from scipy.stats import expon
from scipy.stats import entropy


def cal_kl_btw_wsi(wsi1_prob, wsi2_prob):
    wsi1_prob = torch.tensor(wsi1_prob, dtype=torch.float64)
    wsi2_prob = torch.tensor(wsi2_prob, dtype=torch.float64)

    if torch.all(wsi1_prob == torch.zeros(3, dtype=torch.float64)) or torch.all(wsi2_prob == torch.zeros(3, dtype=torch.float64)):
        return 0

    kl1 = wsi1_prob * torch.log(wsi1_prob / wsi2_prob)
    kl2 = wsi2_prob * torch.log(wsi2_prob / wsi1_prob)
    kl = -0.5 * (torch.sum(kl1)) - 0.5 * (torch.sum(kl2))
    return abs(kl).item()

def cal_wsi_rank(df, num_slides) :
    new_df = df.copy() 
    new_df.loc[:, 'N_distance'] = new_df['N_distance'] / new_df['N_distance'].max()
    new_df.loc[:, 'D_distance'] = new_df['D_distance'] / new_df['D_distance'].max()
    new_df.loc[:, 'M_distance'] = new_df['M_distance'] / new_df['M_distance'].max()
    new_df['rank_score'] = 1+ new_df[['N_distance', 'D_distance', 'M_distance']].sum(axis=1) - 2*new_df[['N_distance', 'D_distance', 'M_distance']].min(axis=1)

    new_df = new_df.sort_values(by='rank_score', ascending=True)
    top_indices = new_df.head(num_slides).index

    return top_indices


def kl_divergence(lambda1, lambda2):
    
    expon1 = expon(scale=1/lambda1)
    expon2 = expon(scale=1/lambda2)

    # Using the probability density function to generate sample data
    # If you don't add 1e-10, it will be inf
    x = np.linspace(0, max(lambda1, lambda2) * 10, 1000)
    pdf1 = expon1.pdf(x) + 1e-10
    pdf2 = expon2.pdf(x) + 1e-10


    pdf1 = pdf1 / np.sum(pdf1) 
    pdf2 = pdf2 / np.sum(pdf2)

    # KL-divergence 계산
    kl_div = entropy(pdf1, pdf2)

    return kl_div

def jensen_shannon_divergence(lambda1, lambda2):

    # Calculating the probability density function
    x = np.linspace(0, max(lambda1, lambda2) * 10, 1000)
    pdf1 = (1 / lambda1) * np.exp(-x / lambda1) + 1e-10
    pdf2 = (1 / lambda2) * np.exp(-x / lambda2) + 1e-10
    
    # Calculating the mean distribution
    pdf_mean = 0.5 * (pdf1 + pdf2)
    
    # Calculate KL-Divergence
    kl1 = np.sum(pdf1 * np.log(pdf1 / pdf_mean))
    kl2 = np.sum(pdf2 * np.log(pdf2 / pdf_mean))
    
    # Calculate Jensen-Shannon Divergence
    jsd = 0.5 * (kl1 + kl2)
    
    return jsd


def calculate_distance(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    distance = np.linalg.norm(point1 - point2)
    return distance

import numpy as np 
def shannon_entropy(probabilities):

    probabilities = np.array(probabilities)
    
    #Entropy calculation for nonzero probability values
    non_zero_probabilities = probabilities[probabilities != 0]

    epsilon = 1e-10
    entropy = -np.sum(non_zero_probabilities * np.log2(non_zero_probabilities)) + epsilon
    
    return entropy

def cal_weighted_prob(filtered_df) : 
    weight_prob = np.zeros(3, dtype=np.float64)
    if len(filtered_df) == 0 : 
        return weight_prob

    sum_entropy = 0
    for i in filtered_df.index:
        prob = filtered_df.loc[i, ['N_prob', 'D_prob', 'M_prob']].tolist()
        entropy_value = shannon_entropy(prob)
        weight_prob += np.array(prob) * entropy_value
        sum_entropy += entropy_value

    weight_prob = weight_prob / sum_entropy
    return weight_prob


def cal_distance(pre_lambda, new_lambda, metric, class_): 
    pre_n_opt,pre_d_opt,pre_m_opt = pre_lambda
    n_opt, d_opt, m_opt = new_lambda
    original_point = [0,0,0]

    if metric in ['kl_rep', 'kl_rank'] : 
        n_diff = kl_divergence(n_opt, pre_n_opt)
        d_diff = kl_divergence(d_opt, pre_d_opt)
        m_diff = kl_divergence(m_opt, pre_m_opt)
        wsi_score = [n_diff, d_diff, m_diff]
        distance = calculate_distance(original_point, wsi_score)

    elif metric in ['js_rep', 'js_rank'] : 
        n_diff = jensen_shannon_divergence(n_opt, pre_n_opt)
        d_diff = jensen_shannon_divergence(d_opt, pre_d_opt)
        m_diff = jensen_shannon_divergence(m_opt, pre_m_opt)
        wsi_score = [n_diff, d_diff, m_diff]
        distance = calculate_distance(original_point, wsi_score)

    elif metric in ['kl_dis_x', 'js_dis_x'] : 
        if class_ == 'N' : distance = n_opt
        elif class_ == 'D' : distance = d_opt
        elif class_ == 'M' : distance = m_opt
    
    return distance


def cal_wsi_score(patch_result_df, wsi_patch_score, wsi_rep_method, label_lambda : list) :
    
    if wsi_patch_score in ['entropy', 'confidence', 'margin'] : 
        patch_result= patch_result_df[str(wsi_patch_score)]
        patch_result = np.array(patch_result.tolist())

        if wsi_rep_method == 'average' : 
            wsi_score = np.mean(patch_result, axis =0)
        
        elif wsi_rep_method == 'var' : 
            wsi_score = np.var(patch_result, axis = 0)


    elif wsi_patch_score == 'cdal' : 
        N_indices = patch_result_df[patch_result_df['label'] == 2].index 
        D_indices = patch_result_df[patch_result_df['label'] == 0].index 
        M_indices = patch_result_df[patch_result_df['label'] == 1].index 
        
        N_filtered_df = patch_result_df.loc[N_indices]
        D_filtered_df = patch_result_df.loc[D_indices]
        M_filtered_df = patch_result_df.loc[M_indices]

        wsi_score = [cal_weighted_prob(N_filtered_df), cal_weighted_prob(D_filtered_df), cal_weighted_prob(M_filtered_df)]
 
    elif wsi_patch_score == 'class_conf' : 
        n_wsi_lambda,d_wsi_lambda,m_wsi_lambda = label_lambda
        n_opt, d_opt, m_opt = cal_opt_lambda(patch_result_df, filter = True, threshold=0.99, train_lambda=n_wsi_lambda)
        new_lambda = [n_opt, d_opt, m_opt]
        
        n_dis = cal_distance(n_wsi_lambda, new_lambda, wsi_rep_method, 'N')
        d_dis = cal_distance(d_wsi_lambda, new_lambda, wsi_rep_method, 'D')
        m_dis = cal_distance(m_wsi_lambda, new_lambda, wsi_rep_method, 'M')
        
        wsi_score = [n_dis, d_dis, m_dis]
    elif wsi_patch_score == 'random' : 
        wsi_score = 0 

    return wsi_score

def log_likelihood(params, data):
    lambda_ = params[0]
    return -np.sum(np.log(lambda_ * np.exp(-lambda_ * np.array(data))))


def cal_opt_lambda(dataframe, filter, threshold, train_lambda = None) : 
    N_prob = dataframe['N_prob'].tolist()
    N_prob = [1 - value for value in N_prob]

    if filter == False : 
        D_prob = dataframe['D_prob'].tolist()
        M_prob = dataframe['M_prob'].tolist()

    elif filter == True: 
        D_prob = dataframe.loc[dataframe['N_prob'] < threshold, 'D_prob'].tolist()
        M_prob = dataframe.loc[dataframe['N_prob'] < threshold, 'M_prob'].tolist()

    D_prob = [1 - value for value in D_prob] 
    M_prob = [1 - value for value in M_prob] 

    
    # Initial value of the exponential distribution coefficient when no probability distribution case existed at all
    initial_lambda = 0.01

    N_result = minimize(log_likelihood, initial_lambda, args=(N_prob,), method='L-BFGS-B').x[0]

    if len(D_prob) == 0 : D_result = train_lambda[1]
    else : D_result = minimize(log_likelihood, initial_lambda, args=(D_prob,), method='L-BFGS-B').x[0]
    

    if len(M_prob) == 0 : M_result = train_lambda[2]
    else :  M_result = minimize(log_likelihood, initial_lambda, args=(M_prob,), method='L-BFGS-B').x[0]

    return N_result, D_result, M_result    

