import torch
import random
import pandas as pd
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm 

import numpy as np

from src.utils_calculation import cal_wsi_rank, cal_wsi_score, cal_opt_lambda
from src.utils_model import get_features, predict_prob_embeddings, get_grad_embeddings, get_model_results
from src.utils_badge import init_centers
from src.utils_cdal import make_cdal_dist
from src.utils_coreset import Coreset_Greedy
from src.dataloader import patch_dataloaders

def select_random_patches_per_class(data: list, patches_per_class: dict, classes: list):

    selected_data = []
    for l, class_ in enumerate(sorted(classes)):
        sublist = [[subset, slide_name, img_path, label] for (subset, slide_name, img_path, label) in data if label==l]
        sublist = random.sample(sublist, k=patches_per_class[class_])
        selected_data.extend(sublist)
    return selected_data


def select_random_slides_per_class(data, classes, num_slides_per_class):

    selected_slides = []
    for label, class_ in enumerate(sorted(classes)):        
        # select slide from unlabeled dataset if WSI_label == label 
        n_slides = list(set([d[1] for d in data if d[3] == label]))
        assert len(n_slides) >= num_slides_per_class
        n_slides = n_slides[:num_slides_per_class]
        selected_slides.extend(n_slides)

    return selected_slides


def select_data_by_random(data, classes, num_slides, num_patches_per_slide):
    slides = list(set([d[1] for d in data]))
    
    # Balance between class 
    selected_slides = select_random_slides_per_class(data, classes, num_slides//3)
    #selected_slides = random.sample(slides, num_slides)
    
    # prevent the case "num_patches_per_slide > len(patches)" 
    selected_data = []        
    for slide in selected_slides:
        patches = [[subset, slide_name, img_path, label] for (subset, slide_name, img_path, label) in data if slide_name==slide]
        selected_data.extend(random.sample(patches, min(num_patches_per_slide, len(patches))))
    return selected_slides, selected_data

def select_data_all(data, classes, num_slides) : 
    slides = list(set([d[1] for d in data]))
    
    # Balance between class 
    selected_slides = select_random_slides_per_class(data, classes, num_slides//3)
    
    # prevent the case "num_patches_per_slide > len(patches)" 
    selected_data = []        
    for slide in selected_slides:
        patches = [[subset, slide_name, img_path, label] for (subset, slide_name, img_path, label) in data if slide_name==slide]
        selected_data.extend(patches)
    return selected_slides, selected_data


def select_wsi_by_strategy(cycle, wsi_patch_score, wsi_rep_method, model, label_data, full_data, unlabel_data, classes, batch_size, num_slides, num_patches_per_wsi) :
    selected_slides = []
    model.eval()

    unlabeled_df = pd.DataFrame(unlabel_data, columns=['subset', 'slide_name', 'img_path', 'label'])
    full_df = pd.DataFrame(full_data, columns=['subset', 'slide_name', 'img_path', 'label'])

    # Filter slide_name with less data than num_patch
    slide_name_counts = unlabeled_df['slide_name'].value_counts()
    filtered_slide_names = slide_name_counts[slide_name_counts >= num_patches_per_wsi].index
    
    filter_df = full_df[full_df['slide_name'].isin(filtered_slide_names)]
    slide_list = list(set(filter_df['slide_name'].tolist()))
    
    n_slides = slide_list

    unlabeled_loader = patch_dataloaders(
                data=filter_df[['img_path', 'label']].values.tolist(), 
                is_train=False,
                batch_size=batch_size)

    # columns=['AL_iter', 'slide_name', 'img_path', 'loss', 'N_prob', 'D_prob', 'M_prob', 'confidence', 'margin', 'label', 'prediction']
    model_result_df = get_model_results(cycle=cycle, model=model, loader=unlabeled_loader)
    
    label_lambda = None
    if wsi_rep_method in ["kl_dis_x", "kl_rep", "kl_rank", "js_dis_x", "js_rep", "js_rank"] : 
        labeled_df = pd.DataFrame(label_data, columns=['subset', 'slide_name', 'img_path', 'label'])
        labeled_loader = patch_dataloaders(
                data=labeled_df[['img_path', 'label']].values.tolist(), 
                is_train=False,
                batch_size=batch_size)
        label_wsi_result_df = get_model_results(cycle=cycle, model=model, loader=labeled_loader)
        

        n_wsi_result_df = label_wsi_result_df.loc[label_wsi_result_df['label']==2]
        label_n_opt, label_d_opt, label_m_opt = cal_opt_lambda(n_wsi_result_df, filter = True, threshold=0.99)
        n_wsi_lambda = [label_n_opt, label_d_opt, label_m_opt]
        
        d_wsi_result_df = label_wsi_result_df.loc[label_wsi_result_df['label']==0]
        label_n_opt, label_d_opt, label_m_opt = cal_opt_lambda(d_wsi_result_df, filter = True, threshold=0.99)
        d_wsi_lambda = [label_n_opt, label_d_opt, label_m_opt]
        
        m_wsi_result_df = label_wsi_result_df.loc[label_wsi_result_df['label']==1]
        label_n_opt, label_d_opt, label_m_opt = cal_opt_lambda(m_wsi_result_df, filter = True, threshold=0.99)
        m_wsi_lambda = [label_n_opt, label_d_opt, label_m_opt]

        label_lambda = [n_wsi_lambda, d_wsi_lambda, m_wsi_lambda]


    # set different base_columns 
    if wsi_patch_score in ['entropy', 'confidence', 'margin', 'random'] : 
        base_columns = ['AL_iter', 'slide_name', 'class', 'wsi_patch_score', 'wsi_rep_method', 'score', 'selected']
    
    elif wsi_patch_score in ['class_conf'] : 
        base_columns = ['AL_iter', 'slide_name', 'class', 'wsi_patch_score', 'wsi_rep_method', 'N_distance', 'D_distance', 'M_distance', 'selected']

    elif wsi_patch_score in ['cdal'] : 
        base_columns = ['AL_iter', 'slide_name', 'class', 'wsi_patch_score', 'wsi_rep_method', 'weighted_N', 'weighted_D', 'weighted_M', 'selected']

    
    wsi_score = [] 
    for wsi in tqdm(n_slides, desc = "calculate representative value for each wsi", ncols = 100): 
        patch_result_per_wsi = model_result_df.loc[model_result_df['slide_name']==wsi]

        score = cal_wsi_score(patch_result_per_wsi, wsi_patch_score, wsi_rep_method, label_lambda)
        class_ = patch_result_per_wsi['label'].iloc[0]
        cls_dic = {0 : "D", 1 : "M", 2: "N"}
        class_ =cls_dic[class_]
        
        if isinstance(score, list): 
            n_opt, d_opt, m_opt = score
            wsi_score.append([cycle, wsi, class_, wsi_patch_score, wsi_rep_method, n_opt, d_opt, m_opt, 0])

        else :  
            wsi_score.append([cycle, wsi, class_, wsi_patch_score, wsi_rep_method, score, 0])
    

    wsi_unlabeled_df = pd.DataFrame(wsi_score, columns=base_columns)
        
    if wsi_rep_method == 'random' : 
        top_indices = random.sample(range(1, len(wsi_unlabeled_df)), min(num_slides, len(n_slides)))
        wsi_unlabeled_df["selected"] = np.where(wsi_unlabeled_df.index.isin(top_indices), 1, 0)
        subdf = wsi_unlabeled_df[wsi_unlabeled_df['selected'] == 1]['slide_name'].tolist()
        selected_slides.extend(subdf)
                
    elif wsi_rep_method in [ 'kl_rep', 'js_rep', 'kl_dis_x', 'js_dis_x'] :
        embedding = torch.Tensor(wsi_unlabeled_df[['N_distance', 'D_distance', 'M_distance']].values.tolist())    
        selected_data_index = init_centers(embedding, num_slides)               
        wsi_unlabeled_df["selected"] = np.where(wsi_unlabeled_df.index.isin(selected_data_index), 1, 0)
        subdf = wsi_unlabeled_df[wsi_unlabeled_df['selected'] == 1]['slide_name'].tolist()
        selected_slides.extend(subdf)

    elif wsi_rep_method in ['kl_rank' , 'js_rank'] :        
        top_indices = cal_wsi_rank(wsi_unlabeled_df, num_slides)
        wsi_unlabeled_df["selected"] = np.where(wsi_unlabeled_df.index.isin(top_indices), 1, 0)

        subdf = wsi_unlabeled_df[wsi_unlabeled_df['selected'] == 1]['slide_name'].tolist()
        selected_slides.extend(subdf)


    elif wsi_rep_method == 'cdal' : 
        embedding = torch.Tensor(wsi_unlabeled_df[['weighted_N', 'weighted_D', 'weighted_M']].values.tolist())
        dist_matrix = make_cdal_dist(embedding) 
        selected_data_index = init_centers(dist_matrix, num_slides)               
        wsi_unlabeled_df["selected"] = np.where(wsi_unlabeled_df.index.isin(selected_data_index), 1, 0)
        subdf = wsi_unlabeled_df[wsi_unlabeled_df['selected'] == 1]['slide_name'].tolist()
        selected_slides.extend(subdf)

    elif wsi_rep_method in ['average'] : 
        if wsi_patch_score in 'confidence': 
            wsi_unlabeled_df = wsi_unlabeled_df.sort_values(by='score', ascending=True)
            
        elif wsi_patch_score == 'entropy' : 
            wsi_unlabeled_df = wsi_unlabeled_df.sort_values(by='score', ascending=False)
                
        top_indices = wsi_unlabeled_df.head(num_slides).index
        wsi_unlabeled_df["selected"] = np.where(wsi_unlabeled_df.index.isin(top_indices), 1, 0)
        subdf = wsi_unlabeled_df[wsi_unlabeled_df['selected'] == 1]['slide_name'].tolist()
        selected_slides.extend(subdf)

    return selected_slides, wsi_unlabeled_df


def select_patch_by_strategy(cycle, selected_slides, patch_strategy, model, data, labeled_loader, classes, batch_size, num_patches_per_slide):
    selected_data = [[subset, slide_name, img_path, label] for (subset, slide_name, img_path, label) in data if slide_name in selected_slides]

    AL_iter_unlabeled_df = pd.DataFrame(selected_data, columns=['subset', 'slide_name', 'img_path', 'label'])
    
    # Iterate over the selected slides (balance) to implement AL strategies for each WSI
    selected_patches = []

    # Initialize an empty csv file for each strategy and each iteration
    base_columns = ['AL_iter', 'slide_name', 'img_path', 'entropy', 'confidence', 'margin', 'label', 'prediction', 'oracle_imitation_score']
    if patch_strategy in {'badge', 'coreset'}:
        base_columns.append('selected')
    all_selected_data_per_iter_in_csv = pd.DataFrame(columns=base_columns)


    for wsi in tqdm(selected_slides, desc = "selecting patches per wsi", ncols = 100):
        AL_WSI_iter_unlabeled_df = AL_iter_unlabeled_df[AL_iter_unlabeled_df["slide_name"] == wsi]

        AL_WSI_iter_unlabeled_loader = patch_dataloaders(
            data=AL_WSI_iter_unlabeled_df[['img_path', 'label']].values.tolist(), 
            is_train=False,
            batch_size=batch_size)

        # ['AL_iter', 'slide_name', 'img_path', 'entropy', 'N_prob', 'D_prob', 'M_prob', 'confidence', 'margin', 'label', 'prediction']
        wsi_model_result_df = get_model_results(cycle=cycle, model=model, loader=AL_WSI_iter_unlabeled_loader)

        if patch_strategy == 'random' : 
            selected_data_index = random.sample(range(len(wsi_model_result_df)), k=num_patches_per_slide)

        elif patch_strategy == 'passive' : 
            selected_data_index = range(len(wsi_model_result_df))

        elif patch_strategy == 'entropy':
            # Highest entropy first
            wsi_model_result_df = wsi_model_result_df.sort_values(by='entropy', ascending=False)

        elif patch_strategy == 'confidence':
            # LEAST CONFIDENCE
            # check - high confidence is better? then ascending should be 'False'
            wsi_model_result_df = wsi_model_result_df.sort_values(by='confidence', ascending=True)
        

        elif patch_strategy == 'badge':
            # BADGE: https://arxiv.org/abs/1906.03671
            # Apply a BADGE to patches for each WSI
            probs, embeddings = predict_prob_embeddings(model, AL_WSI_iter_unlabeled_loader, num_classes=len(classes))
            _, idxs = probs.sort(descending=True)
            gradEmbeddings = get_grad_embeddings(model, AL_WSI_iter_unlabeled_loader, num_classes=len(classes))
            selected_data_index = init_centers(gradEmbeddings, num_patches_per_slide)

        elif patch_strategy == 'coreset':
            # Coreset: https://arxiv.org/abs/1708.00489
            # In the x-th iteration, utilize labeled data from the previous (x-1) iterations and incorporate patches from each WSI as unlabeled data
            labeled_features = get_features(model, labeled_loader)
            unlabeled_features = get_features(model, AL_WSI_iter_unlabeled_loader)
            all_features = labeled_features + unlabeled_features
            labeled_indices = np.arange(0, len(labeled_features))

            coreset = Coreset_Greedy(all_features)
            data_index, max_distance = coreset.sample(labeled_indices, num_patches_per_slide)

            # Unlabeled rows start after labeled_rows in all_features, so offset the indices
            selected_data_index = [i - len(labeled_features) for i in data_index]
    

        if patch_strategy in {'random', 'passive'} : 
            # random sample 한 것의 index 을 구해야함 
            wsi_model_result_df["selected"] = np.where(wsi_model_result_df.index.isin(selected_data_index), 1, 0)

        elif patch_strategy in {'entropy', 'confidence', 'oracle_imitation'}:
            # Record whether the patches are selected by either 'entropy', or 'confidence' strategy -> based on the top(num_patches_per_slide) after sorting
            selected_data_index = wsi_model_result_df.head(num_patches_per_slide).index
            wsi_model_result_df["selected"] = np.where(wsi_model_result_df.index.isin(selected_data_index), 1, 0)


        elif patch_strategy in {'badge', 'coreset'}:
            # Record whether the patches are selected by either 'badge' or 'coreset' strategy -> based on the selected data index
            wsi_model_result_df["selected"] = np.where(wsi_model_result_df.index.isin(selected_data_index), 1, 0)
            # Filter out the patches based on the selected data index
        
        wsi_model_result_df_selected = wsi_model_result_df[wsi_model_result_df['selected'] == 1]
        subdf = wsi_model_result_df_selected['img_path'].tolist()
        #selected_patches.extend(subdf[:min(num_patches_per_slide, len(subdf))])
        selected_patches.extend(subdf)
        
        all_selected_data_per_iter_in_csv = pd.concat([all_selected_data_per_iter_in_csv, wsi_model_result_df], ignore_index=True)    
        
    selected_data = [[subset, slide_name, img_path, label] for (subset, slide_name, img_path, label) in data if img_path in selected_patches]

    return selected_data, all_selected_data_per_iter_in_csv
