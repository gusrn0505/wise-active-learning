from pathlib import Path
import random
import os
import torch
import torchvision
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import StepLR
import time
import config as cfg

from src.utils_selection_per_slide import select_random_patches_per_class, select_random_slides_per_class, select_wsi_by_strategy, select_patch_by_strategy
from src.dataloader import patch_dataloaders
from src.utils_train import train, eval_model


#PATCH_STRATEGY = ['passive', 'random','confidence', 'entropy', 'coreset', 'badge'] 
PATCH_STRATEGY = ['confidence', 'random', 'badge', 'coreset'] 

#WSI_PATCH_SCORE = ["random", "confidence", "entropy", "cdal", "class_conf"] 
WSI_PATCH_SCORE = ['entropy'] # 'entropy', 'confidence', 'class_conf'

# WSI_REP_METHOD = ["random", "average", "cdal", "kl_dis_x", "kl_rep", "kl_rank", "js_dis_x", "js_rep", "js_rank"]
WSI_REP_METHOD = [ 'random'] 
CLASSES = ['D', 'M', 'N']

NUM_RUNS = 1


def main(cfg, patch_strategy: str, wsi_patch_score :str, wsi_rep_method:str, logger):
    TRAIN_DIR = cfg.TRAIN_DIR
    VAL_DIR = cfg.VAL_DIR
    TEST_DIR = cfg.TEST_DIR
    INITIAL_WSI_PER_CLASS = cfg.INITIAL_WSI_PER_CLASS
    NUM_WSI_PER_GENERATION = cfg.NUM_WSI_PER_GENERATION
    NUM_PATCHES_PER_WSI = cfg.NUM_PATCHES_PER_WSI
    BASE_DIR =cfg.BASE_DIR
    LOAD_WSI_CSV = cfg.LOAD_WSI_CSV
    RESTORE_EXTRACTION = cfg.RESTORE_EXTRACTION
    CHECK_NUM_PATCHES = cfg.CHECK_NUM_PATCHES  # check whether the number of patches in a WSI is under 40
    NUM_PATCH_FILTER = cfg.NUM_PATCH_FILTER

    RANDOM_SEED = cfg.RANDOM_SEED
    MODEL_NAME = cfg.MODEL_NAME 
    BATCH_SIZE = cfg.BATCH_SIZE
    LEARNING_RATE = cfg.LEARNING_RATE
    NUM_EPOCHS = cfg.NUM_EPOCHS
    CYCLES = cfg.CYCLES

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)  


    model_dir = os.path.join(BASE_DIR, "models")
    result_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    experiment_results = []
    csv_dir = f"{BASE_DIR}/csv"
    os.makedirs(csv_dir, exist_ok=True)

    data = []
    for subset in ['train', 'val', 'test']:
        for label, class_ in enumerate(sorted(CLASSES)):
            data_dir = f"{TRAIN_DIR}/{class_}" if subset == 'train' else  f"{VAL_DIR}/{class_}" if subset == 'val' else f"{TEST_DIR}/{class_}"
            slides = [p for p in Path(data_dir).glob('*') if p.is_dir()]
            assert len(slides), f"There's no data in {data_dir}"
            for slide in slides:
                patches = list(slide.glob('*.jpg'))
                if CHECK_NUM_PATCHES and subset == 'train' and (len(patches) < NUM_PATCH_FILTER): 
                    continue
                for patch in patches:
                    data.append([subset, slide.stem, str(patch), label])

    full_train_pool =[d for d in data if d[0] == 'train'] 
    full_slides = list(set([d[1] for d in full_train_pool]))
    
    unlabeled_pool = full_train_pool.copy()
    unlabeled_slides = full_slides.copy()

    unlabeled_slides = random.sample(unlabeled_slides, len(unlabeled_slides))    # select all with random order
    
    val_data = [[img_path, label] for (subset, slide_name, img_path, label) in data if subset == 'val']
    assert len(val_data)

    test_data = [[img_path, label] for (subset, slide_name, img_path, label) in data if subset == 'test']
    assert len(test_data)

    val_loader = patch_dataloaders(data=val_data, is_train=False, batch_size=BATCH_SIZE)
    test_loader = patch_dataloaders(data=test_data, is_train=False, batch_size=BATCH_SIZE)   

    labeled_slides = []
    labeled_patch = []

    # Initialize an empty csv file for each strategy in all iterations
    # change DB setting
    base_columns = ['AL_iter', 'slide_name', 'img_path', 'loss', 'confidence', 'margin', 'label', 'prediction', 'oracle_imitation_score']
    wsi_base_columns = ['AL_iter', 'class', 'slide_name', 'selected']
    if patch_strategy in {'badge', 'coreset', 'cdal'}:
        base_columns.append('selected')
    all_selected_patch_in_csv = pd.DataFrame(columns=base_columns)
    all_selected_wsi_in_csv = pd.DataFrame(columns=wsi_base_columns)

    for cycle in range(1, CYCLES+1): 
        start_time = time.time()         
        if cycle == 1:
            # select data for initial train dataset

            selected_slides = select_random_slides_per_class(
                data=unlabeled_pool,
                classes=CLASSES,
                num_slides_per_class=INITIAL_WSI_PER_CLASS)
                
            
            selected_patch = [[subset, slide_name, img_path, label] 
                             for (subset, slide_name, img_path, label) in data if slide_name in selected_slides]
            
            selected_patch = select_random_patches_per_class(
                data=selected_patch, 
                patches_per_class={'D': 100, 'M': 100, 'N': 100}, 
                classes=CLASSES)  
            

            """
            # When fixing an initial learning dataset

            ini_train_loc = f"initial_trainset.csv"
            df = pd.read_csv(ini_train_loc)
            selected_slides = list(set(df['slide_name'].tolist()))
            selected_patch_info = df['img_path'].tolist()
            
            selected_patch = [[subset, slide_name, img_path, label] 
                             for (subset, slide_name, img_path, label) in data if img_path in selected_patch_info]
            """
            
        elif cycle > 1:
            label_patch_wsi = [[subset, slide_name, img_path, label] 
                             for (subset, slide_name, img_path, label) in full_train_pool if slide_name in labeled_slides]

            
            # To check effectiveness of patch active learning for selected WSI by each WSI AL methods
            if LOAD_WSI_CSV == True : 
                selected_wsi_loc = cfg.LOAD_WSI_LOCATION
                wsi_df = pd.read_csv(selected_wsi_loc)
                selected_wsi_per_iter_in_csv = wsi_df[wsi_df['AL_iter'] == cycle]
                selected_slides = selected_wsi_per_iter_in_csv[selected_wsi_per_iter_in_csv["selected"]  ==1]["slide_name"].tolist()


            else : 
                selected_slides, selected_wsi_per_iter_in_csv = select_wsi_by_strategy(                    
                            cycle=cycle,
                            wsi_patch_score = wsi_patch_score, 
                            wsi_rep_method = wsi_rep_method, 
                            model = model, 
                            label_data = label_patch_wsi,
                            full_data = full_train_pool,
                            unlabel_data= unlabeled_pool,
                            classes=CLASSES,
                            batch_size=BATCH_SIZE,
                            num_slides=NUM_WSI_PER_GENERATION, 
                            num_patches_per_wsi = NUM_PATCHES_PER_WSI
                            )

            selected_patch, selected_patch_per_iter_in_csv = select_patch_by_strategy(
                            cycle=cycle,
                            selected_slides = selected_slides,
                            patch_strategy=patch_strategy,
                            model=model,
                            data=unlabeled_pool,
                            labeled_loader=train_loader,
                            classes=CLASSES,
                            batch_size=BATCH_SIZE,
                            num_patches_per_slide=NUM_PATCHES_PER_WSI)

            print("Finish to select patch from each wsi")

            all_selected_patch_in_csv = pd.concat([all_selected_patch_in_csv, selected_patch_per_iter_in_csv], ignore_index=True)
            all_selected_wsi_in_csv = pd.concat([all_selected_wsi_in_csv, selected_wsi_per_iter_in_csv], ignore_index=True)
              
        # Update unlabeled pool
        labeled_patch.extend(selected_patch)
        labeled_slides.extend(selected_slides)

        # To compare pool-based AL methodogy with proposed method 
        if RESTORE_EXTRACTION is True : 
            unlabeled_slides = full_slides
            #unlabeled_pool = [d for d in unlabeled_pool if d[2] not in labeled_patch]
            unlabeled_pool = [d for d in unlabeled_pool if d[2] not in [labeled[2] for labeled in labeled_patch]]

            print("# of unlabled_pool: ", len(unlabeled_pool))

        else : 
            unlabeled_slides = [s for s in unlabeled_slides if s not in labeled_slides]
            unlabeled_pool = [d for d in unlabeled_pool if d[1] not in labeled_slides]
            print("restore_extraction : X ")


        labeled_df = pd.DataFrame(labeled_patch, columns=['subset', 'slide_name', 'img_path', 'label'])
        labeled_df.to_csv(f"{csv_dir}/{MODEL_NAME}_initialWSI{INITIAL_WSI_PER_CLASS}_WSI_{wsi_patch_score}{NUM_WSI_PER_GENERATION}__Patch_{patch_strategy}{NUM_PATCHES_PER_WSI}_cycle{cycle}.csv", index=False)

        print(f"Cycle: {cycle}. Unlabeled WSI: {len(unlabeled_slides)}. # of training WSI: {len(labeled_slides)}, # of training patches: {len(labeled_patch)}")  

        # TRAINING
        train_data = labeled_df[['img_path', 'label']].values.tolist()
        train_loader = patch_dataloaders(data=train_data, is_train=True, batch_size=BATCH_SIZE)

        if MODEL_NAME == 'resnet':
            backbone = torchvision.models.resnet18() # Since we hope to check early stage of learning curve of mode, we will not use pretrained model 
            backbone.fc = torch.nn.Linear(backbone.fc.in_features, len(CLASSES))

        elif MODEL_NAME == 'vgg':        
            backbone = torchvision.models.vgg16() # Since we hope to check early stage of learning curve of mode, we will not use pretrained model 
            in_features = backbone.classifier[-1].in_features
            backbone.classifier[-1] = torch.nn.Linear(in_features, len(CLASSES))

        backbone.cuda()

        
        optimizer = torch.optim.SGD(params=backbone.parameters(), lr=LEARNING_RATE, momentum=0.9, nesterov=True)
        # optimizer = torch.optim.SGD(params=backbone.parameters(), lr=LEARNING_RATE)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.2)

        model = train(
            backbone=backbone,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            num_epochs=NUM_EPOCHS,
            model_dir=model_dir,
            scheduler=scheduler,
            use_val=False 
        )

        test_acc, auroc_score = eval_model(
            model=model,
            test_loader=test_loader,
            classes=CLASSES
        )


        end_time = time.time()
        elapsed_time = end_time - start_time
        
        logger.write(f"{cycle},{len(labeled_slides)},{NUM_PATCHES_PER_WSI},{len(labeled_patch)}, {elapsed_time}, {auroc_score:.4f},{test_acc:.4f}\n")


        experiment_results.append([cycle, len(labeled_slides), NUM_PATCHES_PER_WSI, auroc_score, test_acc])

        print()

    # save in log folder
    all_selected_patch_in_csv.to_csv(f"{result_dir}/Patch_{wsi_patch_score}_{wsi_rep_method}_{patch_strategy}.csv", index=False)
    all_selected_wsi_in_csv.to_csv(f"{result_dir}/WSI_{wsi_patch_score}_{wsi_rep_method}_{patch_strategy}.csv", index=False)
    

if __name__=='__main__':  
    for _ in range(NUM_RUNS):
        trial_number = 1

        while Path(f"{cfg.BASE_DIR}{trial_number}").exists():
            trial_number += 1
        log_dir = f"{cfg.BASE_DIR}{trial_number}"
        Path(log_dir).mkdir(exist_ok=False, parents=True)
        print(f"Log dir: {log_dir}")
        
        for wsi_rep_method in WSI_REP_METHOD : 
            for wsi_patch_score in WSI_PATCH_SCORE:
                for patch_strategy in PATCH_STRATEGY : 

                    if wsi_patch_score == "random" : 
                        if wsi_rep_method != "random" : continue 

                    elif wsi_patch_score in ["confidence", "entropy"] : 
                        if wsi_rep_method != 'average' : continue 

                    elif wsi_patch_score == "cdal" : 
                        if wsi_rep_method != 'cdal' : continue 
                    
                    else : 
                        if wsi_rep_method not in ["kl_dis_x", "kl_rep", "kl_rank", "js_dis_x", "js_rep", "js_rank"] : continue


                    logger_name = f"{wsi_patch_score}_{wsi_rep_method}_{patch_strategy}_{cfg.INITIAL_WSI_PER_CLASS}WSI"
                    log_file = f"{log_dir}/test_{cfg.MODEL_NAME}_{logger_name}.txt"
                    

                    print(f"{wsi_patch_score}, {patch_strategy}, {cfg.INITIAL_WSI_PER_CLASS} initial wsi per class, {cfg.NUM_WSI_PER_GENERATION} wsi per generation, {cfg.NUM_PATCHES_PER_WSI} patches per wsi")
                    with open(log_file, "a") as logger:
                        logger.write("cycle, number_of_wsi,number_of_patches_per_wsi, train_patches, time, auroc_score, accuracy\n")
                        main(
                        patch_strategy=patch_strategy,
                        wsi_patch_score = wsi_patch_score, 
                        wsi_rep_method = wsi_rep_method,
                        logger=logger) 


