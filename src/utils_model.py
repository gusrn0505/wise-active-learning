import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pathlib import Path

def get_model_results(cycle, model, loader):
    model.cuda()
    model.eval()
    results = []
    with torch.no_grad():
        for batch in loader : 
        #for batch in tqdm(loader, desc = "calculate patch", ncols = 100):
            inputs = batch[0].cuda()
            labels = batch[1].cuda()
            img_paths = batch[2]
            # e.g.: "2019S000042301-28_69.jpg"  ==> 2019S000042301
            slides = [Path(img_path).stem.split('-')[0] for img_path in img_paths]
            outputs = model(inputs)
            losses = F.cross_entropy(outputs, labels, reduction='none')
            # softmax 추가하기 
            total_prob = F.softmax(outputs, dim=1)
            confs, preds = total_prob.data.max(1)
            
            for i in range(len(img_paths)):
                D_prob,M_prob,N_prob = total_prob[i]
                top_probs, top_classes = total_prob[i].topk(2)  # 가장 높은 2개 클래스의 확률과 인덱스를 가져옵니다.
                margin = top_probs[0] - top_probs[1]  # margin은 가장 높은 확률 - 두 번째로 높은 확률 입니다.
                results.append([cycle, slides[i], img_paths[i], losses[i].item(), round(N_prob.item(), 5), round(D_prob.item(),5), round(M_prob.item(), 5), confs[i].item(), margin.item(), labels[i].item(), preds[i].item()])

    return pd.DataFrame(results, columns=['AL_iter', 'slide_name', 'img_path', 'entropy', 'N_prob', 'D_prob', 'M_prob', 'confidence', 'margin', 'label', 'prediction'])



class ModifiedModel(nn.Module):
    def __init__(self, model, num_classes):
        super(ModifiedModel, self).__init__()
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
        self.fc = nn.Linear(model.fc.in_features, num_classes)

    def forward(self, x):
        embeddings = self.feature_extractor(x).squeeze()
        logits = self.fc(embeddings)
        return embeddings, logits

def get_features(model, loader):
    model.cuda()
    model.eval()

    features = []
    feature_extractor = nn.Sequential(*list(model.children())[:-1])

    with torch.no_grad():
        for batch in loader:
            inputs = batch[0].cuda()
            labels = batch[1].cuda()
            outputs = feature_extractor(inputs)
            outputs = torch.flatten(outputs, 1)
            for feature in outputs:
                features.append(feature.cpu().numpy())
    
    return features

def predict_prob_embeddings(model, loader, num_classes):
    model.cuda()
    model.eval()

    num_samples = len(loader.dataset)
    probs = torch.zeros([num_samples, num_classes])
    embeddings = torch.zeros([num_samples, model.fc.in_features]) # model.fc.in_features: 512 (extracted number of features from resnet18)

    modified_model = ModifiedModel(model, num_classes).cuda()

    sample_idx = 0
    with torch.no_grad():
        for batch in loader:

            inputs = batch[0].cuda()
            labels = batch[1].cuda()
            embeds, logits = modified_model(inputs)
            batch_size = inputs.shape[0]

            if len(labels) > 1:
                prob = F.softmax(logits, dim=1)
            else :
                prob = F.softmax(logits, dim=0)
            #prob = F.softmax(logits, dim=1)
            probs[sample_idx: sample_idx + batch_size] = prob.cpu()

            embeddings[sample_idx: sample_idx + batch_size] = embeds.cpu()
            sample_idx += batch_size

    return probs, embeddings

def get_grad_embeddings(model, loader, num_classes):
    model.cuda()
    model.eval()

    embeddings = np.zeros([len(loader.dataset), model.fc.in_features * num_classes]) # model.fc.in_features: 512 (extracted number of features from resnet18)

    modified_model = ModifiedModel(model, num_classes).cuda()

    sample_idx = 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch[0].cuda()
            labels = batch[1].cuda()
            #print("num of labels :", len(labels))
            embeds, logits = modified_model(inputs)
            embeds = embeds.data.cpu().numpy()
            if len(labels) > 1: 
                batchProbs = F.softmax(logits, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs, 1)
                for j in range(len(labels)):
                    for c in range(num_classes):
                        if c == maxInds[j]:
                            embeddings[sample_idx + j][model.fc.in_features * c: model.fc.in_features * (c + 1)] = copy.deepcopy(embeds[j]) * (1 - batchProbs[j][c])
                        else:
                            embeddings[sample_idx + j][model.fc.in_features * c: model.fc.in_features * (c + 1)] = copy.deepcopy(embeds[j]) * (-1 * batchProbs[j][c])

            else : 
                batchProbs = F.softmax(logits, dim=0).data.cpu().numpy()
                maxInds = np.argmax(batchProbs, axis=0)
                for c in range(num_classes) : 
                    if c == maxInds:
                        embeddings[sample_idx][model.fc.in_features * c: model.fc.in_features * (c + 1)] = copy.deepcopy(embeds) * (1 - batchProbs[c])
                    else:
                        embeddings[sample_idx][model.fc.in_features * c: model.fc.in_features * (c + 1)] = copy.deepcopy(embeds) * (-1 * batchProbs[c])

            sample_idx += len(labels)
    
    return torch.Tensor(embeddings)

