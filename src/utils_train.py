from torchmetrics import AUROC, F1Score
from copy import deepcopy
from tqdm import tqdm
import time
import torch
import pandas as pd
import torch.nn.functional as F

def train(backbone, train_loader, val_loader, optimizer, num_epochs, model_dir, scheduler = None, use_val=False, logger=None):
    lowest_loss = 1e2
    best_model = deepcopy(backbone)

    loss_func = torch.nn.CrossEntropyLoss()
    backbone = backbone.cuda()

    # Training
    for epoch in range(1, num_epochs+1):

        since = time.perf_counter()
        running_loss = 0.0
        running_corrects = 0
        total_inputs = 0
        backbone.train()
        progress_bar = tqdm(train_loader, total=len(train_loader), desc=f"[{epoch}/{num_epochs}] Train", leave=False)
        for batch in progress_bar:
            inputs = batch[0].cuda()
            labels = batch[1].cuda()

            optimizer.zero_grad(set_to_none=True)
            outputs = backbone(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = outputs.data.max(1)
            running_loss += loss.item() * labels.size(0)
            running_corrects += torch.sum(preds==labels).item()
            total_inputs += labels.size(0)

        train_loss = running_loss / total_inputs
        train_acc = running_corrects / total_inputs
        if scheduler != None : 
            scheduler.step()

        timestamp = time.perf_counter() - since

        if use_val:
            running_loss = 0.0
            running_corrects = 0
            total_inputs = 0 
            backbone.eval()
            progress_bar = tqdm(val_loader, total=len(val_loader), desc=f"[{epoch}/{num_epochs}] Val", leave=False)
            with torch.no_grad():
                for batch in progress_bar:
                    inputs = batch[0].cuda()
                    labels = batch[1].cuda()
                    outputs = backbone(inputs)
                    loss = F.cross_entropy(outputs, labels)

                    _, preds = outputs.data.max(1)
                    running_loss += loss.item() * labels.size(0)
                    running_corrects += torch.sum(preds==labels).item()
                    total_inputs += labels.size(0)

            val_loss = running_loss / total_inputs
            val_acc = running_corrects / total_inputs

            if val_loss < lowest_loss:
                lowest_loss = deepcopy(val_loss)
                best_model = deepcopy(backbone)
                # torch.save(best_model.state_dict(), model_dir)
        else:
            val_loss = 0 
            val_acc = 0 

        print(f"[{epoch}/{num_epochs}] train loss: {train_loss:.3f}, train acc: {train_acc:.3f}, "
            f"val loss: {val_loss:.3f}, val acc: {val_acc:.3f}. "
            f"Time: {timestamp:.6f}sec")
        
        if logger is not None:
            print(f"[{epoch}/{num_epochs}] train loss: {train_loss:.3f}, train acc: {train_acc:.3f}, "
                f"val loss: {val_loss:.3f}, val acc: {val_acc:.3f}. "
                f"Time: {timestamp:.6f}sec", file=logger)

    if use_val:
        return best_model
    else:
        return backbone



def eval_model(model, test_loader, classes, logger=None):

    model.cuda()
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_inputs = 0
    all_preds = []
    all_labels = []
    all_outputs = []
    progress_bar = tqdm(test_loader, total=len(test_loader), desc="Evaluation", leave=False)
    with torch.no_grad():
        for batch in progress_bar:
            inputs = batch[0].cuda()
            labels = batch[1].cuda()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)

            _, preds = outputs.data.max(1)
            running_loss += loss.item() * labels.size(0)
            total_inputs += labels.size(0)
            running_corrects += torch.sum(preds==labels).item()
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())
            all_outputs.extend(outputs.tolist())

    test_loss = running_loss / total_inputs
    test_acc = running_corrects / total_inputs

    auroc = AUROC(task="multiclass", num_classes=len(classes))
    auroc_score = auroc(F.softmax(torch.tensor(all_outputs), dim=1), torch.tensor(all_labels))

    # f1 = F1Score(task="multiclass", num_classes=len(classes))
    # f1_score = f1(torch.tensor(all_preds), torch.tensor(all_labels))

    print(f"Test - loss: {test_loss:.3f}, acc {test_acc:.3f}, auc: {auroc_score:.3f}")

    # cm = confusion_matrix(all_labels, all_preds)
    # cm = pd.DataFrame(cm, index=[c for c in classes], columns=[c for c in classes])
    # print(cm)

    if logger is not None:
        print(f"Test - loss: {test_loss:.3f}, acc {test_acc:.3f}, auc: {auroc_score:.3f}", file=logger)
    
    return test_acc, auroc_score








