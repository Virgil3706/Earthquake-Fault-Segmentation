import torch

from functions import iou_pytorch,acc_metrics

best_iou_threshold=0.5

def IoU(predicted_mask, masks):
    c =iou_pytorch(predicted_mask.squeeze(1).byte(), masks.squeeze(1).byte())
    return torch.mean(c)

def accuracy1(outputs,labels):
    outputs=outputs.squeeze(1).byte()
    labels=labels.squeeze(1).byte()
    # acc = ((outputs==labels).sum().item() )/ ((outputs==labels).sum().item()+(outputs!=labels).sum().item())
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    # TP    predict 和 label 同时为1
    TP += ((outputs == 1) & (labels == 1)).sum().item()
    # TN    predict 和 label 同时为0
    TN += ((outputs == 0) & (labels == 0)).sum().item()
    # FN    predict 0 label 1
    FN += ((outputs == 0) & (labels == 1)).sum().item()
    # FP    predict 1 label 0
    FP += ((outputs == 1) & (labels == 0)).sum().item()

    acc = (TP + TN) / (TP + TN + FP + FN)

    return acc