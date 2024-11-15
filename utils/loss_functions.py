#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 11:58:17 2023

@author: cgravelmiguel
@description: This script defines the different loss functions we can use.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# DEFINE METRICS TO COMPUTE

"""
Notes:
    
In (preds, 1), the 1 says that the maximum value will be calculated per row (the second dimension in a [[],[]]
Then, the [1] returns a tensor with the indices of the maximum values instead of the max values themselves.

Therefore, torch.max(preds, 1)[1] == targets will return true when the index of the max value is the same as the target.

"""

def compute_metrics(preds, targets):
    
    '''
    Calculates the number of True Positives, False Positives, and False Negatives, which allows then calculating recall, precision, and F1

    Parameters
    ----------
    preds : PyTorch array
        Array of the predicted location of objects (predicted mask).
    targets : PyTorch array
        Location of the annotated objects to compare to (actual mask).

    Returns
    -------
    recall : double
        Percentage of actual objects that are correctly predicted by the model.
    precision : double
        Percentage of predicted objects that are correct.
    f1 : double
        Hamornic mean of recall and precision.
    mcc : double
        Matthews Correlation Coefficient

    '''
    
    TP = torch.sum((targets == 1) & (torch.max(preds, 1)[1] == 1))
    TN = torch.sum((targets == 0) & (torch.max(preds, 1)[1] == 0))
    FP = torch.sum((targets == 0) & (torch.max(preds, 1)[1] == 1))
    FN = torch.sum((targets == 1) & (torch.max(preds, 1)[1] == 0))
    
    #print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    
    # metrics
    recall = TP / (TP + FN + 1e-7)
    precision = TP / (TP + FP + 1e-7)
    f1 = (2 * recall * precision)/(recall + precision + 1e-7)
    mcc = (TP*TN-FP*FN)/torch.sqrt(float(TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    #print(f"recall: {recall}, precision: {precision}, f1: {f1}, mcc: {mcc}")
    return recall, precision, f1, mcc

# DEFINE THE LOSS FUNCTIONS

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss

        if self.reduction == 'mean':
            focal_loss = torch.mean(focal_loss)
        elif self.reduction == 'sum':
            focal_loss = torch.sum(focal_loss)

        return focal_loss

# This function come from https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch#Jaccard/Intersection-over-Union-(IoU)-Loss)
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)     
        
        # Extracts only the positive layer of the predicted mask
        inputs = inputs[:, 1, :, :]  # assuming class 1 is the positive class
        
        # Flatten the H and W of each channel
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1) # Then change the shape by multiplying the B*H*W
        
        # calculate the intersection   
        intersection = (inputs * targets).sum()
                  
        # calculate the dice loss for each channel
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

# This function come from https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch#Jaccard/Intersection-over-Union-(IoU)-Loss)
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        # Extracts only the positive layer of the predicted mask
        inputs = inputs[:, 1, :, :]  # assuming class 1 is the positive class
        
        # Flatten the H and W of each channel
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1) # Then change the shape by multiplying the B*H*W
        
        # calculate the intersection
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()    
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

def UnetLoss(preds, targets, loss_type):
    '''
    Calls the correct metrics function based on the needs.

    Parameters
    ----------
    preds : PyTorch array
        Array of the predicted location of objects (predicted mask).
    targets : PyTorch array
        Location of the annotated objects to compare to (actual mask).
    loss_type : str
        Name of the loss computed (can be focal, dice, or iou).

    Returns
    -------
    loss (focal, dice, or iou) : double
        Computed loss.
    acc : double
        Percentage of pixels correctly predicted.
    recall : double
        Percentage of actual objects that are correctly predicted by the model.
    precision : double
        Percentage of predicted objects that are correct.
    f1 : double
        Hamornic mean of recall and precision.
    mcc : double
        Matthews Correlation Coefficient

    '''
    
    acc = (torch.max(preds, 1)[1] == targets).float().mean()
    recall, precision, f1, mcc = compute_metrics(preds, targets)
    
    if loss_type == "focal":
        focal_loss = FocalLoss()(preds, targets)
        return focal_loss, acc, recall, precision, f1, mcc
    if loss_type == "dice":
        dice_loss = DiceLoss()(preds, targets, smooth = 1)
        return dice_loss, acc, recall, precision, f1, mcc
    if loss_type == "iou":
        iou_loss = IoULoss()(preds, targets, smooth = 1)
        return iou_loss, acc, recall, precision, f1, mcc

# THE END
