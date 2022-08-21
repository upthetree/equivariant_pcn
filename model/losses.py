#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
import sys
sys.path.append("..")
import dataset, data_util

import extension.emd.emd_module as emd
import extension.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as chamfer_3D


def fscore(dist1, dist2, threshold=0.001):
    """
    Calculates the F-score between two point clouds with the corresponding threshold value.
    :param dist1: Batch, N-Points
    :param dist2: Batch, N-Points
    :param th: float
    :return: fscore, precision, recall
    """
    # NB : In this depo, dist1 and dist2 are squared pointcloud euclidean distances, so you should adapt the threshold accordingly.
    precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
    precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0
    return fscore, precision_1, precision_2



def calc_cd(output, gt, calc_f1=False):
    cham_loss = chamfer_3D.chamfer_3DDist()
    dist1, dist2, _, _ = cham_loss(gt, output)
    cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    cd_t = (dist1.mean(1) + dist2.mean(1))
    if calc_f1:
        f1, _, _ = fscore(dist1, dist2)
        return cd_p, cd_t, f1
    else:
        return cd_p, cd_t


def calc_emd(output, gt, eps=0.005, iterations=50):
    emd_loss = emd.emdModule()
    dist, _ = emd_loss(output, gt, eps, iterations)
    emd_out = torch.sqrt(dist).mean(1)
    return emd_out


def cross_entropy_loss (pred_labels, gt_labels, label_smooth, n_class=6):
    '''
    pred_labels: (b, n, 6)
    gt_labels: (b, n)
    --------
    loss: value
    '''
    pred_labels = pred_labels.reshape(-1, n_class)     #(b*n, 6)
    gt_labels = gt_labels.reshape(-1)    #(b*n)
    
    if label_smooth:
        eps = 0.2
        one_hot = torch.zeros_like(pred_labels).scatter(1, gt_labels.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1) 
        log_prb = F.log_softmax(pred_labels, dim=1)
        
        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred_labels, gt_labels, reduction='mean')
    
    return loss
