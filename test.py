#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
import numpy as np
import dataset, data_util
import collections
import sys, os, time, logging, csv

from scipy.stats import special_ortho_group

'''Load module'''
import argparse, importlib
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='model')
parser.add_argument('--model_name', default='equcomp')
parser.add_argument('--novel_input', type=bool, default=False)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--result_path', default='./test_result/equnet_all')
parser.add_argument('--load_dir', default=None)
parser.add_argument('--cat', default=None)
args = parser.parse_args()

sys.path.append(args.model_path)
running_model = importlib.import_module(args.model_name)
import hyper_encoder, hyper_decoder

'''Save path'''
os.makedirs(args.result_path) if not os.path.exists(args.result_path) else None
csv_path = os.path.join(args.result_path, 'csv')
pcd_path = os.path.join(args.result_path, 'pcd')
os.makedirs(csv_path) if not os.path.exists(csv_path) else None
os.makedirs(pcd_path) if not os.path.exists(pcd_path) else None


'''Dataset'''
if args.cat is None:
    mvp_test = dataset.MVP2('mvp_dataset', train=False, npoints=2048, novel_input=args.novel_input)
    csv_name = os.path.join(csv_path, 'loss_all_30.csv')
else:
    mvp_test = dataset.MVP_cat('mvp_dataset', train=False, npoints=2048, cat=args.cat)
    csv_name = os.path.join(csv_path, 'loss_%s_30.csv'%args.cat)
dataloader_test = torch.utils.data.DataLoader(mvp_test, batch_size=args.batch_size, shuffle=False)
f = open(csv_name,'w')
writer = csv.writer(f)


'''Load Model'''
MODEL = running_model.model_completion(hyper_encoder, hyper_decoder, 'VTR')
MODEL.cuda()
MODEL.eval()

checkpoint = torch.load(os.path.join('info', args.load_dir, 'model/%s_cd.pth'%args.model_name))
start_epoch = checkpoint['epoch']
MODEL.load_state_dict(checkpoint['model_state_dict'])
print ('Restoring model of %s, epoch %d'%(args.load_dir, start_epoch))

pred_coarse_all = []
pred_fine_all = []
cat_name_all = ['airplane', 'cabinet', 'car', 'chair', 'lamp', 'sofa', 'table', 'vessel']

for i, data in enumerate(dataloader_test):
    
    label, points_xyz_raw, gt_coarse_raw, gt_fine_raw = data
    label = label.long()
    
    #####################
    # if i<5:
    #     data_util.view_pcd(points_xyz[2])
    #     data_util.view_pcd(gt_coarse[2])
    #     data_util.view_pcd(gt_fine[2])
    #####################
    
    emd_all = []
    cd_all = []
    
    for j in range(30):
        rot_mat = special_ortho_group.rvs(3)
        # rot_mat_t = rot_mat.T
        points_xyz = points_xyz_raw @ rot_mat
        gt_coarse = gt_coarse_raw @ rot_mat
        gt_fine = gt_fine_raw @ rot_mat
    
        points_xyz = points_xyz.cuda().float()
        gt_coarse = gt_coarse.cuda().float()
        gt_fine = gt_fine.cuda().float()
        with torch.no_grad():
            pred_coarse, pred_fine, loss_emd, loss_cdp = MODEL(points_xyz, gt_coarse, gt_fine, istraining=False)
        
        # pred_coarse = pred_coarse.detach().cpu().numpy()
        # pred_fine = pred_fine.detach().cpu().numpy()
        loss_emd = loss_emd.detach().cpu().numpy()[:, None]
        loss_cdp = loss_cdp.detach().cpu().numpy()[:, None]
        
        cd_all.append(loss_cdp)
        emd_all.append(loss_emd)
        
        print ('%d/%d'%(i, j))
    
    # pred_coarse_all.append(pred_coarse)
    # pred_fine_all.append(pred_fine)
    
    emd_all = np.hstack(emd_all)  #(b, 30)
    cd_all = np.hstack(cd_all)    #(b, 30)
    
    max_emd = np.max(emd_all, axis=1)
    min_emd = np.min(emd_all, axis=1)
    max_cd = np.max(cd_all, axis=1)
    min_cd = np.min(cd_all, axis=1)
    
    for j in range(label.shape[0]):
        writer.writerow([str(i*args.batch_size+j), '', cat_name_all[label[j]],'', 
                         str(max_emd[j]), '', str(min_emd[j]), '',
                         str(max_cd[j]), '', str(min_cd[j]), '',
                         str(max_emd[j]-min_emd[j]), '', str(max_cd[j]-min_cd[j])])
    

# pred_coarse_all = np.concatenate(pred_coarse_all, axis=0)
# pred_fine_all = np.concatenate(pred_fine_all, axis=0)
f.close()

# import h5py
# fp = h5py.File(os.path.join(pcd_path, 'output_%s.hdf5'%args.cat),'w')
# fp.create_dataset("fine", data = pred_fine_all)
# fp.create_dataset("coarse", data = pred_coarse_all)
# fp.close()
