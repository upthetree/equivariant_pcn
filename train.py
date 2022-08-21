#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
import numpy as np
import dataset, data_util
import collections
import sys, os, time, logging
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from scipy.stats import special_ortho_group

from tensorboardX import SummaryWriter

'''Load module'''
import argparse, importlib
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='model')
parser.add_argument('--model_name', default='equcomp')
parser.add_argument('--novel_input', type=bool, default=False)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_epoch', type=int, default=60)
parser.add_argument('--load_dir', default=None)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_decay', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--opt_name', default='Adam')
parser.add_argument('--decay_type', default='step')
parser.add_argument('--step_size', type=int, default=10)
parser.add_argument('--use_emd', type=bool, default=True)

### Normal training ###
parser.add_argument('--cat', default=None)
parser.add_argument('--sparse', type=bool, default=False)

args = parser.parse_args()

sys.path.append(args.model_path)
running_model = importlib.import_module(args.model_name)
import hyper_encoder, hyper_decoder

'''Save path'''
time_folder = time.strftime("%Y%m%d_%H%M%S", time.localtime())
if args.cat is not None:
    time_folder = '%s_%s'%(time_folder, args.cat)
    
if not os.path.exists(os.path.join('info', time_folder)):
    os.makedirs(os.path.join('info', time_folder, 'log'))
    os.makedirs(os.path.join('info', time_folder, 'model'))
    os.makedirs(os.path.join('info', time_folder, 'data'))
    
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_name = os.path.join('info', time_folder, 'log/%s.log'%args.model_name)
fh = logging.FileHandler(log_name, mode='w')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s: %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)

writer_train = SummaryWriter(os.path.join('info', time_folder, 'log/run'))


'''Dataset'''
if args.cat is None:
    mvp_train = dataset.MVP2('mvp_dataset', train=True, npoints=2048, novel_input=args.novel_input)
    mvp_test = dataset.MVP2('mvp_dataset', train=False, npoints=2048, novel_input=args.novel_input)
    dev = 1
else:
    mvp_train = dataset.MVP_cat('mvp_dataset', train=True, npoints=2048, cat=args.cat)
    mvp_test = dataset.MVP_cat('mvp_dataset', train=False, npoints=2048, cat=args.cat)
    dev = 8

if args.novel_input:
    num_train_pt = 62400//dev
    num_test_pt = 41600//dev
else:
    num_train_pt = 41600//dev
    num_test_pt = 31200//dev

if args.novel_input:
    num_cat = 16
else:
    num_cat = 8
    

'''Model and Optimizer'''
if args.sparse:
    MODEL = running_model.model_completion(hyper_encoder, hyper_decoder, 'PCN')
else:
    MODEL = running_model.model_completion(hyper_encoder, hyper_decoder, 'VTR')
MODEL.cuda()

def new_optimizer(opt_name, lr):
    if opt_name == 'Adam':
        optimizer = torch.optim.Adam(MODEL.parameters(), lr=lr)
    elif args.opt_name == 'SGD':
        optimizer = torch.optim.SGD(MODEL.parameters(), lr=lr, momentum=0.9)
    print ('create a new optimizer: %s'%opt_name)
    return optimizer

def adjust_lr(epoch, decay_type='step', lr_start=1e-3, lr_min=1e-5, decay=0.5, steps=15, max_epoch=50):
    assert decay_type in ['step', 'cosine']
    if decay_type=='step':
        lr = np.max([lr_start*np.power(decay, epoch//steps), lr_min])
    if decay_type=='cosine':
        lr = lr_min + (lr_start-lr_min)*0.5*(1+np.cos(np.pi*epoch/max_epoch))
    return lr


lr = args.lr
optimizer = new_optimizer(args.opt_name, args.lr)
logger.info('Optimizer: %s\n'%args.opt_name)
    
if args.load_dir is not None:
    checkpoint = torch.load(os.path.join('info', args.load_dir, 'model/%s.pth'%args.model_name))
    start_epoch = checkpoint['epoch']+1
    MODEL.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    logger.info('Restoring model of %s'%time_folder)
else:
    logger.info('Training a new model')
    start_epoch = 0

global_epoch = 0
best_emd = 10.0
best_cd = 10.0


'''Train and Test'''
for epoch in range(start_epoch, args.num_epoch):
    ### -------- ###
    print ('Training...')
    ### -------- ###
    if epoch < 5:
        alpha = 0.001
    elif epoch < 10:
        alpha = 0.01
    elif epoch < 20:
        alpha = 0.1
    elif epoch < 30:
        alpha = 0.2
    elif epoch < 40:
        alpha = 0.5
    else:
        alpha = 1.0
    
    lr = adjust_lr(epoch, args.decay_type, args.lr, 1e-5, args.lr_decay, args.step_size, args.num_epoch)
    if epoch in [5, 10, 20, 30, 40]:
        optimizer = new_optimizer(args.opt_name, lr)
    else:
        for param_group in optimizer.param_groups:
        	param_group['lr'] = lr
    
    dataloader = torch.utils.data.DataLoader(mvp_train, batch_size=args.batch_size, shuffle=True)
    
    logger.info('Epoch %d/%s:' % (epoch+1, args.num_epoch))
    logger.info('Learning rate: %f, alpha: %f'%(optimizer.param_groups[0]['lr'], alpha))
    print ('Learning rate: %f'%optimizer.param_groups[0]['lr'])
    
    all_emd = []
    all_cd = []
    
    MODEL.train()
    
    for i, data in enumerate(dataloader, 0):
        label, points_xyz, gt_coarse, gt_fine = data
        points_xyz = points_xyz.cuda().float()
        gt_coarse = gt_coarse.cuda().float()
        gt_fine = gt_fine.cuda().float()
        points_xyz, _ = dataset.shuffle_points(points_xyz, batch_label=None)
        
        if args.sparse:
            points_xyz, _ = dataset.shuffle_points(points_xyz, batch_label=None)
            points_xyz = points_xyz[:, 0:256, :]
        
        pred_coarse, pred_fine, loss_emd, loss_cdp = MODEL(points_xyz, gt_coarse, gt_fine, istraining=True, use_emd=args.use_emd)
        loss_emd = loss_emd.mean()
        loss_cdp = loss_cdp.mean()
        
        loss = loss_emd + alpha*loss_cdp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        all_emd.append(loss_emd.item()*100)
        all_cd.append(loss_cdp.item()*100)
        print ('Epoch %d, %d/%d: alpha %.3f, emd %.4f, cd %.4f'%(epoch, i+1, num_train_pt//args.batch_size+1, alpha, loss_emd.item()*100, loss_cdp.item()*100))
                
    all_emd = np.hstack(all_emd)
    all_cd = np.hstack(all_cd)
    
    writer_train.add_scalar('train_emd', np.mean(all_emd), global_step=epoch)
    writer_train.add_scalar('train_cd', np.mean(all_cd), global_step=epoch)
    
    ### -------- ###
    print ('Test...')
    ### -------- ###
    dataloader = torch.utils.data.DataLoader(mvp_test, batch_size=args.batch_size, shuffle=False)
    
    all_emd = []
    all_cd = []
    
    MODEL.eval()
    
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            label, points_xyz, gt_coarse, gt_fine = data
            
            '''
            This block can be commented in training, rotations will be implemented in test.py
            '''
            # rot_mat = special_ortho_group.rvs(3)
            # points_xyz = points_xyz @ rot_mat
            # gt_coarse = gt_coarse @ rot_mat
            # gt_fine = gt_fine @ rot_mat
            
            points_xyz = points_xyz.cuda().float()
            gt_coarse = gt_coarse.cuda().float()
            gt_fine = gt_fine.cuda().float()
            
            if args.sparse:
                points_xyz, _ = dataset.shuffle_points(points_xyz, batch_label=None)
                points_xyz = points_xyz[:, 0:256, :]
            
            pred_coarse, pred_fine, loss_emd, loss_cdp = MODEL(points_xyz, gt_coarse, gt_fine, istraining=False, use_emd=args.use_emd)
            loss_emd = loss_emd.mean()
            loss_cdp = loss_cdp.mean()

            all_emd.append(loss_emd.item()*100)
            all_cd.append(loss_cdp.item()*100)
            print ('Test epoch %d, %d/%d'%(epoch, i+1, num_test_pt//args.batch_size+1))
        
    all_emd = np.hstack(all_emd)
    all_cd = np.hstack(all_cd)
        
    writer_train.add_scalar('test_emd', np.mean(all_emd), global_step=epoch)
    writer_train.add_scalar('test_cd', np.mean(all_cd), global_step=epoch)
    
    if (np.mean(all_emd) < best_emd):
        state = {'epoch': epoch,
                  'model_state_dict': MODEL.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()}
        torch.save(state, os.path.join('info', time_folder, 'model/%s_emd.pth'%args.model_name))
        best_emd = np.mean(all_emd)
    
    if (np.mean(all_cd) < best_cd):
        state = {'epoch': epoch,
                  'model_state_dict': MODEL.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()}
        torch.save(state, os.path.join('info', time_folder, 'model/%s_cd.pth'%args.model_name))
        best_cd = np.mean(all_cd)
        
    
    state = {'epoch': epoch,
             'model_state_dict': MODEL.state_dict(),
             'optimizer_state_dict': optimizer.state_dict()}
    torch.save(state, os.path.join('info', time_folder, 'model/%s.pth'%args.model_name))
            
