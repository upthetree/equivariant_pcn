#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import sys
sys.path.append("..")
from extension.pointnet2 import pointnet2_utils


def knn(x, k):
    '''
    x: (b, dim_fts, n)
    -------
    idx: (b, n, k)
    '''
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1, sorted=True)[1]   #(batch_size, num_points, k)
    return idx


def knn_cross(x, k, x_q=None):
    '''
    x: (b, dim_fts, n)
    x_q: (b, dim_fts, m)
    -------
    idx: (b, n, k) or (b, m, k)
    '''
    if x_q is None:
        idx = knn(x, k)   #(b, n, k)
    else:
        inner = -2*torch.matmul(x_q.transpose(2, 1), x)          #(b, m, n)
        xqxq = torch.sum(x_q**2, dim=1, keepdim=True)            #(b, 1, m)
        xx = torch.sum(x**2, dim=1, keepdim=True)                #(b, 1, n)
        pairwise_distance = -xx - inner - xqxq.transpose(2, 1)   #(b, m, n)
        
        idx = pairwise_distance.topk(k=k, dim=-1, sorted=True)[1]#(b, m, k)
    return idx
    
    


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


#---------****---------#


def get_graph_feature_xyz(xyz, fts, k=20, query='xyz', idx=None):

    batch_size = xyz.shape[0]
    num_points = xyz.shape[1]
    num_dims = fts.shape[1]
    xyz = xyz.permute(0, 2, 1).contiguous()
    fts = fts.reshape(batch_size, num_dims*3, num_points)
    
    if idx is None:
        if query == 'xyz':
            idx = knn(xyz, k=k)
        elif query == 'fts':
            idx = knn(fts, k=k)
        else:
            raise NotImplementedError
    
        device = torch.device('cuda')
        
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    
        idx = idx + idx_base
    
        idx = idx.view(-1)
    
    xyz = xyz.transpose(2, 1).contiguous()
    group_xyz = xyz.view(batch_size*num_points, -1)[idx, :]
    group_xyz = group_xyz.view(batch_size, num_points, k, 3)
    
    fts = fts.transpose(2, 1).contiguous()
    feature = fts.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3) 
    fts = fts.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    group_fts = torch.cat((feature-fts, fts), dim=3).permute(0, 3, 4, 1, 2).contiguous()

    return group_fts, group_xyz, idx



def get_graph_feature_xyz_new(xyz, fts, new, k=20, query='xyz', idx=None):
    '''
    xyz: (b, n, 3)
    fts: (b, dim_fts, 3, n)
    new: [(b, npoint, 3) if query=='xyz']  or  [(b, dim_fts, 3, npoint) if query=='fts']
    -------
    group_fts: (b, dim_fts or 2*dim_fts, 3, npoint, k)
    group_xyz: (b, npoint, k, 3)
    '''
    b = xyz.shape[0]
    n = xyz.shape[1]
    dim_fts = fts.shape[1]
    xyz = xyz.permute(0, 2, 1).contiguous()
    fts = fts.reshape(b, dim_fts*3, n)
    npoint = new.shape[-1] if query == 'fts' else new.shape[1]
    
    if idx is None:
        if query == 'xyz':
            new = new.permute(0, 2, 1).contiguous()
            idx = knn_cross(xyz, k, new)
            # idx = knn(xyz, k=k)
        elif query == 'fts':
            new = new.reshape(b, dim_fts*3, npoint)
            idx = knn_cross(fts, k, new)
            # idx = knn(fts, k=k)
        else:
            raise NotImplementedError
    
        device = torch.device('cuda')
        
        idx_base = torch.arange(0, b, device=device).view(-1, 1, 1)*n
    
        idx = idx + idx_base
    
        idx = idx.view(-1)
    
    xyz = xyz.transpose(2, 1).contiguous()
    group_xyz = xyz.view(b*n, -1)[idx, :]
    group_xyz = group_xyz.view(b, npoint, k, 3)
    
    fts = fts.transpose(2, 1).contiguous()
    feature = fts.view(b*n, -1)[idx, :]
    feature = feature.view(b, npoint, k, dim_fts, 3)
    feature = feature.permute(0, 3, 4, 1, 2).contiguous()
    
    if query == 'xyz':
        group_fts = feature
    if query == 'fts':
        new = new.view(b, dim_fts, 3, npoint, 1).repeat(1, 1, 1, 1, k)
        group_fts = torch.cat((feature-new, new), dim=1).contiguous()
        # group_fts = torch.cat((feature-new, feature), dim=1).contiguous()

    return group_fts, group_xyz, idx
