#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import ptlayers, dgcnn_util, transformer, eigenvec

import sys
sys.path.append("..")
from extension.pointnet2 import pointnet2_utils
# from extension.invariant import inv_local
    


class get_local_area(nn.Module):
    def __init__(self, hypers_group):
        super().__init__()
        self.group_type = hypers_group.group_type
        
        if self.group_type=='ball':
            self.grouper = pointnet2_utils.QueryAndGroup(hypers_group.radius, hypers_group.nsample, use_xyz=True)
        elif self.group_type=='knn':
            self.query = hypers_group.query
        else:
            raise NotImplementedError
        
        self.npoint = hypers_group.npoint
        self.nsample = hypers_group.nsample
    
    def forward(self, points_xyz, points_fts):
        b, dim_fts, _, n = points_fts.size()
        if self.npoint is not None:
            fps_ids = pointnet2_utils.furthest_point_sample(points_xyz, self.npoint)
            new_xyz = pointnet2_utils.gather_operation(points_xyz.permute(0, 2, 1).contiguous(), fps_ids)
            new_xyz = new_xyz.permute(0, 2, 1).contiguous()
            new_fts = pointnet2_utils.gather_operation(points_fts.reshape(b, dim_fts*3, n), fps_ids)
            new_fts = new_fts.reshape(b, dim_fts, 3, self.npoint)
        else:
            new_xyz = points_xyz
            new_fts = points_fts
            
        if self.group_type=='ball':
            point_fts = points_fts.reshape(b, -1, n).contiguous()
            new_group = self.grouper(points_xyz, new_xyz, point_fts)
            group_xyz = new_group[:, 0:3, ...].permute(0, 2, 3, 1).contiguous()
            group_fts = new_group[:, 3:, ...].reshape(b, dim_fts, 3, self.npoint, self.nsample)
        elif self.group_type=='knn':
            group_fts, group_xyz, _ = dgcnn_util.get_graph_feature_xyz(points_xyz, points_fts, k=self.nsample, query=self.query)
            new_fts = torch.cat([new_fts, new_fts-new_fts], dim=1).contiguous()
        else:
            raise NotImplementedError
            
        return group_xyz, group_fts, new_xyz, new_fts
    


class get_local_area_new(nn.Module):
    def __init__(self, hypers_group, source='origin'):
        super().__init__()
        self.group_type = hypers_group.group_type
        
        if self.group_type=='ball':
            self.grouper = pointnet2_utils.QueryAndGroup(hypers_group.radius, hypers_group.nsample, use_xyz=True)
        elif self.group_type=='knn':
            self.query = hypers_group.query
        else:
            raise NotImplementedError
        
        self.npoint = hypers_group.npoint
        self.nsample = hypers_group.nsample
        self.source = source
        assert self.source in ['origin', 'new']
    
    def forward(self, points_xyz, points_fts):
        b, dim_fts, _, n = points_fts.size()
        if self.npoint == n:
            new_xyz = points_xyz
            new_fts = points_fts
        elif self.npoint < n:
            fps_ids = pointnet2_utils.furthest_point_sample(points_xyz, self.npoint)
            new_xyz = pointnet2_utils.gather_operation(points_xyz.permute(0, 2, 1).contiguous(), fps_ids)
            new_xyz = new_xyz.permute(0, 2, 1).contiguous()
            new_fts = pointnet2_utils.gather_operation(points_fts.reshape(b, dim_fts*3, n), fps_ids)
            new_fts = new_fts.reshape(b, dim_fts, 3, self.npoint)
        else:
            raise Exception('Sample too many points')
            
        if self.group_type=='ball':
            '''
            Ball query based on xyz (PointNet++)
            '''
            point_fts = points_fts.reshape(b, -1, n).contiguous()
            new_group = self.grouper(points_xyz, new_xyz, point_fts)
            group_xyz = new_group[:, 0:3, ...].permute(0, 2, 3, 1).contiguous()
            group_fts = new_group[:, 3:, ...].reshape(b, dim_fts, 3, self.npoint, self.nsample)
        elif self.group_type=='knn':
            if self.query == 'xyz':
                '''
                Knn based on xyz
                '''
                points_new = new_xyz
                if self.source == 'origin':
                    group_fts, group_xyz, _ = dgcnn_util.get_graph_feature_xyz_new(points_xyz, points_fts, points_new, self.nsample, self.query)
                else:
                    group_fts, group_xyz, _ = dgcnn_util.get_graph_feature_xyz_new(new_xyz, new_fts, points_new, self.nsample, self.query)
            elif self.query == 'fts':
                '''
                Knn based on fts (DGCNN)
                '''
                points_new = new_fts
                if self.source == 'origin':
                    group_fts, group_xyz, _ = dgcnn_util.get_graph_feature_xyz_new(points_xyz, points_fts, points_new, self.nsample, self.query)
                else:
                    group_fts, group_xyz, _ = dgcnn_util.get_graph_feature_xyz_new(new_xyz, new_fts, points_new, self.nsample, self.query)
            elif self.query == 'both':
                '''
                Geometry and semantic grouping (xyz|fts)
                '''
                points_new = new_xyz
                if self.source == 'origin':
                    group_fts_g, group_xyz_g, _ = dgcnn_util.get_graph_feature_xyz_new(points_xyz, points_fts, points_new, self.nsample, 'xyz')
                else:
                    group_fts_g, group_xyz_g, _ = dgcnn_util.get_graph_feature_xyz_new(new_xyz, new_fts, points_new, self.nsample, 'xyz')                    
                
                points_new = new_fts
                if self.source == 'origin':
                    group_fts_s, group_xyz_s, _ = dgcnn_util.get_graph_feature_xyz_new(points_xyz, points_fts, points_new, self.nsample, 'fts')
                else:
                    group_fts_s, group_xyz_s, _ = dgcnn_util.get_graph_feature_xyz_new(new_xyz, new_fts, points_new, self.nsample, 'fts')                                    
                group_xyz = torch.cat([group_xyz_g.unsqueeze(1), group_xyz_s.unsqueeze(1)], dim=1)
                group_fts = torch.cat([group_fts_g, group_fts_s], dim=1)
            new_fts = group_fts[..., 0:1]
        else:
            raise NotImplementedError
            
        return group_xyz, group_fts, new_xyz, new_fts

    
    
def pca(xyz, fts=None, k=8):
    '''
    cov: 
    frame: (b, n, 3, 3)
    '''
    b, n, _ = xyz.size()
    if fts is None:
        fts = xyz.permute(0, 2, 1).contiguous().unsqueeze(1)
    
    _, group_xyz, _ = dgcnn_util.get_graph_feature_xyz(xyz, fts, k, query='xyz')
    neighbours = group_xyz.reshape(b*n, k, 3)
    centers = xyz.reshape(b*n, 1, 3)
    
    dist = neighbours-centers
    dist_T = dist.permute(0, 2, 1).contiguous()
    cov = torch.matmul(dist_T, dist)/(n-1)
    _, frame = eigenvec.fast_eigen(cov)
    cov = cov.reshape(b, n, 3, 3)
    frame = frame.reshape(b, n, 3, 3)
    return cov, frame


import warnings
warnings.filterwarnings("ignore", category=UserWarning)



class PointUpsample(Function):
    @staticmethod
    def forward(ctx, xyz, k, up_ratio, half_grid_size):
    
        assert xyz.is_contiguous()
        b, n, _ = xyz.size()
        cov, frame = pca(xyz, k=k)  #(b, n, 3, 3)
        t = np.sqrt(up_ratio)       #2
        line = torch.range(-half_grid_size, half_grid_size, 2*half_grid_size/(t-1))
        ys, xs = torch.meshgrid(line, line)
        grid = torch.stack([xs, ys], dim=-1).reshape(1, 1, up_ratio, 2).repeat(b, n, 1, 1)
        grid = torch.cat([torch.zeros(b, n, up_ratio, 1), grid], dim=3).cuda()
        grid = torch.einsum('bmij,bmjk->bmik', grid, frame).reshape(b, n*up_ratio, 3)
        xyz = xyz.unsqueeze(2).repeat(1, 1, up_ratio, 1).reshape(b, n*up_ratio, 3)
        dense_xyz = xyz + grid
        return grid, dense_xyz

    @staticmethod
    def backward(ctx, grad_grid=None, grad_dense_xyz=None):
        return None, None, None, None
    
point_upsample = PointUpsample.apply


def upsample(xyz, k=8, up_ratio=4, half_grid_size=0.1):
    b, n, _ = xyz.size()
    cov, frame = pca(xyz, k=k)
    t = np.sqrt(up_ratio)
    line = torch.range(-half_grid_size, half_grid_size, 2*half_grid_size/(t-1))
    ys, xs = torch.meshgrid(line, line)
    grid = torch.stack([xs, ys], dim=-1).reshape(1, 1, up_ratio, 2).repeat(b, n, 1, 1)
    grid = torch.cat([torch.zeros(b, n, up_ratio, 1), grid], dim=3).cuda()
    grid = torch.einsum('bmij,bmjk->bmik', grid, frame).reshape(b, n*up_ratio, 3)
    xyz = xyz.unsqueeze(2).repeat(1, 1, up_ratio, 1).reshape(b, n*up_ratio, 3)
    xyz = xyz + grid
    return grid, xyz
    
    

class VectorFPModule(nn.Module):
    """Propigates the features of one set to another"""
    def __init__(self, hyper_vnn, use_nn=True):
        """
        :param mlp: list of int
        :param bn: whether to use batchnorm
        """
        super().__init__()
        self.use_nn = use_nn
        if self.use_nn:
            self.mlp = transformer.vnn_block(hyper_vnn)

    def forward(self, unknown, known, unknow_feats, known_feats):
    
        n = unknown.shape[1]
        
        if known is not None:
            '''
            known_feats: (b, cm, 3, m)
            '''
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            
            b, cm, _, m = known_feats.size()
            known_feats = known_feats.reshape(b, cm*3, m)
            interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)
        else:
            '''
            known_feats: (b, cm, 3)
            '''
            b, cm, _ = known_feats.size()
            known_feats = known_feats.reshape(b, cm*3, 1)
            interpolated_feats = known_feats.repeat(1, 1, n)
        
        interpolated_feats = interpolated_feats.reshape(b, cm, 3, n)

        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)
        else:
            new_features = interpolated_feats
            
        if self.use_nn:
            new_features = self.mlp(new_features)

        return new_features  
    

class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another"""

    def __init__(self, hyper_mlp):
        """
        :param mlp: list of int
        :param bn: whether to use batchnorm
        """
        super().__init__()
        self.mlp = ptlayers.conv_1d(hyper_mlp)

    def forward(self, unknown, known, unknow_feats, known_feats):
        """
        :param unknown: (B, n, 3) tensor of the xyz positions of the unknown features
        :param known: (B, m, 3) tensor of the xyz positions of the known features
        :param unknow_feats: (B, C1, n) tensor of the features to be propigated to
        :param known_feats: (B, C2, m) tensor of features to be propigated
        :return:
            new_features: (B, mlp[-1], n) tensor of the features of the unknown features
        """
        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))

        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = self.mlp(new_features)

        return new_features
    

