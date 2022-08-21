#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import geometry, transformer, vnnlayers, ptlayers, losses
import numpy as np

import sys
sys.path.append("..")
from extension.pointnet2 import pointnet2_utils


class PCN_encoder (nn.Module):
    def __init__(self, vn_encoder):
        super().__init__()
        self.e_1 = transformer.vnn_block(vn_encoder.layer1)
        self.mp_1 = vnnlayers.VNMaxPool(vn_encoder.layer1['layers'][-1])
        self.e_2 = transformer.vnn_block(vn_encoder.layer2)
        # self.mp_2 = vnnlayers.VNMaxPool(vn_encoder.layer2['layers'][-1])
    
    def forward(self, x):
        '''
        x: (b, 2048, 3)
        --------
        x: equ (b, 1024, 3)
        '''
        n = x.shape[1]
        x = x.permute(0, 2, 1).contiguous().unsqueeze(1)
        x = self.e_1(x)
        x_max_1 = self.mp_1(x).unsqueeze(-1).repeat(1, 1, 1, n)
        x = torch.cat([x, x_max_1], dim=1)
        x = self.e_2(x)
        # x_max_2 = self.mp_2(x)
        return x
        
        
class VN_encoder (nn.Module):
    def __init__(self, encoder_l1, encoder_l2, use_std=False):
        super().__init__()
        self.e_1 = transformer.vnn_block(encoder_l1)
        self.mp_1 = vnnlayers.VNMaxPool(encoder_l1['layers'][-1])
        self.e_2 = transformer.vnn_block(encoder_l2)
        self.use_std = use_std
        
        in_channels = encoder_l2['layers'][-1]
        if self.use_std:
            self.std = vnnlayers.VNStdFeatureLin(in_channels, dim=3, normalize_frame=True)
    
    def forward(self, x):
    
        b = x.shape[0]
        x = x.permute(0, 2, 1).contiguous().unsqueeze(1)
        x = self.e_1(x)
        x_max_1 = self.mp_1(x).unsqueeze(-1).repeat(1, 1, 1, 2048)
        x = torch.cat([x, x_max_1], dim=1)
        x = self.e_2(x)
        if self.use_std:
            std_fts, z0 = self.std(x)
            std_fts = std_fts.reshape(b, -1)
            z0 = z0.transpose(1, 2).contiguous()
            return x, std_fts, z0
        else:
            return x
    
    
class VTR_encoder(nn.Module):
    def __init__(self, hypers_encoder):
        super().__init__()
        self.grouper_l1 = geometry.get_local_area_new(hypers_encoder.group_1)
        self.grouper_l2 = geometry.get_local_area_new(hypers_encoder.group_2)
        self.grouper_l3 = geometry.get_local_area_new(hypers_encoder.group_3)
        self.grouper_l4 = geometry.get_local_area_new(hypers_encoder.group_4)
        
        self.vn_1 = transformer.vnn_block(hypers_encoder.vn_1)
        self.vn_2 = transformer.vnn_block(hypers_encoder.vn_2)
        self.vn_3 = transformer.vnn_block(hypers_encoder.vn_3)
        self.vn_4 = transformer.vnn_block(hypers_encoder.vn_4)
        
        self.pool1 = vnnlayers.mean_pool()
        self.pool2 = vnnlayers.mean_pool()
        self.pool3 = vnnlayers.mean_pool()
        self.pool4 = vnnlayers.mean_pool()
        
        self.vn_5 = transformer.vnn_block(hypers_encoder.vn_5)
        vn_channels = hypers_encoder.vn_5['layers'][-1]
        self.gl_maxpool = vnnlayers.VNMaxPool(vn_channels)
        self.gl_meanpool = vnnlayers.mean_pool()
    
    def forward(self, points_xyz):
        '''
        points_xyz: (b, 2048, 3)
        ----------
        gl_fts: equ (b, 1024, 3)
        '''
        points_fts = points_xyz.permute(0, 2, 1).contiguous().unsqueeze(1)
        group_xyz_1, group_fts_1, _, _ = self.grouper_l1(points_xyz, points_fts)
        group_fts_1 = self.vn_1(group_fts_1)
        mean_fts_1 = self.pool1(group_fts_1)
        
        group_xyz_2, group_fts_2, _, _ = self.grouper_l2(points_xyz, mean_fts_1)
        group_fts_2 = self.vn_2(group_fts_2)
        mean_fts_2 = self.pool2(group_fts_2)
        
        group_xyz_3, group_fts_3, _, _ = self.grouper_l3(points_xyz, mean_fts_2)
        group_fts_3 = self.vn_3(group_fts_3)
        mean_fts_3 = self.pool3(group_fts_3)
        
        group_xyz_4, group_fts_4, _, _ = self.grouper_l4(points_xyz, mean_fts_3)
        group_fts_4 = self.vn_4(group_fts_4)
        mean_fts_4 = self.pool4(group_fts_4)
        
        cat_fts = torch.cat((mean_fts_1, mean_fts_2, mean_fts_3, mean_fts_4), dim=1)
        cat_fts = self.vn_5(cat_fts)
        
        cat_fts_max = self.gl_maxpool(cat_fts)
        cat_fts_mean = self.gl_meanpool(cat_fts)
        gl_fts = torch.cat([cat_fts_max, cat_fts_mean], dim=1)
        return gl_fts    

    
class coarse_decoder(nn.Module):
    def __init__(self, hyper_decoder):
        super().__init__()
        
        self.up_lift = transformer.vnn_block(hyper_decoder.vn_uplift)
        self.k = transformer.vnn_block(hyper_decoder.vn_qk)
        self.q_1 = transformer.vnn_block(hyper_decoder.vn_qk)
        self.q_2 = transformer.vnn_block(hyper_decoder.vn_qk)
        self.f = transformer.vnn_block(hyper_decoder.vn_fw)
        
        
    def forward(self, gl_fts):
        '''
        gl_fts: equ (b, 1024, 3)
        ----------
        pts: equ (b, 1024, 3)
        '''
        up_fts = self.up_lift(gl_fts)
        k = self.k(up_fts)
        q_1 = self.q_1(up_fts)
        q_2 = self.q_2(up_fts)
        
        s_1 = torch.sum(q_1*k, dim=-1, keepdim=True)
        s_2 = torch.sum(q_2*k, dim=-1, keepdim=True)
        s = F.softmax(torch.cat([s_1, s_2], dim=-1), dim=-1)
        pts = q_1*s[..., 0:1] + q_2*s[..., 1:2] + self.f(gl_fts)
        
        return pts


class VTR_decoder(nn.Module):
    def __init__(self, hypers_tr_1, hypers_tr_2, hypers_dec):
        super().__init__()
        
        self.grouper_l1 = geometry.get_local_area_new(hypers_tr_1.group)
        self.grouper_l2 = geometry.get_local_area_new(hypers_tr_2.group, source='new')
        
        self.tr_1 = transformer.vnn_block(hypers_tr_1.vnn_l)
        self.tr_2 = transformer.transformer_base(hypers_tr_2.vnn, None, hypers_tr_2.transformer)
        # self.gl_1 = transformer.vnn_block(hypers_tr_1.vnn_g)
        self.gl_1 = VN_encoder(hypers_tr_1.layer1, hypers_tr_1.layer2, use_std=False)
        
        self.fp_2 = geometry.VectorFPModule(hypers_dec.fp_2)
        self.fp_1 = geometry.VectorFPModule(hypers_dec.fp_1)      
        self.k = hypers_dec.k
        self.hg = hypers_dec.hg
        
        self.vnn_d = hypers_dec.vnn_d
        self.use_emb = hypers_dec.use_emb
        if self.use_emb:
            self.vnn_d['layers'][0] += 1024
        self.refine = transformer.vnn_block(self.vnn_d)
        
    
    def forward(self, pred_coarse, coarse_fts):
        '''
        pred_coarse: (b, 2048, 3)
        coarse_fts: (b, 512, 3)
        ------
        output_fine: (b, 8192, 3)
        '''
        b, n, _ = pred_coarse.size()
        pred_tile = pred_coarse.unsqueeze(2).repeat(1, 1, 4, 1).reshape(b, n*4, 3)
        pred_fts = pred_coarse.permute(0, 2, 1).contiguous().unsqueeze(1)
        grid, pred_fine = geometry.point_upsample(pred_coarse, self.k, 4, self.hg)
        
        gl_fts = self.gl_1(pred_coarse)
        gl_fts_exp = gl_fts.unsqueeze(3).repeat(1, 1, 1, n*4)
        grid_fts = grid.unsqueeze(2).permute(0, 2, 3, 1).contiguous()
        
        ''' Equ Surface SA layer_1'''
        group_xyz_l1, group_fts_l1, points_xyz_l1, points_fts_l1 = self.grouper_l1(pred_coarse, pred_fts)
        tr_fts_l1 = self.tr_1(group_fts_l1)
        ''' Equ Structure SA layer_2'''
        group_xyz_l2, group_fts_l2, points_xyz_l2, points_fts_l2 = self.grouper_l2(points_xyz_l1, tr_fts_l1)
        tr_fts_l2 = self.tr_2(group_xyz_l2, group_fts_l2, points_xyz_l2)
        ''' Feature propogation '''
        points_fp_l2 = self.fp_2(points_xyz_l2, None, tr_fts_l2, gl_fts)
        points_fp_l1 = self.fp_1(points_xyz_l1, points_xyz_l2, tr_fts_l1, points_fp_l2)
        points_fp_l1 = points_fp_l1.unsqueeze(4).repeat(1, 1, 1, 1, 4).reshape(b, -1, 3, n*4)
        points_fp_l1 = torch.cat([gl_fts_exp, points_fp_l1], dim=1)
        
        cat_fts = torch.cat([grid_fts, points_fp_l1], dim=1)
        if self.use_emb:
            cat_fts = torch.cat([cat_fts, coarse_fts.unsqueeze(3).repeat(1, 1, 1, n*4)], dim=1)
        
        offset = self.refine(cat_fts).squeeze(1).permute(0, 2, 1).contiguous()
        output_fine = pred_tile + offset
        
        return output_fine, offset


class model_completion(nn.Module):
    def __init__(self, encoder, decoder, enc_type='VTR'):
        super().__init__()
        assert enc_type in ['VTR', 'PCN']
        if enc_type=='PCN':
            self.shape_encoder = PCN_encoder(encoder.vn_encoder)
            print ('using PCN encoder')
        else:
            self.shape_encoder = VTR_encoder(encoder.vtr_encoder)
            print ('using DGCNN encoder')
        self.coarse_decoder = coarse_decoder(decoder.coarse)
        self.fine_decoder = VTR_decoder(decoder.tr_1, decoder.tr_2, decoder.dec)
        
        
    def forward(self, points_xyz, gt_coarse, gt_fine, istraining=True, use_emd=True):
        '''
        points_xyz: (b, 2048, 3)
        gt_coarse: (b, 2048, 3)
        gt_fine: (b, 8192, 3) 
        '''
        b, n, _ = points_xyz.size()
        
        coarse_fts = self.shape_encoder(points_xyz)
        pred_coarse = self.coarse_decoder(coarse_fts)
        
        concat_xyz = torch.cat([points_xyz, pred_coarse], dim=1)
        fps_ids = pointnet2_utils.furthest_point_sample(concat_xyz, 2048)
        new_xyz = pointnet2_utils.gather_operation(concat_xyz.permute(0, 2, 1).contiguous(), fps_ids)
        new_xyz = new_xyz.permute(0, 2, 1).contiguous()
        
        pred_fine, offset = self.fine_decoder(new_xyz, coarse_fts)
        
        ''' Losses '''
        if istraining:
            if use_emd:
            	loss_coarse = losses.calc_emd(pred_coarse, gt_coarse, eps=0.05, iterations=200)
            else:
            	loss_coarse, _ = losses.calc_cd(pred_coarse, gt_coarse)
        else:
            if use_emd:
            	loss_coarse = losses.calc_emd(pred_coarse, gt_coarse, eps=0.004, iterations=3000)
            else:
            	loss_coarse, _ = losses.calc_cd(pred_coarse, gt_coarse)
            
        loss_cdp, _ = losses.calc_cd(pred_fine, gt_fine)
        
        return pred_coarse, pred_fine, loss_coarse, loss_cdp
        
        
