#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import vnnlayers, geometry, ptlayers
import numpy as np


class vnn_block(nn.Module):
    def __init__(self, hypers_vnn_block):
        super().__init__()
        layers = hypers_vnn_block['layers']
        data_dim = hypers_vnn_block['data_dim']
        share_nonlinearity = hypers_vnn_block['share_nonlinearity']
        negative_slope = hypers_vnn_block['negative_slope']
        
        sequence = []
        for i in range(len(layers)-2):
            d_in = layers[i]
            d_out = layers[i+1]
            if hypers_vnn_block['hid_layer'] == 'linear':
                conv = vnnlayers.VNLinear(d_in, d_out)
            elif hypers_vnn_block['hid_layer'] == 'relu':
                conv = vnnlayers.VNLinearAndLeakyReLU(d_in, d_out, data_dim[i], share_nonlinearity[i], 'none', negative_slope[i])
            elif hypers_vnn_block['hid_layer'] == 'bn_relu':
                conv = vnnlayers.VNLinearLeakyReLU(d_in, d_out, data_dim[i], share_nonlinearity[i], negative_slope[i])
            else:
                raise NotImplementedError
            sequence.append(conv)
            
        if hypers_vnn_block['last_layer'] == 'linear':
            conv = vnnlayers.VNLinear(layers[-2], layers[-1])
        elif hypers_vnn_block['last_layer'] == 'relu':
            conv = vnnlayers.VNLinearAndLeakyReLU(layers[-2], layers[-1], data_dim[-1], share_nonlinearity[-1], 'none', negative_slope[-1])
        elif hypers_vnn_block['last_layer'] == 'bn_relu':
            conv = vnnlayers.VNLinearLeakyReLU(layers[-2], layers[-1], data_dim[-1], share_nonlinearity[-1], negative_slope[-1])
        else:
            raise NotImplementedError
        sequence.append(conv)
        
        if hypers_vnn_block['pool'] == 'max':
            sequence.append(vnnlayers.VNMaxPool(layers[-1]))
        elif hypers_vnn_block['pool'] == 'mean':
            sequence.append(vnnlayers.mean_pool())
        else:
            None
        
        self.model = nn.Sequential(*sequence)
    
    def forward(self, x):
    
        x =self.model(x)
        return x
        
    
    
class transformer_base(nn.Module):
    def __init__(self, hypers_vnn, hypers_centroid, hypers_transformer):
        super().__init__()
        self.vnn_e = vnn_block(hypers_vnn.e)
        self.shortcut = hypers_transformer.shortcut
        if self.shortcut == 'cat':
            self.max_e = vnnlayers.VNMaxPool(hypers_vnn.e['layers'][-1])
        
        self.vnn_q = vnn_block(hypers_vnn.qkv)
        self.vnn_k = vnn_block(hypers_vnn.qkv)
        self.vnn_v = vnn_block(hypers_vnn.qkv)
        self.ca = hypers_vnn.qkv['layers'][-1]
        
        self.hypers_centroid = hypers_centroid
        self.overdeterm = hypers_transformer.overdeterm
        
        if hypers_transformer.relation_func == 'sub':
            if hypers_transformer.use_geometry:
                self.pos_e = nn.Sequential(
                    nn.Conv2d(7+6*self.overdeterm, self.ca, 1), nn.ReLU(),
                    nn.Conv2d(self.ca, self.ca*3, 1)
                    )
            self.attn_std = vnnlayers.VNStdFeatureLin(hypers_vnn.qkv['layers'][-1], dim=5, normalize_frame=False)
            self.attn_fwd = nn.Sequential(
                nn.Conv2d(self.ca*3, self.ca, 1), nn.ReLU(),
                nn.Conv2d(self.ca, self.ca, 1)
                )
            if hypers_transformer.use_position:
                self.pos_r = vnn_block(hypers_vnn.r)
            
        elif hypers_transformer.relation_func == 'dot':
            if hypers_transformer.use_geometry:
                self.pos_e = nn.Sequential(
                    nn.Conv2d(7+6*self.overdeterm, self.ca, 1), nn.ReLU(),
                    nn.Conv2d(self.ca, self.ca, 1)
                    )
            self.pos_std = vnnlayers.VNStdFeatureLin(1, dim=5, normalize_frame=False)
            self.attn_fwd = nn.Sequential(
                nn.Conv2d(self.ca, self.ca, 1), nn.ReLU(),
                nn.Conv2d(self.ca, self.ca, 1)
                )
            if hypers_transformer.use_position:
                self.pos_r = nn.Sequential(
                    nn.Conv2d(3, self.ca*2, 1), nn.ReLU(),
                    nn.Conv2d(self.ca*2, self.ca, 1)
                    )
        
        else:
            raise NotImplementedError
        
        self.scale_mode = hypers_transformer.scale_mode
        self.relation_func = hypers_transformer.relation_func
        self.use_geometry = hypers_transformer.use_geometry
        self.use_position = hypers_transformer.use_position
              
        self.vnn_f = vnn_block(hypers_vnn.f)  #feed forward network  
            
    
    def forward(self, group_xyz, group_fts, query_xyz):
        '''
        out_fts: (b, ca, 3, npoint)
        '''
        group_emb = self.vnn_e(group_fts)
        query_emb = group_emb[..., 0].unsqueeze(-1)
        
        q = self.vnn_q(query_emb)
        k = self.vnn_k(group_emb)
        v = self.vnn_v(group_emb)
        b, _, _, npoint, nsample = k.size()
        if self.scale_mode == 'none':
            scale = 1.0
        elif self.scale_mode == 'dimfts':
            scale = np.sqrt(k.size(1))
        elif self.scale_mode == 'dimvec':
            scale = np.sqrt(k.size(1))*3.0
        else:
            raise NotImplementedError
            
        #invariant geometry encoding
        if self.use_geometry:
            None ### other implementation, not used in this work
        
        if self.relation_func == 'sub':
            #equivariant position encoding
            if self.use_position:
                xyz_rel = group_xyz - query_xyz.unsqueeze(2)
                xyz_rel = xyz_rel.unsqueeze(3).permute(0, 3, 4, 1, 2).contiguous()
                pos_rel = self.pos_r(xyz_rel)
            #attention module
            attn = q-k
            if self.use_position:
                attn = attn + pos_rel
            attn, _ = self.attn_std(attn)
            if self.use_geometry:
                None ### other implementation, not used in this work
                
            attn = attn.reshape(b, self.ca*3, npoint, nsample)
            attn = self.attn_fwd(attn).unsqueeze(2)
            attn = F.softmax(attn/scale, dim=-1)
            if self.use_position:
                v = v + pos_rel
            resi = torch.sum(attn*v, dim=-1)
            
        elif self.relation_func == 'dot':
            #invariant position encoding
            if self.use_position:
                xyz_rel = group_xyz - query_xyz.unsqueeze(2)
                xyz_rel = xyz_rel.unsqueeze(3).permute(0, 3, 4, 1, 2).contiguous()
                pos_rel = self.pos_std(xyz_rel)[0].squeeze(1)
                pos_rel = self.pos_r(pos_rel)
            #attention module
            q = q.repeat(1, 1, 1, 1, nsample)
            attn = torch.sum(q*k, dim=2)
            if self.use_position:
                attn = attn + pos_rel
            if self.use_geometry:
                None ### other implementation, not used in this work
                
            attn = self.attn_fwd(attn).unsqueeze(2)
            attn = F.softmax(attn/scale, dim=-1)
            resi = torch.sum(attn*v, dim=-1)
        else:
            raise NotImplementedError
        
        if self.shortcut == 'cat':
            return torch.cat([self.max_e(group_emb), resi], dim=1)
        else:
            return resi
    
    
class self_attention(nn.Module):
    def __init__(self, hypers_attn):
        super().__init__()
        self.vnn_e = vnn_block(hypers_attn.e)
        
        self.num_head = hypers_attn.num_head
        self.q = nn.ModuleList([vnn_block(hypers_attn.qkv) for i in range(self.num_head)])
        self.k = nn.ModuleList([vnn_block(hypers_attn.qkv) for i in range(self.num_head)])
        self.v = nn.ModuleList([vnn_block(hypers_attn.qkv) for i in range(self.num_head)])
        
        layers = hypers_attn.qkv['layers'][-1]
        self.attn_std = nn.ModuleList([vnnlayers.VNStdFeatureLin(layers) for i in range(self.num_head)])
        
        self.use_position = hypers_attn.use_position
        if self.use_position is True:
            self.pos_r = nn.ModuleList([vnn_block(hypers_attn.r) for i in range(self.num_head)])
        
        self.attn_fwd = nn.ModuleList([ptlayers.conv_2d(hypers_attn.fwd) for i in range(self.num_head)])
        self.vnn_f = nn.ModuleList([vnn_block(hypers_attn.f) for i in range(self.num_head)])
        
    
    def forward(self, fts, xyz):
        '''
        fts: (b, c, 3, npoint)
        xyz: (b, npoint, 3)
        '''
        b, npoint, _ = xyz.size()
        xyz_rel = xyz.unsqueeze(2).permute(0, 2, 3, 1).contiguous()
        
        fts = self.vnn_e(fts)
        
        resi_list = []
        for i in range(self.num_head):
            q = self.q[i](fts)
            k = self.q[i](fts)
            v = self.q[i](fts)
            
            scale = np.sqrt(k.size(1))
            attn = q-k
            
            if self.use_position:
                pos_rel = self.pos_r[i](xyz_rel)
                attn = attn + pos_rel
                v = v + pos_rel
            
            attn, _ = self.attn_std[i](attn)
            attn = self.attn_fwd[i](attn)
            attn = F.softmax(attn/scale, dim=-1)
            
            resi = torch.sum(attn*v, dim=-1)
            resi = self.vnn_f[i](resi)
            resi_list.append(resi)
        
        resi = torch.cat(resi_list, dim=1)
        return resi
        
        
        
