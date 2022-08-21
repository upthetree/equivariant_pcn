#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn

class coarse:
    vn_uplift = {'layers': [1024, 1024, 2048],
                  'data_dim': [3, 3],
                  'share_nonlinearity': [False, False],
                  'negative_slope': [0.2, 0.2],
                  'hid_layer': 'relu',
                  'last_layer': 'relu',
                  'pool': 'none'}
    
    vn_qk = {'layers': [2048, 2048],
              'data_dim': [3],
              'share_nonlinearity': [False],
              'negative_slope': [0.2],
              'hid_layer': 'linear',
              'last_layer': 'linear',
              'pool': 'none'}
    
    vn_fw = {'layers': [1024, 1024, 2048],
              'data_dim': [3, 3],
              'share_nonlinearity': [False, False],
              'negative_slope': [0.2, 0.2],
              'hid_layer': 'relu',
              'last_layer': 'linear',
              'pool': 'none'}
    

class tr_1:
    class group:
        group_type = 'knn'
        query = 'fts'
        npoint = 2048
        nsample = 24
    
    vnn_l = {'layers': [2, 32, 64, 64],
             'data_dim': [5, 5, 5],
             'share_nonlinearity': [False, False, False],
             'negative_slope': [0.2, 0.2, 0.2],
             'hid_layer': 'relu',
             'last_layer': 'linear',
             'pool': 'max'}
    
    vnn_g = {'layers': [64, 128, 256, 512],
             'data_dim': [4, 4, 4],
             'share_nonlinearity': [False, False, False],
             'negative_slope': [0.2, 0.2, 0.2],
             'hid_layer': 'relu',
             'last_layer': 'linear',
             'pool': 'max'}
    
    layer1 = {'layers': [1, 64, 128],
              'data_dim': [4, 4],
              'share_nonlinearity': [False, False],
              'negative_slope': [0.2, 0.2],
              'hid_layer': 'relu',
              'last_layer': 'linear',
              'pool': 'none'}
    
    layer2 = {'layers': [256, 256, 512],
              'data_dim': [4, 4],
              'share_nonlinearity': [False, False],
              'negative_slope': [0.2, 0.2],
              'hid_layer': 'relu',
              'last_layer': 'linear',
              'pool': 'max'}


class tr_2:
    class transformer:
        overdeterm = False
        relation_func = 'sub'
        scale_mode = 'dimfts'
        shortcut = None
        use_geometry = False
        use_position = True
        
    class group:
        group_type = 'knn'
        query = 'fts'
        npoint = 512
        nsample = 16
        
    class vnn:
        e = {'layers': [128, 64, 64],
             'data_dim': [5, 5],
             'share_nonlinearity': [False, False],
             'negative_slope': [0.2, 0.2],
             'hid_layer': 'relu',
             'last_layer': 'relu',
             'pool': 'none'}
        
        qkv = {'layers': [64, 64],
               'data_dim': [5],
               'share_nonlinearity': [False],
               'negative_slope': [0.2],
               'hid_layer': 'linear',
               'last_layer': 'linear',
               'pool': 'none'}
        
        r = {'layers': [1, 64, 64],
            'data_dim': [5, 5],
            'share_nonlinearity': [False, False],
            'negative_slope': [0.2, 0.2],
            'hid_layer': 'relu',
            'last_layer': 'linear',
            'pool': 'none'}
        
        f = {'layers': [64, 64],
            'data_dim': [4],
            'share_nonlinearity': [False],
            'negative_slope': [0.2],
            'hid_layer': 'linear',
            'last_layer': 'linear',
            'pool': 'none'}

                
class dec:
    k = 8
    hg = 0.05
    use_emb = True
    
    fp_2 = {'layers': [576, 256, 128],
            'data_dim': [4, 4],
            'share_nonlinearity': [False, False],
            'negative_slope': [0.2, 0.2],
            'hid_layer': 'relu',
            'last_layer': 'linear',
            'pool': 'none'}
    
    fp_1 = {'layers': [192, 128, 64],
            'data_dim': [4, 4],
            'share_nonlinearity': [False, False],
            'negative_slope': [0.2, 0.2],
            'hid_layer': 'relu',
            'last_layer': 'linear',
            'pool': 'none'}
    
    vnn_d = {'layers': [577, 512, 128, 1],
             'data_dim': [4, 4, 4],
             'share_nonlinearity': [False, False, False],
             'negative_slope': [0.2, 0.2, 0.2],
             'hid_layer': 'relu',
             'last_layer': 'linear',
             'pool': 'none'}
        
