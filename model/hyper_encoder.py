#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class vtr_encoder:
    class group_1:
        group_type = 'knn'
        query = 'fts'
        npoint = 2048
        nsample = 32
        
    class group_2:
        group_type = 'knn'
        query = 'fts'
        npoint = 2048
        nsample = 16
        
    class group_3:
        group_type = 'knn'
        query = 'fts'
        npoint = 2048
        nsample = 16
        
    class group_4:
        group_type = 'knn'
        query = 'fts'
        npoint = 2048
        nsample = 8
        
    vn_1 = {'layers': [2, 32],
            'data_dim': [5],
            'share_nonlinearity': [False],
            'negative_slope': [0.2],
            'hid_layer': 'relu',
            'last_layer': 'relu',
            'pool': 'none'}
    
    vn_2 = {'layers': [64, 64],
            'data_dim': [5],
            'share_nonlinearity': [False],
            'negative_slope': [0.2],
            'hid_layer': 'relu',
            'last_layer': 'relu',
            'pool': 'none'}
    
    vn_3 = {'layers': [128, 128],
            'data_dim': [5],
            'share_nonlinearity': [False],
            'negative_slope': [0.2],
            'hid_layer': 'relu',
            'last_layer': 'relu',
            'pool': 'none'}
    
    vn_4 = {'layers': [256, 256],
            'data_dim': [5],
            'share_nonlinearity': [False],
            'negative_slope': [0.2],
            'hid_layer': 'relu',
            'last_layer': 'relu',
            'pool': 'none'}
    
    vn_5 = {'layers': [480, 512],
            'data_dim': [4],
            'share_nonlinearity': [True],
            'negative_slope': [0.2],
            'hid_layer': 'relu',
            'last_layer': 'relu',
            'pool': 'none'}
        

class vn_encoder:
    layer1 = {'layers': [1, 64, 128],
              'data_dim': [4, 4],
              'share_nonlinearity': [False, False],
              'negative_slope': [0.2, 0.2],
              'hid_layer': 'relu',
              'last_layer': 'linear',
              'pool': 'none'}
    
    
    layer2 = {'layers': [256, 512, 1024],
              'data_dim': [4, 4],
              'share_nonlinearity': [False, False],
              'negative_slope': [0.2, 0.2],
              'hid_layer': 'relu',
              'last_layer': 'linear',
              'pool': 'max'}