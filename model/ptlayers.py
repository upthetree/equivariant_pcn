#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class conv_1d(nn.Module):
    def __init__(self, hypers_conv):
        super().__init__()
        
        sequence = []
        for l in range(len(hypers_conv.layers)-1):
            sequence.append(nn.Conv1d(hypers_conv.layers[l], hypers_conv.layers[l+1], 1, bias=hypers_conv.bias[l]))
            sequence.append(nn.BatchNorm1d(hypers_conv.layers[l+1])) if hypers_conv.bn[l]==True else None
            sequence.append(hypers_conv.func[l]) if hypers_conv.func[l] is not None else None
            sequence.append(nn.Dropout(p=hypers_conv.dp[l])) if hypers_conv.dp[l] is not None else None
                
        self.model = nn.Sequential(*sequence)
    
    def forward(self, x):
        x =self.model(x)
        return x
    
    
    
class conv_2d(nn.Module):
    def __init__(self, hypers_conv):
        super().__init__()
        
        sequence = []
        for l in range(len(hypers_conv.layers)-1):
            sequence.append(nn.Conv2d(hypers_conv.layers[l], hypers_conv.layers[l+1], hypers_conv.kernel[l], bias=hypers_conv.bias[l]))
            sequence.append(nn.BatchNorm2d(hypers_conv.layers[l+1])) if hypers_conv.bn[l]==True else None
            sequence.append(hypers_conv.func[l]) if hypers_conv.func[l] is not None else None
            sequence.append(nn.Dropout(p=hypers_conv.dp[l])) if hypers_conv.dp[l] is not None else None
                
        self.model = nn.Sequential(*sequence)
    
    def forward(self, x):
        x =self.model(x)
        return x
    

if __name__ == '__main__':
    class conv:
        layers = [None, 256, 256, 128, 6]
        bias = [False, False, False, False]
        bn = [True, True, True, False]
        func = [nn.LeakyReLU(negative_slope=0.2),
                nn.LeakyReLU(negative_slope=0.2),
                nn.LeakyReLU(negative_slope=0.2),
                None]
        dp = [0.5, 0.5, None, None]
    
    conv.layers[0] = 2326
    decoder = conv_1d(conv)
    print (decoder)
