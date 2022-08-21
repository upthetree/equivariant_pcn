#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np

def compute_evec0(A, eval0):
    '''
    A : (b, 3, 3)
    eval0 : (b)
    -------
    evec0 : (b, 3)
    '''
    row0 = torch.stack([A[:, 0, 0]-eval0, A[:, 0, 1], A[:, 0, 2]], dim=1)
    row1 = torch.stack([A[:, 0, 1], A[:, 1, 1]-eval0, A[:, 1, 2]], dim=1)
    row2 = torch.stack([A[:, 0, 2], A[:, 1, 2], A[:, 2, 2]-eval0], dim=1)
    
    r0xr1 = torch.cross(row0, row1)
    r0xr2 = torch.cross(row0, row2)
    r1xr2 = torch.cross(row1, row2)

    d0 = torch.sum(r0xr1**2, dim=1, keepdim=True)
    d1 = torch.sum(r0xr2**2, dim=1, keepdim=True)
    d2 = torch.sum(r1xr2**2, dim=1, keepdim=True)
    
    d0 = d0 + 1e-10*(d0==0)
    d1 = d1 + 1e-10*(d1==0)
    d2 = d2 + 1e-10*(d2==0)
    
    d012 = torch.cat([d0, d1, d2], dim=1)
    imax = torch.argmax(d012, dim=1, keepdim=True)
    
    evec0 = r0xr1/torch.sqrt(d0)*(imax==0) + r0xr2/torch.sqrt(d1)*(imax==1) + r1xr2/torch.sqrt(d2)*(imax==2)
    return evec0


def compute_evec1(A, evec0, eval1):
    '''
    A : (b, 3, 3)
    evec0 : (b, 3)
    eval1 : (b)
    -------
    evec1 : (b, 3)
    '''
    b = A.shape[0]
    mask_1 = torch.abs(evec0[:, 0])>torch.abs(evec0[:, 1])
    mask_1 = mask_1.unsqueeze(1)
    inv_length = 1/torch.sqrt(evec0[:, 0]**2+evec0[:, 2]**2+1e-10);
    U_1 = torch.stack([-evec0[:, 2]*inv_length, torch.zeros(b).cuda(), evec0[:, 0]*inv_length], dim=1)
    mask_2 = torch.abs(evec0[:, 0])<=torch.abs(evec0[:, 1])
    mask_2 = mask_2.unsqueeze(1)
    inv_length = 1/torch.sqrt(evec0[:, 1]**2+evec0[:, 2]**2+1e-10);
    U_2 = torch.stack([torch.zeros(b).cuda(), evec0[:, 2]*inv_length, -evec0[:, 1]*inv_length], dim=1)
    U = U_1*mask_1 + U_2*mask_2
    V = torch.cross(evec0, U)
    
    AU = torch.stack([A[:, 0, 0] * U[:, 0] + A[:, 0, 1] * U[:, 1] + A[:, 0, 2] * U[:, 2],
                      A[:, 0, 1] * U[:, 0] + A[:, 1, 1] * U[:, 1] + A[:, 1, 2] * U[:, 2],
                      A[:, 0, 2] * U[:, 0] + A[:, 1, 2] * U[:, 1] + A[:, 2, 2] * U[:, 2]], dim=1)
    AV = torch.stack([A[:, 0, 0] * V[:, 0] + A[:, 0, 1] * V[:, 1] + A[:, 0, 2] * V[:, 2],
                      A[:, 0, 1] * V[:, 0] + A[:, 1, 1] * V[:, 1] + A[:, 1, 2] * V[:, 2],
                      A[:, 0, 2] * V[:, 0] + A[:, 1, 2] * V[:, 1] + A[:, 2, 2] * V[:, 2]], dim=1)
    
    m00 = U[:, 0] * AU[:, 0] + U[:, 1] * AU[:, 1] + U[:, 2] * AU[:, 2] - eval1
    m01 = U[:, 0] * AV[:, 0] + U[:, 1] * AV[:, 1] + U[:, 2] * AV[:, 2]
    m11 = V[:, 0] * AV[:, 0] + V[:, 1] * AV[:, 1] + V[:, 2] * AV[:, 2] - eval1
    
    absM00 = torch.abs(m00)
    absM01 = torch.abs(m01)
    absM11 = torch.abs(m11)
    
    m00 = m00 + 1e-10*(m00==0)
    m01 = m01 + 1e-10*(m01==0)
    m11 = m11 + 1e-10*(m11==0)
    
    max_abs_comp = absM00*(absM00>absM01) + absM01*(absM01>=absM00)
    M01 = m01/m00*(1/torch.sqrt(1+(m01/m00)**2))*(absM00>=absM01) + 1/torch.sqrt(1+(m00/m01)**2)*(absM00<absM01)
    M00 = 1/torch.sqrt(1+(m01/m00)**2)*(absM00>=absM01) + m00/m01*(1/torch.sqrt(1+(m00/m01)**2))*(absM00<absM01)
    output_1 = (M01[:,None]*U - M00[:,None]*V)*(max_abs_comp[:,None]>0) + U*(max_abs_comp[:,None]<=0)
    
    max_abs_comp = absM11*(absM11>absM01) + absM01*(absM01>=absM11)
    M01 = m01/m11*(1/torch.sqrt(1+(m01/m11)**2))*(absM11>=absM01) + 1/torch.sqrt(1+(m11/m01)**2)*(absM11<absM01)
    M11 = 1/torch.sqrt(1+(m01/m11)**2)*(absM11>=absM01) + m11/m01*(1/torch.sqrt(1+(m11/m01)**2))*(absM11<absM01)
    output_2 = (M11[:,None]*U - M01[:,None]*V)*(max_abs_comp[:,None]>0) + U*(max_abs_comp[:,None]<=0)
    
    evec1 = output_1*(absM00[:,None] >= absM11[:,None]) + output_2*(absM00[:,None] < absM11[:,None])
    return evec1


# evec1 = compute_evec1(A, evec0, 0.3)
# print (evec1)
# print (torch.sum(evec1*evec0))
    

def fast_eigen(A):
    '''
    A : (b, 3, 3)
    --------
    frame : (b, 3, 3)
    evals : (b, 3)
    '''
    b = A.shape[0]
    max_coeff, _ = torch.max(A.reshape(b, -1), dim=1)
    max_coeff = max_coeff + 1e-10*(max_coeff==0)
    A = A/(max_coeff[:, None, None])
    
    norm = A[:, 0, 1]**2 + A[:, 0, 2]**2 + A[:, 1, 2]**2
    
    # if (norm > 0)
    q = (A[:, 0, 0] + A[:, 1, 1] + A[:, 2, 2]) / 3
    b00 = A[:, 0, 0] - q
    b11 = A[:, 1, 1] - q
    b22 = A[:, 2, 2] - q
    
    p = torch.sqrt((b00**2 + b11**2 + b22**2 + norm*2) / 6)
    p = p + 1e-10*(p==0)
    
    c00 = b11*b22 - A[:, 1, 2]**2
    c01 = A[:, 0, 1]*b22 - A[:, 1, 2]*A[:, 0, 2]
    c02 = A[:, 0, 1]*A[:, 1, 2] - b11*A[:, 0, 2]
    det = (b00*c00 - A[:, 0, 1]*c01 + A[:, 0, 2]*c02) / (p*p*p)
    
    half_det = det * 0.5
    half_det = torch.clamp(half_det, min=-1.0, max=1.0)
    
    angle = torch.acos(half_det)/3.0
    two_thirds_pi = 2*np.pi/3
    beta2 = torch.cos(angle)*2
    beta0 = torch.cos(angle + two_thirds_pi)*2
    beta1 = -(beta0 + beta2)
    
    evals = torch.stack([q+p*beta0, q+p*beta1, q+p*beta2], dim=1)
    
    ### when half_det >= 0
    evec2 = compute_evec0(A, evals[:, 2])
    evec1 = compute_evec1(A, evec2, evals[:, 1])
    evec0 = torch.cross(evec1, evec2)
    
    evecs = torch.stack([evec0, evec1, evec2], dim=1).reshape(b*3, 3)
    base = torch.arange(0, b).reshape(b, 1).repeat(1, 3).reshape(b*3)*3
    evals_ind = torch.argsort(evals, dim=1).reshape(b*3) + base.cuda()
    frame_1 = evecs[evals_ind].reshape(b, 3, 3)
    
    ### when half_det < 0
    evec0 = compute_evec0(A, evals[:, 0])
    evec1 = compute_evec1(A, evec0, evals[:, 1])
    evec2 = torch.cross(evec0, evec1)
    
    evecs = torch.stack([evec0, evec1, evec2], dim=1).reshape(b*3, 3)
    base = torch.arange(0, b).reshape(b, 1).repeat(1, 3).reshape(b*3)*3
    evals_ind = torch.argsort(evals, dim=1).reshape(b*3) + base.cuda()
    frame_2 = evecs[evals_ind].reshape(b, 3, 3)
    
    half_det = half_det.reshape(b, 1, 1)
    frame = frame_1*(half_det>=0) + frame_2*(half_det<0)
    
    A = A*max_coeff[:, None, None]
    
    # if (norm <= 0)
    diag = torch.stack([A[:, 0, 0], A[:, 1, 1], A[:, 2, 2]], dim=1)
    eyes = torch.eye(3).repeat(b, 1).cuda()
    base = torch.arange(0, b).reshape(b, 1).repeat(1, 3).reshape(b*3)*3
    diag_ind = torch.argsort(diag, dim=1).reshape(b*3) + base.cuda()
    eyes = eyes[diag_ind].reshape(b, 3, 3)
    
    norm = norm.reshape(b, 1, 1)
    frame = frame*(norm > 0) + eyes*(norm<=0)
    
    return evals, frame
