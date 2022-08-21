#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import hydra
import open3d as o3d
import numpy as np
from matplotlib import cm
from plyfile import PlyData

def so3rot(points, alpha, beta, gamma):
    '''
    rotate points using Euler angles
    points : (n,3)
    '''
    def R_z(ang):
        return np.array([[np.cos(ang),-np.sin(ang),0],[np.sin(ang),np.cos(ang),0],[0,0,1]])
    def R_y(ang):
        return np.array([[np.cos(ang),0,np.sin(ang)],[0,1,0],[-np.sin(ang),0,np.cos(ang)]])
    R = np.matmul(np.matmul(R_z(alpha), R_y(beta)), R_z(gamma))
    points = np.matmul(R, points.T).T
    return points
    
def load_ply(file_name):
    ply_data = PlyData.read(file_name)
    vertices = ply_data['vertex']
    points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    ret_val = [points]
    if len(ret_val) == 1:  # Unwrap the list
        ret_val = ret_val[0]
    pc = ret_val
    return pc

def read_pcd(path):
    pcd = o3d.io.read_point_cloud(path)
    pc_array = np.asarray(pcd.points, dtype=np.float32)
    return pc_array

def save_pcd(pc_array, path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_array)
    o3d.io.write_point_cloud(path, pcd)

def view_pcd(pc_array, color_map=None):
    '''
    pc_array : (n, 3)
    color_map : (n)
    '''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_array)
    if color_map is not None:
        rgb_map = cm.rainbow(np.linspace(0, 1, np.max(color_map)+1))[:, 0:3]
        colors = np.array([rgb_map[color_map[i]] for i in range(len(color_map))])
        pcd.colors = o3d.utility.Vector3dVector(colors)    
    o3d.visualization.draw_geometries([pcd])
