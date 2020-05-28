import numpy as np
import scipy.sparse as sp

import sys, os
sys.path.append('../')

import cv2
import sys
import argparse

from time import time
from tqdm import tqdm
import pickle

from os import listdir
from os.path import isfile, join


class OBJLoader():
    '''Simplified SMPL model. All pose-induced transformation is ignored. Also ignore beta.'''
    def __init__(self, path_to_obj, path_to_colors, path_to_faces):
        self.path_to_obj = path_to_obj
        self.all_objs = [f for f in listdir(path_to_obj) if isfile(join(path_to_obj, f))]
        self.all_objs.sort()

        self.colors = np.load(path_to_colors)
        self.faces = np.int32(np.load(path_to_faces))
        self.vertices = None

        self.num_verts = self.colors.shape[0]
        self.num_faces = self.faces.shape[0]
        self.num_frames = len(self.all_objs)

    def get_faces(self, mesh_path):
        with open(mesh_path, "r") as f:
            content = f.readlines()
        faces = [[np.float32(f_i) for f_i in f.split('\n')[0].split(' ')[1:]] 
                    for f in content if f.startswith('f ')]
        faces = np.array(faces)
        return faces

    def get_vertices(self, mesh_path, with_texture = False):
        with open(mesh_path, "r") as f:
            content = f.readlines()
        vertices = [[np.float32(v_i) for v_i in v.split('\n')[0].split(' ')[1:]] 
                    for v in content if v.startswith('v ')]
        vertices = np.array(vertices)
        if with_texture:
            return vertices
        else:
            return vertices[:,:3]

    def imitate(self, idx):
        vertices = self.get_vertices(join(self.path_to_obj, self.all_objs[idx]))

        bbox_x_min, bbox_x_max = np.min(vertices[:,0]), np.max(vertices[:,0])
        bbox_y_min, bbox_y_max = np.min(vertices[:,1]), np.max(vertices[:,1])
        bbox_z_min, bbox_z_max = np.min(vertices[:,2]), np.max(vertices[:,2])

        # center = 0.5* np.array([(bbox_x_min + bbox_x_max), 
        #                           (bbox_y_min + bbox_y_max),
        #                           (bbox_z_min + bbox_z_max)])
        center = 0.5* np.array([(bbox_x_min + bbox_x_max), 
                                  (bbox_y_min + bbox_y_max),
                                  (0)])

        self.vertices = vertices - center

        # if self.faces is None:
        #     self.faces = self.get_faces(join(self.path_to_obj, self.all_objs[idx]))


if __name__ == '__main__':
    obj_loader = OBJLoader('all_meshes/running_sequence', 'all_meshes/colors.npy', 'all_meshes/faces.npy')
    obj_loader.imitate(1)
    print(np.max(obj_loader.faces), np.min(obj_loader.faces))
