import numpy as np
import scipy.sparse as sp

import sys, os
sys.path.append('./tex2shape/')

import cv2
import sys
import argparse

from lib.mesh_from_maps import MeshFromMaps
from lib import mesh
from lib.maps import map_densepose_to_tex, normalize

import pickle
import time


class Tex2Shape():
    '''Simplified SMPL model. All pose-induced transformation is ignored. Also ignore beta.'''
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            self.dd = pickle.load(f, encoding="latin-1")
        self.mfm = MeshFromMaps()
        clothed = self.mfm.get_mesh(self.dd['normal_map'], self.dd['displacement_map'], betas=self.dd['betas'])

        self.v = clothed['v']
        self.f = clothed['f']
        self.vn = clothed['vn']
        self.vt = clothed['vt']
        self.ft = clothed['ft']

        color_path = os.path.join('/'.join(model_path.split('/')[:-2]), 'colors.npy')
        self.colors = np.load(color_path)

    def imitate(self, theta):
        clothed = self.mfm.get_mesh(self.dd['normal_map'], self.dd['displacement_map'], betas=self.dd['betas'], pose=theta)
        self.v = clothed['v']
        self.f = clothed['f']
        self.vn = clothed['vn']
        self.vt = clothed['vt']
        self.ft = clothed['ft']


if __name__ == '__main__':
    t2s = Tex2Shape('./tex2shape/results/pkls/rp_lee_posed_004_30k.pkl')
    print(t2s.colors.shape)