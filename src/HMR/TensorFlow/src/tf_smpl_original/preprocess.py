#!/usr/bin/python2

import numpy as np
import cPickle as pickle
import sys


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python2 preprocess.py <path-to-neutral-SMPL-model>')
        sys.exit()
    smpl_path = sys.argv[1]

    with open(smpl_path, 'rb') as f:
        src_data = pickle.load(f)
    smpl_model_np = {
        'v_template': np.array(src_data['v_template']),
        'shapedirs': np.array(src_data['shapedirs']),
        'J_regressor': src_data['J_regressor'],
        'posedirs': np.array(src_data['posedirs']),
        'kintree_table': src_data['kintree_table'],
        'weights': np.array(src_data['weights']),
        'cocoplus_regressor': src_data['cocoplus_regressor']
    }
    with open('./model_neutral_np.pkl', 'wb') as f:
        pickle.dump(smpl_model_np, f)
