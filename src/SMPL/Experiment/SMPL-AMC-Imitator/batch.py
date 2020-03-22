from imitator import Imitator
from smpl_np import SMPLModel
from tqdm import tqdm
import reader
import os
import numpy as np

if __name__ == '__main__':
  target_folder = './pose'

  lv0 = './data'
  lv1s = os.listdir(lv0)
  for lv1 in tqdm(lv1s, ncols=120):
    lv2s = os.listdir('/'.join([lv0, lv1]))
    asf_path = '%s/%s/%s.asf' % (lv0, lv1, lv1)
    joints = reader.parse_asf(asf_path)
    im = Imitator(
      joints,
      SMPLModel('./model.pkl')
    )
    for lv2 in tqdm(lv2s, ncols=120):
      pose = []
      if lv2.split('.')[-1] != 'amc':
        continue
      amc_path = '%s/%s/%s' % (lv0, lv1, lv2)
      save_path = '%s/%s' % (target_folder, lv2.split('.')[0]+'.npy')
      motions = reader.parse_amc(amc_path)
      for idx, motion in enumerate(motions):
        pose.append(im.motion2theta(motions[idx]))
      np.save(save_path, np.array(pose))
      
      im.imitate(motions[0])
      im.smpl.output_mesh('.'.join(save_path.split('.')[:-1])+'.obj')
