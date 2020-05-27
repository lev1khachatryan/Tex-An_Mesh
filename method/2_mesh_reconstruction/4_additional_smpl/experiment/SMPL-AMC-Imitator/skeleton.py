import numpy as np
import transforms3d


smpl_keypoint_semantic = {
  0: 'root',
  1: 'llegroot',
  2: 'rlegroot',
  3: 'lowerback',
  4: 'lknee',
  5: 'rknee',
  6: 'upperback',
  7: 'lankle',
  8: 'rankle',
  9: 'thorax',
  10: 'ltoes',
  11: 'rtoes',
  12: 'lowerneck',
  13: 'lclavicle',
  14: 'rclavicle',
  15: 'upperneck',
  16: 'larmroot',
  17: 'rarmroot',
  18: 'lelbow',
  19: 'relbow',
  20: 'lwrist',
  21: 'rwrist',
  22: 'lhand',
  23: 'rhand'
}


smpl_asf_map = {
  0: 'root',
  1: 'lfemur',
  2: 'rfemur',
  3: 'upperback',
  4: 'ltibia',
  5: 'rtibia',
  6: 'thorax',
  7: 'lfoot',
  8: 'rfoot',
  9: 'lowerneck',
  10: 'ltoes',
  11: 'rtoes',
  12: 'upperneck',
  13: 'lclavicle',
  14: 'rclavicle',
  15: 'head',
  16: 'lhumerus',
  17: 'rhumerus',
  18: 'lradius',
  19: 'rradius',
  20: 'lwrist',
  21: 'rwrist',
  22: 'lhand',
  23: 'rhand'
}


asf_smpl_map = {
  'root': 0,
  'lfemur': 1,
  'rfemur': 2,
  'upperback': 3,
  'ltibia': 4,
  'rtibia': 5,
  'thorax': 6,
  'lfoot': 7,
  'rfoot': 8,
  'lowerneck': 9,
  'ltoes': 10,
  'rtoes': 11,
  'upperneck': 12,
  'lclavicle': 13,
  'rclavicle': 14,
  'head': 15,
  'lhumerus': 16,
  'rhumerus': 17,
  'lradius': 18,
  'rradius': 19,
  'lwrist': 20,
  'rwrist': 21,
  'lhand': 22,
  'rhand': 23
}

class Joint:
  def __init__(self, name, direction, length, axis, dof, limits):
    self.name = name
    self.direction = np.matrix(direction)
    self.length = length
    axis = np.deg2rad(axis)
    self.C = np.matrix(transforms3d.euler.euler2mat(*axis))
    self.Cinv = np.linalg.inv(self.C)
    self.limits = np.zeros([3, 2])
    self.movable = len(dof) == 0
    for lm, nm in zip(limits, dof):
      if nm == 'rx':
        self.limits[0] = lm
      elif nm == 'ry':
        self.limits[1] = lm
      else:
        self.limits[2] = lm
    self.parent = None
    self.children = []
    # bone's far end's cooridnate
    self.coordinate = None
    self.matrix = None
    self.relative_R = None

  def set_motion(self, motion):
    if self.name == 'root':
      self.coordinate = np.array(motion['root'][:3])
      rotation = np.deg2rad(motion['root'][3:])
      self.matrix = self.C * np.matrix(transforms3d.euler.euler2mat(*rotation)) * self.Cinv
      self.relative_R = np.array(self.matrix)
    else:
      # set rx ry rz according to degree of freedom
      idx = 0
      rotation = np.zeros(3)
      for axis, lm in enumerate(self.limits):
        if not np.array_equal(lm, np.zeros(2)):
          rotation[axis] = motion[self.name][idx]
          idx += 1
      rotation = np.deg2rad(rotation)
      self.relative_R = np.array(self.C * np.matrix(transforms3d.euler.euler2mat(*rotation)) * self.Cinv)
      self.matrix = self.parent.matrix * np.matrix(self.relative_R)
      self.coordinate = np.squeeze(np.array(np.reshape(self.parent.coordinate, [3, 1]) + self.length * self.matrix * np.reshape(self.direction, [3, 1])))

    for child in self.children:
      child.set_motion(motion)

  def to_dict(self):
    ret = {self.name: self}
    for child in self.children:
      ret.update(child.to_dict())
    return ret

  def reset_pose(self):
    if self.name == 'root':
      self.coordinate = np.zeros(3)
    else:
      self.coordinate = self.parent.coordinate + self.length * np.squeeze(np.array(self.direction))
    self.relative_R = np.eye(3)
    self.matrix = np.matrix(self.relative_R)
    for child in self.children:
      child.reset_pose()

  def pretty_print(self):
    print('===================================')
    print('joint: %s' % self.name)
    print('direction:')
    print(self.direction)
    print('limits:', self.limits)
    print('parent:', self.parent)
    print('children:', self.children)


class SMPLJoints:
  def __init__(self, idx):
    self.idx = idx
    self.to_parent = None
    self.parent = None
    self.coordinate = None
    self.matrix = None
    self.children = []
    self.align_R = np.eye(3)
    self.motion_R = None

  def init_bone(self):
    if self.parent is not None:
      self.to_parent = self.coordinate - self.parent.coordinate

  def set_motion_R(self, motion):
    self.motion_R = motion[self.idx]
    if self.parent is not None:
      self.motion_R = self.parent.motion_R.dot(self.motion_R)
    for child in self.children:
      child.set_motion_R(motion)

  def update_coord(self):
    if self.parent is not None:
      absolute_R = self.parent.motion_R.dot(self.parent.align_R)
      self.coordinate = self.parent.coordinate + np.squeeze(absolute_R.dot(np.reshape(self.to_parent, [3,1])))
    for child in self.children:
      child.update_coord()

  def to_dict(self):
    ret = {self.idx: self}
    for child in self.children:
      ret.update(child.to_dict())
    return ret

  def export_G(self):
    G = np.zeros([4, 4])
    G[:3,:3] = self.motion_R.dot(self.align_R)
    G[:3,3] = self.coordinate
    G[3,3] = 1
    return G

  def export_theta(self):
    self_relative_G = None
    if self.parent is None:
      self_relative_G = self.export_G()[:3,:3]
    else:
      parent_G = self.parent.export_G()[:3,:3]
      self_G = self.export_G()[:3,:3]
      # parent_G * relative_G = self_G
      self_relative_G = np.linalg.inv(parent_G).dot(self_G)
    ax, rad = transforms3d.axangles.mat2axangle(self_relative_G)
    ax = ax[:3]
    axangle = ax / np.linalg.norm(ax) * rad
    return axangle


def setup_smpl_joints(smpl, rescale=True):
  joints = {}
  for i in range(24):
    joints[i] = SMPLJoints(i)
  for child, parent in smpl.parent.items():
    joints[child].parent = joints[parent]
    joints[parent].children.append(joints[child])
  if rescale:
    J = smpl.J / 0.45 * 10
  else:
    J = smpl.J
  for j in joints.values():
    j.coordinate = J[j.idx]
  for j in joints.values():
    j.init_bone()
  return joints
