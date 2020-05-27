import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def imshow(img):
  plt.imshow(img)
  plt.show()


def move_skeleton(joints, distance):
  distance = np.array(distance)
  for j in joints.values():
    j.coordinate += distance


def combine_skeletons(skeletons):
  joints = {}
  for idx, skeleton in enumerate(skeletons):
    for k, v in skeleton.items():
      joints['%s_%d' % (k, idx)] = v
  return joints


def obj_save(path, vertices, faces=None):
  with open(path, 'w') as fp:
    for v in vertices:
      fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
    if faces is not None:
      for f in faces + 1:
        fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


def draw_body(joints, xr=(-20, 40), yr=(-10, 50), zr=(-40, 20)):
  fig = plt.figure()
  ax = Axes3D(fig)

  # ax.set_xlim3d(-30, 50)
  # ax.set_ylim3d(0, 30)
  # ax.set_zlim3d(0, 30)

  ax.set_xlim3d(*xr)
  ax.set_ylim3d(*yr)
  ax.set_zlim3d(*zr)

  xs, ys, zs = [], [], []
  for joint in joints.values():
    if joint.coordinate is None:
      continue
    xs.append(joint.coordinate[0])
    ys.append(joint.coordinate[1])
    zs.append(joint.coordinate[2])
  plt.plot(zs, xs, ys, 'b.')

  for joint in joints.values():
    if joint.coordinate is None:
      continue
    child = joint
    if child.parent is not None:
      parent = child.parent
      xs = [child.coordinate[0], parent.coordinate[0]]
      ys = [child.coordinate[1], parent.coordinate[1]]
      zs = [child.coordinate[2], parent.coordinate[2]]
      plt.plot(zs, xs, ys, 'r')
  plt.show()

