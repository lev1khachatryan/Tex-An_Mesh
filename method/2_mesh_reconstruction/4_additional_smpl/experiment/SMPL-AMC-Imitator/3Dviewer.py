import pygame
import numpy as np
import time
import transforms3d.euler as euler
import graphics_np
import copy
import cv2
import os
import random
from tqdm import tqdm

import reader
from vistool import *
from skeleton import *
from smpl_np import SMPLModel
from imitator import Imitator

from OpenGL.GL import *
from OpenGL.GLU import *


class SkeletonViewer:
  def __init__(self, imitator, motions):
    self.imitator = imitator
    self.smpl_joints = copy.copy(self.imitator.smpl_joints)
    self.asf_joints = copy.copy(self.imitator.asf_joints)
    self.motions = motions
    self.frame = 0
    self.playing = False
    self.fps = 120

    self.rorate_dragging = False
    self.translate_dragging = False
    self.old_x = 0
    self.old_y = 0
    self.global_rx = 0
    self.global_ry = 0
    self.rotation_R = np.eye(3)
    self.speed_rx = np.pi / 90
    self.speed_ry = np.pi / 90
    self.speed_trans = 0.25
    self.speed_zoom = 0.5
    self.done = False
    self.default_translate = np.array([0, 0, -200], dtype=np.float32)
    self.translate = np.copy(self.default_translate)

    self.smpl_default_translate = np.array([-40, 0, 0], dtype=np.float32)
    self.smpl_translate = np.copy(self.smpl_default_translate)

    pygame.init()
    self.screen_size = (1024, 768)
    self.screen = pygame.display.set_mode(
      self.screen_size, pygame.DOUBLEBUF | pygame.OPENGL
    )
    pygame.display.set_caption(
      'AMC Parser - frame %d / %d' % (self.frame, len(self.motions))
    )
    self.clock = pygame.time.Clock()

    glClearColor(0, 0, 0, 0)
    glShadeModel(GL_SMOOTH)
    glMaterialfv(
      GL_FRONT, GL_SPECULAR, np.array([1, 1, 1, 1], dtype=np.float32)
    )
    glMaterialfv(
      GL_FRONT, GL_SHININESS, np.array([100.0], dtype=np.float32)
    )
    glMaterialfv(
      GL_FRONT, GL_AMBIENT, np.array([0.7, 0.7, 0.7, 0.7], dtype=np.float32)
    )
    glEnable(GL_POINT_SMOOTH)

    glLightfv(GL_LIGHT0, GL_POSITION, np.array([1, 1, 1, 0], dtype=np.float32))
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST)
    gluPerspective(45, (self.screen_size[0]/self.screen_size[1]), 0.1, 500.0)

    glPointSize(10)
    glLineWidth(2.5)

  def process_event(self):
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        self.done = True
      elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_RETURN:
          self.translate = self.default_translate
          self.smpl_translate = self.smpl_default_translate
          self.global_rx = 0
          self.global_ry = 0
        elif event.key == pygame.K_SPACE:
          self.playing = not self.playing
      elif event.type == pygame.MOUSEBUTTONDOWN:
        if event.button == 1:
          self.rorate_dragging = True
        else:
          self.translate_dragging = True
        self.old_x, self.old_y = event.pos
      elif event.type == pygame.MOUSEBUTTONUP:
        if event.button == 1:
          self.rorate_dragging = False
        else:
          self.translate_dragging = False
      elif event.type == pygame.MOUSEMOTION:
        if self.translate_dragging:
          # haven't figure out best way to implement this
          pass
        elif self.rorate_dragging:
          new_x, new_y = event.pos
          self.global_ry -= (new_x - self.old_x) / \
              self.screen_size[0] * np.pi
          self.global_rx -= (new_y - self.old_y) / \
              self.screen_size[1] * np.pi
          self.old_x, self.old_y = new_x, new_y
    pressed = pygame.key.get_pressed()
    if pressed[pygame.K_DOWN]:
      self.global_rx -= self.speed_rx
    if pressed[pygame.K_UP]:
      self. global_rx += self.speed_rx
    if pressed[pygame.K_LEFT]:
      self.global_ry += self.speed_ry
    if pressed[pygame.K_RIGHT]:
      self.global_ry -= self.speed_ry
    if pressed[pygame.K_a]:
      self.translate[0] -= self.speed_trans
    if pressed[pygame.K_d]:
      self.translate[0] += self.speed_trans
    if pressed[pygame.K_w]:
      self.translate[1] += self.speed_trans
    if pressed[pygame.K_s]:
      self.translate[1] -= self.speed_trans
    if pressed[pygame.K_q]:
      self.translate[2] += self.speed_zoom
    if pressed[pygame.K_e]:
      self.translate[2] -= self.speed_zoom
    if pressed[pygame.K_COMMA]:
      self.frame -= 1
      if self.frame >= len(self.motions):
        self.frame = 0
    if pressed[pygame.K_PERIOD]:
      self.frame += 1
      if self.frame < 0:
        self.frame = len(self.motions) - 1
    if pressed[pygame.K_KP8]:
      self.smpl_translate[1] += self.speed_trans
    if pressed[pygame.K_KP2]:
      self.smpl_translate[1] -= self.speed_trans
    if pressed[pygame.K_KP4]:
      self.smpl_translate[0] -= self.speed_trans
    if pressed[pygame.K_KP6]:
      self.smpl_translate[0] += self.speed_trans

    grx = euler.euler2mat(self.global_rx, 0, 0)
    gry = euler.euler2mat(0, self.global_ry, 0)
    self.rotation_R = grx.dot(gry)

  def set_asf_joints(self, asf_joints):
    self.asf_joints = asf_joints

  def set_smpl_joints(self, smpl_joints):
    self.smpl_joints = smpl_joints

  def set_motion(self, motions):
    self.motions = motions

  def draw(self):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glBegin(GL_POINTS)
    self.draw_joints(self.asf_joints)
    self.draw_joints(self.smpl_joints)
    glEnd()

    glBegin(GL_LINES)
    self.draw_bones(self.asf_joints)
    self.draw_bones(self.smpl_joints)
    glEnd()

  def draw_joints(self, joints):
    for j in joints.values():
      coord = np.array(np.squeeze(j.coordinate).dot(self.rotation_R)+self.translate, dtype=np.float32)
      glVertex3f(*coord)

  def draw_bones(self, joints):
    for j in joints.values():
      child = j
      parent = j.parent
      if parent is not None:
        coord_x = np.array(np.squeeze(child.coordinate).dot(self.rotation_R)+self.translate, dtype=np.float32)
        coord_y = np.array(np.squeeze(parent.coordinate).dot(self.rotation_R)+self.translate, dtype=np.float32)
        glVertex3f(*coord_x)
        glVertex3f(*coord_y)

  def update_motion(self):
    self.imitator.imitate(self.motions[self.frame])
    self.smpl_joints = copy.deepcopy(self.imitator.smpl_joints)
    self.asf_joints = copy.copy(self.imitator.asf_joints)

    for j in self.smpl_joints.values():
      j.init_bone()
    for j in self.smpl_joints.values():
      if j.parent is not None:
        j.coordinate = j.parent.coordinate + 25 * j.to_parent
    move_skeleton(self.smpl_joints, self.smpl_translate)

    if self.playing:
      self.frame += 1
      if self.frame >= len(self.motions):
        self.frame = 0

  def run(self):
    while not self.done:
      self.process_event()
      self.update_motion()
      self.draw()
      pygame.display.set_caption('AMC Parser - frame %d / %d' % (self.frame, len(self.motions)))
      pygame.display.flip()
      self.clock.tick(self.fps)
    pygame.quit()


class MeshViewer:
  def __init__(self, imitator, motions):
    """
    Display 3D SMPL model mesh using `pygame`.

    Prameter
    --------
    imitator: Instance of `imitator.Imitator`, used to transfer asf motions to
    SMPL model.

    motions: A list of motions returned from 'reader.parse_amc'. Each item
    should be a dict whose keys are asf joint names and values are rotation
    degrees parsed from amc file.

    """
    self.imitator = imitator
    self.motions = motions
    self.num_verts = 6890
    self.num_faces = 13776
    self.num_frames = len(self.motions)
    # whether is dragging and the kind
    self.rorate_dragging = False
    self.translate_dragging = False
    self.old_x = 0
    self.old_y = 0

    storage = 9
    # how many faces each vertex belongs to
    vert_cnt = np.zeros([self.num_verts], dtype=np.int)
    # the faces each vertex belongs to
    # this is used to compute vertex normal later
    # which is the average of faces' normal it belongs to
    self.vftable = np.zeros((self.num_verts, storage), dtype=np.int32)
    for idx in range(self.num_faces):
      v1, v2, v3 = self.imitator.smpl.faces[idx]
      self.vftable[v1, vert_cnt[v1]] = idx
      self.vftable[v2, vert_cnt[v2]] = idx
      self.vftable[v3, vert_cnt[v3]] = idx
      vert_cnt[v1] += 1
      vert_cnt[v2] += 1
      vert_cnt[v3] += 1
    # we use a "virtual face" to fill "blanks"
    # this face's normal will be set to (0,0,0)
    for i in range(self.num_verts):
      for j in range(vert_cnt[i], storage):
        self.vftable[i, j] = self.num_faces

    pygame.init()
    self.screen_size = (1024, 768)
    self.screen = pygame.display.set_mode(
      self.screen_size, pygame.DOUBLEBUF | pygame.OPENGL
    )
    pygame.display.set_caption('Frame %d / %d' % (0, self.num_frames))
    self.clock = pygame.time.Clock()

    glClearColor(0.0, 0.0, 0.0, 0.0)
    glShadeModel(GL_SMOOTH)

    glMaterialfv(
      GL_FRONT,
      GL_AMBIENT,
      np.array([0.192250, 0.192250, 0.192250], dtype=np.float32)
    )
    glMaterialfv(
      GL_FRONT,
      GL_DIFFUSE,
      np.array([0.507540, 0.507540, 0.507540], dtype=np.float32)
    )
    glMaterialfv(
      GL_FRONT,
      GL_SPECULAR,
      np.array([.5082730, .5082730, .5082730], dtype=np.float32)
    )
    glMaterialf(
      GL_FRONT,
      GL_SHININESS,
      .4 * 128.0
    )

    glLightfv(GL_LIGHT0, GL_POSITION, np.array([1, 1, 1, 0], dtype=np.float32))

    glLightfv(
      GL_LIGHT0,
      GL_SPECULAR,
      np.array([0, 0, 0, 1], dtype=np.float32)
    )
    glLightfv(
      GL_LIGHT0, GL_DIFFUSE,
      np.array([1, 1, 1, 1], dtype=np.float32)
    )
    glLightfv(
      GL_LIGHT0, GL_AMBIENT,
      np.array([1, 1, 1, 1], dtype=np.float32)
    )

    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHTING)
    glEnable(GL_CULL_FACE)
    glEnable(GL_DEPTH_TEST)
    # glEnable(GL_VERTEX_ARRAY)
    # glEnable(GL_NORMAL_ARRAY)
    gluPerspective(45, (self.screen_size[0]/self.screen_size[1]), 0.1, 50.0)

  def run(self, translate=False, video_path=None, video_fps=120, render_fps=120,
          auto_run=False, auto_rerun=False, close_after_run=False):
    if video_path is not None:
      video = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*'DIVX'),
        video_fps,
        self.screen_size
      )
    global_rx = 0
    global_ry = 0
    # default mesh tranlsation to put the mesh in the center of the window
    # if translate is True, this translation need to be set carefully to make
    # sure the mesh is within the screen.
    std_trans = np.array([0, 0.2, -3], dtype=np.float32)
    global_trans = std_trans
    speed_rx = np.pi / 90
    speed_ry = np.pi / 90
    speed_trans = 0.01
    speed_zoom = 0.1
    frame = 0
    done = False
    playing = True if auto_run else False
    while not done:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          done = True
        elif event.type == pygame.KEYDOWN:
          if event.key == pygame.K_SPACE: # pause / play
            if playing:
              playing = False
            else:
              playing = True
          if event.key == pygame.K_RETURN: # reset camera
            global_trans = std_trans
            global_rx = 0
            global_ry = 0
        elif event.type == pygame.MOUSEBUTTONDOWN: # dragging
          if event.button == 1:
            self.rorate_dragging = True
          else:
            self.translate_dragging = True
          self.old_x, self.old_y = event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
          if event.button == 1:
            self.rorate_dragging = False
          else:
            self.translate_dragging = False
        elif event.type == pygame.MOUSEMOTION:
          if self.translate_dragging:
            # haven't figure out best way to implement this
            pass
          elif self.rorate_dragging:
            new_x, new_y = event.pos
            global_ry -= (new_x - self.old_x) / self.screen_size[0] * np.pi
            global_rx -= (new_y - self.old_y) / self.screen_size[1] * np.pi
            self.old_x, self.old_y = new_x, new_y
        pressed = pygame.key.get_pressed()
        # rotation
        if pressed[pygame.K_DOWN]:
          global_rx -= speed_rx
        if pressed[pygame.K_UP]:
          global_rx += speed_rx
        if pressed[pygame.K_LEFT]:
          global_ry += speed_ry
        if pressed[pygame.K_RIGHT]:
          global_ry -= speed_ry
        # translation
        if pressed[pygame.K_a]:
          global_trans[0] -= speed_trans
        if pressed[pygame.K_d]:
          global_trans[0] += speed_trans
        if pressed[pygame.K_w]:
          global_trans[1] += speed_trans
        if pressed[pygame.K_s]:
          global_trans[1] -= speed_trans
        # zoom
        if pressed[pygame.K_q]:
          global_trans[2] += speed_zoom
        if pressed[pygame.K_e]:
          global_trans[2] -= speed_zoom

      pygame.display.set_caption('Frame %d / %d' % (frame, self.num_frames))

      motion = self.motions[frame]

      if playing:
        frame += 1
      if frame >= self.num_frames:
        frame = 0
        if auto_rerun:
          playing = True
        else:
          playing = False
        if close_after_run:
          done = True

      grx = euler.euler2mat(global_rx, 0, 0)
      gry = euler.euler2mat(0, global_ry, 0)

      self.imitator.imitate(motion)
      verts = self.imitator.smpl.verts
      verts = verts.dot(grx).dot(gry) + global_trans
      faces_coor = verts[self.imitator.smpl.faces]

      face_normals = graphics_np.get_normal(faces_coor)
      face_normals = face_normals / \
        np.linalg.norm(face_normals, axis=1, keepdims=True)
      # a virtual face to boost vertex normal computation
      face_normals = np.append(face_normals, np.zeros((1,3)), axis=0)

      vert_normals = face_normals[self.vftable]
      vert_normals = np.sum(vert_normals, axis=1)
      vert_normals = vert_normals / \
        np.linalg.norm(vert_normals, axis=1, keepdims=True)

      verts = verts.astype(np.float32)
      vert_normals = vert_normals.astype(np.float32)

      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
      glVertexPointerf(verts)
      glNormalPointerf(vert_normals)
      glDrawElementsui(GL_TRIANGLES, self.imitator.smpl.faces)

      # using pygame.surfarray.array2d will always invoke a segmentation fault
      if video_path is not None:
        tmp = pygame.image.tostring(self.screen, 'RGB')
        tmp = pygame.image.fromstring(
          tmp,
          self.screen_size,
          'RGB'
        )
        img = np.transpose(pygame.surfarray.array3d(tmp), [1, 0, 2])
        video.write(img)

      pygame.display.flip()
      if render_fps > 0:
        self.clock.tick(freq)

    pygame.quit()
    video.release()


def test_skeleton():
  subject = '01'
  im = Imitator(
    reader.parse_asf('./data/%s/%s.asf' % (subject, subject)),
    SMPLModel('./model.pkl')
  )
  sequence = '01'
  motions = reader.parse_amc(
    './data/%s/%s_%s.amc' % (subject, subject, sequence)
  )
  viewer = SkeletonViewer(im, motions)
  viewer.run()


def test_mesh():
  subject = '01'
  im = Imitator(
    reader.parse_asf('./data/%s/%s.asf' % (subject, subject)),
    SMPLModel('./model.pkl')
  )
  sequence = '01'
  motions = reader.parse_amc(
    './data/%s/%s_%s.amc' % (subject, subject, sequence)
  )
  viewer = MeshViewer(im, motions)
  viewer.run()


def video_example():
  target_folder = './video'
  try:
    os.makedirs(target_folder)
  except:
    pass

  lv0 = './data'
  lv1s = os.listdir(lv0)
  for lv1 in tqdm(lv1s, ncols=120):
    lv2s = os.listdir('/'.join([lv0, lv1]))
    asf_path = os.path.join(lv0, lv1, lv1+'.asf')
    joints = reader.parse_asf(asf_path)
    im = Imitator(
      joints,
      SMPLModel('./model.pkl')
    )
    random.shuffle(lv2s)
    lv2 = None
    for lv2 in lv2s:
      if lv2.split('.')[-1] == 'amc':
        break
    amc_path = os.path.join(lv0, lv1, lv2)
    video_path = os.path.join(
      target_folder,
      '%s.avi' % lv2.split('.')[0]
    )
    motions = reader.parse_amc(amc_path)
    viewer = MeshViewer(im, motions)
    viewer.run(
      video_path=video_path,
      render_fps=-1,
      auto_run=True,
      auto_rerun=False,
      close_after_run=True
    )


if __name__ == '__main__':
  # test_mesh()
  # test_skeleton()
  video_example()
