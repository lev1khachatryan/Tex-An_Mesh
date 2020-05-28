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
from obj_loader import OBJLoader

from OpenGL.GL import *
from OpenGL.GLU import *

from os import listdir
from os.path import isfile, join


class MeshViewer:
  def __init__(self, objLoader):
    """
    Display 3D SMPL model mesh using `pygame`.

    Prameter
    --------
    t2s: Instance of `tex2shape.Tex2Shape`

    thetas: A list of poses. (N, 1, 72) where N is number of motions

    """
    self.obj_loader = objLoader

    self.num_verts = self.obj_loader.num_verts
    self.num_faces = self.obj_loader.num_faces
    self.num_frames = self.obj_loader.num_frames
    
    self.colors = self.obj_loader.colors
    self.faces = self.obj_loader.faces
    
    self.rorate_dragging = False
    self.translate_dragging = False
    self.old_x = 0
    self.old_y = 0

    storage = 11
    # how many faces each vertex belongs to
    vert_cnt = np.zeros([self.num_verts], dtype=np.int)
    # the faces each vertex belongs to
    # this is used to compute vertex normal later
    # which is the average of faces' normal it belongs to
    self.vftable = np.zeros((self.num_verts, storage), dtype=np.int32)

    for idx in range(self.num_faces):
      v1, v2, v3 = np.int32(self.faces[idx])
      self.vftable[v1, vert_cnt[v1]] = idx
      self.vftable[v2, vert_cnt[v2]] = idx
      self.vftable[v3, vert_cnt[v3]] = idx
      vert_cnt[v1] += 1
      vert_cnt[v2] += 1
      vert_cnt[v3] += 1
    # we use a "virtual face" to fill "blanks"
    # this face's normal will be set to (0,0,0)
    # print(vert_cnt)
    # exit()
    for i in range(self.num_verts):
      for j in range(vert_cnt[i], storage):
        self.vftable[i, j] = self.num_faces

    pygame.init()
    # w = 336
    # h = 720
    w = 1024
    h = 768
    self.screen_size = (w, h)
    self.screen = pygame.display.set_mode(
      self.screen_size, pygame.DOUBLEBUF | pygame.OPENGL
    )
    pygame.display.set_caption('Frame %d / %d' % (0, self.num_frames))
    self.clock = pygame.time.Clock()

    # glClearColor(0.7, 0.7, 0.7, 1.0)
    glClearColor(1.0, 1.0, 1.0, 1.0)
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
      GL_LIGHT0, 
      GL_DIFFUSE,
      np.array([1, 1, 1, 1], dtype=np.float32)
    )
    glLightfv(
      GL_LIGHT0, 
      GL_AMBIENT,
      np.array([1, 1, 1, 1], dtype=np.float32)
    )

    # glEnable(GL_LIGHT0)
    # glEnable(GL_LIGHTING)
    glEnable(GL_CULL_FACE)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_VERTEX_ARRAY)
    glEnable(GL_NORMAL_ARRAY)
    glEnable(GL_COLOR_ARRAY)
    gluPerspective(45, (self.screen_size[0]/self.screen_size[1]), 0.1, 50.0)

  def run(self, translate=False, video_path=None, video_fps=24, render_fps=120,
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
    std_trans = np.array([0, -0.2, -3], dtype=np.float32)
    # std_trans = np.array([0.7264086, 0.01124804, 0.12840584], dtype=np.float32)
    # std_trans = np.array([1792.38024181,  168.48374273,  351.3065274], dtype=np.float32)
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

      # theta = self.thetas[frame, :, :]

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

      self.obj_loader.imitate(frame)
      verts = self.obj_loader.vertices
      verts = verts.dot(grx).dot(gry) + global_trans
      faces_coor = verts[self.faces]

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

      alpha = [1.0]
      alphas = np.repeat(alpha, self.num_verts, axis=0).reshape((self.num_verts, 1))
      clr = np.concatenate((self.colors, alphas), axis=1)

      # cols = np.ones_like(self.colors,dtype='int8')
      clr = clr.astype(np.float32)

      # verts = np.concatenate((verts, self.colors), axis=1)
      # vert_normals = np.concatenate((vert_normals, self.colors), axis=1)

      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
      glVertexPointerf(verts)
      glNormalPointerf(vert_normals)
      glColorPointerf(clr)
      glDrawElementsui(GL_TRIANGLES, self.faces) # GL_QUADS GL_TRIANGLES

      # using pygame.surfarray.array2d will always invoke a segmentation fault
      if video_path is not None:
        img_id = 0
        tmp = pygame.image.tostring(self.screen, 'RGB')
        tmp = pygame.image.fromstring(
          tmp,
          self.screen_size,
          'RGB'
        )
        img = np.transpose(pygame.surfarray.array3d(tmp), [1, 0, 2])
        img = img[:, :, ::-1]

        # # for "gevor" character
        # bkg = cv2.imread('characters/gevor/original_bg.jpg')#[:, :, ::-1]
        # x1, x2, y1, y2 = 260, 650, 420, 600

        # for "dinklage" character
        # bkg = cv2.imread('characters/dinklage/original_bg.jpg')#[:, :, ::-1]
        # x1, x2, y1, y2 = 285, 615, 400, 620

        # # for "ronaldo" character
        # bkg = cv2.imread('characters/ronaldo/original_bg.jpg')#[:, :, ::-1]
        # x1, x2, y1, y2 = 290, 620, 350, 900

        # # for "messi" character
        # bkg = cv2.imread('characters/messi/original_bg.jpg')#[:, :, ::-1]
        # x1, x2, y1, y2 = 290, 605, 400, 620

        # # # for "putin" character
        # bkg = cv2.imread('characters/putin/original_bg.jpg')#[:, :, ::-1]
        # x1, x2, y1, y2 = 200, 600, 150, 750


        # img_to_mask = img.copy()
        # img_to_mask[~((img_to_mask[:, :, 0] >= 245) & (img_to_mask[:, :, 1] >= 245) & (img_to_mask[:, :, 2] >= 245))] = [0, 0, 0]
        # img_to_mask[((img_to_mask[:, :, 0] >= 245) & (img_to_mask[:, :, 1] >= 245) & (img_to_mask[:, :, 2] >= 245))] = [255, 255, 255]
        # kernel = np.ones((5,5),np.uint8)
        # img_to_mask = cv2.morphologyEx(img_to_mask, cv2.MORPH_CLOSE, kernel)
        # mask = np.int8(img_to_mask[x1:x2, y1:y2, :] / 255)
        # bkg = cv2.resize(bkg, (mask.shape[1], mask.shape[0]))
        # img_to_mask[x1:x2, y1:y2, :] = bkg * mask
        # img_to_mask[img_to_mask==0] = img[img_to_mask==0]
        # img=img_to_mask

        # cv2.imwrite('a.jpg', img[:, :, ::-1])
        # video.write(img)

      pygame.display.flip()
      if render_fps > 0:
        self.clock.tick(freq)

    pygame.quit()
    video.release()


def video_example(path_to_obj, path_to_colors, path_to_faces, target_folder):
	try:
		os.makedirs(target_folder)
	except:
		pass

	objLoader = OBJLoader(path_to_obj, path_to_colors, path_to_faces)
	viewer = MeshViewer(objLoader)

	video_path = os.path.join(
      target_folder,
      'output.avi'
    )

	viewer.run(
      video_path=video_path,
      render_fps=-1,
      auto_run=True,
      auto_rerun=False,
      close_after_run=False
    )


if __name__ == '__main__':
  # path_to_obj    = 'characters/dinklage/motions/roll/sequence'
  # video_path     = 'characters/dinklage/motions/roll/video'
  # path_to_colors = 'characters/dinklage/colors.npy'
  # path_to_faces  = 'characters/dinklage/faces.npy'

  # path_to_obj    = 'characters/gevor/motions/uprock/sequence'
  # video_path     = 'characters/gevor/motions/uprock/video'
  # path_to_colors = 'characters/gevor/colors.npy'
  # path_to_faces  = 'characters/gevor/faces.npy'

  # path_to_obj    = 'characters/karo/motions/samba/sequence'
  # video_path     = 'characters/karo/motions/samba/video'
  # path_to_colors = 'characters/karo/colors.npy'
  # path_to_faces  = 'characters/karo/faces.npy'

  # path_to_obj    = 'characters/ronaldo/motions/running/sequence'
  # video_path     = 'characters/ronaldo/motions/running/video'
  # path_to_obj    = 'characters/ronaldo/motions/bellydancing/sequence'
  # video_path     = 'characters/ronaldo/motions/bellydancing/video'
  # path_to_colors = 'characters/ronaldo/colors.npy'
  # path_to_faces  = 'characters/ronaldo/faces.npy'

  # path_to_obj    = 'characters/messi/motions/running/sequence'
  # video_path     = 'characters/messi/motions/running/video'
  # path_to_colors = 'characters/messi/colors.npy'
  # path_to_faces  = 'characters/messi/faces.npy'

  path_to_obj    = 'characters/putin/motions/walking/sequence'
  video_path     = 'characters/putin/motions/walking/video'
  path_to_colors = 'characters/putin/colors.npy'
  path_to_faces  = 'characters/putin/faces.npy'

  video_example(path_to_obj, path_to_colors, path_to_faces, video_path)