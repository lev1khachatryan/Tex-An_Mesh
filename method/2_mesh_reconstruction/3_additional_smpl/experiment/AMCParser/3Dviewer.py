import pygame
import numpy as np
import time
import transforms3d.euler as euler
from amc_parser import *

from OpenGL.GL import *
from OpenGL.GLU import *


class Viewer:
  def __init__(self, joints=None, motions=None):
    """
    Display motion sequence in 3D.

    Parameter
    ---------
    joints: Dict returned from `amc_parser.parse_asf`. Keys are joint names and
    values are instance of Joint class.

    motions: List returned from `amc_parser.parse_amc. Each element is a dict
    with joint names as keys and relative rotation degree as values.

    """
    self.joints = joints
    self.motions = motions
    self.frame = 0 # current frame of the motion sequence
    self.playing = False # whether is playing the motion sequence
    self.fps = 120 # frame rate

    # whether is dragging
    self.rotate_dragging = False
    self.translate_dragging = False
    # old mouse cursor position
    self.old_x = 0
    self.old_y = 0
    # global rotation
    self.global_rx = 0
    self.global_ry = 0
    # rotation matrix for camera moving
    self.rotation_R = np.eye(3)
    # rotation speed
    self.speed_rx = np.pi / 90
    self.speed_ry = np.pi / 90
    # translation speed
    self.speed_trans = 0.25
    self.speed_zoom = 0.5
    # whether the main loop should break
    self.done = False
    # default translate set manually to make sure the skeleton is in the middle
    # of the window
    # if you can't see anything in the screen, this is the first parameter you
    # need to adjust
    self.default_translate = np.array([0, -20, -100], dtype=np.float32)
    self.translate = np.copy(self.default_translate)

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
    """
    Handle user interface events: keydown, close, dragging.

    """
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        self.done = True
      elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_RETURN: # reset camera
          self.translate = self.default_translate
          self.global_rx = 0
          self.global_ry = 0
        elif event.key == pygame.K_SPACE:
          self.playing = not self.playing
      elif event.type == pygame.MOUSEBUTTONDOWN: # dragging
        if event.button == 1:
          self.rotate_dragging = True
        else:
          self.translate_dragging = True
        self.old_x, self.old_y = event.pos
      elif event.type == pygame.MOUSEBUTTONUP:
        if event.button == 1:
          self.rotate_dragging = False
        else:
          self.translate_dragging = False
      elif event.type == pygame.MOUSEMOTION:
        if self.translate_dragging:
          # haven't figure out best way to implement this
          pass
        elif self.rotate_dragging:
          new_x, new_y = event.pos
          self.global_ry -= (new_x - self.old_x) / \
              self.screen_size[0] * np.pi
          self.global_rx -= (new_y - self.old_y) / \
              self.screen_size[1] * np.pi
          self.old_x, self.old_y = new_x, new_y
    pressed = pygame.key.get_pressed()
    # rotation
    if pressed[pygame.K_DOWN]:
      self.global_rx -= self.speed_rx
    if pressed[pygame.K_UP]:
      self. global_rx += self.speed_rx
    if pressed[pygame.K_LEFT]:
      self.global_ry += self.speed_ry
    if pressed[pygame.K_RIGHT]:
      self.global_ry -= self.speed_ry
    # moving
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
    # forward and rewind
    if pressed[pygame.K_COMMA]:
      self.frame -= 1
      if self.frame < 0:
        self.frame = len(self.motions) - 1
    if pressed[pygame.K_PERIOD]:
      self.frame += 1
      if self.frame >= len(self.motions):
        self.frame = 0
    # global rotation
    grx = euler.euler2mat(self.global_rx, 0, 0)
    gry = euler.euler2mat(0, self.global_ry, 0)
    self.rotation_R = grx.dot(gry)

  def set_joints(self, joints):
    """
    Set joints for viewer.

    Parameter
    ---------
    joints: Dict returned from `amc_parser.parse_asf`. Keys are joint names and
    values are instance of Joint class.

    """
    self.joints = joints

  def set_motion(self, motions):
    """
    Set motion sequence for viewer.

    Paramter
    --------
    motions: List returned from `amc_parser.parse_amc. Each element is a dict
    with joint names as keys and relative rotation degree as values.

    """
    self.motions = motions

  def draw(self):
    """
    Draw the skeleton with balls and sticks.

    """
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glBegin(GL_POINTS)
    for j in self.joints.values():
      coord = np.array(
        np.squeeze(j.coordinate).dot(self.rotation_R) + \
        self.translate, dtype=np.float32
      )
      glVertex3f(*coord)
    glEnd()

    glBegin(GL_LINES)
    for j in self.joints.values():
      child = j
      parent = j.parent
      if parent is not None:
        coord_x = np.array(
          np.squeeze(child.coordinate).dot(self.rotation_R)+self.translate,
          dtype=np.float32
        )
        coord_y = np.array(
          np.squeeze(parent.coordinate).dot(self.rotation_R)+self.translate,
          dtype=np.float32
        )
        glVertex3f(*coord_x)
        glVertex3f(*coord_y)
    glEnd()

  def run(self):
    """
    Main loop.

    """
    while not self.done:
      self.process_event()
      self.joints['root'].set_motion(self.motions[self.frame])
      if self.playing:
        self.frame += 1
        if self.frame >= len(self.motions):
          self.frame = 0
      self.draw()
      pygame.display.set_caption(
        'AMC Parser - frame %d / %d' % (self.frame, len(self.motions))
      )
      pygame.display.flip()
      self.clock.tick(self.fps)
    pygame.quit()


if __name__ == '__main__':
  asf_path = './data/01/01.asf'
  amc_path = './data/01/01_01.amc'
  joints = parse_asf(asf_path)
  motions = parse_amc(amc_path)
  v = Viewer(joints, motions)
  v.run()
