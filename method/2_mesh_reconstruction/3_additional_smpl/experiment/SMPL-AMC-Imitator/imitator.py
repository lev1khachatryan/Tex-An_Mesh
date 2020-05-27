from smpl_np import SMPLModel
from skeleton import Joint
from skeleton import SMPLJoints
from skeleton import asf_smpl_map
from skeleton import smpl_asf_map
import numpy as np
import transforms3d

class Imitator:
  def __init__(self, asf_joints, smpl):
    """
    Transfer amc/asf motions into SMPL model pose.

    Paramters
    ---------
    asf_joints: Dict returned from `reader.parse_asf`. Keys are joint names and
    values are instance of Joint class.

    smpl: `SMPLModel` class in `smpl_np.py`.

    """
    asf_joints['root'].reset_pose()
    self.smpl = smpl
    self.asf_joints = asf_joints
    self.smpl_joints = self.setup_smpl_joints()
    self.align_smpl_asf()

  def setup_smpl_joints(self):
    """
    Initialize a dict of `SMPLJoint`.

    Return
    ------
    Initilized dict of `SMPLJoint`. Keys are names, values are SMPLJoints.

    """
    joints = {}
    for i in range(24):
      joints[i] = SMPLJoints(i)
    for child, parent in self.smpl.parent.items():
      joints[child].parent = joints[parent]
      joints[parent].children.append(joints[child])
    J = np.copy(self.smpl.J)
    for j in joints.values():
      j.coordinate = J[j.idx]
    for j in joints.values():
      j.init_bone()
    return joints

  def align_smpl_asf(self, axis_rotation=False):
    """
    Align SMPL default pose to asf default pose. We process legs only (femur and
    tibia). `SMPLJoint.align_R` of leg root and knee are set.

    """
    for bone_name in ['lfemur', 'rfemur']:
      asf_dir = self.asf_joints[bone_name].direction

      smpl_leg_root = self.smpl_joints[asf_smpl_map[bone_name]]
      if axis_rotation:
        if bone_name == 'lfemur':
          smpl_leg_root.align_R = transforms3d.euler.axangle2mat(
            [0, 1, 0], -np.pi/16
          )
        else:
          smpl_leg_root.align_R = transforms3d.euler.axangle2mat(
            [0, 1, 0], +np.pi/16
          )

      smpl_knee = smpl_leg_root.children[0]
      smpl_dir = smpl_knee.to_parent / np.linalg.norm(smpl_knee.to_parent)

      smpl_leg_root.align_R = smpl_leg_root.align_R.dot(
        self.compute_rodrigues(smpl_dir, asf_dir)
      )

    for bone_name in ['ltibia', 'rtibia']:
      asf_tibia_dir = self.asf_joints[bone_name].direction
      asf_femur_dir = self.asf_joints[bone_name].parent.direction
      if not np.allclose(asf_femur_dir, asf_tibia_dir):
        # this case shouldn't happend in CMU dataset
        # so we just leave it here
        print('warning: femur and tibia are different!')

      smpl_knee = self.smpl_joints[asf_smpl_map[bone_name]]
      smpl_ankle = smpl_knee.children[0]
      smpl_tibia_dir = smpl_ankle.to_parent
      smpl_femur_dir = smpl_knee.to_parent

      smpl_knee.align_R = smpl_knee.parent.align_R.dot(
        self.compute_rodrigues(smpl_tibia_dir, smpl_femur_dir)
      )

  def compute_rodrigues(self, x, y):
    """
    Compute rotation matrix R such that y = Rx.

    Parameter
    ---------
    x: Ndarray to be rotated.
    y: Ndarray after rotation.

    """
    theta = np.arccos(np.inner(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
    axis = np.squeeze(np.cross(x, y))
    return transforms3d.axangles.axangle2mat(axis, theta)

  def map_R_asf_smpl(self):
    """
    Map asf joints' rotation matrices to SMPL joints'. In other words,
    'transfer' rotation of each asf joint to corresponding SMPL joint. Also
    transfer global translation.

    Return
    ------
    A tuple of (R, T), where R is a dict of name-rotation_matrix pair, and T is
    the gloval translation for root joint.

    """
    R = {}
    for k, v in smpl_asf_map.items():
      R[k] = self.asf_joints[v].relative_R
    return R, np.copy(np.squeeze(self.asf_joints['root'].coordinate))

  def smpl_joints_to_mesh(self):
    """
    Extract motions from SMPL joints to guide skinning.

    """
    G = np.empty([len(self.smpl_joints), 4, 4])
    for j in self.smpl_joints.values():
      G[j.idx] = j.export_G()
    self.smpl.do_skinning(G)

  def extract_theta(self):
    """
    Extract SMPL model's theta parameter, or 'pose'. An axis-angle presentation
    of each joint's rotation relative it's parent joint.

    Return
    ------
    A numpy ndarray of shape (24, 3).

    """
    theta = np.empty([len(self.smpl_joints), 3])
    for j in self.smpl_joints.values():
      theta[j.idx] = j.export_theta()
    return theta

  def motion2theta(self, motion):
    """
    A high level wrapper to convert asf motion into SMPL pose parameter.

    Parameter
    ---------
    motion: A dict whose keys are joint names and values are rotation degrees
    parsed from amc file. Should be a element of the list returned from
    `reader.parse_amc`.

    Return
    ------
    Pose parameter theta, an ndarray of shape (24, 3).

    """
    self.asf_joints['root'].set_motion(motion)
    self.asf_to_smpl_joints(False)
    return self.extract_theta()

  def asf_to_smpl_joints(self, translate):
    """
    Transfer asf joints' pose to SMPL joints. The coordinate of SMPL joints will
    be updated.

    """
    R, offset = self.map_R_asf_smpl()
    if translate:
      self.smpl_joints[0].coordinate = offset
    self.smpl_joints[0].set_motion_R(R)
    self.smpl_joints[0].update_coord()

  def set_asf_motion(self, motion, translate):
    """
    Set SMPL model joints and mesh to asf motion.

    Prameter
    --------
    motion: A dict whose keys are joint names and values are rotation degrees
    parsed from amc file. Should be a element of the list returned from
    `reader.parse_amc`.

    translate: Whether add global translate to the mesh. Not recommended since
    the mesh is likely to move beyond the screen.

    """
    self.asf_joints['root'].set_motion(motion)
    self.asf_to_smpl_joints(translate)
    self.smpl_joints_to_mesh()

  def imitate(self, motion, translate=False):
    """
    A warpper for `set_asf_motion` with a cool name.

    Prameter
    --------
    translate: Whether add global translate to the mesh. Not recommended since
    the mesh is likely to move beyond the screen.

    """
    self.set_asf_motion(motion, translate)


if __name__ == '__main__':
  import reader
  import pickle

  subject = '01'
  im = Imitator(
    reader.parse_asf('./data/%s/%s.asf' % (subject, subject)),
    SMPLModel('./model.pkl')
  )

  sequence = '01'
  frame_idx = 0
  motions = reader.parse_amc('./data/%s/%s_%s.amc' % (subject, subject, sequence))
  theta = im.motion2theta(motions[frame_idx])
  np.save('./pose.npy', theta)
