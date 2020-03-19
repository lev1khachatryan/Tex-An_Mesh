from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def batch_orth_proj_idrot(X, camera, name=None):
    """
    X is N x P x 3 (for our case P = 19)
    camera is N x 3
    same as applying orth_proj_idrot to each N 
    """
    with tf.name_scope(name, "batch_orth_proj_idrot", [X, camera]):
        # reshape to (N, 1, 3)
        camera = tf.reshape(camera, [-1, 1, 3], name="cam_adj_shape")

        # (1, 19, 2) + (1, 1, 2) = (1, 19, 2)
        X_trans = X[:, :, :2] + camera[:, :, 1:]

        shape = tf.shape(X_trans)
        
        # camera[:, :, 0] is (1, 1)
        # tf.reshape(X_trans, [shape[0], -1]) is (1, 38)
        # camera[:, :, 0] * tf.reshape(X_trans, [shape[0], -1]) is (1, 38)
        return tf.reshape(
            camera[:, :, 0] * tf.reshape(X_trans, [shape[0], -1]), shape)