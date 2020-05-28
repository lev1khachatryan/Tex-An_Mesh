import numpy as np
import pickle


def depth2cloud(depth, dfx=365.456, dfy=365.456, dcx=254.878, dcy=205.395):
    _, height, width = depth.shape
    x_template = np.tile(np.reshape(np.array(range(width)), (1, -1)) - dcx, (height, 1)) / dfx
    y_template = np.tile(np.reshape(np.array(range(height)), (-1, 1)) - dcy, (1, width)) / dfy
    sequence = []
    for depth_frame in depth:
        x_slice = x_template * depth_frame
        y_slice = y_template * depth_frame
        z_slice = -depth_frame
        sequence.append(np.dstack((x_slice, y_slice, z_slice)))
    clouds = np.array(sequence)
    return clouds


def depth_color_to_cloud(depth, color):
    '''Should be 480x640, otherwise need to multiply by a rate.'''
    dfx = 1 / 558.666
    dfy = 1 / 556.919
    dcx = 329.522
    dcy = 231.313
    batch, height, width = depth.shape
    x_template = np.tile(np.reshape(np.array(range(width)), (1, -1)) - dcx, (height, 1)) * dfx
    y_template = np.tile(np.reshape(np.array(range(height)), (-1, 1)) - dcy, (1, width)) * dfy
    sequence = []
    for frame in range(batch):
        depth_frame = depth[frame]
        x_slice = x_template * depth_frame
        y_slice = y_template * depth_frame
        z_slice = -depth_frame
        geometry = np.dstack((x_slice, y_slice, z_slice))
        package = np.concatenate((geometry, color[frame]), axis=2)
        sequence.append(package)
    clouds = np.array(sequence)
    return clouds


def construct_surface_frame(vertices, faces):
    '''Give vertices' coordinates and face indices, return all faces in a vector style.'''
    ret = []
    for face in faces:
        v1, v2, v3 = face
        layer = np.vstack((vertices[v1], vertices[v2], vertices[v3]))  # 3x3
        ret.append(layer)
    ret = np.dstack(ret)  # 3x3xN
    ret = np.transpose(ret, axes=(2, 0, 1))
    return ret


def get_normal(faces):
    '''Given faces, return normal vector(NOT UNIT VECTOR).'''
    p1x, p1y, p1z = faces[:, 0, 0], faces[:, 0, 1], faces[:, 0, 2]
    p2x, p2y, p2z = faces[:, 1, 0], faces[:, 1, 1], faces[:, 1, 2]
    p3x, p3y, p3z = faces[:, 2, 0], faces[:, 2, 1], faces[:, 2, 2]
    x_stick = (p2y - p1y) * (p3z - p1z) - (p2z - p1z) * (p3y - p1y)
    y_stick = (p2z - p1z) * (p3x - p1x) - (p2x - p1x) * (p3z - p1z)
    z_stick = (p2x - p1x) * (p3y - p1y) - (p2y - p1y) * (p3x - p1x)
    normals = np.stack((x_stick, y_stick, z_stick), axis=1)
    return normals


def get_D(faces, normals):
    '''Plane: Ax + By + Cz + D = 0. Give normal (A, B, C) and 1 point, return D.'''
    p = faces[:, 0, :]
    D = np.negative(np.sum(normals*p, axis=1))
    return D


def cloud_surface_dist_all(cloud, normals, Ds):
    dists = np.dot(cloud, np.transpose(normals)) + np.reshape(Ds, (1, -1))
    mod = np.reshape(np.linalg.norm(normals, axis=1), (1, -1))
    dists_norm = np.abs(dists / mod)
    return dists_norm


def cloud_surface_dist_min(cloud_frame, normals, Ds):
    '''Compute the distance between each point in the frame to mesh surface, which is the minimal distance to all faces.'''
    # cloud_frame: height x width x 3
    # normals: N x 3
    # Ds: N
    cloud = np.reshape(cloud_frame, (-1, 3))
    dists_norm = cloud_surface_dist_all(cloud, normals, Ds)
    dists_min = np.reshape(np.amin(dists_norm, axis=1), (-1, 1))
    return dists_min


def compute_distance_all(faces, cloud):
    normals = get_normal(faces)
    Ds = get_D(faces, normals)
    dists = cloud_surface_dist_all(cloud, normals, Ds)
    return dists


def single_filter(arr, strict=1):
    '''Returns a mask'''
    mean = np.mean(arr, axis=(1, 2), keepdims=True)
    std = np.std(arr, axis=(1, 2), keepdims=True)
    return arr > mean + strict * std


def double_filter(depth, color, strict=1):
    '''Returns a mask'''
    mask_d = single_filter(depth, strict)
    mask_c = single_filter(color, strict)
    mask = np.logical_and(mask_c, mask_d)
    return mask


def compute_edges_direction(e):
    v1 = e[:, 0, :]
    v2 = e[:, 1, :]
    diff = v1 - v2
    return diff / np.linalg.norm(diff, axis=1, keepdims=True)


def edges_distance(e1, e2):
    #e: n x 2 x 3
    v1 = compute_edges_direction(e1)
    v2 = compute_edges_direction(e2)
    c_normal = np.cross(v1, v2) # N x 3
    connection = e1[:, 0, :] - e2[:, 0, :] # N x 3
    dists = np.sum(c_normal * connection, axis=1)
    return dists
        

def face_visibility_mask(faces):
    normals = get_normal(faces)
    direction = np.array((0, 0, 1)).reshape((3, 1))
    dot_product = np.dot(normals, direction)
    return (dot_product > 0)


def vertex_visibility_mask(verts, face_indices):
    '''0 MEAN VISIBLE!!! This is compatible with SMPLSolver'''
    faces = np.empty((face_indices.shape[0], 3, 3))
    for fidx, vidx in enumerate(face_indices):
        v0, v1, v2 = vidx
        faces[fidx, 0] = verts[v0]
        faces[fidx, 1] = verts[v1]
        faces[fidx, 2] = verts[v2]
    fmask = face_visibility_mask(faces)
    vmask = np.ones((1, verts.shape[0]))
    for fidx, visible in enumerate(fmask):
        if visible:
            v0, v1, v2 = face_indices[fidx]
            vmask[0, v0] = 0
            vmask[0, v1] = 0
            vmask[0, v2] = 0
    return vmask


def euler2R(theta):
    cos = np.cos(theta)
    sin = np.sin(theta)
    cosu, cosv, cosw = cos[:,0], cos[:,1], cos[:,2]
    sinu, sinv, sinw = sin[:,0], sin[:,1], sin[:,2]
    zero = np.zeros(theta.shape[0], dtype=theta.dtype)
    one = np.ones(theta.shape[0], dtype=theta.dtype)
    cube_A = np.reshape(np.stack([one, zero, zero, zero, cosu, -sinu, zero, sinu, cosu], axis=1), [-1, 3, 3])
    cube_B = np.reshape(np.stack([cosv, zero, sinv, zero, one, zero, -sinv, zero, cosv], axis=1), [-1, 3, 3])
    cube_G = np.reshape(np.stack([cosw, -sinw, zero, sinw, cosw, zero, zero, zero, one], axis=1), [-1, 3, 3])
    R = np.matmul(cube_G, np.matmul(cube_B, cube_A))
    return R
