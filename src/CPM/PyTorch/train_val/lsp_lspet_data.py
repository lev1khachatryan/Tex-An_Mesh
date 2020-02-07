# -*-coding:UTF-8-*-
import os
import scipy.io
import numpy as np
import glob
import torch.utils.data as data
import scipy.misc
from PIL import Image
import cv2
import Mytransforms


def read_data_file(root_dir):
    """get train or val images
        return: image list: train or val images list
    """
    image_arr = np.array(glob.glob(os.path.join(root_dir, 'images/*.jpg')))
    image_nums_arr = np.array([float(s.rsplit('/')[-1][2:-4]) for s in image_arr])
    sorted_image_arr = image_arr[np.argsort(image_nums_arr)]
    return sorted_image_arr.tolist()

def read_mat_file(mode, root_dir, img_list):
    """
        get the groundtruth

        mode (str): 'lsp' or 'lspet'
        return: three list: key_points list , centers list and scales list

        Notice:
            lsp_dataset differ from lspet dataset
    """
    mat_arr = scipy.io.loadmat(os.path.join(root_dir, 'joints.mat'))['joints']
    # lspnet (14,3,10000)
    if mode == 'lspet':
        lms = mat_arr.transpose([2, 1, 0])
        kpts = mat_arr.transpose([2, 0, 1]).tolist()
    # lsp (3,14,2000)
    if mode == 'lsp':
        mat_arr[2] = np.logical_not(mat_arr[2])
        lms = mat_arr.transpose([2, 0, 1])
        kpts = mat_arr.transpose([2, 1, 0]).tolist()

    centers = []
    scales = []
    for idx in range(lms.shape[0]):
        im = Image.open(img_list[idx])
        w = im.size[0]
        h = im.size[1]
        # lsp and lspet dataset doesn't exist groundtruth of center points
        center_x = (lms[idx][0][lms[idx][0] < w].max() +
                    lms[idx][0][lms[idx][0] > 0].min()) / 2
        center_y = (lms[idx][1][lms[idx][1] < h].max() +
                    lms[idx][1][lms[idx][1] > 0].min()) / 2
        centers.append([center_x, center_y])

        scale = (lms[idx][1][lms[idx][1] < h].max() -
                lms[idx][1][lms[idx][1] > 0].min() + 4) / 368.0
        scales.append(scale)

    return kpts, centers, scales


def guassian_kernel(size_w, size_h, center_x, center_y, sigma):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)


class LSP_Data(data.Dataset):
    """
        Args:
            root_dir (str): the path of train_val dateset.
            stride (float): default = 8
            transformer (Mytransforms): expand dataset.
        Notice:
            you have to change code to fit your own dataset except LSP

    """

    def __init__(self, mode, root_dir, stride, transformer=None ):

        self.img_list = read_data_file(root_dir)
        self.kpt_list, self.center_list, self.scale_list = read_mat_file(mode, root_dir, self.img_list)
        self.stride = stride
        self.transformer = transformer
        self.sigma = 3.0


    def __getitem__(self, index):

        img_path = self.img_list[index]
        img = np.array(cv2.imread(img_path), dtype=np.float32)

        kpt = self.kpt_list[index]
        center = self.center_list[index]
        scale = self.scale_list[index]

        # expand dataset
        img, kpt, center = self.transformer(img, kpt, center, scale)
        height, width, _ = img.shape

        heatmap = np.zeros((height / self.stride, width / self.stride, len(kpt) + 1), dtype=np.float32)
        for i in range(len(kpt)):
            # resize from 368 to 46
            x = int(kpt[i][0]) * 1.0 / self.stride
            y = int(kpt[i][1]) * 1.0 / self.stride
            heat_map = guassian_kernel(size_h=height / self.stride, size_w=width / self.stride, center_x=x, center_y=y, sigma=self.sigma)
            heat_map[heat_map > 1] = 1
            heat_map[heat_map < 0.0099] = 0
            heatmap[:, :, i + 1] = heat_map

        heatmap[:, :, 0] = 1.0 - np.max(heatmap[:, :, 1:], axis=2)  # for background

        centermap = np.zeros((height, width, 1), dtype=np.float32)
        center_map = guassian_kernel(size_h=height, size_w=width, center_x=center[0], center_y=center[1], sigma=3)
        center_map[center_map > 1] = 1
        center_map[center_map < 0.0099] = 0
        centermap[:, :, 0] = center_map

        img = Mytransforms.normalize(Mytransforms.to_tensor(img), [128.0, 128.0, 128.0],
                                     [256.0, 256.0, 256.0])
        heatmap = Mytransforms.to_tensor(heatmap)
        centermap = Mytransforms.to_tensor(centermap)
        return img, heatmap, centermap

    def __len__(self):
        return len(self.img_list)


