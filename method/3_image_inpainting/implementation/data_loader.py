import os
import glob
import numpy as np
from skimage.io import imread
import random
import tensorflow as tf
import re
import scipy.misc
from skimage.transform import resize
import matplotlib.pyplot as plt
import cv2



class DataLoader:

    def __init__(self, train_images_dir, val_images_dir, test_images_dir, train_batch_size, val_batch_size, 
            test_batch_size, height_of_image, width_of_image, num_channels):

        self.train_paths = glob.glob(os.path.join(train_images_dir, "**/*.png"), recursive=True)
        self.val_paths = glob.glob(os.path.join(val_images_dir, "**/*.png"), recursive=True)
        self.test_paths = glob.glob(os.path.join(test_images_dir, "**/*.png"), recursive=True)

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        # self.num_classes=num_classes
        self.height_of_image=height_of_image
        self.width_of_image=width_of_image
        self.num_channels=num_channels
    def load_mask(self):
        mask=np.ones((1,self.height_of_image, self.width_of_image, 1)) 
        return mask


    def load_image(self, path):
        image=imread(path)
        img=resize(image,[self.height_of_image, self.width_of_image, self.num_channels])
        return img
    
    def load_image_mask(self, path):
        image=imread(path)
        w=int(image.shape[1]/2)
        h=int(image.shape[0])
        img=image[0:h,0:w,:]
        mask=image[0:h,w:,:]
        img=resize(img,[self.height_of_image, self.width_of_image, self.num_channels])
        mask=cv2.resize(mask[:,:,0],(self.height_of_image, self.width_of_image))
        mask=mask.reshape(self.height_of_image, self.width_of_image,1)
        return img , mask 
             

    def batch_data_loader(self, batch_size, file_paths, index):
        images=[]
        
        for i in range(int(index*batch_size),int((index+1)*batch_size)):
            image=self.load_image(file_paths[i])
            images.append(image)
           
        return images

    ### this function can be used if mask is also generated
    def batch_data_loader_test(self, batch_size, file_paths, index):
        images=[]
        masks=[]
        
        for i in range(int(index*batch_size),int((index+1)*batch_size)):
            image,mask=self.load_image_mask(file_paths[i])
            images.append(image)
            masks.append(mask)

        return images , masks




    def train_data_loader(self, index):
        return self.batch_data_loader(self.train_batch_size, self.train_paths, index)

    def val_data_loader(self, index):
        return self.batch_data_loader(self.val_batch_size, self.val_paths, index)

    def test_data_loader(self, index):
        return self.batch_data_loader(self.test_batch_size, self.test_paths, index)




