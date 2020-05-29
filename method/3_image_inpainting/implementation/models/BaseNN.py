import tensorflow as tf
from data_loader import *
from abc import abstractmethod
import math
import numpy as np
import random
from numpy import array
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
import glob
import os
from scipy.signal import convolve2d
from tqdm.autonotebook import tqdm
import cv2
import pandas as pd
from random import randint, seed
from scipy.spatial import distance
from skimage.transform import resize
from time import time

class BaseNN:
    def __init__(self, train_images_dir, val_images_dir, test_images_dir, num_epochs, train_batch_size,
                 val_batch_size, test_batch_size, height_of_image, width_of_image, num_channels, 
                  learning_rate, base_dir, max_to_keep, model_name):

        self.data_loader = DataLoader(train_images_dir, val_images_dir, test_images_dir, train_batch_size, 
                val_batch_size, test_batch_size, height_of_image, width_of_image, num_channels)
        self.num_epochs=num_epochs
        self.train_paths = glob.glob(os.path.join(train_images_dir, "**/*.png"), recursive=True)
        self.val_paths = glob.glob(os.path.join(val_images_dir, "**/*.png"), recursive=True)
        self.test_paths = glob.glob(os.path.join(test_images_dir, "**/*.png"), recursive=True)
        self.base_dir=base_dir

        self.train_batch_size=train_batch_size
        self. val_batch_size= val_batch_size
        self. test_batch_size= test_batch_size

        self.learning_rate=learning_rate
        self.width_of_image=width_of_image
        self.height_of_image=height_of_image
        self.max_to_keep=max_to_keep
        self.model_name=model_name
        self.num_channels=num_channels


    def create_network(self,model_type):
        if model_type == "train":

            self.x = tf.placeholder("float",[None, self.height_of_image, self.width_of_image, self.num_channels] ,name="x")
            self.mask = tf.placeholder("float",[None, self.height_of_image, self.width_of_image, 1] ,name="mask")

            encoder_op,mask_op = self.encoder(self.x,self.mask)
            logit,decoder = self.decoder(encoder_op)
            # Prediction
            self.prediction = decoder

            self.loss=self.metrics(self.x,self.prediction)

            self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        
        else:
            self.x = tf.placeholder("float",[None, self.height_of_image, self.width_of_image, self.num_channels] ,name="x")
            self.mask = tf.placeholder("float",[None, self.height_of_image, self.width_of_image, 1] ,name="mask")
            self.encoder_new=tf.placeholder("float",[None, self.height_of_image//16, self.width_of_image//16, 256] ,name="mask")

            self.encoder_op,self.mask_op = self.encoder(self.x,self.mask)
            
            logit,self.decoder_= self.decoder(self.encoder_new)
            # # Prediction
            self.prediction = self.decoder_
            self.loss = self.metrics(self.x,self.prediction)
           
        
    def load(self):
        print(" [*] Reading checkpoint...")
        self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)
        checkpoint_dir = os.path.join(os.getcwd(), self.base_dir, self.model_name, 'chekpoints')
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(ckpt_name)
            return True
        else:
            return False

    def initialize_network(self):
        self.sess= tf.InteractiveSession()
        if os.path.exists(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'chekpoints'))== False:
            self.sess.run(tf.global_variables_initializer())
           
        else:
            self.load()
           
        

    def train_model(self, display_step, validation_step, checkpoint_step, summary_step):
        num_complete_minibatches_train = len(self.train_paths)// self.train_batch_size
        num_complete_minibatches_val = len(self.val_paths)// self.val_batch_size
        
        mask_train=self.data_loader.load_mask()
        mini_batches=[]
        for k in range(num_complete_minibatches_train):
                mini_batches.append(self.data_loader.train_data_loader(k))


        for epoch in range(self.num_epochs):
            random.shuffle(self.data_loader.train_paths)   
            
            epoch_cost = 0
            epoch_acc = 0
            
            for minibatch in mini_batches:
                minibatch_x = minibatch
                minibatch_optimizer, minibatch_cost,minibatch_prediction_image = self.sess.run([self.optim, self.loss,self.prediction], feed_dict = {self.x: minibatch_x, self.mask: mask_train})
                epoch_cost += minibatch_cost     

            epoch_cost = epoch_cost/num_complete_minibatches_train
           
            
            if epoch%display_step ==0:
                print("cost after epoch %i :  %.5f" % (epoch + 1, epoch_cost))
                 

            if epoch%validation_step ==0:
                random.shuffle(self.data_loader.val_paths)
                k=random.randint(1,num_complete_minibatches_val)
                k=k-1
                x_val=self.data_loader.val_data_loader(k)
                val_loss = self.sess.run([self.loss], feed_dict = {self.x: x_val, self.mask: mask_train})

                print("  val cost   :  %.3f" % (val_loss[0]))
               
                

            if epoch%checkpoint_step ==0:
                self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)
                if os.path.isfile(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'chekpoints'))== False:
                    
                    os.makedirs(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'chekpoints'),exist_ok=True)
                    self.saver.save(self.sess, os.path.join(os.getcwd(),self.base_dir, self.model_name, 'chekpoints','my-model'))
                else:
                    self.saver.save(self.sess, os.path.join(os.getcwd(), self.base_dir, self.model_name, 'chekpoints','my-model'))


            if epoch%summary_step ==0:
               
                if os.path.isfile(os.path.join(os.getcwd(),self.base_dir,self.model_name, 'summaries'))== False:
                    os.makedirs(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'summaries'),exist_ok=True)
                    tf.summary.FileWriter(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'summaries'), self.sess.graph)
                    tf.summary.scalar("cost", minibatch_cost)
                    merged_summary_op = tf.summary.merge_all()
                else:
                    tf.summary.FileWriter(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'summaries'), self.sess.graph)
                    tf.summary.scalar("cost", minibatch_cost)
                    merged_summary_op = tf.summary.merge_all()
          
        print("network trained")
        

    def test_model(self):
        def concat_images(imga, imgb):
            ha,wa = imga.shape[:2]
            hb,wb = imgb.shape[:2]
            max_height = np.max([ha, hb])
            total_width = wa+wb
            new_img = np.zeros(shape=(max_height, total_width, 3))
            new_img[:ha,:wa]=imga
            new_img[:hb,wa:wa+wb]=imgb
            return new_img


        for i in range(len(self.test_paths) // self.test_batch_size):
            path=self.base_dir

            x_test1=self.data_loader.test_data_loader(i)
            ###### generate mask
            mask_test = np.ones((self.height_of_image, self.width_of_image))
            mask_test[150:200,150:200]=0
            # print(np.unique(mask_op_test,return_counts=True))
            mask_test=mask_test.reshape(1,self.height_of_image, self.width_of_image,1)
            ###### generate image with hole
            x_test=x_test1*mask_test
            # plt.imshow(x_test[0])
            # plt.show()
            start = time()
            encoder_test,mask_op_test= self.sess.run([self.encoder_op,self.mask_op], feed_dict={self.x: x_test,self.mask: mask_test})
            where_are_NaNs = np.isnan(mask_op_test)
            mask_op_test[where_are_NaNs] = 0
            pm=self.patch_match(encoder_test,mask_op_test,1,0.00009)
            image= self.sess.run(self.prediction, feed_dict={self.encoder_new: pm})
            # plt.imshow(image[0])
            # plt.show()
            new_image=concat_images(x_test[0],image[0])

            imsave(os.path.join(path+"/or_and_rec"+str(int(i))+".png"),new_image)
            # imsave(os.path.join(path+"/rec"+str(int(i))+".png"),image[0])
            imsave(os.path.join(path+"/or"+str(int(i))+".png"),x_test1[0])
            print(time()-start)

       




    @abstractmethod
    # def network(self, X):
    #     raise NotImplementedError('subclasses must override network()!')
    def encoder(self, X, mask):
        raise NotImplementedError('subclasses must override network()!')
    def decoder(self, X):
        raise NotImplementedError('subclasses must override network()!')
    def patch_match(self,img,mask,kernel_radius,eps):
        raise NotImplementedError('subclasses must override network()!')
   
    @abstractmethod
    def metrics(self, Y, y_pred):
        raise NotImplementedError('subclasses must override metrics()!')
    
   
   
        


