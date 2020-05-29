from .BaseNN import *
import math
import numpy as np
import functools
import scipy.io
import pdb
class DNN(BaseNN):
   
   
   def encoder(self,X,mask):
    ### Encoder
    ########## layer -------1
    conv1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=(7, 7),strides=(2,2), padding='same', activation=tf.nn.relu)
    update_mask1 = tf.layers.conv2d(mask, filters=1,
                                               kernel_size=(7, 7), kernel_initializer=tf.constant_initializer(1.0),strides=(2,2),
                                                padding='same', use_bias=False, trainable=False, activation=tf.nn.relu)
    mask_ratio1 = 9 / (update_mask1 )
    um1=update_mask1/update_mask1
    mask_ratio1 = mask_ratio1 * um1
    conv1 = conv1 * mask_ratio1

    ########## layer -------2
    conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=(5, 5),strides=(2,2), padding='same', activation=tf.nn.relu)
    update_mask2 = tf.layers.conv2d(um1, filters=1,
                                               kernel_size=(5, 5),strides=(2,2), kernel_initializer=tf.constant_initializer(1.0),
                                                padding='same', use_bias=False, trainable=False, activation=tf.nn.relu)
    mask_ratio2 = 9 / (update_mask2 )
    um2=update_mask2/update_mask2
    mask_ratio2 = mask_ratio2 * um2
    conv2 = conv2 * mask_ratio2
    ########## layer -------3
    conv3 = tf.layers.conv2d(inputs=conv2, filters=128, kernel_size=(5,5),strides=(2,2), padding='same', activation=tf.nn.relu)
    update_mask3 = tf.layers.conv2d(um2, filters=1,
                                               kernel_size=(5, 5),strides=(2,2), kernel_initializer=tf.constant_initializer(1.0),
                                                padding='same', use_bias=False, trainable=False, activation=tf.nn.relu)
    mask_ratio3 = 9 / (update_mask3 )
    um3=update_mask3/update_mask3
    mask_ratio3 = mask_ratio3 * um3
    
    conv3 = conv3 * mask_ratio3
   
    ########## layer -------4
  
    conv4 = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=(3,3),strides=(2,2), padding='same', activation=tf.nn.relu)
    update_mask4 = tf.layers.conv2d(um3, filters=1,
                                               kernel_size=(3, 3),strides=(2,2), kernel_initializer=tf.constant_initializer(1.0),
                                                padding='same', use_bias=False, trainable=False, activation=tf.nn.relu)
    mask_ratio4 = 9 / (update_mask4 )
    um4=update_mask4/update_mask4
    mask_ratio4 = mask_ratio4 * um4
    
    conv4 = conv4 * mask_ratio4

    encoded = conv4
    mask_encoded = um4

    return encoded,mask_encoded

# Building the decoder
   def decoder(self,X):
    ### Decoder
    upsample1 = tf.image.resize_images(X, size=(self.height_of_image//8, self.width_of_image//8), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    conv4 = tf.layers.conv2d(inputs=upsample1, filters=128, kernel_size=(3,3), padding='same', activation=tf.nn.relu)

    upsample2 = tf.image.resize_images(conv4, size=(self.height_of_image//4, self.width_of_image//4), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    conv5 = tf.layers.conv2d(inputs=upsample2, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)

    upsample3 = tf.image.resize_images(conv5, size=(self.height_of_image//2, self.width_of_image//2), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    conv6 = tf.layers.conv2d(inputs=upsample3, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)

    upsample4 = tf.image.resize_images(conv6, size=(self.height_of_image, self.width_of_image), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    logits = tf.layers.conv2d(inputs=upsample4, filters=self.num_channels, kernel_size=(3,3), padding='same', activation=None)

    # Pass logits through sigmoid to get reconstructed image
    decoded = tf.nn.sigmoid(logits)

    return logits, decoded  

       
   def metrics(self, Y, Y_pred):     
       loss = tf.losses.mean_squared_error(Y,Y_pred)
       return loss

   def get_boundary(self,mask):
    kernel=np.ones((3,3), np.uint8)
    inner_part = cv2.erode(mask,kernel, iterations=1)
    return mask - inner_part

   def get_where_indices(sels,arr):
    arr = np.where(arr)
    return np.array([[arr[0][i], arr[1][i]] for i in range(len(arr[0]))])

   def ssd(self,arr1, arr2):
    return np.sum((arr1 - arr2)**2)
   
   def fill_the_boundary(self,img, mask_to_fill, kernel_radius = 10, eps = 1e-3):
    
    kernel_size = 2*kernel_radius + 1
    crop_mask = cv2.dilate(mask_to_fill, np.ones((kernel_radius*4,4*kernel_radius)))
    
    kernel1=np.ones((kernel_size*2, kernel_size*2))
    
    boundary = self.get_boundary(mask_to_fill)
    dilate=cv2.dilate(crop_mask,kernel1, iterations=1)
    mask_to_get_valid=dilate-crop_mask
    
    masked_img = img

    M_pad = np.zeros((img.shape[0]+kernel_radius*2, img.shape[1]+kernel_radius*2,img.shape[2]))
    M_pad[kernel_radius:-kernel_radius,kernel_radius:-kernel_radius,:] = masked_img

    M_pad_mask = np.zeros((img.shape[0]+kernel_radius*2, img.shape[1]+kernel_radius*2,img.shape[2]))
    M_pad_mask[kernel_radius:-kernel_radius,kernel_radius:-kernel_radius,:] = mask_to_fill

    count_tensor = convolve2d(1.0 - mask_to_fill[:,:,0], np.ones((kernel_size, kernel_size)), mode = 'same')
    
    where_to_fill = self.get_where_indices(boundary[:,:,0])
    where_to_get = self.get_where_indices(mask_to_get_valid[:,:,0])
    
    for ind_fill in tqdm(where_to_fill):
        i, j = ind_fill
        patch_to_replace = M_pad[i:i+kernel_size, j:j+kernel_size,:]
        patch_to_replace_mask = M_pad_mask[i:i+kernel_size, j:j+kernel_size,:]
        patch_mask = M_pad_mask
        scores = []
    
        for idx_get in range(len(where_to_get)):
            i_get, j_get = where_to_get[idx_get]
            patch_to_get = M_pad[i_get:i_get+kernel_size, j_get:j_get+kernel_size,:]
            patch_to_get_mask = patch_mask[i_get:i_get+kernel_size, j_get:j_get+kernel_size,:]
            score = self.ssd(patch_to_get[patch_to_replace_mask[:,:,0]==0] ,patch_to_replace[patch_to_replace_mask[:,:,0]==0])/count_tensor[i, j]
            scores.append(score) 
        b=np.array(scores)
        i_match, j_match = where_to_get[int(np.argwhere(scores==random.choice(b[np.argwhere(b<=(np.min(b)+eps))]))[0])]
               
        masked_img[i,j,:] = masked_img[i_match, j_match,:]
        mask_to_fill[i,j,:] = 0

    
    return masked_img, mask_to_fill


   def patch_match(self,img,mask,kernel_radius,eps):   
    mask_to_fill=1-mask

    img_with_hole = img.reshape(img.shape[1:])
    mask_to_fill=mask_to_fill.reshape(mask_to_fill.shape[1:3])
    res_mask_to_fill1 = np.stack([mask_to_fill]*img.shape[3], axis = -1)
    
    while res_mask_to_fill1[:,:,0].sum() != 0.:
        print(res_mask_to_fill1[:,:,0].sum())
        img_with_hole, res_mask_to_fill1 = self.fill_the_boundary(img_with_hole, res_mask_to_fill1, kernel_radius = kernel_radius, eps = eps)
        print(img_with_hole.shape)
        h,w,c=img_with_hole.shape
        
    img_with_hole=img_with_hole.reshape(img.shape[0], img.shape[1], img.shape[2],img.shape[3])

    return img_with_hole



    



