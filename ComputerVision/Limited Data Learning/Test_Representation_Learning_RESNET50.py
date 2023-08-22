# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 02:24:56 2022

@author: vijmr
"""

############################################################
####***************************************************#####
############################################################
import tensorflow
import tensorflow as tf
import tensorflow_addons as tfa
model1=tensorflow.keras.models.load_model('SGD_AV_wo_aug.h5')

for layer in model1.layers:
  print(layer.name,end=",")

last_layer = tf.keras.models.Model(inputs=model1.input, outputs=model1.get_layer('avg_pool').output)  
last_layer.summary()
  
## creating true labels
with open('./Imagenet50/ILSVRC2012_test_ground_truth.txt') as f:
    lis=[list(map(int,x.split()))[0] for x in f if x.strip()]   # if x.strip() to skip blank lines


#print(model.summary())

def normalize_image(image, mean, std):
    for channel in range(3):
        image[:,:,channel] = (image[:,:,channel] - mean[channel]) / std[channel]
    return image

def center_crop_normalize(img, crop_size,mean,std):
    height, width = img.shape[0], img.shape[1]
    dy, dx = crop_size
    x = (width - dx + 1) // 2
    y = (height - dy + 1) // 2
    return normalize_image(img[y:(y+dy), x:(x+dx), :],mean,std)

## testing over test data--single test image

import numpy as np
from keras.preprocessing import image
"""
test_image = image.load_img('./Imagenet50/test/ILSVRC2012_val_00000002.JPEG',target_size=(256,256))
test_image = image.img_to_array(test_image)
test_image=test_image/255
test_image=center_crop_normalize(test_image, (224,224), mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
test_image = np.expand_dims(test_image, axis = 0)
result = last_layer.predict(test_image)
result=np.hstack([result,np.array([lis[1]])[:,None]])
"""

import tqdm
import os
results=[]
baseline_results=[]
c=1
file_name_list=os.listdir('./test_resized/')
for i in tqdm.tqdm(range(len(file_name_list))):
    test_image = image.load_img('./test_resized/'+file_name_list[i])
    test_image = image.img_to_array(test_image)
    test_image=test_image/255
    #test_image=center_crop_normalize(test_image, (224,224), mean=[0.485, 0.456, 0.406],
    #                                     std=[0.229, 0.224, 0.225])
    test_image=normalize_image(test_image, mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    
    test_image = np.expand_dims(test_image, axis = 0)
    
    result = last_layer.predict(test_image)
    result=np.hstack([result,np.array([lis[1]])[:,None]])
    
    baseline_results=baseline_results+[result]
    
final=np.array(baseline_results).squeeze(axis=1)

final[:,2048]=lis

import pandas as pd
final_df=pd.DataFrame(final)
final_df.rename(columns={2048:'category_class'},inplace=True)
final_df.to_csv('test_representation_marcin.csv',index=False)







