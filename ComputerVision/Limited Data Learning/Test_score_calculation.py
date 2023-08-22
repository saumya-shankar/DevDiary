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
model1=tensorflow.keras.models.load_model('cosine_SGD_AV_wo_aug.h5')
#model2=tensorflow.keras.models.load_model('SGD_Siamese_contrastive_embeddings.h5')
#model3=tensorflow.keras.models.load_model('SGD_siamese_contrastive_embeddings_transferLearning2.h5')
"""
for layer in model1.layers:
  print(layer.name,end=",")

last_layer = tf.keras.models.Model(inputs=model1.input, outputs=model1.get_layer('avg_pool').output)  
last_layer.summary()
 """ 
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
"""test_image = image.load_img('./test_resized/ILSVRC2012_val_00000002.JPEG')
test_image = image.img_to_array(test_image)
test_image=test_image/255
test_image=center_crop_normalize(test_image, (224,224), mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
test_image = np.expand_dims(test_image, axis = 0)
result = model3.predict(model2.predict(last_layer.predict(test_image)))
print(np.argmax(result))
"""
import tqdm
import os
results=[]
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
    results=results+[model1.predict(test_image)]
    #results = results+[model3.predict(model2.predict(last_layer.predict(test_image)))]
    
top1 = 0.0
top5 = 0.0    
for i, l in enumerate(lis):
    result = results[i]
    top_values = list((-result).argsort()[0])[:5]
    if top_values[0] == l:
        top1 += 1.0
    if np.isin(np.array([l]), top_values):
        top5 += 1.0

print("top1 acc", top1/len(lis))
print("top5 acc", top5/len(lis))

top1 = 0.0
top5 = 0.0    
for i, l in enumerate(lis):
    result = results[i]
    top_values = list((-result).argsort()[0])[:5]
    if top_values[0] == l:
        top1 += 1.0
    if np.isin(np.array([l]), top_values):
        top5 += 1.0

print("top1 acc", top1/len(lis))
print("top5 acc", top5/len(lis))







