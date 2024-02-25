# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 02:08:06 2022

@author: vijmr
"""

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.utils import shuffle
from scipy import signal

data_dir="./Dataset/animated_motionv7/"

padding_dim=20
padding_val=1e6

##Create mapping dataset from adjacencies matrix
adj_mat=[]

for i in range(1000):
    df=np.array(pd.read_csv(data_dir+"adj_mat"+str(i)+".csv").iloc[:,1:])
    df=np.pad(df, [(0,padding_dim-df.shape[0]),(0,padding_dim-df.shape[0])], 'constant', constant_values=padding_val)    
    adj_mat.append(df)    
    
adj_mat=np.stack(adj_mat)
    
##Creating a time series data matrix
def calc_freq_signal(arr):
    freqs, psd = signal.welch(arr, fs=10)
    return psd

def custom_scaler(X, range=(0, 1),min=None,max=None):
    mi, ma = range
    if(min==None):
      min=X.min()
      max=X.max()
      print("\nTrain_Scaling:- min=",min," max=",max)
    X_std = (X - min) / (max - min)
    X_scaled = X_std * (ma - mi) + mi
    return min,max,X_scaled

ts_data=[]
ts_data_f=[]

min_x=np.inf
min_y=np.inf
min_x1d=np.inf
min_y1d=np.inf
min_xf=np.inf
min_yf=np.inf

max_x=-np.inf
max_y=-np.inf
max_x1d=-np.inf
max_y1d=-np.inf
max_xf=-np.inf
max_yf=-np.inf

for i in range(1000):
    df=pd.read_csv(data_dir+"ts_data"+str(i)+".csv").iloc[:,1:]
    x_coord=df[df.columns[0::2]].T
    y_coord=df[df.columns[1::2]].T
    temp=np.dstack([x_coord,y_coord])
    temp_f=np.apply_along_axis(calc_freq_signal, 1, temp)
    temp_1d=np.gradient(temp,axis=1)
    temp=np.dstack([temp,temp_1d])
    ts_data.append(temp)
    ts_data_f.append(temp_f)
    
ts_data=np.array(ts_data)
ts_data_f=np.array(ts_data_f)

"""
## For different topologies in train and test; Uncomment these for unseen topologies
train_indices=np.append(np.arange(250),np.arange(375,1000))
test_indices=np.arange(250,375)
"""

#For same topologeies at both train and test time
train_indices=np.hstack([125*0+np.arange(15,125),
                         125*1+np.arange(15,125),
                         125*2+np.arange(15,125),
                         125*3+np.arange(15,125),
                         125*4+np.arange(15,125),
                         125*5+np.arange(15,125),
                         125*6+np.arange(15,125),
                         125*7+np.arange(15,125)])

test_indices=np.hstack([125*0+np.arange(15),
                         125*1+np.arange(15),
                         125*2+np.arange(15),
                         125*3+np.arange(15),
                         125*4+np.arange(15),
                         125*5+np.arange(15),
                         125*6+np.arange(15),
                         125*7+np.arange(15)])

ts_data_train=ts_data[train_indices]
ts_data_test=ts_data[test_indices]

ts_data_f_train=ts_data_f[train_indices]
ts_data_f_test=ts_data_f[test_indices]

adj_mat_train=adj_mat[train_indices,:,:]
adj_mat_test=adj_mat[test_indices,:,:]

for i in range(len(ts_data_train)):
    temp=ts_data_train[i]
    temp_f=ts_data_f_train[i]
    
    min_x=min(min_x,temp[:,:,0].min())
    min_y=min(min_y,temp[:,:,1].min())
    min_x1d=min(min_x1d,temp[:,:,2].min())
    min_y1d=min(min_y1d,temp[:,:,3].min())
    min_xf=min(min_xf,temp_f[:,:,0].min())
    min_yf=min(min_yf,temp_f[:,:,1].min())
    
    max_x=max(max_x,temp[:,:,0].max())
    max_y=max(max_y,temp[:,:,1].max())
    max_x1d=max(max_x1d,temp[:,:,2].max())
    max_y1d=max(max_y1d,temp[:,:,3].max())
    max_xf=max(max_xf,temp_f[:,:,0].max())
    max_yf=max(max_yf,temp_f[:,:,1].max())
    
for i in range(len(ts_data_train)):
    _,_,ts_data_train[i][:,:,0]=custom_scaler(ts_data_train[i][:,:,0],range=(0,1),min=min_x,max=max_x)
    _,_,ts_data_train[i][:,:,1]=custom_scaler(ts_data_train[i][:,:,1],range=(0,1),min=min_y,max=max_y)
    _,_,ts_data_train[i][:,:,2]=custom_scaler(ts_data_train[i][:,:,2],range=(0,1),min=min_x1d,max=max_x1d)
    _,_,ts_data_train[i][:,:,3]=custom_scaler(ts_data_train[i][:,:,3],range=(0,1),min=min_y1d,max=max_y1d)
    _,_,ts_data_f_train[i][:,:,0]=custom_scaler(ts_data_f_train[i][:,:,0],range=(0,1),min=min_xf,max=max_xf)
    _,_,ts_data_f_train[i][:,:,1]=custom_scaler(ts_data_f_train[i][:,:,1],range=(0,1),min=min_yf,max=max_yf)
    
    ts_data_train[i]=np.pad(ts_data_train[i], [(0,padding_dim-ts_data_train[i].shape[0]),(0,0),(0,0)], 'constant', constant_values=padding_val)
    ts_data_f_train[i]=np.pad(ts_data_f_train[i], [(0,padding_dim-ts_data_f_train[i].shape[0]),(0,0),(0,0)], 'constant', constant_values=padding_val)

ts_data_train=np.stack(ts_data_train)
ts_data_f_train=np.stack(ts_data_f_train)

for i in range(len(ts_data_test)):
    _,_,ts_data_test[i][:,:,0]=custom_scaler(ts_data_test[i][:,:,0],range=(0,1),min=min_x,max=max_x)
    _,_,ts_data_test[i][:,:,1]=custom_scaler(ts_data_test[i][:,:,1],range=(0,1),min=min_y,max=max_y)
    _,_,ts_data_test[i][:,:,2]=custom_scaler(ts_data_test[i][:,:,2],range=(0,1),min=min_x1d,max=max_x1d)
    _,_,ts_data_test[i][:,:,3]=custom_scaler(ts_data_test[i][:,:,3],range=(0,1),min=min_y1d,max=max_y1d)
    _,_,ts_data_f_test[i][:,:,0]=custom_scaler(ts_data_f_test[i][:,:,0],range=(0,1),min=min_xf,max=max_xf)
    _,_,ts_data_f_test[i][:,:,1]=custom_scaler(ts_data_f_test[i][:,:,1],range=(0,1),min=min_yf,max=max_yf)
    
    ts_data_test[i]=np.pad(ts_data_test[i], [(0,padding_dim-ts_data_test[i].shape[0]),(0,0),(0,0)], 'constant', constant_values=padding_val)
    ts_data_f_test[i]=np.pad(ts_data_f_test[i], [(0,padding_dim-ts_data_f_test[i].shape[0]),(0,0),(0,0)], 'constant', constant_values=padding_val)

ts_data_test=np.stack(ts_data_test)
ts_data_f_test=np.stack(ts_data_f_test)


#-------------------------------------------------------------
#For different topologies in train and test
## For different topologies in train and test; Uncomment these for unseen topologies
"""np.save("GraphWise_X_trainv7_exp2.npy",ts_data_train)
np.save("GraphWise_X_testv7_exp2.npy",ts_data_test)
np.save("GraphWise_X_f_trainv7_exp2.npy",ts_data_f_train)
np.save("GraphWise_X_f_testv7_exp2.npy",ts_data_f_test)
np.save("GraphWise_y_trainv7_exp2.npy",adj_mat_train)
np.save("GraphWise_y_testv7_exp2.npy",adj_mat_test)
"""

#For similar topologies in train and test
np.save("GraphWise_X_trainv7_exp1_strat.npy",ts_data_train)
np.save("GraphWise_X_testv7_exp1_strat.npy",ts_data_test)
np.save("GraphWise_X_f_trainv7_exp1_strat.npy",ts_data_f_train)
np.save("GraphWise_X_f_testv7_exp1_strat.npy",ts_data_f_test)
np.save("GraphWise_y_trainv7_exp1_strat.npy",adj_mat_train)
np.save("GraphWise_y_testv7_exp1_strat.npy",adj_mat_test)

