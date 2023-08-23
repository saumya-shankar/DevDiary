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
import copy

data_dir="./Dataset/animated_motionv7/"

padding_dim=30 ##Changed padding dim to a higher number
padding_val=1e6

##Create different video id combination that will be used for Multi Object Data Training
#875 Training instances, 125 Training instances

## For different topologies in train and test
train_indices=np.append(np.arange(250),np.arange(375,1000))

test_indices=np.arange(250,375)

train_df1=pd.DataFrame({'v1':train_indices})
train_df2=pd.DataFrame({'v2':train_indices})
train_df=pd.merge(train_df1, train_df2,how='cross')
train_df['set']=np.apply_along_axis(lambda x:set(x),1,train_df)
train_df['len_set']=np.apply_along_axis(lambda x:len(x[-1]),1,train_df)
train_df['set']=train_df['set'].astype(str)
train_df['status']=True
train_df.loc[train_df['len_set']==1,'status']=False
train_df=train_df.loc[train_df['status']==True]
train_df=train_df.drop_duplicates('set').reset_index(drop=True)
train_df=train_df.sample(875).reset_index(drop=True)
train_df=train_df.iloc[:,:2]

test_df1=pd.DataFrame({'v1':test_indices})
test_df2=pd.DataFrame({'v2':test_indices})
test_df=pd.merge(test_df1, test_df2,how='cross')
test_df['set']=np.apply_along_axis(lambda x:set(x),1,test_df)
test_df['len_set']=np.apply_along_axis(lambda x:len(x[-1]),1,test_df)
test_df['set']=test_df['set'].astype(str)
test_df['status']=True
test_df.loc[test_df['len_set']==1,'status']=False
test_df=test_df.loc[test_df['status']==True]
test_df=test_df.drop_duplicates('set').reset_index(drop=True)
test_df=test_df.sample(125).reset_index(drop=True)
test_df=test_df.iloc[:,:2]

complete_df=pd.concat([train_df,test_df],axis=0).reset_index(drop=True)

##Load the adjacency matrix
adj_mat=[]
for i in range(complete_df.shape[0]):
    v1=complete_df.iloc[i,0]
    v2=complete_df.iloc[i,1]
    df1=np.array(pd.read_csv(data_dir+"adj_mat"+str(v1)+".csv").iloc[:,1:])
    df2=np.array(pd.read_csv(data_dir+"adj_mat"+str(v2)+".csv").iloc[:,1:])
    
    s1=df1.shape[0]
    df1=np.pad(df1, [(0,df2.shape[0]),(0,df2.shape[0])], 
               'constant', constant_values=0)
    
    df1[s1:s1+df2.shape[0],s1:s1+df2.shape[0]]=df2
    df1=np.pad(df1, [(0,padding_dim-df1.shape[0]),(0,padding_dim-df1.shape[0])], 
               'constant', constant_values=padding_val)

    adj_mat.append(df1)

    
adj_mat=np.stack(adj_mat)

## Load the time series data
    
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

for i in range(complete_df.shape[0]):
    v1=complete_df.iloc[i,0]
    v2=complete_df.iloc[i,1]
    df1_orig=pd.read_csv(data_dir+"ts_data"+str(v1)+".csv").iloc[:,1:]
    df2_orig=pd.read_csv(data_dir+"ts_data"+str(v2)+".csv").iloc[:,1:]
    
    df1_x_max=df1_orig.iloc[:,0::2].max().max()
    df2_x_min=df2_orig.iloc[:,0::2].min().min()
    
    df1_x_min=df1_orig.iloc[:,0::2].min().min()
    df2_x_max=df2_orig.iloc[:,0::2].max().max()
    
    df1_y_min=df1_orig.iloc[:,1::2].min().min()
    df2_y_max=df2_orig.iloc[:,1::2].max().max()
    
    df1=copy.copy(df1_orig)
    df2=copy.copy(df2_orig)

    switch=np.random.randint(0,4,1)
    if(switch==0):        
        df1.iloc[:,0::2]=df1.iloc[:,0::2]-df1_x_max
        df2.iloc[:,0::2]=df2.iloc[:,0::2]-df2_x_min
    if(switch==1):        
        df1.iloc[:,1::2]=df1.iloc[:,1::2]-df1_y_min
        df2.iloc[:,1::2]=df2.iloc[:,1::2]-df2_y_max
    if(switch==2):        
        df1.iloc[:,0::2]=df1.iloc[:,0::2]-df1_x_max
        df2.iloc[:,0::2]=df2.iloc[:,0::2]-df2_x_min
        df1.iloc[:,1::2]=df1.iloc[:,1::2]-df1_y_min
        df2.iloc[:,1::2]=df2.iloc[:,1::2]-df2_y_max
    if(switch==3):        
        df1.iloc[:,0::2]=df1.iloc[:,0::2]-df1_x_min
        df2.iloc[:,0::2]=df2.iloc[:,0::2]-df2_x_max
        df1.iloc[:,1::2]=df1.iloc[:,1::2]-df1_y_min
        df2.iloc[:,1::2]=df2.iloc[:,1::2]-df2_y_max
    
    df=pd.concat([df1,df2],axis=1)
    df=df.T.reset_index(drop=True).T
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

## For different topologies in train and test
train_indices=np.arange(875)
test_indices=np.arange(875,1000)

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

np.save("GraphWise_X_trainv7_multiobject_exp2_v2.npy",ts_data_train)
np.save("GraphWise_X_testv7_multiobject_exp2_v2.npy",ts_data_test)
np.save("GraphWise_X_f_trainv7_multiobject_exp2_v2.npy",ts_data_f_train)
np.save("GraphWise_X_f_testv7_multiobject_exp2_v2.npy",ts_data_f_test)
np.save("GraphWise_y_trainv7_multiobject_exp2_v2.npy",adj_mat_train)
np.save("GraphWise_y_testv7_multiobject_exp2_v2.npy",adj_mat_test)

