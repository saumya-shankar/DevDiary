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

data_dir="./Dataset/animated_motion_dynamic_len/"

##Create mapping dataset from adjacencies matrix
filenames=os.listdir(data_dir)
filenames=[x for x in filenames if x.startswith("adj_mat")]

def create_metadata(filename):
    l=[]
    df=pd.read_csv(data_dir+filename).iloc[:,1:]
    for i in range(len(df)):
        for j in range(i+1,len(df)):
            l.append(np.array([filename.split(".")[0][7:],i,j,df.iloc[i,j]]))
    l=np.array(l)
    return(l)

adj_mat=np.empty((0,4))

for i in filenames:
    adj_mat=np.vstack([adj_mat,create_metadata(i)])    
    
adj_mat=pd.DataFrame(adj_mat,columns=['id','n1','n2','edge'])
adj_mat=adj_mat.astype(int)
#adj_mat.to_csv('adj_mat_dataframe_dynamic_len.csv',index=False)

###Creating final 3d numpy array (samples,timesteps,features)
x1=np.empty((0,50))
x2=np.empty((0,50))
y1=np.empty((0,50))
y2=np.empty((0,50))
y=np.empty((0,4))
for i in tqdm(range(len(adj_mat))):
    df=pd.read_csv(data_dir+'ts_data'+str(adj_mat.loc[i,'id'])+'.csv').iloc[:,1:]
    n1=adj_mat.loc[i,'n1']
    n2=adj_mat.loc[i,'n2']
    y=np.vstack([y,adj_mat.loc[i,['edge','id','n1','n2']][None,:]])
    x1=np.vstack([x1,df[str(2*n1)].values[None,:]])
    y1=np.vstack([y1,df[str(2*n1+1)].values[None,:]])
    x2=np.vstack([x2,df[str(2*n2)].values[None,:]])
    y2=np.vstack([y2,df[str(2*n2+1)].values[None,:]])    
    
X=np.dstack([x1,x2,y1,y2])

#seen topolgies in both train and test

train_indices=np.hstack([125*0+np.arange(0,125-15),
                         125*1+np.arange(0,125-15),
                         125*2+np.arange(0,125-15),
                         125*3+np.arange(0,125-15),
                         125*4+np.arange(0,125-15),
                         125*5+np.arange(0,125-15),
                         125*6+np.arange(0,125-15),
                         125*7+np.arange(0,125-15)])

test_indices=np.hstack([125*0+np.arange(125-15,125),
                         125*1+np.arange(125-15,125),
                         125*2+np.arange(125-15,125),
                         125*3+np.arange(125-15,125),
                         125*4+np.arange(125-15,125),
                         125*5+np.arange(125-15,125),
                         125*6+np.arange(125-15,125),
                         125*7+np.arange(125-15,125)])

X_train=X[np.isin(y[:,1],train_indices)]
y_train=y[np.isin(y[:,1],train_indices)]

X_test=X[np.isin(y[:,1],test_indices)]
y_test=y[np.isin(y[:,1],test_indices)]

np.save("all_pairs_ts_data_X_trainv7_exp1_exp5_strat_v2.npy",X_train)
np.save("all_pairs_ts_data_y_trainv7_exp1_exp5_strat_v2.npy",y_train)
np.save("all_pairs_ts_data_X_testv7_exp1_exp5_strat_v2.npy",X_test)
np.save("all_pairs_ts_data_y_testv7_exp1_exp5_strat_v2.npy",y_test)

#unseen topolgies in both train and test

#-------------------------------------------------------------
train_indices=np.append(np.arange(250),np.arange(375,1000))
test_indices=np.arange(250,375)

X_train=X[np.isin(y[:,1],train_indices)]
y_train=y[np.isin(y[:,1],train_indices)]

X_test=X[np.isin(y[:,1],test_indices)]
y_test=y[np.isin(y[:,1],test_indices)]

np.save("all_pairs_ts_data_X_trainv7_exp2_exp5.npy",X_train)
np.save("all_pairs_ts_data_y_trainv7_exp2_exp5.npy",y_train)
np.save("all_pairs_ts_data_X_testv7_exp2_exp5.npy",X_test)
np.save("all_pairs_ts_data_y_testv7_exp2_exp5.npy",y_test)

