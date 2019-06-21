#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 20:36:52 2019

@author: atul
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 17:24:49 2019
@author: curaj
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
df=pd.read_csv('/home/atul/Documents/BDA 2nd sem/FDS/LAB TASKS/Lab2/letter-recognition.data',header=None)
df1=df.iloc[:,1:]
error_vec=[]
train,test = train_test_split(df1, test_size=0.2)
for i in range(len(train.columns)):
    train.iloc[:,i]=(train.iloc[:,i]-train.iloc[:,i].mean())/train.iloc[:,i].std()
S=np.cov(train.transpose())
eig=np.linalg.eig(S)
eig_val=eig[0]
eig_vec=eig[1]
idx=eig_val.argsort()[-1::-1] #index vector of eigen values in descending order
eig_val_desc=eig_val[idx]
eig_vec_desc=eig_vec[idx]
d=len(eig_vec_desc)+1
for i in range(1,d):
    G=eig_vec_desc[:,0:i]
    ggx=(np.dot(G,np.transpose(G)))
    xbar=np.dot(ggx,np.transpose(train))
    x=np.transpose(train)
    error_vec.append(np.square(np.linalg.norm(x-xbar)))
plt.plot(list(range(1,17)),error_vec)