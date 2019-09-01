# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 21:06:11 2019

@author: user
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

#import tensorflow as tf
import numpy as np
# tf.device('/gpu:1')
from pprint import pprint
#%matplotlib inline
#import matplotlib.pyplot as plt
from pyfinite import ffield
#from Field import * # Finite Field class
from pyldpc import (make_ldpc, 
                    binaryproduct, 
                    encode_random_message, 
                    decode,
                    get_message, #input：tG, x
                    encode,
                    utils,
                    coding_matrix_systematic,
                    coding_matrix)

def GFdiv(i,j):
    if i==0:
        return 0;
    else:
        if j==0:
            return 0;
        else:
            return 1;
        
#from Field import FField 
ffield_p = ffield.FField(1)
def GussElim(H):
    n = H.shape[1]   # define number of information symbols n 
    m = H.shape[0]   # define number of measruemennts symbols m
    J=np.array(range(n));
    Index=np.zeros([m])
    redun=0;
    tmp=np.zeros([n])
    for k in range(m):
        if H[k][J[k-redun]]==0:
            d=k;
            for i in range(k+1-redun,n):
                if H[k][J[i]]!=0:
                    d=i;
                    break;
            if(d==k):
                redun += 1;
                Index[k]=1;
                continue;
            else:
                med=J[k-redun];
                J[k-redun]=J[d];
                J[d]=med;
                
        if H[k][J[k-redun]]==0:
            print('H[{}][{}]==0'.format(k,J[k-redun]));
        else:
            for i in range(k+1,m):
                if H[i][J[k-redun]]!=0:
                    z=0;temp=0;
                    #z=field_p.Divide(H[i][J[k-redun]],H[k][J[k-redun]]);
                    z=GFdiv(H[i][J[k-redun]],H[k][J[k-redun]])
                    for j in range(k-redun,n):
                        temp=field_p.Multiply(H[k][J[j]],z);
                        H[i][J[j]]=field_p.Subtract(H[i][J[j]],temp);
        
    index=0;
    for i in range(m):
        if Index[i]==0:
            for j in range(n):
                tmp[j]=H[i][J[j]];
            for j in range(n):
                H[index][j]=tmp[j];
            index += 1;
                
    print(index)
    for k in range(index-1,0,-1):
        for i in range(k-1,-1,-1):
            if H[i,k]!=0:
                z=0;temp=0;
                z=GFdiv(H[i,k],H[k,k]);
                for j in range(k,n):
                    temp=field_p.Multiply(H[k,j],z);
                    H[i,j]=field_p.Subtract(H[i,j],temp);
        
    for i in range(m):
        for j in range(n-1,i,-1):
            H[i][j]=GFdiv(H[i][j],H[i][i]);
                
    return H;                   
        