from copy import deepcopy
from math import *
import pandas as pd
import numpy as np


# uniform the probability in each bucket (to smooth the survival probability)(this function is not used now)
def changeBucketUniform(x,step):
    if step==1:
        return deepcopy(x)
    b = int(ceil(float(len(x))/step))
    y = [0.]*len(x)
    for i in range(0,b):
        bsum = 0.
        for j in range(i*step,(i+1)*step):
            if j>=len(x):
                break
            bsum += x[j]
        for j in range(i*step,(i+1)*step):
            if j>=len(x):
                break
            y[j] = bsum/step
    return y

def KLDivergence(_P, _Q, step = 1):
    if len(_P)!=len(_Q):
        print('kld:',len(_P),len(_Q))
        return 0
    P = changeBucketUniform(_P, step)
    Q = changeBucketUniform(_Q, step)
    KLD = 0.0
    for i in range(0, len(P)):
        if Q[i] == 0:
            print(i)
            continue
        KLD += P[i]*log(P[i]/Q[i])

    return KLD

def KMeans(s:pd.DataFrame, feature_idx):
    #random split s into s1 and s2, and calculate minPrice,maxPrice
    if len(s) == 0:
        return 0
    
    s1 = s.iloc[:len(s) // 2, :]
    s1 = s.iloc[len(s) // 2 :, :]



if __name__ == "__main__":
    pass