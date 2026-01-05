import torch
import numpy as np
import os

def SubscriptTrans0(m) :
    #H(m,n) and H(dot(m),dot(k));for
    #two to ten
    j = np.zeros(1,dtype=int)
    a = np.zeros(1,dtype=int)
    N = m.size
    for i in m :
        a = a + i*torch.pow(2,N-j-1)
        j = j + 1
    return a

def SubscriptTrans0_torch(m) :
    shape = m.size()[-1]
    a = torch.zeros(shape,dtype=torch.int32,device=m.device)
    for i in range(shape) : 
        a[i] = torch.pow(2,torch.tensor([shape-i-1]))
    return torch.sum(m*a,dim=-1)

def SubscriptTrans1(a,N) :
    #ten to two
    m = np.zeros(N,dtype=int)
    i = 0
    while a > 0 :
        m[N-i-1] = np.mod(a,2)
        # a = int(np.div(a,2,rounding_mode='floor'))
        a = int(np.floor_divide(a, 2))
        i += 1
    return m


def SubscriptTrans1_torch(a,N) :
    #ten to two
    m = torch.zeros(N,dtype=int)
    i = 0
    while a > 0 :
        m[N-i-1] = a%2
        a = int(torch.div(a,2,rounding_mode='floor'))
        i += 1
    return m