# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 20:01:59 2020

@author: osoomin
"""

import matplotlib.image as img
import numpy as np

# x 만들기
x=np.zeros((125,900)) 
for i in range(1,126):
    data = img.imread('data{}.png'.format(i))
    x1 = 1-data[:,:,0]
    x2 = x1.reshape(900)
    x[i-1] = np.array(x2)

# t 만들기 
t = np.zeros([125],dtype=int)
for j in range(125):
    if j<25: t[j] = 19 # S
    elif 25<=j<50: t[j] = 15 # O
    elif 50<=j<75: t[j] = 13 # M
    elif 75<=j<100: t[j] = 9 # I
    elif 100<=j<125: t[j] = 14 # N    

np.savez('input',x=x,t=t)

