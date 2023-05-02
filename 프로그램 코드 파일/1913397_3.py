# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 15:44:42 2020

@author: osoomin
"""

import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np

def p(u): #Sigmoid 함수
    return 1/(1+np.exp(-u))
def Dp(u):
    return np.exp(-u)/(1+np.exp(-u))**2
def E(y,t):
    return np.sum((y-t)**2)
def tvector(k):
    tt=np.zeros([10])
    tt[k]=1
    return tt

data = np.load('learning.npz')
w = data['w']
v = data['v']

# test = img.imread("test1.png")
# test = img.imread("test2.png")
# test = img.imread("test3.png")
# test = img.imread("test4.png")
test = img.imread("test5.png")

t1 = 1-test[:,:,0]
x= t1.reshape(900)
x1= np.hstack([[1],x])
xh= np.dot(w,x1)
u=p(xh)
u1 = np.hstack([[1],u])
uh = np.dot(v,u1)
y = p(uh)
xx= list(range(26))
plt.plot(xx,y,"*")
print(y)


