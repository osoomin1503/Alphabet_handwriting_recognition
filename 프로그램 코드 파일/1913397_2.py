# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 15:41:30 2020

@author: osoomin
"""

import matplotlib.image as img
import numpy as np

x=np.zeros((125,900)) 
for i in range(1,126):
    data = img.imread('data{}.png'.format(i))
    x1 = 1-data[:,:,0]
    x2 = x1.reshape(900)
    x[i-1] = np.array(x2)

t = np.zeros([125],dtype=int)
for j in range(125):
    if j<25: t[j] = 19 # S
    elif 25<=j<50: t[j] = 15 # O
    elif 50<=j<75: t[j] = 13 # M
    elif 75<=j<100: t[j] = 9 # I
    elif 100<=j<125: t[j] = 14 # N    

np.savez('input',x=x,t=t)

def p(u): 
    return 1/(1+np.exp(-u))
def Dp(u):
    return np.exp(-u)/(1+np.exp(-u))**2
def E(y,t):
    return np.sum((y-t)**2)
def tvector(k): 
    tt=np.zeros([26]) 
    tt[k]=1
    return tt

eta = 0.1
ndata = len(x) 

# 입력값:900개(30*30),은닉층:40개,출력층:26개

w = 0.1*(2*np.random.random([40,901])-1)
v = 0.1*(2*np.random.random([26,41])-1) 

# 초기값에 대해 y 계산
E1 = 0
for i in range(ndata):
    x1= np.hstack([[1],x[i]]) 
    xh= np.dot(w,x1)
    u=p(xh) 
    u1 = np.hstack([[1],u]) 
    uh = np.dot(v,u1)
    y = p(uh)
    tt= tvector(t[i])
    E1 += E(y,tt)
E1 /= ndata 
# 한 데이터에 대한 업데이트가 끝남

Tol = 0.000001 
Resid = 10*Tol
noiter = 0 

while Resid > Tol and noiter<50:
    for i in range(ndata):
        x1= np.hstack([[1],x[i]]) 
        xh= np.dot(w,x1)
        u=p(xh)
        u1 = np.hstack([[1],u])
        uh = np.dot(v,u1)
        y = p(uh)
        tt= tvector(t[i])
        dEv =((y-tt)*Dp(uh)).reshape(26,1) *u1
        dEw= (np.dot(v.T[1:],(y-tt)*
              Dp(uh))*Dp(xh)).reshape(40,1)*x1
        v -= eta*dEv
        w -= eta*dEw
    # 모든 데이터에 대해 업데이트가 끝남
    
    E2 = 0
    for i in range(ndata):
        x1= np.hstack([[1],x[i]])
        xh= np.dot(w,x1)
        u=p(xh)
        u1 = np.hstack([[1],u])
        uh = np.dot(v,u1)
        y = p(uh)
        tt= tvector(t[i])
        E2 += E(y,tt)
    E2 /= ndata
    Resid = abs(E2-E1)
    E1 = E2
    noiter +=1
    print('{0}th iteration : Error = {1}'
          .format(noiter,E1))
    
np.savez('learning',w=w,v=v)
