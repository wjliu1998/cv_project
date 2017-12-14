#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 13:44:01 2017

@author: wjliu
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np


file = open("data.txt","r")
loss = file.read()
len(loss)

loss = loss[1:-1]
loss = loss.split(',')
len(loss)
loss

x = np.linspace(0, len(loss)-1, len(loss))
plt.plot(x, loss, 'r')

f = open("dnn.pkl", 'rb')
info = pickle.load(f)
f.close()


for k,v in info.items():
    print(k)    #打印网络中的变量名
print(info['layer1.0.weight'])   #打印conv1的weight
weights = info['layer1.0.weight']
list = []

for a in range(0,weights.size()[0]):
    for b in range(0, weights.size()[1]):
        for c in range(0, weights.size()[2]):
            for d in range(0, weights.size()[3]):
                list.append(weights[a][b][c][d])

weights = info['layer1.1.weight']
list = []

for a in range(0,weights.size()[0]):
    for b in range(0, weights.size()[1]):
        for c in range(0, weights.size()[2]):
            for d in range(0, weights.size()[3]):
                list.append(weights[a][b][c][d])
                
weights = info['layer1.2.weight']
list = []

for a in range(0,weights.size()[0]):
    for b in range(0, weights.size()[1]):
        for c in range(0, weights.size()[2]):
            for d in range(0, weights.size()[3]):
                list.append(weights[a][b][c][d])

len(list)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(list, bins=11)
plt.title('Age distribution')
plt.xlabel('Age')
plt.ylabel('Employee')
plt.show()
