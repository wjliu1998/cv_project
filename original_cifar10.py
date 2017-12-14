#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 10:03:11 2017

@author: wjliu
"""

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

transform = transforms.Compose([
        transforms.Scale(40),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()])

train_dataset = dsets.CIFAR10(root='./data/',
                               train=True, 
                               transform=transform,
                               download=True
                               )

test_dataset = dsets.CIFAR10(root='./data/',
                              train=False, 
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100, 
                                          shuffle=False)

def Local_Response_Norm(out, n, k, alpha, beta):
    print("process")
    #func_out = torch.FloatTensor(out.size())
    pics, channels, height, width = out.size()
    for i in range(0, pics):
        print(i)
        for j in range(0, channels):
            for d in range(0, height):
                for t in range(0, width):
                    #new = out[i][max(0, (int)(i-n/2)):min(channels, (int)(i+n/2))][d][t]
                    #print(max(0, (int)(j-n/2)))
                    #print(min(channels, (int)(j+n/2)))
                    new = out[i].narrow(0, max(0, (int)(j-n/2)),min(channels, (int)(j+n/2)))
                    new = new.narrow(1,d,d+1).narrow(2,t,t+1)
                    norm = new.mul(new)
                    norm = norm.sum()
                    out[i][j][d][t].div(pow((k + alpha*norm), beta))
    return out
    

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
                #nn.Conv2d(3, 96, kernel_size = (1,3)),
                #nn.Conv2d(96, 96, kernel_size = (3,1)),
                #nn.Conv2d(96, 96, kernel_size = (1,3)),
                #nn.Conv2d(96, 96, kernel_size = (3,1)),
                #nn.Conv2d(3, 96, kernel_size = 7),
                nn.Conv2d(3, 96, kernel_size = 3),
                nn.Conv2d(96, 96, kernel_size = 3),
                nn.Conv2d(96, 96, kernel_size = 3),
                nn.ReLU(),
                nn.BatchNorm2d(96),
                nn.MaxPool2d(2)
                )
        self.layer2 = nn.Sequential(
                #nn.Conv2d(96, 256, kernel_size = (1,3)),
                #nn.Conv2d(256, 256, kernel_size = (3,1)),
                #nn.BatchNorm2d(256),
                #nn.Conv2d(96, 256, kernel_size = 5),
                nn.Conv2d(96, 256, kernel_size = 3),
                nn.Conv2d(256, 256, kernel_size = 3),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.MaxPool2d(2)
                )
        self.layer3 = nn.Sequential(
                nn.Linear(4*4*256, 256),
                )
        self.layer4 = nn.Sequential(
                nn.Linear(256, 128),
                )
        self.layer5 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        #out = Local_Response_Norm(out, 5, 2, 0.0001, 0.75)
        #print(out.size())
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        #out = Local_Response_Norm(out, 5, 2, 0.0001, 0.75)
        #print(out.size())
        out = self.layer3(out)
        #print(out.size())
        out = self.layer4(out)
        #print(out.size())
        out = self.layer5(out)
        #print(out.size())
        
        '''out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)'''
        return out
        
alexnet = AlexNet()
alexnet.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(alexnet.parameters(), lr=learning_rate)
lossList = []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        
        
        optimizer.zero_grad()
        outputs = alexnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        lossList.append(loss.data[0])
        optimizer.step()
        if (i+1) % 100 == 0:
            print ('cnn1: Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))
            
        
alexnet.eval()
correct = 0;
total = 0
for images, labels in test_loader:
    images = Variable(images).cuda()
    outputs = alexnet(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()
    
print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))        

x = np.linspace(0, len(lossList)-1, len(lossList))
plt.plot(x, lossList, 'r')
'''torch.save(alexnet.state_dict(), 'rnn.pkl')

params=alexnet.state_dict() 
for k,v in params.items():
    print(k)    #打印网络中的变量名
print(params['layer1.0.weight'])   #打印conv1的weight
print(params['conv1.bias']) '''