#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 08:59:35 2017

@author: wjliu
"""

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import pickle

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
    

class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size = 3,padding = 1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(2)
                )
        self.layer2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size = 3,padding = 1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.MaxPool2d(2)
                )
        self.layer3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size = 3,padding = 1),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, 256, kernel_size = 3,padding = 1),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.MaxPool2d(2)
                )
        self.layer4 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size = 3,padding = 1),
                nn.ReLU(),
                nn.BatchNorm2d(512),
                nn.Conv2d(512, 512, kernel_size = 3,padding = 1),
                nn.ReLU(),
                nn.BatchNorm2d(512),
                nn.MaxPool2d(2)
                )
        self.layer5 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size = 3,padding = 1),
                nn.ReLU(),
                nn.BatchNorm2d(512),
                nn.Conv2d(512, 512, kernel_size = 3,padding = 1),
                nn.ReLU(),
                nn.BatchNorm2d(512),
                nn.MaxPool2d(2)
                )
        self.layer6 = nn.Sequential(
                #nn.Linear(4*4*256, 256),
                nn.AvgPool2d(kernel_size=1, stride=1),
                )
        self.layer7 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = out.view(out.size(0), -1)
        out = self.layer7(out)
        return out
        
vggnet = VGGNet()
vggnet.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vggnet.parameters(), lr=learning_rate)
lossList = []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        
        
        optimizer.zero_grad()
        outputs = vggnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()
        lossList.append(loss.data[0])
        if (i+1) % 100 == 0:
            print ('cnn1: Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))
            
        
vggnet.eval()
correct = 0;
total = 0
for images, labels in test_loader:
    images = Variable(images).cuda()
    outputs = vggnet(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()
    
print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))        

f = open("vgg11_3_3_33_33.pkl", 'wb')
pickle.dump(vggnet.state_dict(), f)
f.close()

file=open('vgg11_3_3_33_33.txt','w')  
file.write(str(lossList));  
file.close()