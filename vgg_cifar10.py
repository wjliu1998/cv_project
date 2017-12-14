#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 12:26:53 2017

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
    

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        #print(out.size())
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
        
vggnet = VGG("VGG19")
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
        
        lossList.append(loss.data[0])
        optimizer.step()
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

x = np.linspace(0, len(lossList)-1, len(lossList))
plt.plot(x, lossList, 'r')