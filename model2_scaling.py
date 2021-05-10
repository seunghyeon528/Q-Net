import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as  np
import torch.nn.utils as ut
import pdb
####################################################################
#                             QNET MODEL
####################################################################
class QNet(nn.Module):
    def __init__(self):
        super(QNet,self).__init__()
        # conv_layer1
        self.layer1 = nn.Sequential(
            nn.Conv2d(2,15,5,stride=1,padding=2),
            nn.BatchNorm2d(15),
            nn.ReLU())
            #nn.MaxPool2d(2,2))
        # conv_layer2
        self.layer2 = nn.Sequential(
            nn.Conv2d(15,25,7,stride=1,padding=3),
            nn.BatchNorm2d(25),
            nn.ReLU())
        # conv_layer3
        self.layer3 = nn.Sequential(
            nn.Conv2d(25,40,9,stride=1,padding=4),
            nn.BatchNorm2d(40),
            nn.ReLU())
        # conv_layer4
        self.layer4 = nn.Sequential(
            nn.Conv2d(40,50,11,stride=1,padding=5),
            nn.BatchNorm2d(50),
            nn.ReLU())

        # global average poolingp
        self.GAP = nn.AdaptiveAvgPool2d(1)

        # fully connected layers
        self.fc1 = ut.spectral_norm(nn.Linear(50,50))
        self.fc2 = ut.spectral_norm(nn.Linear(50,10))
        self.fc3 = ut.spectral_norm(nn.Linear(10, 1))
        self.sigmoid = nn.Sigmoid() # output scaled to range of 0-1

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.GAP(out)
        out = out.squeeze(2)
        out = out.squeeze(2)
        # leaky_relu nodes
        out = F.leaky_relu(self.fc1(out))
        out = F.leaky_relu(self.fc2(out))
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out
