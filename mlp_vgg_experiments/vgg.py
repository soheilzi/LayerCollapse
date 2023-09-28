# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 13:49:13 2019

@author: Aakash
"""

'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


cfg = {
    'VGG6AS': ['A', 8, 'M', 32, 'A', 128, 'M', 512, 'A'],
    'VGG6AM': [32, 'A', 64, 'M', 128, 'M', 512, 'A', 'M'],
    'VGG6A': [64, 'A', 128, 'A', 256, 'A', 512, 'A', 'M'],
    'VGG6': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 'A'],
    'VGG7': [64, 'M', 64, 'M', 128, 'M', 256, 'M', 512, 'M'],
    'VGG8': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG16A': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 'A', 512, 512, 512, 512, 512, 512, 'A', 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, classes=1000):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        # self.classifier = nn.Linear(512, classes)
        self.classifier = self._make_classifier(25088)
        self.fts = None

    def forward(self, x):
        # print("x.shape", x.shape)
        out = self.features(x)
        # print("out.shape", out.shape)
        out = out.view(out.size(0), -1)
        # print("out.shape", out.shape)
        #self.fts = out
        out = self.classifier(out)
#         print("out.shape", out.shape)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif x == 'A':
                layers += [nn.Conv2d(in_channels, in_channels, kernel_size=1),
                           nn.BatchNorm2d(in_channels),
                           nn.ReLU(inplace=True),
                           nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
#                 layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
#                            nn.ReLU(inplace=True)]
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AdaptiveAvgPool2d((7, 7))]
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]        
        return nn.Sequential(*layers)

    def _make_classifier(self, in_features):
        layers = []
#         layers += [nn.Linear(in_features=in_features, out_features=4096, bias=True), nn.ReLU(inplace=True), nn.Dropout(p=0.5, inplace=False)]
#         layers += [nn.Linear(in_features=4096, out_features=4096, bias=True), nn.ReLU(inplace=True), nn.Dropout(p=0.5, inplace=False)]
#         layers += [nn.Linear(in_features=4096, out_features=1000, bias=True)]


        layers += [nn.Identity() for i in range(6)]
        layers += [nn.Linear(in_features=25088, out_features=1000, bias=True)]

        return nn.Sequential(*layers)



def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)