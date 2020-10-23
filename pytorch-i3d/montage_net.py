import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from pytorch_i3d import InceptionI3d

import os
import sys
from collections import OrderedDict

class MontageNet(nn.Module):
    def __init__(self, mode, num_classes=400):
        super(MontageNet, self).__init__()

        self.mode = mode
        self.num_classes = num_classes
        if self.mode == 'flow':
            self.i3d = InceptionI3d(400, in_channels=2)
            self.i3d.load_state_dict(torch.load('models/flow_imagenet.pt'))
        else:
            self.i3d = InceptionI3d(400, in_channels=3)
            self.i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
        self.i3d.replace_logits(self.num_classes)
        # self.lstm = nn.LSTM(1024, 1024, 2)
        
    def forward(self, x):
        x = self.i3d.extract_features(x)
        x = x.squeeze(3).squeeze(3).mean(2)

        # print(x.shape)
        mean = x.mean(dim=1).reshape(-1, 1)
        std = x.std(dim=1).reshape(-1, 1)
        x = (x - mean)/std

        return x
        # x, _ = self.lstm(x)
        # print("after lstm",x.shape)

class PredictNet(nn.Module):
    def __init__(self, feature_dim=1024, hidden_dim=2048, n_layer=2):
        super(PredictNet, self).__init__()

        self.lstm = nn.LSTM(feature_dim, hidden_dim, n_layer)
        self.line = nn.Linear(hidden_dim, feature_dim)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[-1]
        x = self.line(x)
        return x