import numpy as np
import pandas as pd
from matplotlib import figure
import matplotlib.pyplot as plt
import seaborn as sns

import math
import os
import time
import random
import gc

import warnings
warnings.filterwarnings(action='ignore')

import sys
sys.path.append('../utils')

import torch
from torch import nn
from torch.utils.data import Dataset, WeightedRandomSampler
import torch.nn.functional as F
           
from utils.Visualize import layer_visualize

SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

class DRCNN(nn.Module):
    def __init__(self, in_features, num_class, out_features, hidden_dim, kernel_size, capsule_dim, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.num_class = num_class
        self.output = out_features
        self.kernel_size = kernel_size
        
        self.hidden_dim = hidden_dim
        self.capsule_dim = capsule_dim
        
        self.conv1 = nn.Conv2d(in_features, self.hidden_dim, kernel_size=(kernel_size, 1), stride=(1,1))
        
        self.conv2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=(9, 1), stride=(2,1))
        
        self.conv3 = nn.Conv2d(self.hidden_dim, num_class*(capsule_dim*2), kernel_size=(1,1), stride=(1,1))
        self.pool1 = nn.AdaptiveMaxPool2d(1) # Routing = Pooling
       
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(num_class*(capsule_dim*2), 512)
        self.fc2 = nn.Linear(512,1024)
        self.out = nn.Linear(1024, out_features)

    def forward(self, x, visualize=False, attack=None, num=0, result_folder = None):
        if self.cfg.data == 'traffic':
            x = x.transpose(1,2).unsqueeze(-1)
        else:
            x = x.unsqueeze(1)
            
        if visualize==True:
            x = F.relu(self.conv1(x))
            layer_visualize(x, self.hidden_dim, "CNN_Conv1", attack, num, result_folder)
            x = F.relu(self.conv2(x))
            layer_visualize(x, self.hidden_dim, "CNN_Conv2", attack, num, result_folder)
            x = self.conv3(x)
            layer_visualize(x, (self.capsule_dim*2*self.num_class), "CNN_Conv3", attack, num, result_folder)
            x = F.relu(self.pool1(x))

            x = self.flatten(x).unsqueeze(1)

            x = F.relu(self.fc1(x))
            #layer_visualize(x, self.hidden_dim, "layer5")
            x = F.relu(self.fc2(x))
            #layer_visualize(x, self.hidden_dim, "layer6")
            x = self.out(x)
            
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.pool1(self.conv3(x)))

            x = self.flatten(x).unsqueeze(1)

            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.out(x)
        
        return x