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

import torch
from torch import nn
from torch.utils.data import Dataset, WeightedRandomSampler
import torch.nn.functional as F
from torch.autograd import Variable

SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

### Classification Dataset Making ###
class classification_MyDataset(Dataset):
    def __init__(self, data):
        self.data = torch.Tensor(data)
        self.used_cols = [x for x in range(data.shape[1]-1)] # X = except target column
        self.target_col = -1 
        
        self.shape = self.__getshape__()
        self.size = self.__getsize__()

    def __getitem__(self, index):
        x = self.data[index, self.used_cols].unsqueeze(1)
        y = self.data[index, self.target_col].long()
        return x, y

    def __len__(self):
        return len(self.data) 
    
    def __getshape__(self):
        return (self.__len__(), *self.__getitem__(0)[0].shape) # row, col
    
    def __getsize__(self):
        return (self.__len__())

### Reconstruction Dataset Making ###
class recon_MyDataset(Dataset):
    def __init__(self, data, attack=None):
        self.data = torch.Tensor(data)
        if attack == None:
            self.used_cols = [x for x in range(data.shape[1]-1)] # X = except target column
        else:
            self.used_cols = [x for x in range(data.shape[1])]
        
        self.shape = self.__getshape__()
        self.size = self.__getsize__()

    def __getitem__(self, index):
        x = self.data[index, self.used_cols].unsqueeze(1)
        return x

    def __len__(self):
        return len(self.data) 
    
    def __getshape__(self):
        return (self.__len__(), *self.__getitem__(0)[0].shape) # row, col
    
    def __getsize__(self):
        return (self.__len__())
    
### Prediction Dataset and Loader Making ### (Special DataLoader for Traffic Data)
def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))

class DataLoaderH(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, horizon, window, pred_step, normalize=2):
        """
        1.Get Data
        2. Normalize 
        3. Split
        """
        self.P = window
        self.h = horizon
        self.pred_step = pred_step
        
        fin = open(file_name)
        self.rawdat = np.loadtxt(fin, delimiter=',')
        self.dat = np.zeros(self.rawdat.shape)

        self.n, self.m = self.dat.shape
        self.normalize = 2
        self.scale = np.ones(self.m)
        self.bias =  np.zeros(self.m)
        self._normalized(normalize)
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        self.scale = torch.from_numpy(self.scale).float()
        self.bias = torch.from_numpy(self.bias).float()
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.h, self.m)

        self.scale = self.scale.cuda()
        self.scale = Variable(self.scale)
        self.bias = self.bias.cuda()
        self.bias = Variable(self.bias)

        tmp = tmp[:, -1, :].squeeze()
        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

    def _normalized(self, normalize):
        """
        Normalize Data
        0. No Normalization
        1. MinMaxScaler - by one scaler
        2. MinMaxScaler - by each sensor
        3. StandardScaler - by each sensor
        """
        if (normalize == 0):
            self.dat = self.rawdat

        if (normalize == 1):
            # normalized by the maximum value of entire matrix.
            self.dat = self.rawdat / np.max(self.rawdat)
        
        if (normalize == 2):
            # normlized by the maximum value of each row (sensor).
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))

        if (normalize == 3):
            # normlized by the mean/std value of each row (sensor).
            for i in range(self.m):
                self.scale[i] = np.std(self.rawdat[:, i]) #std
                self.bias[i] = np.mean(self.rawdat[:, i])
                self.dat[:, i] = (self.rawdat[:, i] - self.bias[i]) / self.scale[i]


    def _split(self, train, valid, test):
        """
        Train-Valid-Test Split
        1. Make as Dataset
        2. Make as Loader
        """
        train_set = range(self.P + self.pred_step, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)

    def _batchify(self, idx_set, horizon):
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.h, self.m))
        for i in range(n):
            end = idx_set[i] - self.pred_step + 1
            start = end - self.P
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            Y[i, :, :] = torch.from_numpy(self.dat[[end+self.pred_step-1], :])
        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        """
        get Batch
        """
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            X = X.cuda()
            Y = Y.cuda()
            yield Variable(X), Variable(Y)
            start_idx += batch_size