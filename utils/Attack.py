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
import copy
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings(action='ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
from torch import nn
from torch.utils.data import Dataset, WeightedRandomSampler
import torch.nn.functional as F
           
from utils.Visualize import *
from utils.Dataset import *
from model.ClassificationModule import *
from model.ReconstructionModule import *
from model.PredictionModule import *
from model.DRModule import *

import mlflow

SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

### Manual Attack ###
import copy
def offset(input_, offset=0.1, window=35, num_sample=1,
           random_noise=True, max_rand_noise=0.01, label_last_col=False, 
           multi_var=0, device=None):
    """
    input: pandas dataframe, its axis must be matched like (time, variable, ...)
    offset: constant value in the range of 0-1, which will be plus added.
    window: time window to be attacked.
    num_sample: number of records to be attacked.
    random_noise: adding brownian motion random noise.
    max_rand_noise: maximum random noise.
    label_last_col: whether last column is target label or not.
    multi_var: number of multivariate sensors to be attacked.
    return: normalized x, [attacked sample indices, time indices for start, and end]
    """
    x = copy.deepcopy(input_)
    assert window > 0 and window < x.shape[1], "Incorrect range of window"
    assert offset >= 0 and offset <= 1, "Incorrect range of offset value"
    if label_last_col==True:
        if len(x.shape)>2:
            label_col = x[:,-1,:]
            x = x[:,:-1,:]
        else:
            label_col = x.iloc[:,-1]
            x = x.iloc[:,:-1]

    idxs = range(num_sample)
    start_idx = np.random.randint(0, x.shape[1]-window, num_sample)
    indices = np.stack((idxs, start_idx, start_idx + window), axis=1)    
    
    if multi_var != 0: # multivariate (PeMS) [samples, time, sensor] =(10358, 168, 862)
        assert multi_var >= 0 and multi_var < x.shape[2], "Incorrect number of multivariate sensors to be attacked"
        sensor_idxs = np.random.randint(0, x.shape[2], multi_var)
        if random_noise == True:
            noise = np.random.normal(0, max_rand_noise, size=(len(idxs), window, multi_var))
            noise = np.around(noise,6)
        else:
            noise = np.zeros(shape=(len(idxs), window, multi_var))
        for p, sensor_idx in enumerate(sensor_idxs):
            for j, [i, start_idx, end_idx] in enumerate(indices):
                x[i, start_idx:end_idx, sensor_idx] += torch.as_tensor(offset).to(device) + torch.Tensor(noise[j,:,p]).to(device)
#         x = x.clamp(min=0.0, max=1.0) # clipping 0 through 1
        if label_last_col==True:
            x = torch.cat((x, label_col.unsqueeze(1)), 1)
        return x, indices, sensor_idxs
    
    else: #univariate (ECG) [sample, time] =(21892, 188)
        if random_noise == True:
            noise = np.random.normal(0, max_rand_noise, size=(len(idxs), window))
            noise = np.around(noise,6)
        else:
            noise = np.zeros(shape=(len(idxs), window))
        for j, [i, start_idx, end_idx] in enumerate(indices):
            x.iloc[i,start_idx:end_idx] += offset + noise[j,:]
            
        x.clip(lower=0.0, upper=1.0, inplace=True) # clipping 0 through 1
        if label_last_col==True:
            x = pd.concat([x,label_col],axis=1)
    
        return x, indices
    
    
def drift(input_, scale=0.1, window=35, num_sample=1, type_='increasing', 
          random_noise=True, max_rand_noise=0.01, label_last_col=False,
          multi_var=0, device=None):
    """
    input: pandas dataframe, its axis must be matched like (time, variable, ...)
    scale: constant value in the range of 0-1, which will be plus increasingly added.
    window: time window to be attacked.
    num_sample: number of records to be attacked.
    type: options consists of 'increasing' and 'decreasing'
    random_noise: adding brownian motion random noise.
    max_rand_noise: maximum random noise.
    label_last_col: whether last column is target label or not.
    multi_var: number of multivariate sensors to be attacked.
    return: normalized x, [attacked sample indices, time indices for start, and end]
    """
    x = copy.deepcopy(input_)
    assert window > 0 and window < x.shape[1], "Incorrect range of window"
    assert scale >= 0 and scale <= 1, "Incorrect range of scale value"
    
    if label_last_col==True:
        if len(x.shape)>2:
            label_col = x[:,-1,:]
            x = x[:,:-1,:]
        else:
            label_col = x.iloc[:,-1]
            x = x.iloc[:,:-1]
    idxs = range(num_sample)
    start_idx = np.random.randint(0, x.shape[1]-window, num_sample)
    if len(x.shape)<3:
        start_idx = np.random.randint(0, x.shape[1]//2, num_sample)
    indices = np.stack((idxs, start_idx, start_idx + window), axis=1)

    if multi_var > 0: # multivariate (PeMS) [samples, time, sensor] =(10358, 168, 862)
        assert multi_var >= 0 and multi_var < x.shape[2], "Incorrect number of multivariate sensors to be attacked"
        sensor_idxs = np.random.randint(0, x.shape[2], multi_var)
        if random_noise == True:
            noise = np.random.normal(0, max_rand_noise, size=(len(idxs), window, multi_var))
            noise = np.around(noise,6)
        else:
            noise = np.zeros(shape=(len(idxs), window, multi_var))
        f = np.linspace(start=0, stop=scale, num=window) 
        if type_ =='decreasing':
            f = -f
        for p, sensor_idx in enumerate(sensor_idxs):
            for j, [i, start_idx, end_idx] in enumerate(indices):
                x[i, start_idx:end_idx, sensor_idx] += torch.Tensor(f).to(device) + torch.Tensor(noise[j,:,p]).to(device)
#         x = x.clamp(min=0.0, max=1.0) # clipping 0 through 1
        if label_last_col==True:
            x = torch.cat((x, label_col.unsqueeze(1)), 1)
        return x, indices, sensor_idxs
    
    else: #univariate (ECG) [sample, time] =(21892, 188)
        if random_noise == True:
            noise = np.random.normal(0, max_rand_noise, size=(len(idxs), window))
            noise = np.around(noise,6)
        else:
            noise = np.zeros(shape=(len(idxs), window))
        f = np.linspace(start=0, stop=scale, num=window) 
        if type_ =='decreasing':
            f = -f
        for j, [i, start_idx, end_idx] in enumerate(indices):
            x.iloc[i, start_idx:end_idx] += f + noise[j,:]
        x.clip(lower=0.0, upper=1.0, inplace=True) # clipping 0 through 1
        if label_last_col==True:
            x = pd.concat([x,label_col],axis=1)
        return x, indices
    

def lagging(input_, lag_time=15, num_sample=1, multi_var=0, type_='backward',
            random_noise=True, max_rand_noise=0.01, label_last_col=False, device=None):
    """
    input: pandas dataframe, its axis must be matched like (time, variable, ...).
    lag_time: integer which is lagging duration.
    num_sample: number of records to be attacked.
    random_noise: adding brownian motion random noise.
    max_rand_noise: maximum random noise.
    label_last_col: whether last column is target label or not.
    multi_var: number of multivariate sensors to be attacked.
    return: normalized x, [attacked sample indices, time indices for start]
    """
    x = copy.deepcopy(input_)
    assert lag_time >= 0, "Incorrect range of lag time"
#     assert type(lag_time) is int, "Incorrect data type of lag time"
    if label_last_col==True:
        if len(x.shape)>2: # multivariate
            label_col = x[:,-1,:]
            x = x[:,:-1,:]
        else: #univariate
            label_col = x.iloc[:,-1]
            x = x.iloc[:,:-1]
    
    idxs = range(num_sample)
    if type_=='backward':
        start_idx = np.random.randint(lag_time, x.shape[1]-2*lag_time, num_sample)
        if len(x.shape)<3:
            start_idx = np.random.randint(lag_time, x.shape[1]//2, num_sample)
    else: # 'forward'
        start_idx = np.random.randint(2*lag_time, x.shape[1]-lag_time, num_sample)
        if len(x.shape)<3:
            start_idx = np.random.randint(lag_time, x.shape[1]//2, num_sample)
    indices = np.stack((idxs, start_idx), axis=1)            
    if multi_var > 0: # multivariate (PeMS) [samples, time, sensor] =(10358, 168, 862)
        assert multi_var >= 0 and multi_var < x.shape[2], "Incorrect number of multivariate sensors to be attacked"
        sensor_idxs = np.random.randint(0, x.shape[2], multi_var)
        
        for _, [i, start_idx] in enumerate(indices):
            if type_=='backward':
                if random_noise == True:
                    noise = np.random.normal(0, max_rand_noise, size=(x[i,start_idx-lag_time:x.shape[1],0].shape[0], multi_var))
                    noise = np.around(noise,6)
                else:
                    noise = np.zeros(shape=(x[i,start_idx-lag_time:x.shape[1],0].shape[0], multi_var))
                for p, sensor_idx in enumerate(sensor_idxs): 
                    x[i,start_idx:x.shape[1],sensor_idx] = copy.deepcopy(x[i,start_idx-lag_time:x.shape[1]-lag_time,sensor_idx])
                    x[i,start_idx-lag_time:start_idx,sensor_idx] = copy.deepcopy(x[i,start_idx-lag_time,sensor_idx])
                    x[i,start_idx-lag_time:x.shape[1],sensor_idx] += torch.Tensor(noise[:,p]).to(device)
            else: # 'forward'
                if random_noise == True:
                    noise = np.random.normal(0, max_rand_noise, size=(x[i,start_idx-lag_time:x.shape[1],0].shape[0], multi_var))
                    noise = np.around(noise,6)
                else:
                    noise = np.zeros(shape=(x[i,start_idx-lag_time:x.shape[1],0].shape[0], multi_var))
                for p, sensor_idx in enumerate(sensor_idxs): 
                    x[i,start_idx-lag_time:x.shape[1]-lag_time,sensor_idx] = x[i,start_idx:x.shape[1],sensor_idx]
                    x[i,x.shape[1]-lag_time:x.shape[1],sensor_idx] = x[i,x.shape[1]-lag_time,sensor_idx]
                    x[i,start_idx-lag_time:x.shape[1],sensor_idx] += torch.Tensor(noise[:,p]).to(device)
#         x = x.clamp(min=0.0, max=1.0) # clipping 0 through 1
        if label_last_col==True:
            x = torch.cat((x, label_col.unsqueeze(1)), 1)
        return x, indices, sensor_idxs
    
    else: #univariate (ECG) [sample, time] =(21892, 188)
        for _, [i, start_idx] in enumerate(indices):
            if type_=='backward':
                if random_noise == True:
                    noise = np.random.normal(0, max_rand_noise, size=(x.iloc[i,start_idx-lag_time:x.shape[1]].shape[0]))
                    noise = np.around(noise,6)
                else:
                    noise = np.zeros(shape=(x.iloc[i,start_idx-lag_tim:x.shape[1]].shape[0]))
                x.iloc[i,start_idx-lag_time:x.shape[1]] = x.iloc[i,start_idx-lag_time:x.shape[1]].shift(periods=lag_time, fill_value=x.iloc[i,start_idx-lag_time])
                x.iloc[i,start_idx-lag_time:x.shape[1]] += noise
            else:
                if random_noise == True:
                    noise = np.random.normal(0, max_rand_noise, size=(x.iloc[i,start_idx:x.shape[1]].shape[0]))
                    noise = np.around(noise,6)
                else:
                    noise = np.zeros(shape=(x.iloc[i,start_idx:x.shape[1]].shape[0]))
                x.iloc[i,start_idx:x.shape[1]] = x.iloc[i,start_idx:x.shape[1]].shift(periods=-lag_time-1, fill_value=0)
                x.iloc[i,start_idx:x.shape[1]] += noise     
        x.clip(lower=0.0, upper=1.0, inplace=True) # clipping 0 through 1
        if label_last_col==True:
            x = pd.concat([x,label_col],axis=1)
        return x, indices
    
### Black Box fgsm Attack ###    
def fgsm_attack(model, criterion, x, y, eps, device, cfg) :
    x = x.to(device)
    y = y.to(device)
    x.requires_grad = True
    
    if cfg.task == 'classirecon':
        
        y_onehot = torch.zeros(y.size(0), cfg.num_class).scatter_(1, torch.Tensor(y.float().view(-1,1).detach().cpu()).type(torch.int64), 1.)
        y_onehot = y_onehot.to(device)
        
        classes, recon = model(x)
        
        model.zero_grad()
        if cfg.loss == 'margin':
            cost = criterion(y_onehot, classes, x, recon, 0.0005*cfg.window).to(device)
        else:
            cost = criterion(classes.squeeze(), y.squeeze())+0.0005*cfg.window*nn.MSELoss()(x.squeeze(), recon.squeeze())
        cost.backward()
        
    else:
        if cfg.model == 'DR':
            _, outputs = model(x)
            outputs = outputs.squeeze()
        else:
            outputs = model(x).squeeze()
    
        model.zero_grad()
        cost = criterion(outputs, y.squeeze()).to(device)
        cost.backward()
    
    attack_x = x + eps * x.grad.sign()    
    return attack_x

def fgsm_makeData(model, criterion, test_loader, eps, cfg, device):
    eps_start = eps[0]
    eps_end = eps[1]
    fgsm_data = []
    eps_list = []
    if cfg.task == 'classification' or cfg.task == 'classirecon':
        for eps in tqdm(np.arange(eps_start, eps_end+0.01, 0.01), leave=False):
            eps_list.append(eps)
            fgsm_x = []
            fgsm_y = []
            for x, y in test_loader:
                attack_x = fgsm_attack(model, criterion, x, y, eps, device, cfg)
                fgsm_x.extend(attack_x.squeeze().detach().cpu().numpy())
                fgsm_y.extend(y.squeeze().detach().cpu().numpy())
            tmp = pd.DataFrame(fgsm_x)
            tmp.insert(len(tmp.columns), 'target', pd.Series(fgsm_y))
            fgsm_data.append(tmp)
    elif cfg.task == 'reconstruction':
        for eps in tqdm(np.arange(eps_start, eps_end+0.01, 0.01), leave=False):
            eps_list.append(eps)
            fgsm_x = []
            for x in test_loader:
                attack_x = fgsm_attack(model, criterion, x, x, eps, device, cfg)
                fgsm_x.extend(attack_x.squeeze().detach().cpu().numpy())
            tmp = pd.DataFrame(fgsm_x)
            fgsm_data.append(tmp)
    print("eps: ", eps_list)
    return fgsm_data, eps_list

### Attack Performance Evaluation ###
def attack_inference(cfg, model, criterion, result_folder,  report, loader, eps_list, device, true_label=None, data=None):
    for attack in ['offset', 'increasing', 'decreasing', 'forward', 'backward', 'fgsm']:                
        if cfg.task == 'classification':
            if attack == 'fgsm':
                fgsm_test_loader = loader[-1]
                for i in range(len(fgsm_test_loader)):
                    attack_loader = fgsm_test_loader[i]
                    report = classification_performance_evaluation(model, criterion, attack_loader, attack+f'_{eps_list[i]}', result_folder, cfg, report, device)
            else:
                if attack == 'offset':
                    attack_loader = loader[0] 
                if attack == 'increasing':
                    attack_loader = loader[1] 
                if attack == 'decreasing':
                    attack_loader = loader[2]
                if attack == 'forward':
                    attack_loader = loader[3] 
                if attack == 'backward':
                    attack_loader = loader[4] 
                               
                report = classification_performance_evaluation(model, criterion, attack_loader, attack, result_folder, cfg, report, device)
                             
        elif cfg.task == 'reconstruction':
            if attack == 'fgsm':
                fgsm_test_loader = loader[-1]
                for i in range(len(fgsm_test_loader)):
                    attack_loader = fgsm_test_loader[i]
                    report, _ = recon_performance_evaluation(model, criterion, attack_loader, attack+f'_{eps_list[i]}', result_folder, cfg, report, true_label, device)
            else:
                if attack == 'offset':
                    attack_loader = loader[0] 
                if attack == 'increasing':
                    attack_loader = loader[1] 
                if attack == 'decreasing':
                    attack_loader = loader[2]
                if attack == 'forward':
                    attack_loader = loader[3] 
                if attack == 'backward':
                    attack_loader = loader[4]              
                report, _ = recon_performance_evaluation(model, criterion, attack_loader, attack, result_folder, cfg, report, true_label, device)
                             
        elif cfg.task == 'prediction':
            if attack == 'fgsm':
                fgsm_test_loader = loader[-1]
                for i in range(len(fgsm_test_loader)):
                    attack_loader = fgsm_test_loader[i]
                    report, _ = pred_performance_evaluation(model, criterion, data, attack_loader, attack+f'_{eps_list[i]}', result_folder, cfg, report, true_label, device)
            else:
                if attack == 'offset':
                    attack_loader = loader[0] 
                if attack == 'increasing':
                    attack_loader = loader[1] 
                if attack == 'decreasing':
                    attack_loader = loader[2]
                if attack == 'forward':
                    attack_loader = loader[3] 
                if attack == 'backward':
                    attack_loader = loader[4]              
                report, _ = pred_performance_evaluation(model, criterion, data, attack_loader, attack, result_folder, cfg, report, true_label, device)
        
        else:
            if attack == 'fgsm':
                fgsm_test_loader = loader[-1]
                for i in range(len(fgsm_test_loader)):
                    attack_loader = fgsm_test_loader[i]
                    report, _ = dr_performance_evaluation(model, criterion, attack_loader, attack+f'_{eps_list[i]}', result_folder, cfg, report, true_label, device)
            else:
                if attack == 'offset':
                    attack_loader = loader[0] 
                if attack == 'increasing':
                    attack_loader = loader[1] 
                if attack == 'decreasing':
                    attack_loader = loader[2]
                if attack == 'forward':
                    attack_loader = loader[3] 
                if attack == 'backward':
                    attack_loader = loader[4]              
                report, _ = dr_performance_evaluation(model, criterion, attack_loader, attack, result_folder, cfg, report, true_label, device)
    return report