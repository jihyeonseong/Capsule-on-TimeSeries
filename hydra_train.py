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
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings(action='ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

import torch
from torch import nn
from torch.utils.data import Dataset, WeightedRandomSampler
import torch.nn.functional as F

# Import Model Related Function #           
from utils.Visualize import *
from utils.Dataset import *
from utils.Attack import *
from model.ClassificationModule import *
from model.ReconstructionModule import *
from model.PredictionModule import *
from model.DRModule import *

# Import Model #
from model.CNN import DRCNN
from model.DRCapsNet import CapsuleNet, caps_loss

# Import Hydra and mlflow #
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
from hydra.utils import get_original_cwd, to_absolute_path
import mlflow
import logging

SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

logger = logging.getLogger(__name__)

# Counts Model Parameters #
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Logging Config Parameters #
def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        _explore_recursive(param_name, element)

def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f'{parent_name}.{k}', v)
            else:
                mlflow.log_param(f'{parent_name}.{k}', v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlflow.log_param(f'{parent_name}.{i}', v)
    else:
        mlflow.log_param(f'{parent_name}', element)
        
### Hydra Start ###
@hydra.main(config_path="config", config_name="config")
def main(cfg : DictConfig) -> None:
    print(f"Hydra {cfg.model} {cfg.task} {cfg.data} Ready to Run!")
    print(f"Current working directory : {os.getcwd()}")
    print(f"Orig working directory    : {get_original_cwd()}")      
    
    ### set GPU and mlflow ###
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)
    mlflow.set_tracking_uri('file://' + get_original_cwd() + '/mlruns')
    mlflow.set_experiment(cfg.exp_name)
    mlflow.start_run(run_name=os.getcwd())
    
    ### Make Result Folder for Model ###
    result_folder = os.getcwd() # f'./outputs/date/time'
    os.makedirs(result_folder, exist_ok=True)
    report = pd.DataFrame()
    
    ### get Data ###
    data_folder = to_absolute_path('data')
    os.makedirs(data_folder, exist_ok=True)
    if cfg.data == 'ECG':
        data_file_train = 'mitbih_train.csv'
        data_file_test = 'mitbih_test.csv'
        df_train = pd.read_csv(os.path.join(data_folder, data_file_train), header=None)
        df_test = pd.read_csv(os.path.join(data_folder, data_file_test), header=None)
        
        # Data Sample img #
        plt.figure(figsize=(10,3))
        plt.plot(df_train.iloc[:10, :].T)
        plt.title("10 Samples", fontsize=15)
        plt.xlabel("Time", fontsize=10)
        plt.ylabel("Values", fontsize=10)
        plt.savefig(os.path.join(result_folder, 'data_sample_img.pdf'))
        mlflow.log_artifact(os.path.join(result_folder, 'data_sample_img.pdf'))
    else:
        data_file = 'traffic.csv'
        df = pd.read_csv(os.path.join(data_folder, data_file), header=None)
        df = df[1:].astype('float').T
        
        # Data Sample img #
        plt.figure(figsize=(20,3))
        plt.plot(df.iloc[:5, :1000].T, alpha=.3)
        plt.title("5 Samples", fontsize=15)
        plt.xlabel("Time", fontsize=10)
        plt.ylabel("Values", fontsize=10)
        plt.savefig(os.path.join(result_folder, 'data_sample_img.pdf'))
        mlflow.log_artifact(os.path.join(result_folder, 'data_sample_img.pdf'))
    
    ### Make Ddataset and Loader###
    if cfg.task == 'prediction':
        batch_size = 64
        num_workers = 4
        pin_memory = True
        num_class = cfg.num_class
        out_shape = cfg.out_shape
        device = 'cuda'
        num_epoch = cfg.num_epoch
        lr = cfg.lr
        train_ratio = 0.6
        valid_ratio = 0.2
        window = cfg.window
        horizon = 1
        pred_step = cfg.pred_step
        normalize = 2
        
        print("Make Dataset and Loader ... ")
        #file_name, train, valid, horizon, window, normalize
        loader = DataLoaderH(os.path.join(data_folder, 'traffic.txt'), train_ratio, valid_ratio, horizon, window, pred_step, normalize)
        
    else:
        # Stratify and make Dataset to Train/Valid/Test and Loader #
        x_valid, x_test, y_valid, y_test = train_test_split(df_test.iloc[:,:-1],df_test.iloc[:,-1], test_size=0.5, shuffle=True, stratify=df_test.iloc[:,-1], random_state=SEED)
        df_valid = pd.DataFrame(x_valid)
        df_valid.insert(len(df_valid.columns), "target", pd.Series(y_valid))

        df_test = pd.DataFrame(x_test)
        df_test.insert(len(df_test.columns), "target", pd.Series(y_test))
        
        print("Make Dataset ... ")
        if cfg.task == 'classification' or cfg.task == 'classirecon':
            train_dataset = classification_MyDataset(df_train.values)
            valid_dataset = classification_MyDataset(df_valid.values)
            test_dataset = classification_MyDataset(df_test.values)
        elif cfg.task == 'reconstruction':
            train_dataset = recon_MyDataset(df_train.values)
            valid_dataset = recon_MyDataset(df_valid.values)
            test_dataset = recon_MyDataset(df_test.values)

        ### Make Dataset to Loader ###
        print("Make Loader ... ")
        batch_size = 64
        num_workers = 4
        pin_memory = True
        num_class = cfg.num_class
        out_shape = cfg.out_shape
        device = 'cuda'
        num_epoch = cfg.num_epoch
        lr = cfg.lr

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                                 drop_last=False, 
                                                 num_workers=num_workers, pin_memory=pin_memory, shuffle=True) 
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, 
                                                 drop_last=False, 
                                                 num_workers=num_workers, pin_memory=pin_memory, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                                                 drop_last=False, 
                                                 num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

    ### Model Initialization ###
    print("Initialize Model...")
    if cfg.model=='CNN':
        model = DRCNN(in_features= cfg.in_shape,
                      num_class = num_class,
                      out_features=out_shape,
                      kernel_size=cfg.kernel_size, hidden_dim = 256, capsule_dim=8, cfg=cfg).to(device) 
    elif cfg.model == 'DR':
        model = CapsuleNet([cfg.in_shape, cfg.window,1], cfg.num_class, cfg.num_routing, cfg).to(device) 
    else:
        raise ValueError(f"Passed undefined model : {cfg.model}")

    
    mlflow.log_param('num_parameters', count_parameters(model))
    log_params_from_omegaconf_dict(cfg)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    ### Pick Loss ###
    if cfg.loss == 'ce':
        y_train = df_train.iloc[:, -1]
        class_weight_vec1 = class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(y_train), y=y_train.values)
        weight1 = torch.Tensor(np.unique(class_weight_vec1)).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight1) 
    elif cfg.loss == 'margin':
        criterion = caps_loss
    elif cfg.loss == 'mse':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Passed undefined loss : {cfg.loss}")
    
    ### Model Training and Evaluation ###
    if cfg.task == 'classification':
        # training #
        print("training...")
        #model, optimizer, criterion, num_epoch, train_loader, valid_loader, result_folder, device, cfg
        classification_train(model, optimizer, criterion, num_epoch, train_loader, valid_loader, result_folder, device, cfg)

        checkpoint = torch.load(os.path.join(result_folder, f'{cfg.model}-kernel{cfg.kernel_size}-{cfg.task}-best.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        criterion = checkpoint['criterion']
        best_epoch, best_loss, best_acc, best_f1 = checkpoint['epoch'], checkpoint['loss'], checkpoint['acc'], checkpoint['f1']
        print(f'Best Model: epoch-{best_epoch} | loss-{best_loss} | acc-{best_acc} | f1-{best_f1}')
        mlflow.pytorch.log_model(model, "model")

        # Performance Evaluation #   
        model = model.eval()
        print("Visualize Feature Map...")
        #model, batch_num, sample_num, test_loader, result_folder, attack, device
        classification_feature_map(model, 0, 0, test_loader, result_folder, None, device)
        
        print("Performance Evaluation...")
        #model, criterion, test_loader, attack, result_folder, cfg, report, device
        report = classification_performance_evaluation(model, criterion, test_loader, None, result_folder, cfg, report, device)
        true_label = None
        
    elif cfg.task == 'reconstruction':
        # training #
        print("training...")
        #model, optimizer, criterion, num_epoch, train_loader, valid_loader, result_folder, device, cfg
        recon_train(model, optimizer, criterion, num_epoch, train_loader, valid_loader, result_folder, device, cfg)
        
        checkpoint = torch.load(os.path.join(result_folder, f'{cfg.model}-kernel{cfg.kernel_size}-{cfg.task}-best.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        criterion = checkpoint['criterion']
        best_epoch, best_loss = checkpoint['epoch'], checkpoint['loss']
        print(f'Best Model: epoch-{best_epoch} | loss-{best_loss}')
        mlflow.pytorch.log_model(model, "model")
        
        # Performance Evaluation #   
        model = model.eval()
        print("Visualize Feature Map...")
        #model, batch_num, sample_num, test_loader, result_folder, attack, device
        recon_feature_map(model, 0, 0, test_loader, result_folder, None, device)
        
        print("Performance Evaluation...")
        #model, criterion, test_loader, attack, result_folder, cfg, report, true_label, device
        report, true_label = recon_performance_evaluation(model, criterion, test_loader, None, result_folder, cfg, report, None, device)
        
    elif cfg.task == 'prediction':
        # training #
        print("training...")
        data = loader
        train_loader = data.train
        valid_loader = data.valid
        
        #model, optimizer, criterion, num_epoch, data, train_loader, valid_loader, result_folder, device, cfg
        pred_train(model, optimizer, criterion, num_epoch, data, train_loader, valid_loader, result_folder, device, cfg)
        
        checkpoint = torch.load(os.path.join(result_folder, f'{cfg.model}-kernel{cfg.kernel_size}-{cfg.task}-best.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        criterion = checkpoint['criterion']
        best_epoch, best_loss = checkpoint['epoch'], checkpoint['loss']
        print(f'Best Model: epoch-{best_epoch} | loss-{best_loss}')
        mlflow.pytorch.log_model(model, "model")
        
        # Performance Evaluation #   
        model = model.eval()
        test_loader = data.test
        print("Visualize Feature Map...")
        #model, batch_num, sample_num, data, test_loader, result_folder, attack, device
        pred_feature_map(model, 0, 0, data, test_loader, result_folder, None, device, cfg)
        
        print("Performance Evaluation...")
        #model, criterion, data, test_loader, attack, result_folder, cfg, report, true_label, device
        report, true_label = pred_performance_evaluation(model, criterion, data, test_loader, None, result_folder, cfg, report, None, device)
    
    else:
        # training #
        print("training...")
        #model, optimizer, criterion, num_epoch, train_loader, valid_loader, result_folder, device, cfg
        dr_train(model, optimizer, criterion, num_epoch, train_loader, valid_loader, result_folder, device, cfg)
        
        checkpoint = torch.load(os.path.join(result_folder, f'{cfg.model}-kernel{cfg.kernel_size}-{cfg.task}-best.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        criterion = checkpoint['criterion']
        best_epoch, best_loss, best_acc, best_f1 = checkpoint['epoch'], checkpoint['loss'], checkpoint['acc'], checkpoint['f1']
        print(f'Best Model: epoch-{best_epoch} | loss-{best_loss} | acc-{best_acc} | f1-{best_f1}')
        mlflow.pytorch.log_model(model, "model")
        
        # Performance Evaluation #   
        model = model.eval()
        print("Visualize Feature Map...")
        #model, batch_num, sample_num, test_loader, result_folder, attack, device
        dr_feature_map(model, 0, 0, test_loader, result_folder, None, device)
        
        print("Performance Evaluation...")
        #model, criterion, test_loader, attack, result_folder, cfg, report, true_label, device
        report, true_label = dr_performance_evaluation(model, criterion, test_loader, None, result_folder, cfg, report, None, device)

    ### Attack Dataset ###
    print("Make Attack Dataset and Loader ...")
    if cfg.task == 'prediction':
        X_test, Y_test = data.test
        loader = []
        for attack in tqdm(['offset', 'increase', 'decrease', 'forward', 'backward'], leave=False):
            with open(os.path.join(data_folder, f'traffic_{attack}Attack.pkl'), 'rb') as f:
                data_ = pickle.load(f)
            X_test_attack = data_['data'].to(device)

            attack_loader = [X_test_attack, Y_test]
            loader.append(attack_loader)
        
        eps = [0.0, 0.05]
        fgsm_test_loader = []
        eps_list = []
        #for eps in tqdm(np.arange(eps[0], eps[1]+0.01, 0.01), leave=False): 
        for eps in [0.05]:
            fgsm_x = []
            for (x, y) in tqdm(data.get_batches(X_test, Y_test, 64, False), leave=False):
                attack_x = fgsm_attack(model, criterion, x, y, eps, device, cfg)
                fgsm_x.append(attack_x)
            fgsm_x = torch.cat(fgsm_x, dim=0)
            fgsm_test_loader.append([fgsm_x, Y_test])
            eps_list.append(eps)
        loader.append(fgsm_test_loader)
        
    else:
        offsets, offset_idx = offset(df_test, offset=0.1, window=55, num_sample=df_test.shape[0], 
                             max_rand_noise=0.01, label_last_col=True)
        increasing, increase_idx = drift(df_test, scale=0.2, window=55, num_sample=df_test.shape[0], type_='increasing',
                                     max_rand_noise=0.01, label_last_col=True)
        decreasing, decrease_idx = drift(df_test, scale=0.2, window=55, num_sample=df_test.shape[0], type_='decreasing',
                                     max_rand_noise=0.01, label_last_col=True)
        backward, backward_idx = lagging(df_test, lag_time=10, num_sample=df_test.shape[0], type_="backward", 
                                     max_rand_noise=0.01, label_last_col=True)
        forward, forward_idx   = lagging(df_test, lag_time=10, num_sample=df_test.shape[0], type_="forward", 
                                     max_rand_noise=0.01, label_last_col=True)
        
        if cfg.task == 'classification' or cfg.task == 'classirecon':
            offset_test_dataset = classification_MyDataset(offsets.values)
            increasing_test_dataset = classification_MyDataset(increasing.values)
            decreasing_test_dataset = classification_MyDataset(decreasing.values)
            backward_test_dataset = classification_MyDataset(backward.values)
            forward_test_dataset = classification_MyDataset(forward.values)
            
            eps = [0.0, 0.05]
            #model, criterion, test_loader, eps, cfg, device
            fgsm_data, eps_list = fgsm_makeData(model, criterion, test_loader, eps, cfg, device)
            fgsm_test_dataset = []                 
            for i in range(len(fgsm_data)):
                fgsm_dataset = classification_MyDataset(fgsm_data[i].values)
                fgsm_test_dataset.append(fgsm_dataset)
                             
        elif cfg.task == 'reconstruction':
            offset_test_dataset = recon_MyDataset(offsets.values)
            increasing_test_dataset = recon_MyDataset(increasing.values)
            decreasing_test_dataset = recon_MyDataset(decreasing.values)
            backward_test_dataset = recon_MyDataset(backward.values)
            forward_test_dataset = recon_MyDataset(forward.values)
            
            eps = [0.0, 0.05]
            #model, criterion, test_loader, eps, cfg, device
            fgsm_data, eps_list = fgsm_makeData(model, criterion, test_loader, eps, cfg, device)
            fgsm_test_dataset = []                 
            for i in range(len(fgsm_data)):
                fgsm_dataset = recon_MyDataset(fgsm_data[i].values, 'fgsm')
                fgsm_test_dataset.append(fgsm_dataset)
            
        offset_test_loader = torch.utils.data.DataLoader(offset_test_dataset, batch_size=batch_size, 
                                                 drop_last=False, 
                                                 num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

        increasing_test_loader = torch.utils.data.DataLoader(increasing_test_dataset, batch_size=batch_size, 
                                                 drop_last=False, 
                                                 num_workers=num_workers, pin_memory=pin_memory, shuffle=False)
        decreasing_test_loader = torch.utils.data.DataLoader(decreasing_test_dataset, batch_size=batch_size, 
                                                 drop_last=False, 
                                                 num_workers=num_workers, pin_memory=pin_memory, shuffle=False)
        backward_test_loader = torch.utils.data.DataLoader(backward_test_dataset, batch_size=batch_size, 
                                                 drop_last=False, 
                                                 num_workers=num_workers, pin_memory=pin_memory, shuffle=False)
        forward_test_loader = torch.utils.data.DataLoader(forward_test_dataset, batch_size=batch_size, 
                                                 drop_last=False, 
                                                 num_workers=num_workers, pin_memory=pin_memory, shuffle=False)
        fgsm_test_loader = []                 
        for i in range(len(fgsm_test_dataset)):
            fgsm_loader = torch.utils.data.DataLoader(fgsm_test_dataset[i], batch_size=batch_size, 
                                             drop_last=False, 
                                             num_workers=num_workers, pin_memory=pin_memory, shuffle=False)
            fgsm_test_loader.append(fgsm_loader)
        loader = [offset_test_loader, increasing_test_loader, decreasing_test_loader, forward_test_loader, backward_test_loader, fgsm_test_loader]
        data = None
        
    ### Attack Performance ###
    print("Attack Inference ...")
    #cfg, model, criterion, result_folder,  report, loader, eps_list, device
    report = attack_inference(cfg, model, criterion, result_folder,  report, loader, eps_list, device, true_label, data)
        
    report.T.to_csv(os.path.join(result_folder, f'{cfg.model}_{cfg.task}_{cfg.data}_report.csv'))
    mlflow.log_artifact(os.path.join(result_folder, f'{cfg.model}_{cfg.task}_{cfg.data}_report.csv'))
    
    print("Finished!!")
                       
    mlflow.end_run()
    
if __name__ == "__main__":
    main()