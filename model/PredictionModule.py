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
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F

import mlflow

SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

### Prediction training ###
def pred_train(model, optimizer, criterion, num_epoch, data, train_loader, valid_loader, result_folder, device, cfg):
    train_loss_list = []
    valid_loss_list = []
    
    X_train, Y_train = train_loader
    X_valid, Y_valid = valid_loader
    
    for epoch in range(0, num_epoch+1):
        model = model.train()

        train_loss = []
        for (x, y) in tqdm(data.get_batches(X_train, Y_train, 64, False), leave=False):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
         
            if cfg.model == 'DR':
                _, outputs = model(x)
            else:
                outputs = model(x)
            
            outputs = outputs.squeeze()
            y = y.squeeze()

            scale = data.scale.expand(outputs.size(0), data.m)
            bias = data.bias.expand(outputs.size(0), data.m)

            loss = torch.sqrt(criterion(outputs*scale+bias, y*scale+bias))

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        train_loss = np.mean(train_loss)

        model = model.eval()
        predictions = []
        answers = []
        valid_loss = []
        with torch.no_grad():
            for (x, y) in tqdm(data.get_batches(X_valid, Y_valid, 64, False), leave=False):
                x = x.to(device)
                y = y.to(device)
                
                if cfg.model == 'DR':
                    _, outputs = model(x)
                else:
                    outputs = model(x)
                    
                outputs = outputs.squeeze()
                y = y.squeeze()

                scale = data.scale.expand(outputs.size(0), data.m)
                bias = data.bias.expand(outputs.size(0), data.m)

                loss = torch.sqrt(criterion(outputs*scale+bias, y*scale+bias))

                predictions.extend((outputs*scale+bias).squeeze().detach().cpu().numpy())
                answers.extend((y*scale+bias).detach().cpu().numpy())
                valid_loss.append(loss.item())    

        valid_loss = np.mean(valid_loss)

        print("epoch: {}/{} | trn_loss: {:.4f} | val_loss: {:.4f}".format(
                    epoch, num_epoch, train_loss, valid_loss
                ))
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("valid_loss", valid_loss, step=epoch)

        if (epoch==0) or (epoch>0 and (min(valid_loss_list[:-1])>valid_loss_list[-1])):
            torch.save({
                'epoch': epoch,
                'loss': valid_loss_list[-1],
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'criterion' : criterion
            }, os.path.join(result_folder, f'{cfg.model}-kernel{int(cfg.kernel_size)}-{cfg.task}-best.pt'))
            
    fig, ax = plt.subplots(1,2,figsize=(20,5))
    ax0 = ax[0]
    ax0.plot(train_loss_list, c= 'blue')
    ax0.plot(valid_loss_list, c='red')

    ax1 = ax[1]
    ax1.plot(valid_loss_list, c='red', marker='o')

    fig.suptitle("Loss", fontsize=20)
    plt.savefig(os.path.join(result_folder, 'loss_graph.pdf'))
    mlflow.log_artifact(os.path.join(result_folder, 'loss_graph.pdf'))
    
    loss_df = pd.DataFrame([train_loss_list, valid_loss_list])
    loss_df.to_csv(os.path.join(result_folder, 'loss.csv'), index=0)
    mlflow.log_artifact(os.path.join(result_folder, 'loss.csv'))
    
### Visualize Feature Map ###
def pred_feature_map(model, batch_num, sample_num, data, test_loader, result_folder, attack, device, cfg):
    X_test, Y_test = test_loader
    with torch.no_grad():
        for i, (x, y) in enumerate(data.get_batches(X_test, Y_test, 64, False)):
            if i != batch_num:
                pass
            else:
                x = x.to(device)

                plt.figure(figsize=(20,3))
                plt.plot(x[sample_num,:,0].detach().cpu().numpy())
                plt.title(f"Sample {batch_num*64+sample_num}", fontsize=20)
                plt.savefig(os.path.join(result_folder, 'feature_map_sample.pdf'))
                mlflow.log_artifact(os.path.join(result_folder, 'feature_map_sample.pdf'))
                
                if cfg.model == 'DR':
                    _, outputs = model(x, y=None, visualize=True, num=sample_num, class_num=0, attack=attack, result_folder=result_folder)
                else:
                    outputs = model(x, True, attack, sample_num, result_folder).squeeze()
                break
                
### Performance Evaluation ###
def MAPE(y_true, y_pred):
    return torch.mean(torch.abs((y_true - y_pred) / (y_true+1e-10))) * 100

def pred_inference(model, criterion, data, test_loader, attack, true_label, device, cfg):    
    model = model.eval()
    
    X_test, Y_test = test_loader
    
    performance = []
    mse_loss = []
    rmse_loss = []
    mae_loss = []
    mape_loss = []
    
    answers = []
    predictions = []
    with torch.no_grad():
        for (x, y) in tqdm(data.get_batches(X_test, Y_test, 64, False), leave=False):
            x = x.to(device)
            y = y.to(device)

            if cfg.model == 'DR':
                _, outputs = model(x)
            else:
                outputs = model(x)
                
            outputs = outputs.squeeze()
            y = y.squeeze()
            
            scale = data.scale.expand(outputs.size(0), data.m).squeeze()
            bias = data.bias.expand(outputs.size(0), data.m).squeeze()
            
            outputs = outputs*scale+bias
            y = y*scale+bias
            
            mse = criterion(outputs, y) 
            rmse = torch.sqrt(criterion(outputs, y))
            mae = nn.L1Loss()(outputs, y)
            mape = MAPE(y, outputs)      

            predictions.extend(outputs.squeeze().detach().cpu().numpy())  
            answers.extend(y.squeeze().detach().cpu().numpy())
            
            mse_loss.append(mse.item())
            rmse_loss.append(rmse.item())
            mae_loss.append(mae.item())
            mape_loss.append(mape.item())
    
    if attack == None:
        mse_loss = np.mean(mse_loss)
        performance.append(mse_loss)
        rmse_loss = np.mean(rmse_loss)
        performance.append(rmse_loss)
        mae_loss = np.mean(mae_loss)
        performance.append(mae_loss)
        mape_loss = np.mean(mape_loss)
        performance.append(mape_loss)
        mlflow.log_metric("test_mseloss", mse_loss, step=-1)
        mlflow.log_metric("test_rmseloss", rmse_loss, step=-1)
        mlflow.log_metric("test_maeloss", mae_loss, step=-1)
        mlflow.log_metric("test_mapeloss", mape_loss, step=-1)
        print(f'test Loss {mse_loss}')
    else:
        mse_loss = np.mean(np.square(predictions - true_label.values))
        performance.append(mse_loss)
        rmse_loss = np.sqrt(mse_loss)
        performance.append(rmse_loss)
        mae_loss = np.mean(np.abs(predictions - true_label.values))
        performance.append(mae_loss)
        mape_loss = np.mean(np.abs(predictions - true_label.values) / (true_label.values+1e-10))*100
        performance.append(mape_loss)
        mlflow.log_metric(f"{attack}_mseloss", mse_loss, step=-1)
        mlflow.log_metric(f"{attack}_rmseloss", rmse_loss, step=-1)
        mlflow.log_metric(f"{attack}_maeloss", mae_loss, step=-1)
        mlflow.log_metric(f"{attack}_mapeloss", mape_loss, step=-1)
        print(f'{attack} test Loss {mse_loss}')
    
    return answers, predictions, performance

### Prediction Result Graph ###
def pred_graph(result_ans, result_pred, result_folder, attack, true_label=None):
    for i in range(3):
        plt.figure(figsize=(20,3))
        if attack == None:
            plt.plot(result_ans.iloc[:1000, i], c='black', ls=':', label = 'GroundTruth')
        else:
            plt.plot(true_label.iloc[:1000, i], c='black', ls=':', label = 'GroundTruth')
            plt.plot(result_ans.iloc[:1000, i], c='blue', ls=':', label = f'{attack}')
        plt.plot(result_pred.iloc[:1000, i], c='red', label='Prediction')
        plt.legend()
        if attack == None:
            plt.savefig(os.path.join(result_folder, f'pred_{i}sample.pdf'))
            mlflow.log_artifact(os.path.join(result_folder, f'pred_{i}sample.pdf'))
        else:
            plt.savefig(os.path.join(result_folder, f'{attack}_pred_{i}sample.pdf'))
            mlflow.log_artifact(os.path.join(result_folder, f'{attack}_pred_{i}sample.pdf'))

### Prediction Performance Evaluation ###
def pred_performance_evaluation(model, criterion, data, test_loader, attack, result_folder, cfg, report, true_label, device):
    answers, pred, perf = pred_inference(model, criterion, data, test_loader, attack, true_label, device, cfg)
    if attack == None:
        performance = pd.DataFrame(perf, columns=['test'], index=['mse', 'rmse', 'mae', 'mape'])
    else:
        performance = pd.DataFrame(perf, columns=[attack], index=['mse', 'rmse', 'mae', 'mape'])
    report = pd.concat([report, performance], axis=1)
    
    result_ans = pd.DataFrame(answers)
    result_pred = pd.DataFrame(pred)

    pred_graph(result_ans, result_pred, result_folder, attack, true_label)
    
    if attack == None:
        result_ans.to_csv(os.path.join(result_folder, f'{cfg.model}_{cfg.task}_ans.csv'), index=0)
        result_pred.to_csv(os.path.join(result_folder, f'{cfg.model}_{cfg.task}_pred.csv'), index=0)
        mlflow.log_artifact(os.path.join(result_folder, f'{cfg.model}_{cfg.task}_ans.csv'))
        mlflow.log_artifact(os.path.join(result_folder, f'{cfg.model}_{cfg.task}_pred.csv'))
    else:
        result_ans.to_csv(os.path.join(result_folder, f'{cfg.model}_{cfg.task}_{attack}_ans.csv'), index=0)
        result_pred.to_csv(os.path.join(result_folder, f'{cfg.model}_{cfg.task}_{attack}_pred.csv'), index=0)
        mlflow.log_artifact(os.path.join(result_folder, f'{cfg.model}_{cfg.task}_{attack}_ans.csv'))
        mlflow.log_artifact(os.path.join(result_folder, f'{cfg.model}_{cfg.task}_{attack}_pred.csv'))
        
    return report, result_ans