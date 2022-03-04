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
import torch.nn.functional as F

from model.ClassificationModule import *
from model.ReconstructionModule import *

import mlflow

SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

### Model Training ###
def dr_train(model, optimizer, criterion, num_epoch, train_loader, valid_loader, result_folder, device, cfg):
    train_loss_list = []
    train_acc_list = []
    train_f1_list = []
    valid_loss_list = []
    valid_acc_list = []
    valid_f1_list = []
    
    for epoch in range(0, num_epoch+1):
        model = model.train()

        train_loss = []
        predictions = []
        answers = []
        prob= []
        for (x, y) in tqdm(train_loader, leave=False):
            x = x.to(device)
            y = y.to(device)            
            answers.extend(y.squeeze().detach().cpu().numpy())
            
            # change to one-hot 
            y_onehot = torch.zeros(y.size(0), cfg.num_class).scatter_(1, torch.Tensor(y.float().view(-1,1).detach().cpu()).type(torch.int64), 1.) 
            y_onehot = y_onehot.to(device)

            optimizer.zero_grad()
            classes, recon = model(x)
            
            if cfg.loss == 'margin':
                loss = criterion(y_onehot, classes, x, recon, 0.0005*cfg.window)
            else:
                loss = criterion(classes.squeeze(), y.squeeze()) + 0.0005*cfg.window*nn.MSELoss()(x.squeeze(),recon.squeeze())
            loss.backward()
            optimizer.step()
            
            prob.extend(classes.detach().cpu().numpy())
            predictions.extend(torch.max(classes,1)[1].detach().cpu().numpy())

            train_loss.append(loss.item())

        train_loss = np.mean(train_loss)
        train_acc = accuracy_score(answers, predictions)
        train_f1 = np.mean(f1_score(answers, predictions, average=None))

        model = model.eval()
        #reconstruction = [] # reconstruction
        predictions = [] # classification
        prob= []
        answers = [] # recon target
        #labels = [] # pred target 
        valid_loss = []
        with torch.no_grad():
            for (x, y) in tqdm(valid_loader, leave=False):
                x = x.to(device)
                y = y.to(device)

                answers.extend(y.squeeze().detach().cpu().numpy())
                #labels.extend(x.detach().cpu().numpy())

                y_onehot = torch.zeros(y.size(0), cfg.num_class).scatter_(1, torch.Tensor(y.float().view(-1,1).detach().cpu()).type(torch.int64), 1.)
                y_onehot = y_onehot.to(device)

                classes, recon = model(x)

                if cfg.loss == 'margin':
                    loss = criterion(y_onehot, classes, x, recon, 0.0005*cfg.window)
                else:
                    loss = criterion(classes.squeeze(), y.squeeze()) + 0.0005*cfg.window*nn.MSELoss()(x.squeeze(),recon.squeeze())

                #reconstruction.extend(recon.squeeze().detach().cpu().numpy())
                predictions.extend(torch.max(classes,1)[1].detach().cpu().numpy())
                prob.extend(classes.detach().cpu().numpy())
                valid_loss.append(loss.item())   

        valid_loss = np.mean(valid_loss)
        valid_acc = accuracy_score(answers, predictions)
        valid_f1 = np.mean(f1_score(answers, predictions, average=None))

        print("epoch: {}/{} | trn: {:.4f} / {:.4f} / {:.4f} | val: {:.4f} / {:.4f} / {:.4f}".format(
                    epoch, num_epoch, train_loss, train_acc, train_f1, valid_loss, valid_acc, valid_f1
                ))
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_acc", train_acc, step=epoch)
        mlflow.log_metric("train_acc", train_f1, step=epoch)
        mlflow.log_metric("valid_loss", valid_loss, step=epoch)
        mlflow.log_metric("valid_acc", valid_acc, step=epoch)
        mlflow.log_metric("valid_acc", valid_f1, step=epoch)
        
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)
        train_f1_list.append(train_f1)
        valid_f1_list.append(valid_f1)

        if epoch % 10 ==0 :
            torch.save({
                'epoch': epoch,
                'loss' : valid_loss_list[-1],
                'acc' : valid_acc_list[-1],
                'f1' : valid_f1_list[-1],
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'criterion' : criterion
            }, os.path.join(result_folder, f'{cfg.model}-kernel{int(cfg.kernel_size)}-{cfg.task}-{epoch}.pt'))

        if (epoch==0) or (epoch>0 and (min(valid_loss_list[:-1])>valid_loss_list[-1])):
            torch.save({
                'epoch': epoch,
                'loss' : valid_loss_list[-1],
                'acc' : valid_acc_list[-1],
                'f1' : valid_f1_list[-1],
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
def dr_feature_map(model, batch_num, sample_num, test_loader, result_folder, attack, device):
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if i != batch_num:
                pass
            else:
                x = x.to(device)
                y = y.to(device)

                plt.figure(figsize=(10,3))
                plt.plot(x[sample_num,:,:].squeeze().detach().cpu().numpy())
                plt.title(f"Sample{batch_num*64+sample_num} - Class:"+str(y[sample_num].item()), fontsize=20)
                plt.savefig(os.path.join(result_folder, 'feature_map_sample.pdf'))
                mlflow.log_artifact(os.path.join(result_folder, 'feature_map_sample.pdf'))

                class_num = int(y[0].item())

                classes, recon = model(x, visualize=True, num=sample_num, class_num=class_num, result_folder=result_folder)
                break

### Performance Evaluation ###
def MAPE(y_true, y_pred):
    return torch.mean(torch.abs((y_true - y_pred) / (y_true+1e-10))) * 100

def dr_inference(model, criterion, test_loader, attack, true_label, device, cfg):
    model = model.eval()
    performance = []
    
    test_loss = []
    mse_loss = []
    rmse_loss = []
    mae_loss = []
    mape_loss = []
    
    labels = []
    predictions = []
    prob = []
    
    answers = []
    reconstruction = []
    
    with torch.no_grad():
        for (x, y) in tqdm(test_loader, leave=False):
            x = x.to(device)
            y = y.to(device)
            
            labels.extend(y.squeeze().cpu().numpy())
            answers.extend(x.squeeze().detach().cpu().numpy())

            y_onehot = torch.zeros(y.size(0), cfg.num_class).scatter_(1, torch.Tensor(y.float().view(-1,1).detach().cpu()).type(torch.int64), 1.)
            y_onehot = y_onehot.to(device)

            classes, recon = model(x)
            
            if cfg.loss == 'margin':
                loss = criterion(y_onehot, classes, x, recon, 0.0005*cfg.window)
            else:
                loss = criterion(classes.squeeze(), y.squeeze()) + 0.0005*cfg.window*nn.MSELoss()(x.squeeze(),recon.squeeze())
            
            x = x.squeeze()
            recon = recon.squeeze()
            mse = nn.MSELoss()(recon, x) 
            rmse = torch.sqrt(mse)
            mae = nn.L1Loss()(recon, x)
            mape = MAPE(x, recon)

            reconstruction.extend(recon.squeeze().detach().cpu().numpy())
            predictions.extend(torch.max(classes,1)[1].detach().cpu().numpy())
            prob.extend(classes.detach().cpu().numpy())

            test_loss.append(loss.item())
            mse_loss.append(mse.item())
            rmse_loss.append(rmse.item())
            mae_loss.append(mae.item())
            mape_loss.append(mape.item())
            
        test_loss = np.mean(test_loss)
        performance.append(test_loss)
        
        test_acc = accuracy_score(labels, predictions)
        performance.append(test_acc)
        
        test_f1macro = f1_score(labels, predictions, average='macro')
        performance.append(test_f1macro)
        test_f1micro = f1_score(labels, predictions, average='micro')
        performance.append(test_f1micro)
        test_f1weight = f1_score(labels, predictions, average='weighted')
        performance.append(test_f1weight)
        test_f1mean = np.mean(f1_score(labels, predictions, average=None))
        performance.append(test_f1mean)
        
        if attack == None:  
            mlflow.log_metric("test_loss", test_loss, step=-1)
            mlflow.log_metric("test_acc", test_acc, step=-1)

            mlflow.log_metric("test_f1macro", test_f1macro, step=-1)
            mlflow.log_metric("test_f1micro", test_f1micro, step=-1)
            mlflow.log_metric("test_f1weight", test_f1weight, step=-1)
            mlflow.log_metric("test_f1mean", test_f1mean, step=-1)
            
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
            
            print(f"test Loss {test_loss} / Acc {test_acc} / F1score {test_f1mean} / mse {mse_loss}")   
        else:
            mlflow.log_metric(f"{attack}_loss", test_loss, step=-1)
            mlflow.log_metric(f"{attack}_acc", test_acc, step=-1)

            mlflow.log_metric(f"{attack}_f1macro", test_f1macro, step=-1)
            mlflow.log_metric(f"{attack}_f1micro", test_f1micro, step=-1)
            mlflow.log_metric(f"{attack}_test_f1weight", test_f1weight, step=-1)
            mlflow.log_metric(f"{attack}_test_f1mean", test_f1mean, step=-1)
            
            mse_loss = np.mean(np.square(reconstruction - true_label.values))
            performance.append(mse_loss)
            rmse_loss = np.sqrt(mse_loss)
            performance.append(rmse_loss)
            mae_loss = np.mean(np.abs(reconstruction - true_label.values))
            performance.append(mae_loss)
            mape_loss = np.mean(np.abs(reconstruction - true_label.values) / (true_label.values+1e-10))*100
            performance.append(mape_loss)
            mlflow.log_metric(f"{attack}_mseloss", mse_loss, step=-1)
            mlflow.log_metric(f"{attack}_rmseloss", rmse_loss, step=-1)
            mlflow.log_metric(f"{attack}_maeloss", mae_loss, step=-1)
            mlflow.log_metric(f"{attack}_mapeloss", mape_loss, step=-1)
            
            print(f"{attack} Loss {test_loss} / Acc {test_acc} / F1score {test_f1mean} / mse {mse_loss}")   
        
        return labels, predictions, prob, answers, reconstruction, performance
                
def dr_performance_evaluation(model, criterion, test_loader, attack, result_folder, cfg, report, true_label, device):
    labels, predictions, prob, answers, recon, perf = dr_inference(model, criterion, test_loader, attack, true_label, device, cfg)
    
    if attack == None:
        performance = pd.DataFrame(perf, columns=['test'], index=['loss', 'acc', 'f1macro', 'f1micro', 'f1weight', 'f1mean']+['mse', 'rmse', 'mae', 'mape'])
    else:
        performance = pd.DataFrame(perf, columns=[attack], index=['loss', 'acc', 'f1macro', 'f1micro', 'f1weight', 'f1mean']+['mse', 'rmse', 'mae', 'mape'])
        
    report = pd.concat([report, performance], axis=1)
        
    plot_confusion_matrix(labels, predictions, attack, True, result_folder)
    plot_confusion_matrix(labels, predictions, attack, False, result_folder)
    
    result_label = pd.DataFrame([labels,predictions])
    result_pred = pd.DataFrame(prob)
    result_ans = pd.DataFrame(answers)
    result_recon = pd.DataFrame(recon)
    
    recon_graph(result_ans, result_recon, result_folder, attack, true_label)
    
    if attack == None:
        result_label.to_csv(os.path.join(result_folder, f'{cfg.model}_{cfg.task}.csv'), index=0)
        result_pred.to_csv(os.path.join(result_folder, f'{cfg.model}_{cfg.task}_prob.csv'), index=0)
        mlflow.log_artifact(os.path.join(result_folder, f'{cfg.model}_{cfg.task}.csv'))
        mlflow.log_artifact(os.path.join(result_folder, f'{cfg.model}_{cfg.task}_prob.csv'))
        
        result_ans.to_csv(os.path.join(result_folder, f'{cfg.model}_{cfg.task}_ans.csv'), index=0)
        result_recon.to_csv(os.path.join(result_folder, f'{cfg.model}_{cfg.task}_recon.csv'), index=0)
        mlflow.log_artifact(os.path.join(result_folder, f'{cfg.model}_{cfg.task}_ans.csv'))
        mlflow.log_artifact(os.path.join(result_folder, f'{cfg.model}_{cfg.task}_recon.csv'))
    else:
        result_label.to_csv(os.path.join(result_folder, f'{cfg.model}_{cfg.task}_{attack}.csv'), index=0)
        result_pred.to_csv(os.path.join(result_folder, f'{cfg.model}_{cfg.task}_{attack}_prob.csv'), index=0)
        mlflow.log_artifact(os.path.join(result_folder, f'{cfg.model}_{cfg.task}_{attack}.csv'))
        mlflow.log_artifact(os.path.join(result_folder, f'{cfg.model}_{cfg.task}_{attack}_prob.csv'))
        
        result_ans.to_csv(os.path.join(result_folder, f'{cfg.model}_{cfg.task}_{attack}_ans.csv'), index=0)
        result_recon.to_csv(os.path.join(result_folder, f'{cfg.model}_{cfg.task}_{attack}_recon.csv'), index=0)
        mlflow.log_artifact(os.path.join(result_folder, f'{cfg.model}_{cfg.task}_{attack}_ans.csv'))
        mlflow.log_artifact(os.path.join(result_folder, f'{cfg.model}_{cfg.task}_{attack}_recon.csv'))
        
    return report, result_ans