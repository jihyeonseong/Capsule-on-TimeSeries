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

import mlflow

SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

### Model Training ###
def classification_train(model, optimizer, criterion, num_epoch, train_loader, valid_loader, result_folder, device, cfg):
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
            answers.extend(y.detach().cpu().numpy())

            optimizer.zero_grad()
            outputs = model(x).squeeze()

            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            prob.extend(torch.nn.Softmax()(outputs).detach().cpu().numpy())
            predictions.extend(torch.max(torch.nn.Softmax()(outputs),1)[1].detach().cpu().numpy())

            train_loss.append(loss.item())

        train_loss = np.mean(train_loss)
        train_acc = accuracy_score(answers, predictions)
        train_f1 = np.mean(f1_score(answers, predictions, average=None))

        model = model.eval()
        predictions = []
        answers = []
        valid_loss = []
        prob = []
        with torch.no_grad():
            for (x, y) in tqdm(valid_loader, leave=False):
                x = x.to(device)
                y = y.to(device)
                answers.extend(y.detach().cpu().numpy())

                outputs = model(x).squeeze()
                loss = criterion(outputs, y) 

                prob.extend(torch.nn.Softmax()(outputs).detach().cpu().numpy())
                predictions.extend(torch.max(torch.nn.Softmax()(outputs),1)[1].detach().cpu().numpy())
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
def classification_feature_map(model, batch_num, sample_num, test_loader, result_folder, attack, device):
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if i != batch_num:
                pass
            else:
                x = x.to(device)
                y = y.to(device)

                plt.figure(figsize=(10,3))
                plt.plot(x[sample_num,:,:].detach().cpu().numpy())
                plt.title("Sample {batch_num*64+sample_num} - Class:"+str(y[sample_num].item()), fontsize=20)
                plt.savefig(os.path.join(result_folder, 'feature_map_sample.pdf'))
                mlflow.log_artifact(os.path.join(result_folder, 'feature_map_sample.pdf'))

                outputs = model(x, True, attack, sample_num, result_folder).squeeze()
                break

### Performance Evaluation ###
def classification_inference(model, criterion, test_loader, attack, device):    
    model = model.eval()
    performance = []
    
    test_loss = []
    predictions = []
    answers = []
    prob = []
    
    with torch.no_grad():
        for (x, y) in tqdm(test_loader, leave=False):
            x = x.to(device)
            y = y.to(device)

            answers.extend(y.detach().cpu().numpy())

            outputs = model(x).squeeze()
            loss = criterion(outputs, y) #torch.reshape(y, (-1,1)

            prob.extend(torch.nn.Softmax()(outputs).detach().cpu().numpy())
            predictions.extend(torch.max(torch.nn.Softmax()(outputs),1)[1].detach().cpu().numpy())

            test_loss.append(loss.item())
            
        test_loss = np.mean(test_loss)
        performance.append(test_loss)
        
        test_acc = accuracy_score(answers, predictions)
        performance.append(test_acc)
        
        test_f1macro = f1_score(answers, predictions, average='macro')
        performance.append(test_f1macro)
        test_f1micro = f1_score(answers, predictions, average='micro')
        performance.append(test_f1micro)
        test_f1weight = f1_score(answers, predictions, average='weighted')
        performance.append(test_f1weight)
        test_f1mean = np.mean(f1_score(answers, predictions, average=None))
        performance.append(test_f1mean)
        
        if attack == None:  
            mlflow.log_metric("test_loss", test_loss, step=-1)
            mlflow.log_metric("test_acc", test_acc, step=-1)

            mlflow.log_metric("test_f1macro", test_f1macro, step=-1)
            mlflow.log_metric("test_f1micro", test_f1micro, step=-1)
            mlflow.log_metric("test_f1weight", test_f1weight, step=-1)
            mlflow.log_metric("test_f1mean", test_f1mean, step=-1)
            
            print(f"test Loss {test_loss} / Acc {test_acc} / F1score {test_f1mean}")   
        else:
            mlflow.log_metric(f"{attack}_loss", test_loss, step=-1)
            mlflow.log_metric(f"{attack}_acc", test_acc, step=-1)

            mlflow.log_metric(f"{attack}_f1macro", test_f1macro, step=-1)
            mlflow.log_metric(f"{attack}_f1micro", test_f1micro, step=-1)
            mlflow.log_metric(f"{attack}_test_f1weight", test_f1weight, step=-1)
            mlflow.log_metric(f"{attack}_test_f1mean", test_f1mean, step=-1)
            
            print(f"{attack} Loss {test_loss} / Acc {test_acc} / F1score {test_f1mean}")   
        
        return answers, predictions, prob, performance

### Confusion Matrix ###
def plot_confusion_matrix(y_label, y_pred, attack = None, normalized=True, result_folder=None):
    mat = confusion_matrix(y_label, y_pred)
    fig, ax = plt.subplots(figsize=(6,5))
    if normalized:
        cmn = mat.astype('float')
        cmn = cmn / mat.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cmn, annot=True, fmt='.2f', cmap='Blues')
    else:
        cmn = mat.astype('int')
        sns.heatmap(cmn, annot=True, cmap='Blues')
    plt.ylabel('Actual', fontsize=15)
    plt.xlabel('Predicted', fontsize=15)
    if normalized:
        plt.title("Confusion Matrix (%)", fontsize=20)
        if attack!= None:
            plt.savefig(os.path.join(result_folder, f'{attack}_confmatrix_norm.pdf'))
            mlflow.log_artifact(os.path.join(result_folder, f'{attack}_confmatrix_norm.pdf'))
        else:
            plt.savefig(os.path.join(result_folder, f'confmatrix_norm.pdf'))
            mlflow.log_artifact(os.path.join(result_folder, f'confmatrix_norm.pdf'))
    else:
        plt.title("Confusion Matrix", fontsize=20)
        if attack != None:
            plt.savefig(os.path.join(result_folder,f'{attack}_confmatrix_counts.pdf'))
            mlflow.log_artifact(os.path.join(result_folder,f'{attack}_confmatrix_counts.pdf'))
        else:
            plt.savefig(os.path.join(result_folder,f'confmatrix_counts.pdf'))
            mlflow.log_artifact(os.path.join(result_folder,f'confmatrix_counts.pdf'))

### Evaluation Whole Report ###
def classification_performance_evaluation(model, criterion, test_loader, attack, result_folder, cfg, report, device):
    answers, predictions, prob, perf = classification_inference(model, criterion, test_loader, attack, device)
    if attack == None:
        performance = pd.DataFrame(perf, columns=['test'], index=['loss', 'acc', 'f1macro', 'f1micro', 'f1weight', 'f1mean'])
    else:
        performance = pd.DataFrame(perf, columns=[attack], index=['loss', 'acc', 'f1macro', 'f1micro', 'f1weight', 'f1mean'])
    report = pd.concat([report, performance], axis=1)
    
    plot_confusion_matrix(answers, predictions, attack, True, result_folder)
    plot_confusion_matrix(answers, predictions, attack, False, result_folder)

    result_label = pd.DataFrame([answers,predictions])
    result_pred = pd.DataFrame(prob)
    
    if attack == None:
        result_label.to_csv(os.path.join(result_folder, f'{cfg.model}_{cfg.task}.csv'), index=0)
        result_pred.to_csv(os.path.join(result_folder, f'{cfg.model}_{cfg.task}_prob.csv'), index=0)
        mlflow.log_artifact(os.path.join(result_folder, f'{cfg.model}_{cfg.task}.csv'))
        mlflow.log_artifact(os.path.join(result_folder, f'{cfg.model}_{cfg.task}_prob.csv'))
    else:
        result_label.to_csv(os.path.join(result_folder, f'{cfg.model}_{cfg.task}_{attack}.csv'), index=0)
        result_pred.to_csv(os.path.join(result_folder, f'{cfg.model}_{cfg.task}_{attack}_prob.csv'), index=0)
        mlflow.log_artifact(os.path.join(result_folder, f'{cfg.model}_{cfg.task}_{attack}.csv'))
        mlflow.log_artifact(os.path.join(result_folder, f'{cfg.model}_{cfg.task}_{attack}_prob.csv'))
        
    return report