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

from sklearn import manifold
import umap
import copy

import mlflow

SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

### Common Layer Visaulzliation ###
def layer_visualize(x, num_filter, layer, attack=None, num=0, result_folder=None):
    x = x.detach().cpu()[num]
    try:
        r = int(math.sqrt(num_filter))
        c = int(math.sqrt(num_filter))
        x = x.reshape(r,c,1, -1)
        _, _, _, t = x.size()
    except:
        try:
            r = 8
            c = 10
            x = x.reshape(8,10,1,-1)
        except:
            r = 7
            c = 16
            x = x.reshape(7,16,1,-1)
        _, _, _, t = x.size()
    fig, ax = plt.subplots(r,c,dpi=150)
    for i in range(r):
        for j in range(c):
            ax[i, j].plot(x[i, j, :, :].T)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    fig.suptitle(layer+f"\n{num_filter}feature map (1box=1channel)", fontsize=15)
    fig.supxlabel(f"Time Length ({t})", fontsize=10)
    fig.supylabel("Values", fontsize=10)
    print(layer)
    if attack == None:
        plt.savefig(os.path.join(result_folder,f'{layer}_featuremap.pdf'))
        mlflow.log_artifact(os.path.join(result_folder, f'{layer}_featuremap.pdf'))
    else:
        plt.savefig(os.path.join(result_folder, f'{attack}_{layer}_featuremap.pdf'))
        mlflow.log_artifact(os.path.join(result_folder, f'{attack}_{layer}_featuremap.pdf'))
        
def visualize_weight(weight, num_filter, layer, classi, attack, result_folder=None):
    class_num, in_num_caps, out_dim_caps, in_dim_cpas = weight.size()
    tmp = weight[classi, :, :, :].detach().cpu()
    tmp = tmp.reshape(4,-1,out_dim_caps, in_dim_cpas)
    fig, ax = plt.subplots(4,4,dpi=150)
    for i in range(4):
        for j in range(4):
            sns.heatmap(tmp[i, j,:,:].T, ax = ax[i, j], cbar=False)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    fig.suptitle(layer+f"\nfeature heatmap - {classi}class (16capsules)", fontsize=15)
    fig.supxlabel("16 Out Capsule Dimension", fontsize=10)
    fig.supylabel("8 In Capsule Dimension", fontsize=10)
    if attack == None:
        plt.savefig(os.path.join(result_folder, f'dr_{layer}_heatmap.pdf'))
        mlflow.log_artifact(os.path.join(result_folder, f'dr_{layer}_heatmap.pdf'))
    else:
        plt.savefig(os.path.join(result_folder, f'dr_{attack}_{layer}_heatmap.pdf'))
        mlflow.log_artifact(os.path.join(result_folder, f'dr_{attack}_{layer}_heatmap.pdf'))        
            
    tmp = weight[:, 0, :, :].detach().cpu()
    tmp = tmp.reshape(class_num, out_dim_caps, in_dim_cpas)
    fig, ax = plt.subplots(2,3,dpi=150)
    cnt = 0
    for i in range(2):
        for j in range(3):
            if cnt == class_num:
                ax[i,j].set_xticks([])
                ax[i,j].set_yticks([])
                break
            sns.heatmap(tmp[cnt, :,:].T, ax = ax[i, j], cbar=False)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            cnt += 1
    fig.suptitle(layer+f"\nfeature heatmap - first capsule ({class_num} classes)", fontsize=15)
    fig.supxlabel("16 Out Capsule Dimension", fontsize=10)
    fig.supylabel("8 In Capsule Dimension", fontsize=10)
    print(layer)
    if attack == None:
        plt.savefig(os.path.join(result_folder, f'dr_{layer}_heatmap2.pdf'))
        mlflow.log_artifact(os.path.join(result_folder, f'dr_{layer}_heatmap2.pdf'))
    else:
        plt.savefig(attack_result_folder + f'dr_{attack}_{layer}_heatmap2.pdf')
        mlflow.log_artifact(os.path.join(result_folder, f'dr_{attack}_{layer}_heatmap2.pdf'))
    
def visualize_uhat(uhat, num_filter, layer, num, class_num, attack, result_folder=None):
    batch, out_num_caps, in_num_caps, out_dim_cpas = uhat.size()
    tmp = uhat[num, class_num, :, :].detach().cpu()
    r = int(math.sqrt(num_filter))
    c = int(math.sqrt(num_filter))

    tmp = tmp.reshape(r,-1,1, out_dim_cpas)
    fig, ax = plt.subplots(r,c,dpi=150)
    for i in range(r):
        for j in range(c):
            ax[i, j].plot(tmp[i, j, :, :].T)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    fig.suptitle(layer+f"\nfeature map - {class_num}class (16capsules)", fontsize=15)
    fig.supxlabel("16 Out Capsule Dimension", fontsize=10)
    fig.supylabel("Values", fontsize=10)
    if attack == None:
        plt.savefig(os.path.join(result_folder, f'dr_{layer}.pdf'))
        mlflow.log_artifact(os.path.join(result_folder, f'dr_{layer}.pdf'))
    else:
        plt.savefig(os.path.join(attack_result_folder, f'dr_{attack}_{layer}.pdf'))
        mlflow.log_artifact(os.path.join(result_folder, f'dr_{attack}_{layer}.pdf'))
          
    tmp = uhat[num, :, 0, :].detach().cpu()
    tmp = tmp.reshape(out_num_caps,1, out_dim_cpas)
    fig, ax = plt.subplots(2,3,dpi=150)
    cnt = 0
    for i in range(2):
        for j in range(3):
            if cnt == 5:
                ax[i,j].set_xticks([])
                ax[i,j].set_yticks([])
                break
            ax[i, j].plot(tmp[cnt, :,:].T)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            cnt += 1
    fig.suptitle(layer+f"\nfeature heatmap - first capsule (5 classes)", fontsize=15)
    fig.supxlabel("16 Out Capsule Dimension", fontsize=10)
    fig.supylabel("Values", fontsize=10)
    print(layer)
    if attack == None:
        plt.savefig(os.path.join(result_folder, f'dr_{layer}2.pdf'))
        mlflow.log_artifact(os.path.join(result_folder, f'dr_{layer}2.pdf'))
    else:
        plt.savefig(os.path.join(attack_result_folder, f'dr_{attack}_{layer}2.pdf'))
        mlflow.log_artifact(os.path.join(result_folder, f'dr_{attack}_{layer}2.pdf'))
        
def visualize_cc(cc, layer, num, attack, result_folder=None):
    batch, out_num_caps, in_num_caps = cc.size()
    tmp = cc[num].detach().cpu().numpy()

    plt.figure(dpi=150)
    sns.heatmap(tmp, cbar=True)
    plt.title(layer+"\nfeature heatmap", fontsize=15)
    plt.xlabel(f"{num}sample's {in_num_caps}(t*h) in-capsules", fontsize=10)
    plt.ylabel(f"{num}sample's 5 out-capsules(classes)", fontsize=10)
    if attack == None:
        plt.savefig(os.path.join(result_folder, f'dr_{layer}_heatmap.pdf'))
        mlflow.log_artifact(os.path.join(result_folder, f'dr_{layer}_heatmap.pdf'))
    else:
        plt.savefig(os.path.join(attack_result_folder, f'dr_{attack}_{layer}_heatmap.pdf'))
        mlflow.log_artifact(os.path.join(result_folder, f'dr_{attack}_{layer}_heatmap.pdf'))

    plt.figure(dpi=150)
    sns.distplot(tmp, norm_hist=False)
    plt.title(layer+"\nfeature heatmap", fontsize=15)
    plt.xlabel(f"{num}sample's {in_num_caps}(t*h) in-capsules", fontsize=10)
    plt.ylabel(f"density", fontsize=10)
    print(layer)
    if attack == None:
        plt.savefig(os.path.join(result_folder, f'dr_{layer}_histogram.pdf'))
        mlflow.log_artifact(os.path.join(result_folder, f'dr_{layer}_histogram.pdf'))
    else:
        plt.savefig(os.path.join(result_folder, f'dr_{attack}_{layer}_histogram.pdf'))
        mlflow.log_artifact(os.path.join(result_folder, f'dr_{attack}_{layer}_histogram.pdf'))
        
def visualize_u_uhat(u, uhat, num, attack, result_folder=None):
    batch, in_num_caps, in_dim_caps = u.size()
    batch, out_num_caps, in_num_caps, out_dim_cpas = uhat.size()
        
    m = nn.ConstantPad1d(4, 0)
    u_ = m(u[num]).detach().cpu().numpy()
    uhat_ = uhat[num]
        
    data = u_
    plt.figure(dpi=150)
    cnt = [len(data)]
    for i in range(out_num_caps):
        data = np.concatenate((data, uhat_[i].detach().cpu().numpy()), axis=0)
        cnt.append(len(data))

    trans_data = umap.UMAP(n_components=3, random_state=SEED, metric='euclidean').fit_transform(data) #cosine, euclidean
    
    c = ['r', 'b', 'orange', 'green', 'black', 'purple', 'grey', 'lightsteepblue']
    
    plt.scatter(trans_data[:,0][:cnt[0]], trans_data[:,1][:cnt[0]], c= c[0], s = 1, marker='.', label='u')
    for i in range(out_num_caps):
        plt.scatter(trans_data[:,0][cnt[i]:cnt[i+1]], trans_data[:,1][cnt[i]:cnt[i+1]], c= c[i], s = 1, marker='.', label=f'{i-1}uhat')
 
    plt.legend()
    plt.title("U-Uhat Relationship", fontsize=15)
    print("U-Uhat Relationship")
    if attack == None:
        plt.savefig(os.path.join(result_folder, f'dr_u_uhat_umap.pdf'))
        mlflow.log_artifact(os.path.join(result_folder, f'dr_u_uhat_umap.pdf'))
    else:
        plt.savefig(os.path.join(result_folder, f'dr_{attack}_u_uhat_umap.pdf'))
        mlflow.log_artifact(os.path.join(result_folder, f'dr_{attack}_u_uhat_umap.pdf'))
            
def visualize_uhat_v(uhat, v, num, attack, result_folder=None):
    batch, out_num_caps, in_num_caps, out_dim_cpas = uhat.size()
    batch_v, out_num_caps_v, w, out_dim_caps_v = v.size()
        
    uhat_ = uhat[num]
    v_ = v[num].detach().cpu().numpy()
        
    plt.figure(dpi=150)       
    data = v_[0]
    for i in range(1,out_num_caps):
        data = np.concatenate((data, v_[i]), axis=0)
    cnt = [len(data)]
            
    for i in range(out_num_caps):
        for j in range(in_num_caps):
            data = np.concatenate((data, uhat_[i, j].unsqueeze(0).detach().cpu().numpy()), axis=0)
        cnt.append(len(data))
            
    trans_data = umap.UMAP(n_components=2, random_state=SEED, metric='cosine').fit_transform(data)
      
    c = ['r', 'b', 'orange', 'green', 'black', 'purple', 'grey', 'lightsteepblue']
    
    for i in range(0, out_num_caps):
        plt.scatter(trans_data[:,0][cnt[i]:cnt[i+1]], trans_data[:,1][cnt[i]:cnt[i+1]], c= c[i], s = 1, marker='.', label=f'{i}uhat')
        
    for i in range(out_num_caps):
        plt.text(trans_data[:,0][i], trans_data[:,1][i], f'v{i}', horizontalalignment='left', size='medium', color='black', weight='semibold')
    
    plt.title("V-Uhat Relationship", fontsize=15)
    print("V-Uhat Relationship")
    plt.legend()
    if attack == None:
        plt.savefig(os.path.join(result_folder, f'dr_v_uhat_umap.pdf'))
        mlflow.log_artifact(os.path.join(result_folder, f'dr_v_uhat_umap.pdf'))
    else:
        plt.savefig(os.path.join(result_folder, f'dr_{attack}_v_uhat_umap.pdf'))
        mlflow.log_artifact(os.path.join(result_folder, f'dr_{attack}_v_uhat_umap.pdf'))
        
def visualize_uhat_cc_v(uhat, cc, v, num, attack, result_folder=None):
    batch, out_num_caps, in_num_caps, out_dim_cpas = uhat.size()
    batch_v, out_num_caps_v, w, out_dim_caps_v = v.size()
        
    uhat_ = uhat[num]
    cc_ = cc[num]
    v_ = v[num]
        
    plt.figure(dpi=150)       
    uhat_cs = torch.empty((0,out_dim_cpas+1)).cuda()
    cnt = [0]
    for j in range(out_num_caps):
        for i in range(in_num_caps):
            uhat_c = torch.cat((uhat_[j,i,:].detach(), cc_[j, i].unsqueeze(0).detach()), dim=0)
            uhat_cs = torch.cat((uhat_cs, uhat_c.unsqueeze(0)),0)
        cnt.append(len(uhat_cs))

    v =torch.cat((uhat_cs[:,:-1],v_[:,:].squeeze().detach()), 0)
            
    trans_data = umap.UMAP(n_components=2, random_state=SEED, metric='cosine').fit_transform(v.detach().cpu().numpy())
    trans_data_df = pd.DataFrame(trans_data)
    
    uhat_tmp = copy.deepcopy(uhat_cs)
    uhat_tmp[uhat_tmp[:,-1] > 0.18] = 0
    idx2 = uhat_tmp.nonzero()[:,0].unique().detach().cpu().numpy()
    uhat_tmp = copy.deepcopy(uhat_cs)
    uhat_tmp[uhat_tmp[:,-1] < 0.22] = 0
    idx1 = uhat_tmp.nonzero()[:,0].unique().detach().cpu().numpy()
      
    c = ['r', 'b', 'orange', 'green', 'black', 'purple', 'grey', 'lightsteepblue']
    
    for i in range(out_num_caps):
        plt.scatter(trans_data[:,0][cnt[i]:cnt[i+1]], trans_data[:,1][cnt[i]:cnt[i+1]], c= c[i], s = 1, marker='.', label=f'{i}uhat')
    
    plt.scatter(trans_data_df.iloc[idx1,0], trans_data_df.iloc[idx1,1], marker = '+', s=50, color='r', alpha = 0.8, label="> 0.22")
    plt.scatter(trans_data_df.iloc[idx2,0], trans_data_df.iloc[idx2, 1], marker = '+', s=50, color='b', alpha = 0.8, label="< 0.18")
    
    for i in range(out_num_caps):
        plt.text(trans_data[:, 0][in_num_caps*out_num_caps+i], trans_data[:, 1][in_num_caps*out_num_caps+i], f'v{i}', horizontalalignment='left', size='medium', color='black', weight='semibold')
    
    plt.title("V-CC-Uhat Relationship", fontsize=15)
    print("V-CC-Uhat Relationship")
    plt.legend()
    if attack == None:
        plt.savefig(os.path.join(result_folder, f'dr_v_cc_uhat_umap.pdf'))
        mlflow.log_artifact(os.path.join(result_folder, f'dr_v_cc_uhat_umap.pdf'))
    else:
        plt.savefig(os.path.join(result_folder, f'dr_{attack}_v_cc_uhat_umap.pdf'))
        mlflow.log_artifact(os.path.join(result_folder, f'dr_{attack}_v_cc_uhat_umap.pdf'))
        
def visualize_primarycaps(x, num_filter, layer, num, attack, result_folder=None):
    x = x.detach().cpu()[num]
    r = int(math.sqrt(num_filter))
    c = int(math.sqrt(num_filter))
    x = x.reshape(r,-1,1, 8)
    fig, ax = plt.subplots(r,c,dpi=150)
    for i in range(r):
        for j in range(c):
            ax[i, j].plot(x[i, j, :, :].T)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    fig.suptitle(layer+"\nfeature map (1box=1capsule)", fontsize=15)
    fig.supxlabel("Capsule Dimension (8)", fontsize=10)
    fig.supylabel("Values", fontsize=10)
    print(layer+"2")
    if attack == None:
        plt.savefig(os.path.join(result_folder, f'dr_{layer}2.pdf'))
        mlflow.log_artifact(os.path.join(result_folder, f'dr_{layer}2.pdf'))
    else:
        plt.savefig(os.path.join(result_folder, f'dr_{attack}_{layer}2.pdf'))
        mlflow.log_artifact(os.path.join(result_folder, f'dr_{attack}_{layer}2.pdf'))

def visualize_digitcaps(x, num_filter, layer, num, attack, result_folder=None):
    x = x.detach().cpu()[num]
    r = num_filter
    x = x.reshape(r,1, 16)
    fig, ax = plt.subplots(2, 3, dpi=150)
    cnt = 0
    for i in range(2):
        for j in range(3):
            if cnt == num_filter:
                ax[i,j].set_xticks([])
                ax[i,j].set_yticks([])
                break
            ax[i,j].plot(x[cnt, :, :].T)
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
            cnt += 1
    fig.suptitle(layer+"\nfeature map (1box=1capsule)", fontsize=15)
    fig.supxlabel("Capsule Dimension (16)", fontsize=10)
    fig.supylabel("Values", fontsize=10)
    print(layer+"2")
    if attack == None:
        plt.savefig(os.path.join(result_folder, f'dr_{layer}2.pdf'))
        mlflow.log_artifact(os.path.join(result_folder, f'dr_{layer}2.pdf'))
    else:
        plt.savefig(os.path.join(result_folder, f'dr_{attack}_{layer}2.pdf'))
        mlflow.log_artifact(os.path.join(result_folder, f'dr_{attack}_{layer}2.pdf'))