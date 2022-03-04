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
from torch.autograd import Variable
           
from utils.Visualize import *

SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

### Needed Function ###
def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)

def squash(inputs, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param inputs: vectors to be squashed
    :param axis: the axis to squash
    :return: a Tensor with same size as inputs
    """
    norm = torch.norm(inputs, p=2, dim=axis, keepdim=True)
    scale = norm**2 / (1 + norm**2) / (norm + 1e-8)
    return scale * inputs

def caps_loss(y_true, y_pred, x, x_recon, lam_recon):
    """
    Capsule loss = Margin loss + lam_recon * reconstruction loss.
    :param y_true: true labels, one-hot coding, size=[batch, classes]
    :param y_pred: predicted labels by CapsNet, size=[batch, classes]
    :param x: input data, size=[batch, channels, width, height]
    :param x_recon: reconstructed data, size is same as `x`
    :param lam_recon: coefficient for reconstruction loss
    :return: Variable contains a scalar loss value.
    """
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()
    x = x.squeeze()
    x_recon = x_recon.squeeze()
    
    L = y_true * torch.clamp(0.9 - y_pred, min=0.) ** 2 + \
        0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.) ** 2
    L_margin = L.sum(dim=1).mean()

    L_recon = nn.MSELoss()(x_recon, x)

    return L_margin + lam_recon * L_recon

### Digit Capsule layer ###
class DenseCapsule(nn.Module):
    """
    The dense capsule layer. It is similar to Dense (FC) layer. Dense layer has `in_num` inputs, each is a scalar, the
    output of the neuron from the former layer, and it has `out_num` output neurons. DenseCapsule just expands the
    output of the neuron from scalar to vector. So its input size = [None, in_num_caps, in_dim_caps] and output size = \
    [None, out_num_caps, out_dim_caps]. For Dense Layer, in_dim_caps = out_dim_caps = 1.
    :param in_num_caps: number of cpasules inputted to this layer
    :param in_dim_caps: dimension of input capsules
    :param out_num_caps: number of capsules outputted from this layer
    :param out_dim_caps: dimension of output capsules
    :param routings: number of iterations for the routing algorithm
    """
    def __init__(self, in_num_caps, in_dim_caps, out_num_caps, out_dim_caps, routings, cfg):
        super(DenseCapsule, self).__init__()
        self.in_num_caps = in_num_caps
        self.in_dim_caps = in_dim_caps
        self.out_num_caps = out_num_caps
        self.out_dim_caps = out_dim_caps
        self.routings = routings
        self.cfg = cfg
        
        if cfg.affine == 'param':
            self.weight = nn.Parameter(0.01 * torch.randn(out_num_caps, in_num_caps, out_dim_caps, in_dim_caps))
        elif cfg.affine == 'shared':
            self.weight = nn.Conv2d(1, cfg.num_class, 1, 1)
        elif cfg.affine == 'constant':
            self.weight = 0.01 * torch.ones(out_num_caps, in_num_caps, out_dim_caps, in_dim_caps).cuda()
        else:
            raise ValueError(f"Passed undefined affine : {cfg.affine}")
        
    def forward(self, x, visualize=False, num = 0, class_num=0, attack=None, result_folder=None):
        """
        # x.size=[batch, in_num_caps, in_dim_caps]
        # expanded to    [batch, 1,            in_num_caps, in_dim_caps,  1]
        # weight.size   =[       out_num_caps, in_num_caps, out_dim_caps, in_dim_caps]
        # torch.matmul: [out_dim_caps, in_dim_caps] x [in_dim_caps, 1] -> [out_dim_caps, 1]
        # => x_hat.size =[batch, out_num_caps, in_num_caps, out_dim_caps]
        """
        
        if visualize == True:
            if self.cfg.affine == 'shared':
                print("You used affine matrix as Conv2d (shared param)")
            else:
                visualize_weight(self.weight, 16, "Affine Transformation Matrix", class_num, attack, result_folder)
        
        if self.cfg.affine == 'shared':
            x_hat = self.weight(x.unsqueeze(1)).repeat(1,1,1,2)
        else:
            x_hat = torch.squeeze(torch.matmul(self.weight, x[:, None, :, :, None]), dim=-1)
            
        if visualize == True:
            visualize_u_uhat(x, x_hat, num, attack, result_folder)
            
        """
        # In forward pass, `x_hat_detached` = `x_hat`;
        # In backward, no gradient can flow from `x_hat_detached` back to `x_hat`.
        """
        x_hat_detached = x_hat.detach()
        
        """
        # The prior for coupling coefficient, initialized as zeros.
        # b.size = [batch, out_num_caps, in_num_caps]
        """
        b = Variable(torch.zeros(x.size(0), self.out_num_caps, self.in_num_caps).cuda())

        assert self.routings > 0, 'The \'routings\' should be > 0.'
        for i in range(self.routings):
            # c.size = [batch, out_num_caps, in_num_caps]
            c = F.softmax(b, dim=1)
            if visualize==True:
                visualize_cc(c, f"Coupling Coefficient after {i}routing", num, attack, result_folder)
            
            # At last iteration, use `x_hat` to compute `outputs` in order to backpropagate gradient
            if i == self.routings - 1:
                """
                # c.size expanded to [batch, out_num_caps, in_num_caps, 1           ]
                # x_hat.size     =   [batch, out_num_caps, in_num_caps, out_dim_caps]
                # => outputs.size=   [batch, out_num_caps, 1,           out_dim_caps]
                """
                if self.cfg.squashfn == 'squash':
                    outputs = squash(torch.sum(c[:, :, :, None] * x_hat, dim=-2, keepdim=True))
                elif self.cfg.squashfn == 'sigmoid':
                    outputs = F.sigmoid(torch.sum(c[:, :, :, None] * x_hat, dim=-2, keepdim=True))
                elif self.cfg.squashfn == 'norm':
                    outputs = torch.sum(c[:, :, :, None] * x_hat, dim=-2, keepdim=True).norm(dim=-2, keepdim=True)
                else:
                    raise ValueError(f"Passed undefined squashfn : {cfg.squashfn}")
            
            # Otherwise, use `x_hat_detached` to update `b`. No gradients flow on this path.
            else:  
                if self.cfg.squashfn == 'squash':
                    outputs = squash(torch.sum(c[:, :, :, None] * x_hat_detached, dim=-2, keepdim=True))
                elif self.cfg.squashfn == 'sigmoid':
                    outputs = F.sigmoid(torch.sum(c[:, :, :, None] * x_hat_detached, dim=-2, keepdim=True))
                elif self.cfg.squashfn == 'norm':
                    outputs = torch.sum(c[:, :, :, None] * x_hat_detached, dim=-2, keepdim=True).norm(dim=-2, keepdim=True)
                else:
                    raise ValueError(f"Passed undefined squashfn : {cfg.squashfn}")
                
                """
                # outputs.size       =[batch, out_num_caps, 1,           out_dim_caps]
                # x_hat_detached.size=[batch, out_num_caps, in_num_caps, out_dim_caps]
                # => b.size          =[batch, out_num_caps, in_num_caps]
                """         
                b = b + torch.sum(outputs * x_hat_detached, dim=-1)
                
        if visualize==True:
            visualize_uhat_v(x_hat, outputs, num, attack, result_folder)
            visualize_uhat_cc_v(x_hat, c, outputs, num, attack, result_folder)

        return torch.squeeze(outputs, dim=-2)

### Primary Capsule layer ###
class PrimaryCapsule(nn.Module):
    """
    Apply Conv2D with `out_channels` and then reshape to get capsules
    :param in_channels: input channels
    :param out_channels: output channels
    :param dim_caps: dimension of capsule
    :param kernel_size: kernel size
    :return: output tensor, size=[batch, num_caps, dim_caps]
    """
    def __init__(self, in_channels, out_channels, dim_caps, kernel_size, stride=1, padding=0):
        super(PrimaryCapsule, self).__init__()
        self.dim_caps = dim_caps
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        outputs = self.conv2d(x)
        outputs = outputs.view(x.size(0), -1, self.dim_caps)
        return squash(outputs)

### DR-CapsuleNetwork architecture ###    
class CapsuleNet(nn.Module):
    """
    A Capsule Network on MNIST.
    :param input_size: data size = [channels, width, height]
    :param classes: number of classes
    :param routings: number of routing iterations
    Shape:
        - Input: (batch, channels, width, height), optional (batch, classes) .
        - Output:((batch, classes), (batch, channels, width, height))
    """
    def __init__(self, input_size, classes, routings, cfg):
        super(CapsuleNet, self).__init__()
        self.input_size = input_size
        self.classes = classes
        self.routings = routings
        self.cfg = cfg

        # Layer 1: Just a conventional Conv2D layer
        self.conv1 = nn.Conv2d(input_size[0], 256, kernel_size=(cfg.kernel_size,1), stride=1, padding=0)

        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_caps, dim_caps]
        self.primarycaps = PrimaryCapsule(256, 32*8, 8, kernel_size=(9,1), stride=(2,1), padding=0)

        # Layer 3: Capsule layer. Routing algorithm works here.
        self.digitcaps = DenseCapsule(in_num_caps=32*int(((input_size[1]-(cfg.kernel_size-1)-1+1)-(9-1)-1)/2+1), in_dim_caps=8,
                                      out_num_caps=classes, out_dim_caps=16, routings=routings, cfg=cfg) 

        # Decoder network.
        self.decoder = nn.Sequential(
            nn.Linear(16*classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, cfg.out_shape),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()
    
    def forward(self, x, y=None, visualize=False, num=0, class_num=0, attack=None, result_folder=None):
        if self.cfg.task == 'prediction':
            x = x.transpose(1,2).unsqueeze(-1)
        else:
            x = x.unsqueeze(1)
        
        if visualize==True:
            x = self.relu(self.conv1(x))
            layer_visualize(x, 256, "ReLU-Conv layer", attack, num, result_folder)
            x = self.primarycaps(x)
            layer_visualize(x, 256, "Primary-Caps layer", attack, num, result_folder)
            visualize_primarycaps(x, 16, 'Primary-Caps layer', num, attack, result_folder)
            x = self.digitcaps(x, True, num, class_num, attack, result_folder)
            layer_visualize(x, 16, "Digit-Caps layer", attack, num, result_folder)
            visualize_digitcaps(x, self.classes, "Digit-Caps layer",num, attack, result_folder)
            length = x.norm(dim=-1)
            if y is None:  # during testing, no label given. create one-hot coding using `length`
                if self.cfg.reconLoss == 'class':
                    index = length.max(dim=1)[1]
                    y = Variable(torch.zeros(length.size()).scatter_(1, index.view(-1, 1).cpu().data, 1.).cuda())
                    reconstruction = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
                elif self.cfg.reconLoss == 'soft':
                    y = Variable(length)
                    reconstruction = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
                elif self.cfg.reconLoss == 'no':
                    reconstruction = self.decoder((x).view(x.size(0), -1))
            else:
                if self.cfg.reconLoss == 'class':
                    reconstruction = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
                elif self.cfg.reconLoss == 'soft':
                    reconstruction = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
                elif self.cfg.reconLoss == 'no':
                    reconstruction = self.decoder((x).view(x.size(0), -1))
                                
        else:    
            x = self.relu(self.conv1(x))
            x = self.primarycaps(x)
            x = self.digitcaps(x)
            length = x.norm(dim=-1)
            if y is None:  # during testing, no label given. create one-hot coding using `length`
                if self.cfg.reconLoss == 'class':
                    index = length.max(dim=1)[1]
                    y = Variable(torch.zeros(length.size()).scatter_(1, index.view(-1, 1).cpu().data, 1.).cuda())
                    reconstruction = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
                elif self.cfg.reconLoss == 'soft':
                    y = Variable(length)
                    reconstruction = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
                elif self.cfg.reconLoss == 'no':
                    reconstruction = self.decoder((x).view(x.size(0), -1))
            else:
                if self.cfg.reconLoss == 'class':
                    reconstruction = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
                elif self.cfg.reconLoss == 'soft':
                    reconstruction = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
                elif self.cfg.reconLoss == 'no':
                    reconstruction = self.decoder((x).view(x.size(0), -1))
                    
        return length, reconstruction.view(-1, self.cfg.out_shape)