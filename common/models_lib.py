#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from tqdm import tqdm
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

import numpy as np
import sys,os
import pandas as pd
import sklearn, sklearn.model_selection, sklearn.neighbors
import sklearn.linear_model, sklearn.ensemble
import collections


# class VAE(nn.Module):
#     def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
#         super(VAE, self).__init__()
        
#         # encoder part
#         self.fc1 = nn.Linear(x_dim, h_dim1)
#         self.fc2 = nn.Linear(h_dim1, h_dim2)
#         self.fc31 = nn.Linear(h_dim2, z_dim)
#         self.fc32 = nn.Linear(h_dim2, z_dim)
#         # decoder part
#         self.fc4 = nn.Linear(z_dim, h_dim2)
#         self.fc5 = nn.Linear(h_dim2, h_dim1)
#         self.fc6 = nn.Linear(h_dim1, x_dim)
        
#     def encoder(self, x):
#         h = F.relu(self.fc1(x))
#         h = F.relu(self.fc2(h))
#         return self.fc31(h), self.fc32(h) # mu, log_var
    
#     def sampling(self, mu, log_var):
#         std = torch.exp(0.5*log_var)
#         eps = torch.randn_like(std)
#         return eps.mul(std).add_(mu) # return z sample
        
#     def decoder(self, z):
#         h = F.relu(self.fc4(z))
#         h = F.relu(self.fc5(h))
#         return F.sigmoid(self.fc6(h)) 
    
#     def forward(self, x):
#         mu, log_var = self.encoder(x.view(-1, 784))
#         z = self.sampling(mu, log_var)
#         return self.decoder(z), mu, log_var


class tabAE(nn.Module):
    def __init__(self, in_channels, n_layers=3, layers_dim=None, z_dim=2, dropout=0.2, seed=0):
        super(tabAE, self).__init__()
        self.in_channels = in_channels
        #
        self.n_layers = n_layers
        if layers_dim:
            if isinstance(layers_dim, list):
                self.layers_dim = layers_dim
            elif isinstance(layers_dim, int):
                self.layers_dim = [layers_dim for i in range(n_layers)]
        else:
            self.n_layers= int(np.floor(np.math.log(in_channels, 2))-1)
            layer_dim = [in_channels//(2**n) for n in range(1,self.n_layers+1)]
            self.layers_dim = layer_dim
        self.z_dim = z_dim
        self.dropout = dropout
        assert self.z_dim <= self.layers_dim[-1], ValueError("bottleneck dim should be smaller than last hidden layer in the encoder")
        # Building the input linear layer
        f_encoder = collections.OrderedDict()
        # layers['norm0'] = nn.BatchNorm1d(self.in_channels, momentum=self.momentum)
        # layers['dropout0'] = nn.Dropout(p=self.dropout)
        f_encoder['encoder0'] = nn.Linear(self.in_channels, self.layers_dim[0])
        f_encoder['relu_e0'] = nn.ReLU()
        # Building all hidden linear layers
        for i in range(1, self.n_layers):
            # layers['norm{}'.format(i)] = nn.BatchNorm1d(self.layer_size, momentum=self.momentum)
            # layers['dropout{}'.format(i)] = nn.Dropout(p=self.dropout)
            f_encoder['encoder{}'.format(i)] = nn.Linear(self.layers_dim[i-1], self.layers_dim[i])
            f_encoder['relu_e{}'.format(i)] = nn.ReLU()
        # Building the output linear layer
        # layers['norm_out'] = nn.BatchNorm1d(self.layer_size, momentum=self.momentum)
        # layers['dropout_out'] = nn.Dropout(p=self.dropout/2)
        self.f_encoder = nn.Sequential(f_encoder)
        self.bottleneck1 = nn.Linear(self.layers_dim[-1], self.z_dim)
        self.bottleneck2 = nn.Linear(self.layers_dim[-1], self.z_dim)
        # decoder part
        f_decoder = collections.OrderedDict()
        f_decoder['decoder0'] = nn.Linear(self.z_dim, self.layers_dim[-1])
        f_decoder['relu_d0'] = nn.ReLU()
        for i in range(1, self.n_layers):
            f_decoder['decoder{}'.format(i)] = nn.Linear(self.layers_dim[-i], self.layers_dim[-(i+1)])
            f_decoder['relu_d{}'.format(i)] = nn.ReLU()
        self.f_decoder = nn.Sequential(f_decoder)
        f_out = collections.OrderedDict()
        f_out['output'] = nn.Linear(self.layers_dim[0], self.in_channels)
        f_out['sig_out'] = nn.Sigmoid()
        self.f_out = nn.Sequential(f_out)
        
    def encoder(self, x):
        # h = x.view(x.size(0), -1)
        h = self.f_encoder(x)
        return self.bottleneck1(h), self.bottleneck2(h)
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
    
    def decoder(self, z):
        h = self.f_decoder(z)
        return self.f_out(h)  
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var    



class tabCNN(nn.Module):
    def __init__(self, in_channels, n_classes, layer_size=32, n_channels=16,
                 n_channels_conv=32, rate=2, dropout_input=0.2, dropout_layers=0.2,
                 dropout_output=0.2, seed=0):
        super(tabCNN, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.rate = rate
        self.dropout_input = dropout_input
        self.dropout_layers = dropout_layers
        self.dropout_output = dropout_output
        # convolution layer 1
        self.L0_size = layer_size * n_channels
        self.n_channels_1 = n_channels * rate
        self.kernel_1 = 5
        self.stride_1 = 1
        self.padding_1 = 2
        # convolution layer 2
        self.L1_size = layer_size
        self.L2_size = layer_size // 2
        self.n_channels_conv = n_channels_conv
        self.kernel_2 = 3
        self.stride_2 = 1
        self.padding_2 = 1
        # post process layer
        self.kernel_Avg = 4
        self.stride_Avg = 2
        self.padding_Avg = 1
        # ouput layer
        self.output_size = (layer_size // 4) * n_channels_conv

        # Building the soft-layer to reorder the input data
        layers_0 = collections.OrderedDict()
        # layers_0['norm0'] = nn.BatchNorm1d(self.in_channels)
        # layers_0['dropout0'] = nn.Dropout(p=self.dropout_input)
        layers_0['dense0'] = nn.utils.weight_norm(nn.Linear(self.in_channels, self.L0_size, bias=False))
        layers_0['CELU0'] = nn.CELU()
        self.layers_0 = nn.Sequential(layers_0)
        # print(self.layers_0)

        # Building the first convolution layer
        layers_1 = collections.OrderedDict()
        # layers_1['norm1'] = nn.BatchNorm1d(self.n_channels)
        conv1 = nn.Conv1d(self.n_channels,
                          self.n_channels_1,
                          kernel_size=self.kernel_1,
                          stride=self.stride_1,
                          padding=self.padding_1,
                          groups=self.n_channels,
                          bias=False)
        layers_1['conv1'] = nn.utils.weight_norm(conv1, dim=None)
        layers_1['relu1'] = nn.ReLU()
        layers_1['AvgPool1'] = nn.AdaptiveAvgPool1d(output_size=self.L2_size)
        self.layers_1 = nn.Sequential(layers_1)
        # print(self.layers_1)

        # Building the second convolution layer
        layers_2 = collections.OrderedDict()
        # layers_2['norm2'] = nn.BatchNorm1d(self.n_channels_1)
        # layers_2['dropout2'] = nn.Dropout(p=self.dropout_layers)
        conv2 = nn.Conv1d(self.n_channels_1,
                          self.n_channels_conv,
                          kernel_size=self.kernel_2,
                          stride=self.stride_2,
                          padding=self.padding_2,
                          bias=False)
        layers_2['conv2'] = nn.utils.weight_norm(conv2, dim=None)
        layers_2['relu2'] = nn.ReLU()
        self.layers_2 = nn.Sequential(layers_2)
        # print(self.layers_2)

        # Building the third convolution layer
        # Layer 2 and layer 3 has the same kernel size, stride and padding and are different with layer 1 and 4
        layers_3 = collections.OrderedDict()
        # layers_3['norm3'] = nn.BatchNorm1d(self.n_channels_conv)
        # layers_3['dropout3'] = nn.Dropout(p=self.dropout_layers)
        conv3 = nn.Conv1d(self.n_channels_conv,
                          self.n_channels_conv,
                          kernel_size=self.kernel_2,
                          stride=self.stride_2,
                          padding=self.padding_2,
                          bias=False)
        layers_3['conv3'] = nn.utils.weight_norm(conv3, dim=None)
        layers_3['relu3'] = nn.ReLU()
        self.layers_3 = nn.Sequential(layers_3)
        # print(self.layers_3)

        # Building the fourth convolution layer
        # Layer 1 and layer 4 has the same kernel size, stride and padding and are different with layer 2 and 3
        layers_4 = collections.OrderedDict()
        # layers_4['norm4'] = nn.BatchNorm1d(self.n_channels_conv)
        conv4 = nn.Conv1d(self.n_channels_conv,
                          self.n_channels_conv,
                          kernel_size=self.kernel_1,
                          stride=self.stride_1,
                          padding=self.padding_1,
                          groups=self.n_channels_conv,
                          bias=False)
        layers_4['conv4'] = nn.utils.weight_norm(conv4, dim=None)
        # Applying convolution, adding x+x_w and then applying relu. We put relu in the forward
        self.layers_4 = nn.Sequential(layers_4)
        # print(self.layers_4)

        # Post process layer
        layers_5 = collections.OrderedDict()
        layers_5['relu'] = nn.ReLU()
        layers_5['AvgPool'] = nn.AvgPool1d(kernel_size=self.kernel_Avg,
                                           stride=self.stride_Avg,
                                           padding=self.padding_Avg)
        layers_5['flatten'] = nn.Flatten()
        self.layers_5 = nn.Sequential(layers_5)
        # print(self.layers_5)

        # Building the output layer
        layers_6 = collections.OrderedDict()
        # layers_6['norm6'] = nn.BatchNorm1d(self.output_size)
        # layers_6['dropout6'] = nn.Dropout(p=self.dropout_output)
        layers_6['dense6'] = nn.utils.weight_norm(nn.Linear(self.output_size, self.n_classes, bias=False))
        self.layers_6 = nn.Sequential(layers_6)
        # print(self.layers_6)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers_0(x)
        x = x.reshape(x.shape[0], self.n_channels, self.L1_size)
        x = self.layers_1(x)
        x = self.layers_2(x)
        x_w = x
        x = self.layers_3(x)
        x = self.layers_4(x)
        x = x + x_w
        x = self.layers_5(x)
        x = self.layers_6(x)
        return x
    
class tabMLP(nn.Module):
    def __init__(self, in_channels, n_classes, layer_size=32, n_layers=3, dropout=0.2, momentum=0.1, seed=0):
        super(tabMLP, self).__init__()

        self.in_channels = in_channels
        self.n_classes = n_classes
        self.layer_size = layer_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.momentum = momentum
        self.seed = seed
        # Building the input linear layer
        layers = collections.OrderedDict()
        # layers['norm0'] = nn.BatchNorm1d(self.in_channels, momentum=self.momentum)
        # layers['dropout0'] = nn.Dropout(p=self.dropout)
        layers['dense0'] = nn.Linear(self.in_channels, self.layer_size, bias=False)
        layers['relu0'] = nn.ReLU()
        # Building all hidden linear layers
        for i in range(1, self.n_layers - 1):
            # layers['norm{}'.format(i)] = nn.BatchNorm1d(self.layer_size, momentum=self.momentum)
            # layers['dropout{}'.format(i)] = nn.Dropout(p=self.dropout)
            layers['dense{}'.format(i)] = nn.Linear(self.layer_size, self.layer_size, bias=False)
            layers['relu{}'.format(i)] = nn.ReLU()
        # Building the output linear layer
        # layers['norm_out'] = nn.BatchNorm1d(self.layer_size, momentum=self.momentum)
        # layers['dropout_out'] = nn.Dropout(p=self.dropout / 2)
        layers['dense_out'] = nn.Linear(self.layer_size, self.n_classes, bias=False)

        self.layers = nn.Sequential(layers)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


## models mostly adapted from this code https://github.com/hsd1503/resnet1d

class MLP(nn.Module):
    """
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
    Output:
        out: (n_samples)
    Pararmetes:
        n_classes: number of classes
    """

    def __init__(self, in_channels, out_channels, n_classes, seed=0):
        super(MLP, self).__init__()

        torch.manual_seed(seed)
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.out_channels = out_channels

        # (batch, channels, length)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, n_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x


class CNN(nn.Module):
    """
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
    Output:
        out: (n_samples)
    Pararmetes:
        n_classes: number of classes
    """

    def __init__(self, in_channels, out_channels, n_layers, n_classes, final_layer, kernel, stride, seed=0):
        super(CNN, self).__init__()

        torch.manual_seed(seed)
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_classes = n_classes
        self.final_layer = final_layer
        self.kernel = kernel
        self.stride = stride
        self.printed = False

        # (batch, channels, length)
        layers_dict = collections.OrderedDict()

        layers_dict["conv0"] = nn.Conv1d(in_channels=self.in_channels,
                                         out_channels=self.out_channels,
                                         kernel_size=self.kernel,
                                         stride=self.stride)
        layers_dict["relu0"] = nn.ReLU()

        # last_size = 0
        for l in range(1, n_layers):
            layers_dict["conv{}".format(l)] = nn.Conv1d(in_channels=self.out_channels,
                                                        out_channels=self.out_channels,
                                                        kernel_size=self.kernel,
                                                        stride=self.stride)
            layers_dict["relu{}".format(l)] = nn.ReLU()
            # last_size=(self.out_channels//(l+1))
            # print(last_size)

        self.layers = nn.Sequential(layers_dict)

        # print(self.layers)

        self.pool = torch.nn.AdaptiveAvgPool1d(128)

        self.dense = nn.Linear(self.final_layer, n_classes)

    def forward(self, x):

        out = x
        # print(out.shape)

        out = self.layers(out)
        # print(out.shape)

        # out = self.pool(out)

        # print(out.shape)

        out = out.view(out.size(0), -1)
        if not self.printed:
            print(out.shape)
            self.printed = True

        out = self.dense(out)

        return out


class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=self.groups)

    def forward(self, x):
        net = x

        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.conv(net)

        return net


class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """

    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        net = x

        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.max_pool(net)

        return net


class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do,
                 is_first_block=False):
        super(BasicBlock, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            groups=self.groups)

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            groups=self.groups)

        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):

        identity = x

        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)

        # the second conv
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)

        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)

        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1, -2)
            ch1 = (self.out_channels - self.in_channels) // 2
            ch2 = self.out_channels - self.in_channels - ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1, -2)

        # shortcut
        out += identity

        return out


class ResNet1D(nn.Module):
    """
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
    Output:
        out: (n_samples)
    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes
    """

    def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_block, n_classes, downsample_gap=2,
                 increasefilter_gap=4, use_bn=True, use_do=True, verbose=False, seed=0):
        super(ResNet1D, self).__init__()

        torch.manual_seed(seed)

        self.verbose = verbose
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do

        self.downsample_gap = downsample_gap  # 2 for base model
        self.increasefilter_gap = increasefilter_gap  # 4 for base model

        # first block
        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters,
                                                kernel_size=self.kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters

        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(base_filters * 2 ** ((i_block - 1) // self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels

            tmp_block = BasicBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                groups=self.groups,
                downsample=downsample,
                use_bn=self.use_bn,
                use_do=self.use_do,
                is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)

        # final prediction
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        # self.do = nn.Dropout(p=0.5)
        self.dense = nn.Linear(out_channels, n_classes)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        out = x

        # first conv
        if self.verbose:
            print('input shape', out.shape)
        out = self.first_block_conv(out)
        if self.verbose:
            print('after first conv', out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)

        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block,
                                                                                                  net.in_channels,
                                                                                                  net.out_channels,
                                                                                                  net.downsample))
            out = net(out)
            if self.verbose:
                print(out.shape)

        # final prediction
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out = out.mean(-1)
        if self.verbose:
            print('final pooling', out.shape)
        # out = self.do(out)
        out = self.dense(out)
        if self.verbose:
            print('dense', out.shape)
        # out = self.softmax(out)
        if self.verbose:
            print('softmax', out.shape)

        return out


class DataWrapper(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = torch.tensor(data, dtype=torch.float)
        self.label = torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
      if hasattr(layer, 'reset_parameters'):
          print(f'Reset trainable parameters of layer = {layer}')
          layer.reset_parameters()
          
class ModelClass():

    def __init__(self, model, n_epoch, batch_size=32, device="cpu", seed=0):
        if (device == "cuda") and (torch.cuda.device_count() > 1):
            model = nn.DataParallel(model)
        self.model = model.to(device)  
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.device = device
        self.seed = seed

    def predict(self, X):
        X = X.values
        dataset = DataWrapper(X[:, None, :], np.zeros(len(X)))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)

        self.model.eval()
        all_pred_prob = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                input_x, input_y = tuple(t.to(self.device) for t in batch)
                pred = self.model(input_x)
                pred = F.softmax(pred, dim=1)
                all_pred_prob.append(pred.cpu().data.numpy())

        all_pred_prob = np.concatenate(all_pred_prob)
        # all_pred = np.argmax(all_pred_prob, axis=1)
        return all_pred_prob

    def fit(self, X, labels):

        torch.manual_seed(self.seed)
        X = X.values
        # class_weight = pd.Series(labels).value_counts().values
        # class_weight = 1 / class_weight / np.max(1 / class_weight)
        # print("class_weight", class_weight)
        ratio = 0.80
        total = len(X)
        dataset = DataWrapper(X[:int(len(X) * ratio), None, :], labels[:int(len(X) * ratio)])
        dataset_valid = DataWrapper(X[int(len(X) * ratio):, None, :], labels[int(len(X) * ratio):])

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=True,
                                                 pin_memory=(self.device == "cuda"))
        dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=self.batch_size, drop_last=False)

        # train and test
        self.model.apply(reset_weights)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        # loss_func = torch.nn.CrossEntropyLoss(weight=torch.Tensor(class_weight).to(self.device))
        loss_func = torch.nn.CrossEntropyLoss()
        best = {}
        best["best_valid_score"] = 99999
        for i in range(self.n_epoch):
            self.model.train()
            losses = []
            with tqdm(dataloader, unit="batch") as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {i}")
                    input_x, input_y = tuple(t.to(self.device) for t in batch)
                    pred = self.model(input_x)
                    loss = loss_func(pred, input_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    tepoch.set_postfix(loss=loss.item())
                    losses.append(loss.detach().item())

            scheduler.step(i)

            # test
            self.model.eval()
            all_pred_prob = []
            all_pred_gt = []
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader_valid):
                    input_x, input_y = tuple(t.to(self.device) for t in batch)
                    pred = self.model(input_x)
                    all_pred_prob.append(pred.cpu().data.numpy())
                    all_pred_gt.append(input_y.cpu().data.numpy())

            all_pred_prob = np.concatenate(all_pred_prob)
            all_pred_gt = np.concatenate(all_pred_gt)
            all_pred = np.argmax(all_pred_prob, axis=1)

            bacc = sklearn.metrics.balanced_accuracy_score(all_pred_gt, all_pred)

            print("loss", np.mean(losses), "valid_bacc", bacc)

            if (best["best_valid_score"] > bacc):
                best["best_model"] = self.model.state_dict()
                best["best_valid_score"] = bacc

        self.model.load_state_dict(best["best_model"])