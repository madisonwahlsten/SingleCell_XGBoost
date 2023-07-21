#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 23:33:45 2022

@author: amin
"""
import numpy as np
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def conv2d(data, kernel, padding=0, strides=1):
    """
    Implements the forward propagation for a convolution function
    
    Arguments:
    data -- input data array, numpy array of shape (m, n)
    kernel -- kernel array, numpy array of shape (f, f)
    padding -- specifies padding size, int
    strides -- specifies whether to perform stide, int
        
    Returns:
    Z -- conv output, numpy array of shape ((m - f + 2 * padding) / strides) + 1,  ((n - f + 2 * padding) / strides) + 1)
    """
    
    # get dimensions from data and kernel
    assert data.ndim == 2, ValueError('ndim does not match')
    xdim, ydim = data.shape
    m, n = kernel.shape
    
    # shape of Output Convolution
    xo = int(((xdim - m + 2 * padding) / strides) + 1)
    yo = int(((ydim - n + 2 * padding) / strides) + 1)
    Z = np.zeros([xo, yo])
    
    # apply equal padding from both sides
    if padding != 0:
      pdata = np.zeros([xdim + padding*2, ydim + padding*2])
      pdata[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = data
    else:
      pdata = data
    
    # Iteration through columns and rows in the data
    for idy in range(ydim):
      if idy > ydim - n:
          break
      # check y movement specified by stride before convolution
      if idy % strides == 0:
          for idx in range(xdim):
              if idx > xdim - m:
                  break
              try:
                  # check x movement specified by stride before convolution
                  if idx % strides == 0:
                      Z[idx, idy] = (kernel * pdata[idx: idx + m, idy: idy + n]).sum()
              except:
                  break
    return Z

def get_layers(m: torch.nn.Module):
    children = dict(m.named_children())
    output_layers = {}
    if children == {}:
        # if module has no children; m is last child! :O
        return m
    else:
        # look for children from children... to the last child!
        for name, child in children.items():
            try:
                output_layers[name] = get_layers(child)
            except TypeError:
                output_layers[name] = get_layers(child)
    return output_layers

def get_weight(layers):
    if isinstance(layers, list):
        output_weight = []
        for ind, layer in enumerate(layers):
            if hasattr(layer, 'weight'):
                output_weight.append(layer.weight.detach().cpu().numpy())
    elif isinstance(layers, dict):
        output_weight = {}
        for ind, layer in layers.items():
            if hasattr(layer, 'weight'):
                output_weight[ind] = layer.weight.detach().cpu().numpy()
    return output_weight

def get_bias(layers):
    if isinstance(layers, list):
        output_bias = []
        for ind, layer in enumerate(layers):
            if hasattr(layer, 'bias'):
                output_bias.append(layer.bias.detach().cpu().numpy())
    elif isinstance(layers, dict):
        output_bias = {}
        for ind, layer in layers.items():
            if hasattr(layer, 'bias'):
                output_bias[ind] = layer.bias.detach().cpu().numpy()
    return output_bias

def relu(Z):
    return np.maximum(0,Z)   

def plot_weights(data, output_w, output_b, markers_list, tpoints, ct, pep, con, params):
    fig, ax = plt.subplots(figsize=(12,8))
    im = ax.imshow(data, cmap='plasma', vmin=0, vmax=1e3)
    ax.set_xticks(np.arange(len(tpoints)))
    ax.set_xticklabels(tpoints)
    ax.set_yticks(np.arange(len(markers_list)))
    ax.set_yticklabels(markers_list)
    plt.colorbar(im, ax=ax, orientation='vertical')
    plt.show()
    if hasattr(params, 'save_output'):
        results_path = os.path.join(params.results_path,'timeseries_plots')
        os.makedirs(results_path, exist_ok=True)
        try:
            plt.savefig(os.path.join(results_path, f'{ct}_{pep}_{con}_timeseries'+f'_fold_{params.fold}.png'), dpi=300, facecolor='w', bbox_inches='tight')
        except:
            plt.savefig(os.path.join(results_path, f'{ct}_{pep}_{con}_timeseries.png'), dpi=300, facecolor='w', bbox_inches='tight')
        plt.close()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,6))
    im = ax1.imshow(output_w['conv0'][0,:,:], cmap='plasma')
    # ax1.set_xticks(np.arange(len(tpoints)))
    # ax1.set_xticklabels(tpoints)
    # ax1.set_yticks(np.arange(len(markers_list)))
    # ax1.set_yticklabels(markers_list)
    plt.colorbar(im, ax=ax1, orientation='horizontal', location='top')
    im0 = ax2.imshow(output_w['conv0'][1,:,:], cmap='plasma')
    plt.colorbar(im0, ax=ax2, orientation='horizontal', location='bottom')
    plt.show()
    if hasattr(params, 'save_output'):
        results_path = os.path.join(params.results_path,'weights')
        os.makedirs(results_path, exist_ok=True)
        try:
            plt.savefig(os.path.join(results_path, f'{ct}_{pep}_{con}_weights'+f'_fold_{params.fold}.png'), dpi=300, facecolor='w', bbox_inches='tight')
        except:
            plt.savefig(os.path.join(results_path, f'{ct}_{pep}_{con}_weights.png'), dpi=300, facecolor='w', bbox_inches='tight')
        plt.close()    
    
    output_1 = conv2d(data, output_w['conv0'][0,:,:], padding=0, strides=1)
    out_rel_1 = relu(output_1)
    output_2 = conv2d(data, output_w['conv0'][1,:,:], padding=0, strides=1)
    out_rel_2 = relu(output_2)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(36,36))
    im = ax1.imshow(output_1, cmap='plasma')
    im2 = ax2.imshow(out_rel_1, cmap='plasma')
    plt.colorbar(im, ax=ax2, orientation='vertical', location='right')
    im3 = ax3.imshow(output_2, cmap='plasma')
    im4 = ax4.imshow(out_rel_2, cmap='plasma')
    plt.colorbar(im3, ax=ax4, orientation='vertical', location='right')
    plt.show()
    if hasattr(params, 'save_output'):
        results_path = os.path.join(params.results_path,'conv_layer')
        os.makedirs(results_path, exist_ok=True)
        try:
            plt.savefig(os.path.join(results_path, f'{ct}_{pep}_{con}_conv_layer'+f'_fold_{params.fold}.png'), dpi=300, facecolor='w', bbox_inches='tight')
        except:
            plt.savefig(os.path.join(results_path, f'{ct}_{pep}_{con}_conv_layer.png'), dpi=300, facecolor='w', bbox_inches='tight')
        plt.close()