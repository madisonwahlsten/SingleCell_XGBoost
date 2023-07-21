#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 05:19:02 2022

@author: amin
"""
import os, sys
import pandas as pd
import pickle
import sklearn.preprocessing
import sklearn, sklearn.neighbors
import sklearn.linear_model, sklearn.ensemble
from common import models_lib_cv as models_lib

def scale_fit(df_train, df_val, df_test, scaling='standard', save_path=None, **kw):
    scaler_dict = {'standard':sklearn.preprocessing.StandardScaler(),
                'minmax':sklearn.preprocessing.MinMaxScaler(),
                'maxabs':sklearn.preprocessing.MaxAbsScaler(), 
                'robust':sklearn.preprocessing.RobustScaler(),
                'norm':sklearn.preprocessing.Normalizer(), 
                'quantile':sklearn.preprocessing.QuantileTransformer(n_quantiles=100, random_state=0, output_distribution="normal"),
                'power':sklearn.preprocessing.PowerTransformer()}
    scaled_data = []
    scaler = scaler_dict[scaling].fit(df_train)
    if save_path:
        pickle.dump(scaler, open(save_path, 'wb'))
    for df in [df_train, df_val, df_test]:
        data = scaler.transform(df)
        if isinstance(df, pd.DataFrame):
            scaled_data.append(pd.DataFrame(data, index=df.index, columns=df.columns))
        else:
            scaled_data.append(data)

    return scaled_data[0], scaled_data[1], scaled_data[2]


def get_model(model_name, mode='network', **kw):
    if mode == 'model':
        if model_name == "knn":
            model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3)
        elif model_name == "lr":
            model = sklearn.linear_model.LogisticRegression(multi_class="auto")
        elif model_name == "adaboost":
            model = sklearn.ensemble.AdaBoostClassifier()
        return model
    elif mode == 'network':
        assert kw['in_channels'], ValueError('number of input channels (in_channels) is not given')
        assert kw['n_classes'], ValueError('number of classes (n_classes) is not given')
       
        if hasattr(kw, 'seed'):
            seed = kw['seed']
        else:
            seed = 0
            
        if model_name == "mlp":
            network = models_lib.MLP(in_channels=kw['in_channels'],
                                     out_channels=10,
                                     n_classes=kw['n_classes'],
                                     seed=seed)
        ## TODO: Adding cutomized kernels; Restructing the CNN to increase the accuracy
        elif model_name == "tabCNN":
            network = models_lib.tabCNN(in_channels=kw['in_channels'],
                                        n_classes=kw['n_classes'],
                                        layer_size=512, n_channels=16,
                                        n_channels_conv=32, rate=2,
                                        dropout_input=0.2, dropout_layers=0.2, dropout_output=0.2,
                                        seed=seed)

        elif model_name == "tabMLP":
            network = models_lib.tabMLP(in_channels=kw['in_channels'],
                                        n_classes=kw['n_classes'],
                                        layer_size=32, n_layers = 3,
                                        dropout = 0.2,
                                        momentum = 0.1,
                                        seed=seed)

        elif model_name == "conv-basic":
            network = models_lib.CNN(in_channels=kw['in_channels'],
                                     n_classes=kw['n_classes'],
                                     n_layers=2, out_channels=8,
                                     final_layer=200,
                                     kernel=3, stride=1, seed=seed)
        ## TODO: Debugging after optimization; Preparing the autoencoder based on resnet
        elif model_name == "conv-resnet":
            network = models_lib.ResNet1D(in_channels=1,
                                          base_filters=128,  # 64 for ResNet1D, 352 for ResNeXt1D
                                          kernel_size=7, stride=1,
                                          groups=16, n_block=24,
                                          n_classes=kw['n_classes'],
                                          downsample_gap=6,
                                          increasefilter_gap=12,
                                          use_do=True,
                                          seed=seed)
        return network
