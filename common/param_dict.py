#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 05:54:43 2022

@author: amin
"""
import sys,os
import collections
import datetime
import torch


models=["tabCNN", "tabMLP", "lgb", "xgb", "knn", "mlp", "lr", "adaboost", "conv-resnet", "conv-basic"]
filters = [None, ('Treatment', 'None'), ('Treatment', 'Sample'), ('Treatment', 'Cytokines'), ('Treatment', 'JAKi'), ('Treatment', 'anti-IFNg+IFNgR')]
scalers = ['standard', 'minmax', 'maxabs', 'robust', 'norm', 'quantile', 'power']

class params():
    def __init__(self, args, **kwargs):
        try:
            self.dir_path, self.fname = os.path.split(args.file_path)
            self.experiment = self.fname.split('.')[0]
            self.file_path = args.file_path
        except:
            raise NameError("File {} does not exist".fomrat(args.file_path))
        for key in args.__dict__:
            setattr(self, key, args.__dict__[key])
        
        self.__dict__.update( kwargs )
        if not self.date:
            self.date = datetime.datetime.now().strftime('%Y%m%d')
        
        # if not hasattr(self, 'output_class'):
        #     self.output_class = None
            
        if not hasattr(self, 'seed'):
            self.seed = 0
            
        if not hasattr(self, 'device'):
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

        if hasattr(self, 'cv'):
            if not hasattr(self, 'k_folds'):
                self.k_folds = 5
        else:
            self.k_folds = 2
        
        if not hasattr(self, 'scaling'):
            print('Normalization is None. For strandard normalization choose scaling = standard')
            self.scaling = None

        if not hasattr(self, 'n_epoch'):
            self.n_epoch = 20
        
        if not hasattr(self, 'batch_size'):
            self.batch_size = 32
        
        if hasattr(self, 'save_output'):
            if not hasattr(self, 'results_path'):
                self.results_path = self.dir_path
            try:
                file_name = self.experiment[self.experiment.rindex('MW'):]
                if hasattr(self, 'filter'):
                    file_name = file_name+'_'+self.filter[1]
                else:
                    file_name = file_name+'_'+'NoFilt'
                self.results_path = os.path.join(self.results_path,file_name,self.model_name+'_'+self.date)
                os.makedirs(self.results_path, exist_ok=True)
                # save_fold_name = os.path.join(save_path,self.model_name+'_model-fold-{fold}.pth')
                # torch.save(self.model.model.state_dict(), save_fold_name)
            except:
                os.makedirs(self.results_path, exist_ok=True)
            # save_fold_name = os.path.join(self.results_path,self.model_name+'_model-fold-{fold}.pth')
            # torch.save(self.model.model.state_dict(), save_fold_name)
            self.model_path = os.path.join(self.results_path, 'model')
            os.makedirs(self.model_path, exist_ok=True)            