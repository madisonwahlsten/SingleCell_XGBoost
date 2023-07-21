import xgboost as xgb
from tqdm import tqdm

import numpy as np
import sys,os
import pandas as pd
import collections


import argparse
import matplotlib.pylab as plt
import sklearn.model_selection as model_selection
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from sklearn import metrics
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import datetime
from common import param_dict, io, visualization, utils


space = {'n_estimators': hp.quniform('n_estimators', 20, 200, 5),
         'eta': hp.quniform('eta', 0.025, 1, 0.05),
         'max_depth':  hp.choice('max_depth', np.arange(1, 10, dtype=int)),
         'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
         'subsample': 0.2,
         'sampling_method':'gradient_based',
         'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
         'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
         'eval_metric': 'auc',
         'objective': 'multi:softprob',
         'num_class': 8,
         'nthread': 4,
         'booster': 'gbtree',
         'tree_method': 'gpu_hist',
         'silent': 1,
         'seed': 0
         }
def objective(xgb_params):
    print("Training with params: ")
    print(xgb_params)
    num_round = int(xgb_params['n_estimators'])
    del xgb_params['n_estimators']
    model = xgb.train(xgb_params, dataloader, num_round, evals = dataset,
                      early_stopping_rounds=10, verbose_eval=True)
    predictions = model.predict(datatest, iteration_range=(0, model.best_iteration + 1))
    test_acc = metrics.accuracy_score(y_test, np.argmax(predictions, axis=1))
    loss = 1 - test_acc
    avg_per_class_acc = metrics.balanced_accuracy_score(y_test, np.argmax(predictions, axis=1))
    print('Test, test acc: %.6f, test avg acc: %.6f, loss: %.6f' % (test_acc, avg_per_class_acc, loss))
    return {'loss': loss, 'status': STATUS_OK, 'model': model }

def best_model(trials):
    valid_trials = [trial for trial in trials if STATUS_OK == trial['result']['status']]
    losses = [float(trial['result']['loss']) for trial in valid_trials]
    min_loss = np.argmin(losses)
    best_trial = valid_trials[min_loss]
    return best_trial['result']['model']

if __name__ == "__main__":
    ## TODO: Model parameters should be updated after tuning

    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', default="/home/amin/projects/Immunodynamics/data/initialSingleCellDf-channel-20220916-MW_018.h5", \
                        type=str, help='path file to read the single-cell dataframe')
    parser.add_argument('--output_class', default="Peptide", type=str,
                        help='level of MultiIndex in input DataFrame to use as model output')
    parser.add_argument('--model_name', type=str, default="xgb", choices= param_dict.models, help='Classifier model structure')
    parser.add_argument('--save_output', type=bool, default=True, help='Save Outputs')
    parser.add_argument('--plots', type=bool, default=True, help='return and save plots from the result')
    parser.add_argument('--results_path', default=os.path.join(os.getcwd(),'data', 'results'), \
                        type=str, help='path to save the result')
    parser.add_argument('--add', nargs='+', default=None, \
                        help='level of MultiIndex in input DataFrame to use as a model feature')
    parser.add_argument('--drop', nargs='+', default=None, \
                        help='Value in output_class to remove as potential output')
    parser.add_argument('--scaling', type=str, default=None, choices= param_dict.scalers, help='sklearn preprocessing scaler to use')
    parser.add_argument('--date', type=str, default=None, help='input date to read results')
    parser.add_argument('--filter', action='append', nargs=2, default=None, help='Filter to keep rows corresponding only to MultiIndex Level value. Requires: Name of Level (1) and Desired Value (2)')
    parser.add_argument('--filterOut', action='append', nargs=2, default=None, help='Filter to remove rows corresponding only to MultiIndex Level value. Requires: Name of Level (1) and Desired Value (2)')
    parser.add_argument('-m', '--mixtures', default=None,\
                        help='Used to separate out cells from single-antigen co-cultures as training data. Pass value to replace in classification level for Ratio 1:0 (index level must be labeled as Ratio)')    
    args = parser.parse_args()
    params = param_dict.params(args, cv=True, k_folds=5, n_epoch=10, batch_size=128)
    if hasattr(params, 'output_class'):
        X, y, le = io.get_data(params)
        space['num_class'] = len(np.unique(y))
    else:
        X = io.get_data(params)
    seed = params.seed
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                        test_size=0.2,
                                                                        stratify=y,
                                                                        random_state=seed)
    X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train,
                                                                      test_size=0.25,
                                                                      stratify=y_train,
                                                                      random_state=seed)
    if params.scaling:
        X_train, X_val, X_test = utils.scale_fit(X_train, X_val, X_test, scaling=params.scaling, save_path='{}/{}_scaler.pkl'.format(params.results_path, params.experiment))
    print(X_train.head())
    if hasattr(params, 'add') or hasattr(params, 'drop'):
        X_train = io.add_drop(X_train, add_val=params.add, drop_val=params.drop, output=params.output_class)
        X_test = io.add_drop(X_test, add_val=params.add, drop_val=params.drop, output=params.output_class)
        X_val = io.add_drop(X_val, add_val=params.add, drop_val=params.drop, output=params.output_class)

    print("train_set", X_train.shape, "test_set", X_test.shape)

    dataloader = xgb.DMatrix(X_train, label=y_train)
    dataloader_valid = xgb.DMatrix(X_val, label=y_val)
    dataset = [(dataloader, 'train'), (dataloader_valid, 'eval')]
    datatest = xgb.DMatrix(X_test, label=y_test)
    trials = Trials()
    best = fmin(fn = objective, space = space, algo = tpe.suggest, max_evals = 100, trials = trials)
    bmodel = best_model(trials)
    # xgb.save(bmodel, "xgboost.model")
    print("The best hyperparameters are: ", "\n")
    print(bmodel)
    fold_prediction = bmodel.predict(datatest, iteration_range=(0, bmodel.best_iteration + 1))
    test_acc = metrics.accuracy_score(y_test, np.argmax(fold_prediction, axis=1))
    avg_per_class_acc = metrics.balanced_accuracy_score(y_test, np.argmax(fold_prediction, axis=1))
    print('test acc: %.6f, test avg acc: %.6f' % (test_acc, avg_per_class_acc))
    if hasattr(params, 'save_output'):
        bmodel.save_model(os.path.join(params.model_path, params.model_name + f'_best_model.model'))

    if hasattr(params, 'save_output'):
        fold_results_name = os.path.join(params.results_path, params.experiment + f'_predictions')
        fold_results = io.make_results(y_test, fold_prediction, le, params.output_class, save_path=fold_results_name)
    else:
        fold_results = io.make_results(y_test, fold_prediction, le, params.output_class)

    if hasattr(params, 'plots'):
        visualization.plot_confusion_matrix(fold_results['True Values'], fold_results['Predicted Values'],
                                            params.output_class, params)
        visualization.plot_ovr_roc_curves(fold_results, fold_prediction, le, params.output_class, params)
        visualization.plot_prediction_distributions(fold_results, params.output_class, params)

    # import shap
    # explainer = shap.TreeExplainer(bmodel)
    # shap_values = explainer.shap_values(datatest)
    # pred = bmodel.predict(datatest, output_margin=True)
    # np.abs(shap_values.sum(1) + explainer.expected_value - pred).max()
    # shap.summary_plot(shap_values, X_test)
    # xgb.to_graphviz(bmodel, num_trees=2)
    # xgb.plot_importance(bmodel)
    # xgb.plot_tree(bmodel, num_trees=2)

