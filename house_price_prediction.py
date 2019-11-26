import os, sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import numpy as np
import json
import pandas as pd

# import local
root_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, root_path)
from pre_data import pre_data
from pre_data_demo import pre_data_demo
from models.dnn_regressor import DNN_regressor
from models.random_forest_model import Randomforest
from models.ridge_regression_model import ridge_regression
from utilities import evaluation
from utilities import reverse_meanstd
from config.default_config import parse_args

def fit_stacked_model(X_stacked, y_train, X_test, y_test):
    model = Ridge(alpha=1.0)
    model.fit(X_stacked, y_train)
    return model

def make_stacked_data(models, features):
    preds = []
    for model in models:
        preds.append(model.predict(features))
    return np.array(preds).transpose()

def stacking(models, data, data_type='HOA'):
    # making stacking dataset
    X_train = data.stacking_train_features
    y_train = data.stacking_train_targets
    X_test = data.test_features
    y_test = data.test_targets

    X_stacked = make_stacked_data(models, X_train)

    #print('====Training stacking model=====')
    stacked_model = fit_stacked_model(X_stacked, y_train, X_test, y_test)
    preds = stacked_model.predict(X_stacked)
    json_file = os.path.join(root_path, 'Data/para_{}.json'.format(data_type))
    paras = json.load(open(json_file, 'r'))
    target_min = paras['min']
    target_max = paras['max']
    target_mean = paras['mean']
    target_std = paras['std']

    X_stacked_test = make_stacked_data(models, X_test)
    preds = stacked_model.predict(X_stacked_test)
    print('Stacking: ', evaluation(preds, y_test,
                                          target_mean,
                                          target_std))

def stacking_demo(models, data, demo_data, data_type='HOA'):
    # making stacking dataset
    X_train = data.stacking_train_features
    y_train = data.stacking_train_targets
    X_test = data.test_features
    y_test = data.test_targets

    X_stacked = make_stacked_data(models, X_train)

    stacked_model = fit_stacked_model(X_stacked, y_train, X_test, y_test)
    preds = stacked_model.predict(X_stacked)
    json_file = os.path.join(root_path, 'Data/para_{}.json'.format(data_type))
    paras = json.load(open(json_file, 'r'))
    target_min = paras['min']
    target_max = paras['max']
    target_mean = paras['mean']
    target_std = paras['std']

    X_demo = demo_data.data
    y_demo = demo_data.target
    X_stacked_demo = make_stacked_data(models, X_demo)
    preds = stacked_model.predict(X_stacked_demo)
    preds = reverse_meanstd(preds, target_mean, target_std)
    y_demo = reverse_meanstd(y_demo, target_mean, target_std)
    print('--------Demo for stacking----------')
    index = np.arange(len(y_demo))
    for i, pred, y in zip(index, preds, y_demo):
        print('#{}  prediction: {}, target: {}'.format(i, pred, y))

def weighted_average(models, data, data_type='HOA'):
    X_test = data.test_features
    y_test = data.test_targets
    json_file = os.path.join(root_path, 'Data/para_{}.json'.format(data_type))
    paras = json.load(open(json_file, 'r'))
    target_min = paras['min']
    target_max = paras['max']
    target_mean = paras['mean']
    target_std = paras['std']


    preds = []
    metrics = []
    for model in models:
        predictions = model.predict(X_test)
        metric = evaluation(predictions, y_test,
                            target_mean,
                            target_std)
        preds.append(predictions)
        metrics.append(metric[0])
        print(model.exp_name, ': ', metric)

    weights = 1 / np.array(metrics)
    weights = weights / np.sum(weights)
    wave_pred = np.matmul(weights, preds)
    print('WAve: ', evaluation(wave_pred, y_test,
                               target_mean,
                               target_std))

def weighted_average_demo(models, data, demo_data, data_type='HOA'):
    X_test = data.test_features
    y_test = data.test_targets
    json_file = os.path.join(root_path, 'Data/para_{}.json'.format(data_type))
    paras = json.load(open(json_file, 'r'))
    target_min = paras['min']
    target_max = paras['max']
    target_mean = paras['mean']
    target_std = paras['std']


    preds = []
    metrics = []
    for model in models:
        predictions = model.predict(X_test)
        metric = evaluation(predictions, y_test,
                            target_mean,
                            target_std)
        preds.append(predictions)
        metrics.append(metric[0])

    weights = 1 / np.array(metrics)
    weights = weights / np.sum(weights)
    X_demo = demo_data.data
    y_demo = demo_data.target
    preds = []
    for model in models:
        predictions = model.predict(X_demo)
        preds.append(predictions)

    wave_pred = np.matmul(weights, preds)
    wave_pred = reverse_meanstd(wave_pred, target_mean, target_std)
    y_demo = reverse_meanstd(y_demo, target_mean, target_std)
    print('--------Demo for weighted average----------')
    index = np.arange(len(y_demo))
    for i, pred, y in zip(index, wave_pred, y_demo):
        print('#{}  prediction: {}, target: {}'.format(i, pred, y))

def train_models(data):
    config = parse_args()

    X_test = data.test_features
    y_test = data.test_targets
    X_train = data.train_features_all
    y_train = data.train_targets_all
    config.dim_features = np.shape(X_train)[1]
    models = []

    for k in range(8):
        dnn_model = DNN_regressor(config=config, exp_name='dnn_cv{}'.format(k))

        if config.dnn_load_models:
            print('Loading dnn_CV{}'.format(k))
            load_model_path = os.path.join(root_path, 'Models', 'dnn_cv{}'.format(k))
            dnn_model.load_model(load_model_path)
        else:
            print('Training dnn_CV{}'.format(k))
            dnn_model.train(data.train_features[k],
                            data.train_targets[k],
                            data.valid_features[k],
                            data.valid_targets[k])

        models.append(dnn_model)

    rfr_model = Randomforest(exp_name='random_forest')
    if config.rfr_load_models:
        print('Loading random forest')
        load_model_path = os.path.join(root_path, 'Models', 'random_forest')
        rfr_model.load_model(load_model_path)
    else:
        print('Training random forest')
        columns = data.data.columns
        rfr_model.train(X_train, y_train, columns)
    models.append(rfr_model)

    rr_model = ridge_regression(exp_name='ridge_regression')
    if config.rr_load_models:
        print('Loading ridge regression')
        load_model_path = os.path.join(root_path, 'Models', 'ridge_regression')
        rr_model.load_model(load_model_path)
    else:
        print('Training ridge regression')
        rr_model.train(X_train, y_train)
    models.append(rr_model)

    return models


if __name__ == '__main__':
    config = parse_args()
    if config.data_type == 'HOA':
        # Preprocessing
        data_file_HOA = os.path.join(root_path, 'Data/SoldData-HOA-V2.0.csv')
        data_file_HOA = os.path.join(root_path, 'Data/Zillow_dataset_v1.0_HOA.csv')
        #data = pre_data(data_file_HOA, data_type='HOA', rebuild=True)
        data = pre_data(data_file_HOA, data_type='HOA')

        # Training
        models = train_models(data)
        input("Press Enter to continue...")
        # Evaluation
        weighted_average(models, data, data_type='HOA')
        stacking(models, data, data_type='HOA')
        input("Press Enter to continue...")

        # Demo
        demo_original_data = pd.read_csv(os.path.join(root_path, 'Data/predict_data0.csv'))
        print(demo_original_data)
        data_file = os.path.join(root_path, 'Data/predict_data0.csv')
        demo_data = pre_data_demo(data_file, config.data_type, rebuild=False)
        weighted_average_demo(models, data, demo_data, data_type='HOA')
        stacking_demo(models, data, demo_data, data_type='HOA')


    else:
        data_file_LOT = os.path.join(root_path, 'Data/SoldData-Lot-V2.0.csv')
        data_file_LOT = os.path.join(root_path, 'Data/Zillow_dataset_v1.0_Lot.csv')

        #data = pre_data(data_file_LOT, data_type='LOT', rebuild=True)
        #data = pre_data(data_file_LOT, data_type='LOT')
        models = train_models(data)
        #weighted_average(models, data, data_type='LOT')
        #stacking(models, data, data_type='LOT')
