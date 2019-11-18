import os, sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import numpy as np
import json

# import local
root_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, root_path)
from pre_data_sold import pre_data
from models.dnn_regressor import DNN_regressor
from models.random_forest_model import Randomforest
from models.ridge_regression_model import ridge_regression
from utilities import evaluation
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

def stacking(models, data):
    # making stacking dataset
    X_train = data.stacking_train_features
    y_train = data.stacking_train_targets
    X_test = data.test_features
    y_test = data.test_targets

    X_stacked = make_stacked_data(models, X_train)

    print('====Training stacking model=====')
    stacked_model = fit_stacked_model(X_stacked, y_train, X_test, y_test)
    preds = stacked_model.predict(X_stacked)
    json_file = os.path.join(root_path, 'Data/para_HOA.json')
    paras = json.load(open(json_file, 'r'))
    target_min = paras['min']
    target_max = paras['max']
    target_mean = paras['mean']
    target_std = paras['std']
    print('eval on training: ', evaluation(preds, y_train,
                                           target_mean,
                                           target_std))

    X_stacked_test = make_stacked_data(models, X_test)
    preds = stacked_model.predict(X_stacked_test)
    print('eval on testing: ', evaluation(preds, y_test,
                                          target_mean,
                                          target_std))

def weighted_average(models, data):
    X_test = data.test_features
    y_test = data.test_targets
    json_file = os.path.join(root_path, 'Data/para_HOA.json')
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

def train_models(data):
    config = parse_args()

    X_test = data.test_features
    y_test = data.test_targets
    X_train = data.train_features_all
    y_train = data.train_targets_all
    config.dim_features = np.shape(X_train)[1]
    models = []

    for k in range(1):
        print('Training dnn_CV{}'.format(k))
        dnn_model = DNN_regressor(config=config, exp_name='dnn_cv{}'.format(k))

        if config.dnn_load_models:
            load_model_path = os.path.join(root_path, 'Models', 'dnn_cv{}'.format(k))
            dnn_model.load_model(load_model_path)
        else:
            dnn_model.train(data.train_features[k],
                            data.train_targets[k],
                            data.valid_features[k],
                            data.valid_targets[k])

        models.append(dnn_model)

    #print('Training random forest')
    #rfr_model = Randomforest(exp_name='random_forest')
    #if config.rfr_load_models:
    #    load_model_path = os.path.join(root_path, 'Models', 'random_forest')
    #    rfr_model.load_model(load_model_path)
    #else:
    #    columns = data.data.columns
    #    rfr_model.train(X_train, y_train, columns)
    #models.append(rfr_model)

    #print('Training ridge regression')
    #rr_model = ridge_regression(exp_name='ridge_regression')
    #if config.rr_load_models:
    #    load_model_path = os.path.join(root_path, 'Models', 'ridge_regression')
    #    rr_model.load_model(load_model_path)
    #else:
    #    rr_model.train(X_train, y_train)
    #models.append(rr_model)

    return models


if __name__ == '__main__':
    data_file_HOA = os.path.join(root_path, 'Data/SoldData-HOA-V2.0.csv')
    data_file_LOT = os.path.join(root_path, 'Data/SoldData-Lot-V2.0.csv')
    data = pre_data(data_file_HOA, data_type='HOA', rebuild=True)
    #data = pre_data(data_file_LOT, data_type='LOT', rebuild=True)

    #data = pre_data(data_file_HOA, data_type='HOA')
    #data = pre_data(data_file_LOT, data_type='LOT')

    models = train_models(data)
    weighted_average(models, data)
    stacking(models, data)
