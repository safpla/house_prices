import os, sys
from sklearn.model_selection import train_test_split
import numpy as np

# import local
root_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, root_path)
from pre_data import pre_data
from models.dnn_regressor import DNN_regressor
from models.random_forest_model import Randomforest
from models.ridge_regression_model import ridge_regression
from utilities import evaluation
from config.default_config import parse_args


def ensemble(preds, metrics, targets, methods=['Ave']):
    for method in methods:
        if method == 'Ave': # average
            preds = np.array(preds)
            ensemble_pred = np.sum(preds, axis=0) / np.shape(preds)[0]
            print('Ave: ', evaluation(ensemble_pred, y_test))
        elif method == 'WAve': # weighted average
            weights = 1 / np.array(metrics)
            weights = weights / np.sum(weights)
            pred = np.matmul(weights, preds)
            print('WAve: ', evaluation(pred, y_test))

if __name__ == '__main__':
    config = parse_args()
    data_file_HOA = os.path.join(root_path, 'Data/Zillow_dataset_v1.0_HOA.csv')
    data_file_LOT = os.path.join(root_path, 'Data/Zillow_dataset_v1.0_Lot.csv')
    #data_HOA = pre_data(data_file_HOA, data_type='HOA', rebuild=True)
    #data_LOT = pre_data(data_file_LOT, data_type='LOT', rebuild=True)

    data_HOA = pre_data(data_file_HOA, data_type='HOA')
    #data_LOT = pre_data(data_file_LOT, data_type='LOT')

    X_test = data_HOA.test_features
    y_test = data_HOA.test_targets
    X_train = data_HOA.train_features_all
    y_train = data_HOA.train_targets_all
    config.dim_features = np.shape(X_test)[1]
    preds = []
    metrics = []

    for k in range(9):
        dnn_model = DNN_regressor(config=config, exp_name='dnn_cv{}'.format(k))

        load_model_path = None
        #dnn_model.train(data_HOA.train_features[k],
        #                data_HOA.train_targets[k],
        #                data_HOA.valid_features[k],
        #                data_HOA.valid_targets[k])

        load_model_path = os.path.join(root_path, 'Models', 'dnn_cv{}'.format(k))
        dnn_predictions = dnn_model.predict(X_test, load_model_path)

        dnn_metrics = evaluation(dnn_predictions, y_test)
        preds.append(dnn_predictions)
        metrics.append(dnn_metrics[0])

        print('DNN{}: '.format(k), dnn_metrics)
#    dnn_model = DNN_regressor(config=config)
#    load_model_path = os.path.join(root_path, 'Models', 'new_exp')
#    dnn_predictions = dnn_model.predict(X_test, load_model_path)
#    dnn_metrics = evaluation(dnn_predictions, y_test)
#    print('DNN: ', dnn_metrics)
#    #dnn_model.evaluation(data_HOA)
#
    rfr_model = Randomforest()
    columns = data_HOA.data.columns
    rfr_model.train(X_train, y_train, columns)
    rfr_predictions = rfr_model.predict(X_test)
    rfr_metrics = evaluation(rfr_predictions, y_test)
    preds.append(rfr_predictions)
    metrics.append(rfr_metrics[0])
    print('Random Forest: ', rfr_metrics)
#    #rfr_model.evaluation(data_HOA)
#
    rr_model = ridge_regression()
    rr_model.train(X_train, y_train)
    rr_predictions = rr_model.predict(X_test, y_test)
    rr_metrics = evaluation(rr_predictions, y_test)
    preds.append(rr_predictions)
    metrics.append(rr_metrics[0])
    print('Ridge Regression: ', rr_metrics)

#    preds = [dnn_predictions, rfr_predictions, rr_predictions]
#    metrics = [dnn_metrics[0], rfr_metrics[0], rr_metrics[0]]
    predictions = ensemble(preds, metrics, y_test,
                           methods=['WAve'])
