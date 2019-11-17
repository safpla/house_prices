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

    dnn_model = DNN_regressor(config=config)
    #dnn_model.train(X_train, y_train, X_test, y_test)
    load_model_path = os.path.join(root_path, 'Models', 'new_exp')
    predictions = dnn_model.predict(X_test, load_model_path)
    print(evaluation(predictions, y_test))
    #dnn_model.evaluation(data_HOA)

    #rfr_model = Randomforest()
    #columns = data_HOA.data.columns
    #rfr_model.train(X_train, y_train, columns)
    #predictions = rfr_model.predict(X_test)
    #print(evaluation(predictions, y_test))
    #rfr_model.evaluation(data_HOA)

    #rr_model = ridge_regression()
    #rr_model.train(X_train, y_train)
    #rr_model.predict(X_test, y_test)
