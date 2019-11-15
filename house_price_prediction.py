import os, sys
from sklearn.model_selection import train_test_split
import numpy as np

# import local
root_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, root_path)
from pre_data import pre_data
from models.dnn_regressor import DNN_regressor
from models.random_forest_model import Randomforest
from utilities import evaluation
from config.default_config import parse_args


if __name__ == '__main__':
    config = parse_args()
    data_file = os.path.join(root_path, 'Data/Zillow_dataset_v1.0_HOA.csv')
    data_HOA = pre_data(data_file, data_type='HOA')
    data_file = os.path.join(root_path, 'Data/Zillow_dataset_v1.0_Lot.csv')
    data_LOT = pre_data(data_file, data_type='LOT')
    X_train, X_test, y_train, y_test = train_test_split(data_HOA.data,
                                                        data_HOA.target,
                                                        test_size=0.7,
                                                        random_state=0)
    dnn_model = DNN_regressor(config=config)
    rfr_model = Randomforest()
    rfr_model.train(X_train, y_train)
    predictions = rfr_model.predict(X_test, y_test)
    print(evaluation(predictions, y_test))
