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

    dnn_model = DNN_regressor(config=config)
    rfr_model = Randomforest()
    rfr_model.evaluation(data_HOA)
