import sys, os
import tensorflow as tf

# local
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, root_path)
from models.basic_regressor import Basic_regressor
from pre_data import pre_data

class DNN_regressor(Basic_regressor):
    def __init__(self, config=None, exp_name='new_exp'):
        self.config = config
        self.exp_name = exp_name

    def train(self, features, response):
        pass

    def predict(self, features):
        pass



if __name__ == "__main__":
    pass
