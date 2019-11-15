""" Basic model object """
import os, sys
root_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, root_path)
from utilities import evaluation

class Basic_regressor():
    def __init__(self, config=None, exp_name='new_exp'):
        self.config = config
        self.exp_name = exp_name

    def train(features, response):
        """
        input:
            features: narray, m x d (num_samples x num_dim)
            response: narray, m (num_samples)
        output:
            None
        """
        raise NotImplementedError

    def predict(features):
        """
        input:
            features: narray, m x d(num_samples x num_dim)
        output:
            predictions: narray, m (num_samples)
        """
        raise NotImplementedError

    def evaluation(self, housedata):
        """
        Input: housedata: instance of class pre_data.Housedata
        """
        k = 9
        performance = []
        for i in range(k):
            self.train(housedata.train_features[i], housedata.train_targets[i])
            predictions = self.predict(housedata.valid_features[i])
            targets = housedata.valid_targets[i]
            performance.append(evaluation(predictions, targets))
        print(performance)


