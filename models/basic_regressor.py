""" Basic model object """
# __Author__ == "Haowen Xu"
# __Date__ == "05-04-2018"

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

