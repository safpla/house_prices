import sys, os
import tensorflow as tf
import numpy as np

# local
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, root_path)
from models.basic_regressor import Basic_regressor
from pre_data import pre_data
from utilities import Dataset, evaluation

class DNN_regressor(Basic_regressor):
    def __init__(self, config=None, exp_name='new_exp'):
        self.config = config
        self.exp_name = exp_name
        self._build_model()

    def _build_model(self):
        if not self.config.neighborhood:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu',
                                      input_shape=(self.config.dim_features,)),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
        self.model = model

    def train(self, train_features, train_targets, valid_features, valid_targets):
        # build up datasets
        config = self.config
        #print(np.shape(train_features))
        #print(np.shape(valid_features))

        self.train_dataset = Dataset()
        self.train_dataset.build_from_data(train_features, train_targets)
        self.valid_dataset = Dataset()
        self.valid_dataset.build_from_data(valid_features, valid_targets)

        stop_flag = False
        batch_size = config.batch_size
        iters = 0
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=config.dnn_lr,
            decay_steps=1000,
            decay_rate=0.90,
            staircase=True)
        #optimizer = tf.keras.optimizers.Adadelta(learning_rate=lr_schedule)
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.dnn_lr)
        best_valid_metric = 1e10
        no_progress_count = 0
        while not stop_flag:
            batch_data = self.train_dataset.next_batch(batch_size)
            x_train = batch_data['input']
            y_train = batch_data['target']
            with tf.GradientTape() as tape:
                predictions = self.model(x_train)
                predictions = tf.reshape(predictions, [-1])
                loss = tf.keras.losses.MSE(y_train, predictions)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            if iters % config.dnn_valid_freq == 0:
                valid_metric = self.test(self.valid_dataset, metrics=['MSE'])[0]
                print('Iter: {}, train_MSE: {}, valid_MSE: {}'.\
                      format(iters, loss.numpy(), valid_metric))
                if valid_metric > best_valid_metric:
                    no_progress_count += 1
                else:
                    no_progress_count = 0
                    best_valid_metric = valid_metric
                if no_progress_count > 10:
                    stop_flag = True

            if iters > config.max_iterations:
                stop_flag = True
            iters += 1

        saved_model_path = os.path.join(root_path, 'Models', self.exp_name)
        tf.saved_model.save(self.model, saved_model_path)
        print('Model saved at {}'.format(saved_model_path))

    def predict(self, features, load_model_path=None):
        if not load_model_path is None:
            self.model = tf.keras.models.load_model(load_model_path)
        predictions = self.model.predict(features)
        predictions = np.reshape(predictions, [-1])
        return predictions

    def load_model(self, load_model_path):
        self.model = tf.keras.models.load_model(load_model_path)

    def test(self, dataset, metrics=['MSE']):
        test_data = dataset.next_batch(dataset._num_examples)
        predictions = self.model.predict_on_batch(test_data['input'])
        predictions = np.reshape(predictions, [-1])
        return evaluation(predictions, test_data['target'], metrics=metrics)




if __name__ == "__main__":
    pass
