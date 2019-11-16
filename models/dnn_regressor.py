import sys, os
import tensorflow as tf

# local
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, root_path)
from models.basic_regressor import Basic_regressor
from pre_data import pre_data
from utilities import Dataset

class DNN_regressor(Basic_regressor):
    def __init__(self, config=None, exp_name='new_exp'):
        self.config = config
        self.exp_name = exp_name
        self._build_model()

    def _build_model(self):
        if not self.config.neighborhood:
            model = tf.keras.model.Sequential([
                tf.keras.layers.Dense(128, activation='relu',
                                      input_shape=(self.config.dim_features,)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        self.model = model


    def train(self, train_features, train_targets, valid_features, valid_targets):
        # build up datasets
        self.train_dataset = Dataset()
        self.train_dataset.build_from_data(train_features, train_targets)
        self.valid_dataset = Dataset()
        self.valid_dataset.build_from_data(valid_features, valid_targets)

        stop_flag = False
        batch_size = self.config.batch_size
        while not stop_flag:
            x_train, y_train = self.train_dataset.next_batch(batch_size)
            with tf.GradientTape() as tape:
                predictions = model(x_train)
                loss = tf.keras.losses.MSE(y_train, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        #tf.saved_model.save(model, output_path)

    def predict(self, features):
        pass



if __name__ == "__main__":
    pass
