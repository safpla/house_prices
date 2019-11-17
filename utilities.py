""" utilities """

import os, sys
import random
import numpy as np

root_path = os.path.dirname((os.path.realpath(__file__)))
sys.path.insert(0, root_path)
from pre_data import pre_data


class Dataset(object):
    def __init__(self):
        pass

    def load_h5py(self, filename):
        f = h5py.File(filename, "r")
        self._dataset_input = f['img']
        self._dataset_target = f['label']
        self._num_examples = len(self._dataset_target)
        print('num of examples: ', self._num_examples)

        self._index = np.arange(self._num_examples)
        self._index_in_epoch = 0
        self._epochs_completed = 0

    def load_npy(self, filename):
        with open(filename, 'rb') as f:
            data = np.load(f)
            self._dataset_input = []
            self._dataset_target = []
            for point in data:
                self._dataset_input.append(point['x'])
                self._dataset_target.append(point['y'])
            self._dataset_input = np.array(self._dataset_input)
            self._dataset_target = np.array(self._dataset_target)
            self._num_examples = len(self._dataset_target)
            print('load {} samples.'.format(self._num_examples))

            self._index = np.arange(self._num_examples)
            self._index_in_epoch = 0
            self._epochs_completed = 0

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def build_from_data(self, input, target):
        self._dataset_input = np.array(input)
        self._dataset_target = np.array(target)
        self._num_examples = len(self._dataset_target)
        self._index = np.arange(self._num_examples)
        self._index_in_epoch = 0
        self._epochs_completed = 0

    def next_batch(self, batch_size, shuffle=True):
        # batch_size is the first dimension
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            if shuffle:
                self.shuffle()
            # Start next epoch
            start = 0
            assert batch_size <= self._num_examples
            self._index_in_epoch = start + batch_size
        end = self._index_in_epoch
        batch_index = self._index[start:end]
        #print('start:{}, end:{}'.format(start, end))
        batch_index = list(np.sort(batch_index))
        target = self._dataset_target[batch_index]
        input = self._dataset_input[batch_index]
        samples = {}
        samples['input'] = input
        samples['target'] = target
        return samples

    def reset(self):
        self._index_in_epoch = 0
        self._epochs_completed = 0

    def shuffle(self):
        np.random.shuffle(self._index)

        #normalization

def reverse_minmax(data,min,max):
    data=data*(max-min)+min
    return data

def reverse_meanstd(data,mean,std):
    data=(data*std)+mean
    return data

def evaluation(predictions, targets, metrics=['MAE', 'MSE', 'MdAPE', '5pct']):
    outputs = []
    predictions = np.array(predictions)
    targets = np.array(targets)
    for metric in metrics:
        if metric == 'MAE':
            output = sum(abs(targets - predictions)) / len(targets)
            outputs.append(output)
        elif metric == 'MSE':
            output = sum(np.square(targets - predictions)) / len(targets)
            outputs.append(output)
        elif metric == 'MdAPE':
            p = abs(targets - predictions) / targets
            outputs.append(np.median(p))
        elif metric == '5pct':
            p = abs(targets - predictions) / targets
            counts = sum(p<0.05)
            outputs.append(counts / len(targets))
        else:
            raise NotImplementedError
    return outputs

