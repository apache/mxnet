
import os
import numpy as np
from utils import generate_synthetic_lorenz

class LorenzMapData(object):
    def __init__(self, options):
        self._options = options

    def generate_train_test_sets(self):
        data = generate_synthetic_lorenz(self._options.lorenz_steps)
        nTrain = data.shape[0] - self._options.test_size
        receptive_field = 2 ** self._options.dilation_depth
        train_data, test_data = data[:nTrain, :], data[nTrain:, :]
        # train_data = np.append(np.zeros((receptive_field, train_data.shape[1])), train_data, axis=0)

        np.savetxt(os.path.join(self._options.assets_dir, 'train_data.txt'), train_data)
        np.savetxt(os.path.join(self._options.assets_dir, 'test_data.txt'), test_data)

        return train_data, test_data
