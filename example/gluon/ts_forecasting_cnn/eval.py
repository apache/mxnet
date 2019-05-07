# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


import os
import numpy as np
from utils import rmse, plot_predictions, plot_losses


class Evaluate(object):
    """

    Load and plot training loss.

    Load predictions and calculate rmse.
    Plot predictions vs ground truth for the model run.
    """
    def __init__(self, options):
        self._options = options

    def evaluate_model(self):

        # train losses
        if self._options.epochs > 1:
            loss_save = np.loadtxt(os.path.join(self._options.assets_dir, 'losses.txt'))
            #plt = plot_losses(loss_save, 'train loss')
            #plt.savefig(os.path.join(self._options.assets_dir, 'train_loss'))
            #plt.show()
            #plt.close()

        # predictions
        preds = np.loadtxt(os.path.join(self._options.assets_dir, 'preds.txt'))
        labels = np.loadtxt(os.path.join(self._options.assets_dir, 'labels.txt'))

        rmse_test = rmse(preds, labels)
        print('RMSE on test set is {}'.format(rmse_test))
        #plt = plot_predictions(preds, labels)
        #plt.savefig(os.path.join(self._options.assets_dir, 'preds_plot'))
        #plt.show()
        #plt.close()

    def __call__(self):
        self.evaluate_model()
