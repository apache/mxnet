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


# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import ndarray as nd
import numpy as np


# context utils
def create_context(num_gpu):
    if num_gpu == 0:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu()
    return ctx

# data utils
def generate_synthetic_lorenz(stepCnt, dt = 0.01, initx = 0., inity = 1., initz = 1.05, s = 10, r = 28, b = 8/3):
    """
    Generate the Lorenz data for toy example in:
    https://arxiv.org/pdf/1703.04691.pdf
    Generate t=1,1500 X, Y and Z as three time series.
    Typical settings: dt = 0.01, initx = 0., inity = 1., initz = 1., s = 5, r = 20, b = 2
    Or: dt = 0.01, initx = 0., inity = 1., initz = 1.05, s = 10, r = 28, b = 8/3
    """
    xs = np.zeros(stepCnt+1)
    ys = np.zeros(stepCnt+1)
    zs = np.zeros(stepCnt+1)
    xs[0], ys[0], zs[0] = (initx, inity, initz)
    for i in range(stepCnt):
        x_dot = s*(ys[i] - xs[i])
        y_dot = r*xs[i] - ys[i] - xs[i]*zs[i]
        z_dot = xs[i]*ys[i] - b*zs[i]
        xs[i+1] = xs[i] + (x_dot * dt)
        ys[i+1] = ys[i] + (y_dot * dt)
        zs[i+1] = zs[i] + (z_dot * dt)
    # Rescale data to [-0.5, 0.5] range
    xs = (xs - np.amax(xs))/(np.amax(xs)-np.amin(xs)) + 0.5
    ys = (ys - np.amax(ys))/(np.amax(ys)-np.amin(ys)) + 0.5
    zs = (zs - np.amax(zs))/(np.amax(zs)-np.amin(zs)) + 0.5
    return np.concatenate([xs.reshape(-1,1), ys.reshape(-1,1), zs.reshape(-1,1)], axis=1)

# metric utils
def rmse(preds, labels):
    '''RMSE metric reported in literature
    '''
    rmse = mx.metric.RMSE()
    rmse.update(labels=nd.array(labels), preds=nd.array(preds))
    return rmse.get()[1]

# plot utils
def plot_losses(losses, label):
    """Plot losses per epoch.

    Train or validation loss or
    another metric.
    """
    x_axis = np.linspace(0, len(losses), len(losses), endpoint=True)
    plt.semilogy(x_axis, losses, label=label)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    return plt

def plot_predictions(preds, labels):
    """Plot predictions vs ground truth.

    """
    T = len(preds)
    time = nd.arange(0, T)
    plt.plot(time.asnumpy(), labels, label='labels')
    plt.plot(time.asnumpy(), preds, label='predictions')
    plt.legend()
    return plt
