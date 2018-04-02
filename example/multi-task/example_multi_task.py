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

# pylint: skip-file
import mxnet as mx
from mxnet.test_utils import get_mnist_iterator
import numpy as np
import logging
import time

logging.basicConfig(level=logging.DEBUG)

def build_network():
    data = mx.symbol.Variable('data')
    fc1 = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
    act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
    fc2 = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
    act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
    fc3 = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10)
    sm1 = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax1')
    sm2 = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax2')

    softmax = mx.symbol.Group([sm1, sm2])

    return softmax

class Multi_mnist_iterator(mx.io.DataIter):
    '''multi label mnist iterator'''

    def __init__(self, data_iter):
        super(Multi_mnist_iterator, self).__init__()
        self.data_iter = data_iter
        self.batch_size = self.data_iter.batch_size

    @property
    def provide_data(self):
        return self.data_iter.provide_data

    @property
    def provide_label(self):
        provide_label = self.data_iter.provide_label[0]
        # Different labels should be used here for actual application
        return [('softmax1_label', provide_label[1]), \
                ('softmax2_label', provide_label[1])]

    def hard_reset(self):
        self.data_iter.hard_reset()

    def reset(self):
        self.data_iter.reset()

    def next(self):
        batch = self.data_iter.next()
        label = batch.label[0]

        return mx.io.DataBatch(data=batch.data, label=[label, label], \
                pad=batch.pad, index=batch.index)

class Multi_Accuracy(mx.metric.EvalMetric):
    """Calculate accuracies of multi label"""

    def __init__(self, num=None):
        self.num = num
        super(Multi_Accuracy, self).__init__('multi-accuracy')

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.num_inst = 0 if self.num is None else [0] * self.num
        self.sum_metric = 0.0 if self.num is None else [0.0] * self.num

    def update(self, labels, preds):
        mx.metric.check_label_shapes(labels, preds)

        if self.num is not None:
            assert len(labels) == self.num

        for i in range(len(labels)):
            pred_label = mx.nd.argmax_channel(preds[i]).asnumpy().astype('int32')
            label = labels[i].asnumpy().astype('int32')

            mx.metric.check_label_shapes(label, pred_label)

            if self.num is None:
                self.sum_metric += (pred_label.flat == label.flat).sum()
                self.num_inst += len(pred_label.flat)
            else:
                self.sum_metric[i] += (pred_label.flat == label.flat).sum()
                self.num_inst[i] += len(pred_label.flat)

    def get(self):
        """Gets the current evaluation result.

        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        if self.num is None:
            return super(Multi_Accuracy, self).get()
        else:
            return zip(*(('%s-task%d'%(self.name, i), float('nan') if self.num_inst[i] == 0
                                                      else self.sum_metric[i] / self.num_inst[i])
                       for i in range(self.num)))

    def get_name_value(self):
        """Returns zipped name and value pairs.

        Returns
        -------
        list of tuples
            A (name, value) tuple list.
        """
        if self.num is None:
            return super(Multi_Accuracy, self).get_name_value()
        name, value = self.get()
        return list(zip(name, value))


batch_size=100
num_epochs=100
device = mx.gpu(0)
lr = 0.01

network = build_network()
train, val = get_mnist_iterator(batch_size=batch_size, input_shape = (784,))
train = Multi_mnist_iterator(train)
val = Multi_mnist_iterator(val)


model = mx.mod.Module(
    context            = device,
    symbol             = network,
    label_names        = ('softmax1_label', 'softmax2_label'))

model.fit(
    train_data         = train,
    eval_data          = val,
    eval_metric        = Multi_Accuracy(num=2),
    num_epoch          = num_epochs,
    optimizer_params   = (('learning_rate', lr), ('momentum', 0.9), ('wd', 0.00001)),
    initializer        = mx.init.Xavier(factor_type="in", magnitude=2.34),
    batch_end_callback = mx.callback.Speedometer(batch_size, 50))

