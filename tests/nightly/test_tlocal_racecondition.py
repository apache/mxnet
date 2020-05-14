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

import mxnet as mx
from mxnet import gluon
from mxnet import image
from mxnet import nd
import numpy as np
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

root_url = ('https://apache-mxnet.s3-accelerate.amazonaws.com/'
            'gluon/dataset/pikachu/')
data_dir = './data/pikachu/'
dataset = {'train.rec': 'e6bcb6ffba1ac04ff8a9b1115e650af56ee969c8',
          'train.idx': 'dcf7318b2602c06428b9988470c731621716c393',
          'val.rec': 'd6c33f799b4d058e82f2cb5bd9a976f69d72d520'}
for k, v in dataset.items():
    gluon.utils.download(root_url+k, data_dir+k, sha1_hash=v)

T = 1
devs = [mx.gpu(i) for i in range(4)]
data_shape = 224 * T
batch_size = 20 * len(devs)
rgb_mean = np.array([1,2,3])

class_names = ['pikachu']
num_class = len(class_names)

def get_iterators(data_shape, batch_size):
    train_iter = image.ImageDetIter(
        batch_size=batch_size,
        data_shape=(3, data_shape, data_shape),
        path_imgrec=data_dir+'train.rec',
        path_imgidx=data_dir+'train.idx',
        shuffle=True,
        mean=True,
        rand_crop=1,
        min_object_covered=0.95,
        max_attempts=200)
    val_iter = image.ImageDetIter(
        batch_size=batch_size,
        data_shape=(3, data_shape, data_shape),
        path_imgrec=data_dir+'val.rec',
        shuffle=False,
        mean=True)
    return train_iter, val_iter, class_names, num_class

train_data, test_data, class_names, num_class = get_iterators(
    data_shape, batch_size)


class MyCustom(mx.operator.CustomOp):
    def __init__(self):
        super(MyCustom, self).__init__()
    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], 0)
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)

@mx.operator.register("MyCustom")
class MyCustomProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(MyCustomProp, self).__init__(need_top_grad = False)
    def list_arguments(self):
        return ["data", "label"]
    def list_outputs(self):
        return ["loss"]
    def infer_shape(self, in_shape):
        return [in_shape[0], in_shape[1]], [(1, )], []
    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype, dtype], [dtype], []
    def create_operator(self, ctx, shapes, dtypes):
        return MyCustom()

class MyMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(MyMetric, self).__init__("MyMetric")
        self.name = ['empty']
    def update(self, labels, preds):
        pass
    def get(self):
        return self.name, [0]

if __name__ == '__main__':
    x = mx.sym.Variable("data")
    label = mx.sym.Variable("label")
    x = mx.sym.FullyConnected(data = x, num_hidden = 100)
    label = mx.sym.Reshape(data = label, shape = (0, -1))
    sym = mx.sym.Custom(data = x, label = label, op_type = "MyCustom")
    model = mx.module.Module(context = devs, symbol = sym, data_names = ('data',), label_names = ('label',))
    model.fit(train_data = train_data, begin_epoch = 0, num_epoch = 20, allow_missing = True, batch_end_callback = mx.callback.Speedometer(batch_size, 5), eval_metric = MyMetric())
