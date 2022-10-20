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

# Disable noisy logging from INC:
import logging
logging.disable(logging.INFO)

import time
import mxnet as mx
from mxnet.gluon.model_zoo.vision import resnet50_v2
from mxnet.gluon.data.vision import transforms
from mxnet.contrib.quantization import quantize_net
import custom_strategy


def save_model(net, data_loader, description, time_spend):
  save_model.count += 1
  print( "{:21s} tuned in {:8.2f}s".format(description, time_spend))
  net.export("__resnet50_v2_{:02}_".format(save_model.count) + description.replace(' ', '_'))

save_model.count = 0

# Preparing input data
start = time.time()
rgb_mean = (0.485, 0.456, 0.406)
rgb_std = (0.229, 0.224, 0.225)
batch_size = 64
num_calib_batches = 9
# Set proper path to ImageNet data set below
dataset = mx.gluon.data.vision.ImageRecordDataset('../imagenet/rec/val.rec')
# Tuning with INC on the whole data set takes too much time. Therefore, we take only a part of the whole data set
# as representative part of it:
dataset = dataset.take(num_calib_batches * batch_size)
transformer = transforms.Compose([transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=rgb_mean, std=rgb_std)])
# Note: as the input data is used many times during tuning it is better to have it prepared earlier.
#       Therefore, lazy parameter in transform_first is set to False.
val_data = mx.gluon.data.DataLoader(
    dataset.transform_first(transformer, lazy=False), batch_size, shuffle=False)
val_data.batch_size = batch_size
time_consumed = time.time() - start
print("Input data prepared in {:.2f}s".format(time_consumed))

net = resnet50_v2(pretrained=True)

start = time.time()
net.hybridize(static_alloc=True, static_shape=True)
time_consumed = time.time() - start
# Run forward path once to cache the graph - required to save the model
net(next(iter(val_data))[0])
save_model(net, val_data, "fp32", time_consumed)

start = time.time()
net.optimize_for(next(iter(val_data))[0], backend='ONEDNN', static_alloc=True, static_shape=True)
time_consumed = time.time() - start
save_model(net, val_data, "fp32 fused", time_consumed)

start = time.time()
qnet = quantize_net(net, calib_mode='naive', calib_data=val_data)
qnet.hybridize(static_alloc=True, static_shape=True)
time_consumed = time.time() - start
save_model(qnet, val_data, 'int8 full naive', time_consumed)

start = time.time()
qnet = quantize_net(net, calib_mode='entropy', calib_data=val_data)
qnet.hybridize(static_alloc=True, static_shape=True)
time_consumed = time.time() - start
save_model(qnet, val_data, 'int8 full entropy', time_consumed)

start = time.time()
qnet = quantize_net(net, calib_mode='naive', quantize_mode='smart', calib_data=val_data)
qnet.hybridize(static_alloc=True, static_shape=True)
time_consumed = time.time() - start
save_model(qnet, val_data, 'int8 smart naive', time_consumed)

start = time.time()
qnet = quantize_net(net, calib_mode='entropy', quantize_mode='smart', calib_data=val_data)
qnet.hybridize(static_alloc=True, static_shape=True)
time_consumed = time.time() - start
save_model(qnet, val_data, 'int8 smart entropy', time_consumed)

def eval_func(model):
  metric = mx.gluon.metric.Accuracy()
  for x, label in val_data:
    output = model(x)
    metric.update(label, output)
  accuracy = metric.get()[1]
  return accuracy

from neural_compressor.experimental import Quantization
quantizer = Quantization("resnet50v2_mse.yaml")
quantizer.model = net
quantizer.calib_dataloader = val_data
quantizer.eval_func = eval_func
for strategy in ['basic', 'mse', 'mycustom', 'bayesian']:
  quantizer.cfg.tuning.strategy.name = strategy
  start = time.time()
  qnet_inc = quantizer.fit().model
  time_consumed = time.time() - start
  save_model(qnet_inc, val_data, "INC " + strategy, time_consumed)
