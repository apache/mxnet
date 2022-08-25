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
from mxnet.gluon.model_zoo.vision import resnet50_v2
from mxnet.gluon.data.vision import transforms
from mxnet.contrib.quantization import quantize_net

# Preparing input data
rgb_mean = (0.485, 0.456, 0.406)
rgb_std = (0.229, 0.224, 0.225)
batch_size = 64
num_calib_batches = 9
# Set proper path to ImageNet data set below
dataset = mx.gluon.data.vision.ImageRecordDataset('../imagenet/rec/val.rec')
# Tuning with INC on the whole data set takes a lot of time. Therefore, we take only a part of the whole data set
# as representative part of it:
dataset = dataset.take(num_calib_batches * batch_size)
transformer = transforms.Compose([transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=rgb_mean, std=rgb_std)])
# Note: as input data is used many times during tuning it is better to have it prepared earlier.
#       Therefore, lazy parameter for transform_first is set to False.
val_data = mx.gluon.data.DataLoader(
    dataset.transform_first(transformer, lazy=False), batch_size, shuffle=False)
val_data.batch_size = batch_size

net = resnet50_v2(pretrained=True)

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
qnet_inc = quantizer.fit().model
print("INC finished")
# You can save the optimized model for the later use:
qnet_inc.export("__quantized_with_inc")
# You can see which configuration was applied by INC and which nodes were excluded from quantization
# to achieve given accuracy loss against floating point calculation.
print(quantizer.strategy.best_qmodel.q_config['quant_cfg'])
