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
from mxnet.gluon.data.vision import transforms
import time
import glob


def test_accuracy(net, data_loader, description):
  count = 0
  acc_top1 = mx.gluon.metric.Accuracy()
  acc_top5 = mx.gluon.metric.TopKAccuracy(5)
  start = time.time()
  for x, label in data_loader:
    output = net(x)
    acc_top1.update(label, output)
    acc_top5.update(label, output)
    count += 1
  time_spend = time.time() - start
  _, top1 = acc_top1.get()
  _, top5 = acc_top5.get()
  print('{:21} Top1 Accuracy: {:.4f} Top5 Accuracy: {:.4f} from {:4} batches in {:8.2f}s'
        .format(description, top1, top5, count, time_spend))

# Preparing input data
rgb_mean = (0.485, 0.456, 0.406)
rgb_std = (0.229, 0.224, 0.225)
batch_size = 64

start = time.time()
# Set proper path to ImageNet data set below
dataset = mx.gluon.data.vision.ImageRecordDataset('../imagenet/rec/val.rec')
transformer = transforms.Compose([transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=rgb_mean, std=rgb_std)])
# Note: as the input data is used many times it is better to prepare it once.
#       Therefore, lazy parameter for transform_first is set to False.
val_data = mx.gluon.data.DataLoader(
    dataset.transform_first(transformer, lazy=False), batch_size, shuffle=False)
val_data.batch_size = batch_size
time_consumed = time.time() - start
print("Input data prepared in {:8.2f}s".format(time_consumed))

print("Measure accuracy on the whole data set could take a long time. Please wait...")
root_path = '__resnet50_v2_'
symbol_part = '-symbol.json'
for symbol in glob.glob(root_path + '*' + symbol_part):
  param = symbol.replace(symbol_part,'-0000.params')
  net_name = symbol.replace(root_path,'').replace(symbol_part,'').replace('_', ' ')
  net = mx.gluon.SymbolBlock.imports(symbol, ['data'], param)
  net.hybridize(static_alloc=True, static_shape=True)
  test_accuracy(net, val_data, net_name)
