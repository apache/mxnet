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

using MXNet

#--------------------------------------------------------------------------------
# Helper functions to construct larger networks

# basic Conv + BN + ReLU factory
function conv_factory(data, num_filter, kernel; stride=(1,1), pad=(0,0), act_type=:relu)
  conv = mx.Convolution(data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad)
  bn   = mx.BatchNorm(conv)
  act  = mx.Activation(bn, act_type=act_type)
  return act
end

# simple downsampling factory
function downsample_factory(data, ch_3x3)
  # conv 3x3
  conv = conv_factory(data, ch_3x3, (3,3), stride=(2,2), pad=(1,1))
  # pool
  pool = mx.Pooling(data, kernel=(3,3), stride=(2,2), pool_type=:max)
  # concat
  concat = mx.Concat(conv, pool)
  return concat
end

# a simple module
function simple_factory(data, ch_1x1, ch_3x3)
  # 1x1
  conv1x1 = conv_factory(data, ch_1x1, (1,1); pad=(0,0))
  # 3x3
  conv3x3 = conv_factory(data, ch_3x3, (3,3); pad=(1,1))
  # concat
  concat = mx.Concat(conv1x1, conv3x3)
  return concat
end


#--------------------------------------------------------------------------------
# Actual architecture
data    = mx.Variable(:data)
conv1   = conv_factory(data, 96, (3,3); pad=(1,1), act_type=:relu)
in3a    = simple_factory(conv1, 32, 32)
in3b    = simple_factory(in3a, 32, 48)
in3c    = downsample_factory(in3b, 80)
in4a    = simple_factory(in3c, 112, 48)
in4b    = simple_factory(in4a, 96, 64)
in4c    = simple_factory(in4b, 80, 80)
in4d    = simple_factory(in4b, 48, 96)
in4e    = downsample_factory(in4d, 96)
in5a    = simple_factory(in4e, 176, 160)
in5b    = simple_factory(in5a, 176, 160)
pool    = mx.Pooling(in5b, pool_type=:avg, kernel=(7,7), name=:global_pool)
flatten = mx.Flatten(pool, name=:flatten1)
fc      = mx.FullyConnected(flatten, num_hidden=10, name=:fc1)
softmax = mx.SoftmaxOutput(fc, name=:loss)


#--------------------------------------------------------------------------------
# Prepare data
filenames = mx.get_cifar10()
batch_size = 128
num_epoch  = 10
num_gpus   = 8

train_provider = mx.ImageRecordProvider(label_name=:loss_label,
        path_imgrec=filenames[:train], mean_img=filenames[:mean],
        rand_crop=true, rand_mirror=true, data_shape=(28,28,3),
        batch_size=batch_size, preprocess_threads=1)
test_provider = mx.ImageRecordProvider(label_name=:loss_label,
        path_imgrec=filenames[:test], mean_img=filenames[:mean],
        rand_crop=false, rand_mirror=false, data_shape=(28,28,3),
        batch_size=batch_size, preprocess_threads=1)


#--------------------------------------------------------------------------------
# Training model
gpus = [mx.Context(mx.GPU, i) for i = 0:num_gpus-1]
model = mx.FeedForward(softmax, context=gpus)

# optimizer
optimizer = mx.SGD(η=0.05, μ=0.9, λ=0.0001)

# fit parameters
mx.fit(model, optimizer, train_provider, n_epoch=num_epoch, eval_data=test_provider,
       initializer=mx.UniformInitializer(0.07), callbacks=[mx.speedometer()])
