#!/usr/bin/env python

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
from common import *

batch_size=100
num_epoch = 2
model_name = 'lenet_cnn_gluon_api'

ctx = [mx.cpu(0)]
mxnet_version = mx.__version__

def clean_up_files (model_files):
    clean_mnist_data()
    clean_model_files(model_files)
    print ('Model files deleted')

def save_model_files(network):
    model_file_name = model_name + '.params'
    network.save_params(model_file_name)

def get_inference_score(iter, model):
    # Use Accuracy as the evaluation metric.
    metric = mx.metric.Accuracy()
    # Reset the validation data iterator.
    iter.reset()
    # Loop over the validation data iterator.
    for batch in iter:
        # Splits validation data into multiple slices along batch_axis
        # and copy each slice into a context.
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        # Splits validation label into multiple slices along batch_axis
        # and copy each slice into a context.
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        for x in data:
            outputs.append(model(x))
        # Updates internal evaluation
        metric.update(label, outputs)
    acc = metric.get()
    return acc[1]

if __name__=='__main__':
    data = prepare_mnist_data(mx.test_utils.get_mnist())

    train = data['train']
    val = data['val']
    test = data['test']

    train_iter = mx.io.NDArrayIter(train['train_X'], train['train_Y'], batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(val['val_X'], val['val_Y'], batch_size, shuffle=True)
    test_iter = mx.io.NDArrayIter(test['test_X'], test['test_Y'])


    net = Net()
    net.initialize(mx.init.Xavier(), ctx=ctx)

    metric = mx.metric.Accuracy()
    softmax_cross_entropy_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.02})

    for i in range(num_epoch):
        train_iter.reset()
        for batch in train_iter:
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
            outputs = []
            # Inside training scope
            with ag.record():
                for x, y in zip(data, label):
                    z = net(x)
                    # Computes softmax cross entropy loss.
                    loss = softmax_cross_entropy_loss(z, y)
                    # Backpropagate the error for one iteration.
                    loss.backward()
                    outputs.append(z)
                    
            metric.update(label, outputs)
            # Make one step of parameter update. Trainer needs to know the
            # batch size of data to normalize the gradient by 1/batch_size.
            trainer.step(batch.data[0].shape[0])
            
        name, acc = metric.get()
        # Reset evaluation result to initial state.
        metric.reset()
        print('training acc at epoch %d: %s=%f'%(i, name, acc))

        save_model_files(net)


    # In[6]:
    val_acc = get_inference_score(val_iter, net)
    print('validation acc: =%f'%val_acc)

    test_acc = get_inference_score(test_iter, net)
    print('test acc: =%f'%test_acc)

    inference_results = dict()
    inference_results['val_acc'] = val_acc
    inference_results['test_acc'] = test_acc

    inference_results_file = model_name + '_inference' + '.json'

    save_inference_results(inference_results_file, inference_results)

    mxnet_folder = str(mxnet_version) + backslash + model_name + backslash

    files = list()
    files.append(model_name + '.params')
    files.append(model_name + '_inference' + '.json')

    upload_model_files_to_s3(bucket_name, files, mxnet_folder)

    clean_up_files(files)