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


import boto3
import mxnet as mx
import numpy as np
import json
import logging
import os
logging.getLogger().setLevel(logging.DEBUG)
mx.random.seed(7)
np.random.seed(7)

bucket_name = 'mxnet-model-backwards-compatibility'
backslash = '/'
model_name = 'mnist_mlp_module_api'
s3 = boto3.resource('s3')
num_epoch = 2
ctx = mx.cpu()

def prepare_mnist_data(mnist_raw_data):
    
    #shuffle the indices
    indices = np.random.permutation(mnist_raw_data['train_label'].shape[0])

    #print indices[0:10]
    train_idx , val_idx = indices[:50000], indices[50000:]

    train_data = mnist_raw_data['train_data'][train_idx,:]
    train_label = mnist_raw_data['train_label'][train_idx]
    
    val_data = mnist_raw_data['train_data'][val_idx,:]
    val_label = mnist_raw_data['train_label'][val_idx]
    
    test_data = mnist_raw_data['test_data']
    test_label = mnist_raw_data['test_label']

    #print len(train_data)
    #print len(val_data)
    
    train = {'train_X' : train_data, 'train_Y' : train_label}
    test = {'test_X' : test_data, 'test_Y' : test_label}
    val = {'val_X' : val_data, 'val_Y' : val_label}
    
    data = dict()
    data['train'] = train
    data['test'] = test
    data['val'] = val
    
    return data

def get_val_test_iter():
    data = prepare_mnist_data(mx.test_utils.get_mnist())
    val = data['val']
    test = data['test']
    batch_size = 100
    val_iter = mx.io.NDArrayIter(val['val_X'], val['val_Y'], batch_size, shuffle=True)
    test_iter = mx.io.NDArrayIter(test['test_X'], test['test_Y'])
    return val_iter, test_iter

val_iter, test_iter = get_val_test_iter()

def get_top_level_folders_in_bucket(s3client, bucket_name):
    '''This function returns the top level folders in the S3Bucket. These folders help us to navigate to the trained model files stored for different MXNet versions. '''
    bucket = s3client.Bucket(bucket_name)
    result = bucket.meta.client.list_objects(Bucket=bucket.name,
                                         Delimiter=backslash)
    folder_list = list()
    for obj in result['CommonPrefixes']:
        folder_list.append(obj['Prefix'].strip(backslash))

    return folder_list

def get_model():
    ##### Old Model ##### : 
    input = mx.symbol.Variable('data')
    input = mx.symbol.Flatten(data=input)

    fc1 = mx.symbol.FullyConnected(data=input, name='fc1', num_hidden=128)
    act1 = mx.symbol.Activation(data=fc1, name='relu1', act_type='relu')

    fc2 = mx.symbol.FullyConnected(data=act1, name='fc2', num_hidden=64)
    output = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')

    ### this is needed since the model is loaded from a checkpoint ###
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_name, num_epoch)
    loaded_model = mx.mod.Module(symbol=output, context=ctx, data_names=['data'], label_names=['softmax_label'])
    loaded_model.bind(data_shapes=test_iter.provide_data, label_shapes=test_iter.provide_label)
    loaded_model.set_params(arg_params, aux_params)
    return loaded_model

def perform_inference(test_iter, val_iter, model, inference_file):
    test_inference_score = model.score(test_iter, ['acc'])
    val_inference_score = model.score(val_iter, ['acc'])

    with open(inference_file, 'r') as file:
        results = json.load(file)

    print ('Validation accuracy on inference is %f while that on the original training file is %f' % (val_inference_score[0][1], results['val_acc']))
    print ('Test accuracy on inference is %f while that on the original training file is %f' % (test_inference_score[0][1], results['test_acc']))
    assert(results['val_acc'] == val_inference_score[0][1])
    assert(results['test_acc'] == test_inference_score[0][1])
    print ('Inference results passed for %s' % model_name) 

def clean_mnist_data():
    if os.path.isfile('train-images-idx3-ubyte.gz'):
        os.remove('train-images-idx3-ubyte.gz')
    if os.path.isfile('t10k-labels-idx1-ubyte.gz'):
        os.remove('t10k-labels-idx1-ubyte.gz')
    if os.path.isfile('train-labels-idx1-ubyte.gz'):
        os.remove('train-labels-idx1-ubyte.gz')
    if os.path.isfile('t10k-images-idx3-ubyte.gz'):
        os.remove('t10k-images-idx3-ubyte.gz')

def clean_model_files(model_files):
    for file in model_files:
        if os.path.isfile(file):
            os.remove(file)

if __name__=='__main__':
    for folder in get_top_level_folders_in_bucket(s3, bucket_name):
        bucket = s3.Bucket(bucket_name)
        prefix = folder + backslash + model_name
        model_files_meta = list(bucket.objects.filter(Prefix = prefix))
        if len(model_files_meta) == 0:
            continue
        model_files = list()
        for obj in model_files_meta:
            file_name = obj.key.split('/')[2]
            model_files.append(file_name)
            ## Download this file---
            bucket.download_file(obj.key, file_name)

        model = get_model()
        perform_inference(test_iter, val_iter, model, model_name + '_inference.json')
        clean_model_files(model_files)
        clean_mnist_data()
