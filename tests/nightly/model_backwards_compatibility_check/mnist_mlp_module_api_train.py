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

ctx = mx.cpu()
batch_size = 100
num_epoch = 2
backslash = '/'
model_name = 'mnist_mlp_module_api'

def clean_up_files ():
    clean_mnist_data()
    files = list()
    for i in range(1, num_epoch+1):
        files.append(model_name + '-000' + str(i) + '.params')
    
    files.append(model_name + '-symbol.json')
    files.append(inference_results_file)
    clean_model_files(files)
    print ('Model files deleted')

def get_model_definition():
    input = mx.symbol.Variable('data')
    input = mx.symbol.Flatten(data=input)

    fc1 = mx.symbol.FullyConnected(data=input, name='fc1', num_hidden=128)
    act1 = mx.symbol.Activation(data=fc1, name='relu1', act_type='relu')

    fc2 = mx.symbol.FullyConnected(data=act1, name='fc2', num_hidden=64)
    output = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')

    model = mx.mod.Module(symbol=output, context=ctx, data_names=['data'], label_names=['softmax_label'])

    return model

if __name__=='__main__':
    data = prepare_mnist_data(mx.test_utils.get_mnist())

    train = data['train']
    val = data['val']
    test = data['test']

    train_iter = mx.io.NDArrayIter(train['train_X'], train['train_Y'], batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(val['val_X'], val['val_Y'], batch_size, shuffle=True)
    test_iter = mx.io.NDArrayIter(test['test_X'], test['test_Y'])

    model = get_model_definition()

    train_iter.reset()
    checkpoint_callback = mx.callback.do_checkpoint(model_name)
    model.fit(train_iter, epoch_end_callback=checkpoint_callback, eval_data=val_iter, optimizer='sgd', optimizer_params={'learning_rate' : 0.1}, eval_metric='acc', num_epoch=num_epoch)

    score_val = model.score(val_iter,['acc'])
    val_acc = score_val[0][1]
    print ('Validation Accuracy is : %f' % val_acc)
    score_test = model.score(test_iter, ['acc'])
    test_acc = score_test[0][1]
    print ('Test Accuracy is : %f' % test_acc)

    inference_results = dict()
    inference_results['val_acc'] = val_acc
    inference_results['test_acc'] = test_acc

    inference_results_file = model_name + '_inference' + '.json'

    save_inference_results(inference_results_file, inference_results)

    model_params_file = model_name + '-000' + str(num_epoch) + '.params'
    model_symbol_file = model_name + '-symbol.json'
    model_inference_file = inference_results_file
    files = list()
    files.append(model_params_file)
    files.append(model_symbol_file)
    files.append(model_inference_file)


    mxnet_folder = str(mxnet_version) + backslash + model_name + backslash

    # Upload the model files to S3
    upload_model_files_to_s3(bucket_name, files, mxnet_folder)
    # Clean up the local files
    clean_up_files()