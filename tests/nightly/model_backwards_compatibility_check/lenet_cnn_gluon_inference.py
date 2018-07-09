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

model_name = 'lenet_cnn_gluon_api'
num_epoch = 2
ctx = [mx.cpu(0)]
batch_size = 100

val_iter, test_iter = get_val_test_iter()

def get_model(model_file):
	net = Net()
	net.load_params(model_file, ctx)

	return net

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

def perform_inference(test_iter, val_iter, model, inference_file):
	test_inference_score = get_inference_score(test_iter, model)
	val_inference_score = get_inference_score(val_iter, model)

	with open(inference_file, 'r') as file:
		results = json.load(file)

	print (test_inference_score, val_inference_score)
	print results['val_acc']
	print ('Validation accuracy on inference is %f while that on the original training file is %f' % (val_inference_score, results['val_acc']))
	print ('Test accuracy on inference is %f while that on the original training file is %f' % (test_inference_score, results['test_acc']))
	assert(results['val_acc'] == val_inference_score)
	assert(results['test_acc'] == test_inference_score)
	print ('Inference results passed for %s' % model_name) 

def clean_up_files (model_files):
    clean_mnist_data()
    clean_model_files(model_files)
    print ('Model files deleted')

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

        model = get_model(model_name + '.params')
        perform_inference(test_iter, val_iter, model, model_name + '_inference.json')
        clean_up_files(model_files)
