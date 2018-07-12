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

def train_module_checkpoint_api():
	model_name = 'module_checkpoint_api'
	print ('Saving files for model %s' %model_name)
	### Prepare data
	test_data = mx.nd.array(np.random.uniform(-1, 1, size=(20, 1)))
	test_label = mx.nd.array(np.random.randint(0, 2, size=(20,)), dtype='float32')
	data_iter = mx.io.NDArrayIter(test_data, test_label, batch_size=10)


	mod = get_module_api_model_definition()
	mod.bind(data_shapes=data_iter.provide_data, label_shapes=data_iter.provide_label)
	weights = mx.initializer.Xavier(magnitude = 2.57)
	mod.init_params(weights)

	mod.save_checkpoint(model_name, 1)
	### Save the data, labels
	save_data_and_labels(test_data, test_label, model_name)
	upload_data_and_labels_to_s3(model_name)

	inference_results = mod.predict(data_iter)
	### Save inference_results
	save_inference_results(inference_results, model_name)
	### upload model and inference files to S3
	files = list()
	files.append(model_name + '-0001.params')
	files.append(model_name + '-symbol.json')
	files.append(model_name + '-inference')

	mxnet_folder = str(mxnet_version) + backslash + model_name + backslash

	upload_model_files_to_s3(files, mxnet_folder)

	clean_model_files(files, model_name)

def train_lenet_gluon_save_params_api():
	model_name = 'lenet_gluon_save_params_api'
	print ('Saving files for model %s' %model_name)
	net = Net()
	weights = mx.initializer.Xavier(magnitude = 2.57)
	net.initialize(weights, ctx = [mx.cpu(0)])
	### Prepare data

	test_data = mx.nd.array(np.random.uniform(-1, 1, size=(20, 1, 30, 30)))
	output = net(test_data)
	# print (y)
 #    ### Save the test data as well.
 #    ### Save the inference output ys
 #    ### Save the model params

	mx.nd.save(model_name + '-data', {'data' : test_data})
	save_inference_results(output, model_name)
	net.save_params(model_name + '-params')

	mxnet_folder = str(mxnet_version) + backslash + model_name + backslash

	files = list()
	files.append(model_name + '-data')
	files.append(model_name + '-inference')
	files.append(model_name + '-params')

	upload_data_and_labels_to_s3(model_name)

	upload_model_files_to_s3(files, mxnet_folder)

	clean_model_files(files, model_name)

def train_lenet_gluon_hybrid_export_api():
	model_name = 'lenet_gluon_hybrid_export_api'
	print ('Saving files for model %s' %model_name)
	net = HybridNet()
	weights = mx.initializer.Xavier(magnitude = 2.57)
	net.initialize(weights, ctx = [mx.cpu(0)])
	net.hybridize()
	### Prepare data
	test_data = mx.nd.array(np.random.uniform(-1, 1, size=(20, 1, 30, 30)))
	output = net(test_data)
	# print (y)
    ### Save the test data as well.
    ### Save the inference output ys
    ### Save the model params

	mx.nd.save(model_name + '-data', {'data' : test_data})
	save_inference_results(output, model_name)
	net.export(model_name, epoch=1)

	mxnet_folder = str(mxnet_version) + backslash + model_name + backslash

	files = list()
	files.append(model_name + '-data')
	files.append(model_name + '-inference')
	files.append(model_name + '-0001.params')
	files.append(model_name + '-symbol.json')


	upload_data_and_labels_to_s3(model_name)

	upload_model_files_to_s3(files, mxnet_folder)

	clean_model_files(files, model_name)

def train_lstm_gluon_save_parameters_api():
	## If this code is being run on version >= 1.2.0 only then execute it, since it uses save_parameters and load_parameters API
    if compare_versions(str(mxnet_version), '1.2.1')  < 0:
        print ('Found MXNet version %s and exiting because this version does not contain save_parameters and load_parameters functions' %str(mxnet_version))
        sys.exit(1)
        
	model_name = 'lstm_gluon_save_parameters_api'
	print ('Saving files for model %s' %model_name)
	net = SimpleLSTMModel()
	weights = mx.initializer.Xavier(magnitude = 2.57)
	net.initialize(weights, ctx = [mx.cpu(0)])

	test_data = mx.nd.array(np.random.uniform(-1, 1, size=(10, 30)))
	output = net(test_data)
	# print output
	mx.nd.save(model_name + '-data', {'data' : test_data})
	save_inference_results(output, model_name)
	net.save_parameters(model_name + '-params')

	mxnet_folder = str(mxnet_version) + backslash + model_name + backslash

	files = list()
	files.append(model_name + '-data')
	files.append(model_name + '-inference')
	files.append(model_name + '-params')

	upload_data_and_labels_to_s3(model_name)

	upload_model_files_to_s3(files, mxnet_folder)

	clean_model_files(files, model_name)


if __name__=='__main__':
	train_module_checkpoint_api()
	train_lenet_gluon_save_params_api()
	train_lenet_gluon_hybrid_export_api()
	train_lstm_gluon_save_parameters_api()
