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

model_name = 'lm_rnn_gluon_api'

context = mx.cpu(0)

def test(test_data, model):
	test_L = eval(test_data, model)
	return test_L, np.exp(test_L)

def get_top_level_folders_in_bucket(s3client, bucket_name):
	'''This function returns the top level folders in the S3Bucket. These folders help us to navigate to the trained model files stored for different MXNet versions. '''
	bucket = s3client.Bucket(bucket_name)
	result = bucket.meta.client.list_objects(Bucket=bucket.name,
										 Delimiter=backslash)
	folder_list = list()
	for obj in result['CommonPrefixes']:
		folder_list.append(obj['Prefix'].strip(backslash))

	return folder_list

def get_model(model_file):
	model_2 = RNNModel(args_model, ntokens, args_emsize, args_nhid,
					   args_nlayers, args_dropout, args_tied)
	model_2.load_parameters(model_name + '.params', context)

	return model_2

def perform_inference(test_data, val_data, model, inference_file):
	test_loss, test_ppl = test(test_data, model)
	val_loss, val_ppl = test(val_data, model)

	with open(inference_file, 'r') as file:
		results = json.load(file)
	val_results = results['val']
	test_results = results['test']

	print ('Validation loss on inference is %f while that on the original training file is %f' % (val_loss, val_results['loss']))
	print ('Test loss on inference is %f while that on the original training file is %f' % (test_loss, test_results['loss']))

	assert(test_loss == test_results['loss'])
	assert(test_ppl == test_results['ppl'])

	assert(val_loss == val_results['loss'])
	assert(val_ppl == val_results['ppl'])

	print ('Inference results passed for %s' % model_name)

def clean_up_files (model_files):
	clean_ptb_data()
	clean_model_files(model_files)
	print ('Model files deleted')
	
def clean_model_files(model_files):
	for file in model_files:
		if os.path.isfile(file):
			os.remove(file)
	
if __name__=='__main__':
	
	corpus = Corpus(args_data)
	train_data = batchify(corpus.train, args_batch_size).as_in_context(context)
	val_data = batchify(corpus.valid, args_batch_size).as_in_context(context)
	test_data = batchify(corpus.test, args_batch_size).as_in_context(context)
	ntokens = len(corpus.dictionary)

	for folder in get_top_level_folders_in_bucket(s3, bucket_name):
		bucket = s3.Bucket(bucket_name)
		prefix = folder + backslash + model_name
		model_files_meta = list(bucket.objects.filter(Prefix = prefix))
		if len(model_files_meta) == 0:
			continue
		model_files = list()
		for obj in model_files_meta:
			# print 
			file_name = obj.key.split('/')[2]
			if file_name is None or len(file_name) == 0:
				continue
			model_files.append(file_name)
			## Download this file---
			bucket.download_file(obj.key, file_name)

		model = get_model(model_name + '.params')
		perform_inference(test_data, val_data, model, model_name + '_inference.json')
	clean_up_files(model_files)
