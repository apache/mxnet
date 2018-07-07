import math
import os
import time
import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn, rnn
import logging
import boto3
import json
logging.getLogger().setLevel(logging.DEBUG)
mx.random.seed(7)
np.random.seed(7)

mxnet_version = mx.__version__
bucket_name = 'mxnet-model-backwards-compatibility'
ctx = mx.cpu()
backslash = '/'
model_name = 'lm_rnn_gluon_api'
s3 = boto3.resource('s3')


args_data = 'ptb.'
args_model = 'rnn_relu'
args_emsize = 100
args_nhid = 100
args_nlayers = 2
args_lr = 1.0
args_clip = 0.2
args_epochs = 2
args_batch_size = 32
args_bptt = 5
args_dropout = 0.2
args_tied = True
args_cuda = 'store_true'
args_log_interval = 500
args_save = model_name + '.params'

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.download_data_from_s3()
        self.train = self.tokenize(path + 'train.txt')
        self.valid = self.tokenize(path + 'valid.txt')
        self.test = self.tokenize(path + 'test.txt')

    def download_data_from_s3(self):
    	print ('Downloading files from bucket : %s' %bucket_name)
    	bucket = s3.Bucket(bucket_name)
    	files = ['test.txt', 'train.txt', 'valid.txt']
    	for file in files:
    		if os.path.exists(args_data + file) :
    			print ('File %s'%(args_data + file), 'already exists. Skipping download')
    			continue
    		file_path = str(mxnet_version) + backslash + model_name + backslash + args_data + file
    		bucket.download_file(file_path, args_data + file) 

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = np.zeros((tokens,), dtype='int32')
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return mx.nd.array(ids, dtype='int32')

class RNNModel(gluon.Block):
    """A model with an encoder, recurrent layer, and a decoder."""

    def __init__(self, mode, vocab_size, num_embed, num_hidden,
                 num_layers, dropout=0.5, tie_weights=False, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        with self.name_scope():
            self.drop = nn.Dropout(dropout)
            self.encoder = nn.Embedding(vocab_size, num_embed,
                                        weight_initializer = mx.init.Uniform(0.1))
            if mode == 'rnn_relu':
                self.rnn = rnn.RNN(num_hidden, num_layers, activation='relu', dropout=dropout,
                                   input_size=num_embed)
            elif mode == 'rnn_tanh':
                self.rnn = rnn.RNN(num_hidden, num_layers, dropout=dropout,
                                   input_size=num_embed)
            elif mode == 'lstm':
                self.rnn = rnn.LSTM(num_hidden, num_layers, dropout=dropout,
                                    input_size=num_embed)
            elif mode == 'gru':
                self.rnn = rnn.GRU(num_hidden, num_layers, dropout=dropout,
                                   input_size=num_embed)
            else:
                raise ValueError("Invalid mode %s. Options are rnn_relu, "
                                 "rnn_tanh, lstm, and gru"%mode)
            if tie_weights:
                self.decoder = nn.Dense(vocab_size, in_units = num_hidden,
                                        params = self.encoder.params)
            else:
                self.decoder = nn.Dense(vocab_size, in_units = num_hidden)
            self.num_hidden = num_hidden

    def forward(self, inputs, hidden):
        emb = self.drop(self.encoder(inputs))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.reshape((-1, self.num_hidden)))
        return decoded, hidden

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)

context = mx.cpu(0)

def batchify(data, batch_size):
    """Reshape data into (num_example, batch_size)"""
    nbatch = data.shape[0] // batch_size
    data = data[:nbatch * batch_size]
    data = data.reshape((batch_size, nbatch)).T
    return data

def get_batch(source, i):
    seq_len = min(args_bptt, source.shape[0] - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len]
    return data, target.reshape((-1,))

def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [i.detach() for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden

def eval(data_source, model):
    total_L = 0.0
    ntotal = 0
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    hidden = model.begin_state(func = mx.nd.zeros, batch_size = args_batch_size, ctx=context)
    for i in range(0, data_source.shape[0] - 1, args_bptt):
        data, target = get_batch(data_source, i)
        output, hidden = model(data, hidden)
        L = loss(output, target)
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
    return total_L / ntotal

def test(test_data, model):
    test_L = eval(test_data, model)
    return test_L, math.exp(test_L)

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
	model_2.load_parameters(args_save, context)

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

def clean_ptb_data():
	files = ['test.txt', 'train.txt', 'valid.txt']
	for file in files: 
		if os.path.isfile(args_data + file):
			os.remove(args_data + file)
    
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
