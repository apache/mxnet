import boto3
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import nd, autograd, gluon
from mxnet.gluon.data.vision import transforms, datasets
import numpy as np
from mxnet import autograd as ag
import logging
import mxnet.ndarray as F
from mxnet.gluon import nn
import json
import os

logging.getLogger().setLevel(logging.DEBUG)
mx.random.seed(7)
np.random.seed(7)

bucket_name = 'mxnet-model-backwards-compatibility'
backslash = '/'
model_name = 'lenet_cnn_gluon_hybrid_api'
s3 = boto3.resource('s3')
num_epoch = 2
ctx = [mx.cpu(0)]
batch_size = 100

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

class Net(gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)
        with self.name_scope():
            # layers created in name_scope will inherit name space
            # from parent layer.
            self.conv1 = nn.Conv2D(20, kernel_size=(5,5))
            self.pool1 = nn.MaxPool2D(pool_size=(2,2), strides = (2,2))
            self.conv2 = nn.Conv2D(50, kernel_size=(5,5))
            self.pool2 = nn.MaxPool2D(pool_size=(2,2), strides = (2,2))
            self.fc1 = nn.Dense(500)
            self.fc2 = nn.Dense(10)

    def hybrid_forward(self, F, x):
        x = self.pool1(F.tanh(self.conv1(x)))
        x = self.pool2(F.tanh(self.conv2(x)))
        # 0 means copy over size from corresponding dimension.
        # -1 means infer size from the rest of dimensions.
        x = x.reshape((0, -1))
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x


def get_model(model_name):
    net = gluon.SymbolBlock.imports(model_name + '-symbol.json', ['data'], model_name + '-000' + str(num_epoch) + '.params')
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

        model = get_model(model_name)
        perform_inference(test_iter, val_iter, model, model_name + '_inference.json')
        clean_up_files(model_files)
