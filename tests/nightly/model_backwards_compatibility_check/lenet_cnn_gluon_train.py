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

batch_size=100
num_epoch = 2
bucket_name = 'mxnet-model-backwards-compatibility'
backslash = '/'
model_name = 'lenet_cnn_gluon_api'

ctx = [mx.cpu(0)]
mxnet_version = mx.__version__

class Net(gluon.Block):
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

    def forward(self, x):
        x = self.pool1(F.tanh(self.conv1(x)))
        x = self.pool2(F.tanh(self.conv2(x)))
        # 0 means copy over size from corresponding dimension.
        # -1 means infer size from the rest of dimensions.
        x = x.reshape((0, -1))
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x


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

def save_model_files(network):
    model_file_name = model_name + '.params'
    network.save_params(model_file_name)

def save_inference_results(test_acc, val_acc):
    inference_results = dict()
    inference_results['val_acc'] = val_acc
    inference_results['test_acc'] = test_acc

    inference_results_file = model_name + '_inference' + '.json'

    # Write the inference results to local json file. This will be cleaned up later
    with open(inference_results_file, 'w') as file:
        json.dump(inference_results, file)

def upload_model_files_to_s3(bucket_name, files, folder_name):
    s3 = boto3.client('s3')
    for file in files:
        s3.upload_file(file, bucket_name, folder_name + file)
    print ('model successfully uploaded to s3')

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

    save_inference_results(test_acc, val_acc)

    mxnet_folder = str(mxnet_version) + backslash + model_name + backslash

    files = list()
    files.append(model_name + '.params')
    files.append(model_name + '_inference' + '.json')

    upload_model_files_to_s3(bucket_name, files, mxnet_folder)

    clean_up_files(files)