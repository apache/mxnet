import boto3
import mxnet as mx
import numpy as np
import json
import os
import logging
logging.getLogger().setLevel(logging.DEBUG)


# Set fixed random seeds. These would be the same for inference files as well
mx.random.seed(7)
np.random.seed(7)

# get the current mxnet version we are running on
mxnet_version = mx.__version__
bucket_name = 'mxnet-model-backwards-compatibility'
ctx = mx.cpu()
batch_size = 100
num_epoch = 2
backslash = '/'
model_name = 'mnist_mlp_module_api'


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

def upload_model_files_to_s3(bucket_name, files, folder_name):
    s3 = boto3.client('s3')
    for file in files:
        s3.upload_file(file, bucket_name, folder_name + file)
    print ('model successfully uploaded to s3')

def clean_up_files ():
    clean_mnist_data()
    clean_model_files()
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

def clean_model_files():
    for i in range(1, num_epoch+1):
        if os.path.isfile(model_name + '-000' + str(i) + '.params'):
            os.remove(model_name + '-000' + str(i) + '.params')

    if os.path.isfile(model_name + '-symbol.json'):
        os.remove(model_name + '-symbol.json')
    if os.path.isfile(inference_results_file):
        os.remove(inference_results_file)

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

    # Write the inference results to local json file. This will be cleaned up later
    with open(inference_results_file, 'w') as file:
        json.dump(inference_results, file)


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