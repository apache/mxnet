'''
Deep Residual Learning for Image Recognition, http://arxiv.org/abs/1512.03385
an exmaple of deep residual network for cifar10

commands & setups:
set following parameters in example/image-classification/train_model.py
    momentum = 0.9,
    wd = 0.0001,
    initializer = mx.init.Xavier(rnd_type="gaussian", factor_type="in", magnitude=2.0)
set n=3(3 for 20 layers, n=9 for 56 layers) in the get_symbol function in example/image-classification/symbol_resnet-28-small.py

#first train the network with lr=0.1 for 80 epoch
python example/image-classification/train_cifar10.py --network resnet-28-small --num-examples 50000 --lr 0.1 --num-epoch 80 --model-prefix cifar10/resnet 

#second train the network with lr=0.01 from epoch 81 to epoch 120, with lr=0.001 from epoch 121 to epoch 160
python example/image-classification/train_cifar10.py --network resnet-28-small --num-examples 50000 --model-prefix cifar10/resnet --load-epoch 80 --lr 0.01 --lr-factor 0.1 --lr-factor-epoch 40 --num-epoch 200 
#in the paper, he train cifar10 for 160 epoch, I set num-epoch to 200 because I want to see whether it is usefull when set lr=0.0001

#since it needs 160 epochs, please be patient
#and I use batch-size of 128, train the models on one GPU
accuracy:
for 20 layers resnet, accuracy=0.905+, 0.9125 in the paper
for 32 layers resnet, accuracy=0.908+, 0.9239 in the paper
for 56 layers resnet, accuracy=0.915+, 0.9303 in the paper

though the numbers are a little bit lower than the paper, but it does obey the rule: the deeper, the better

differences to the paper on cifar10 network setup
1. in the paper, the author use identity shortcut when dealing with increasing dimensions, while I use 1*1 convolutions to deal with it
2. in the paper, 4 pixels are padded on each side and a 32*32 crop is randomly sampled from the padded image, while I use the dataset provided by mxnet, so the input is 28*28, as a results for 3 different kinds of 2n layers output map sizes are 28*28, 14*14, 7*7, instead of 32*32, 16*16, 8*8 in the paper.

the above two reason might answer why the accuracy is a bit lower than the paper, I suppose.
Off course, there might be other reasons(for example the true network architecture may be different from my script, since my script is just my understanding of the paper), if you find out, please tell me, declanxu@gmail.com or declanxu@126.com, thanks

'''
import mxnet as mx
import find_mxnet

def conv_factory(data, num_filter, kernel, stride, pad, act_type = 'relu', conv_type = 0):
    if conv_type == 0:
        conv = mx.symbol.Convolution(data = data, num_filter = num_filter, kernel = kernel, stride = stride, pad = pad)
        bn = mx.symbol.BatchNorm(data=conv)
        act = mx.symbol.Activation(data = bn, act_type=act_type)
        return act
    elif conv_type == 1:
        conv = mx.symbol.Convolution(data = data, num_filter = num_filter, kernel = kernel, stride = stride, pad = pad)
        bn = mx.symbol.BatchNorm(data=conv)
        return bn

def residual_factory(data, num_filter, dim_match):
    if dim_match == True: # if dimension match
        identity_data = data
        conv1 = conv_factory(data=data, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1), act_type='relu', conv_type=0)
        
        conv2 = conv_factory(data=conv1, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1), conv_type=1)
        new_data = identity_data + conv2
        act = mx.symbol.Activation(data=new_data, act_type='relu')
        return act
    else:        
        conv1 = conv_factory(data=data, num_filter=num_filter, kernel=(3,3), stride=(2,2), pad=(1,1), act_type='relu', conv_type=0)
        conv2 = conv_factory(data=conv1, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1), conv_type=1)

        # adopt project method in the paper when dimension increased
        project_data = conv_factory(data=data, num_filter=num_filter, kernel=(1,1), stride=(2,2), pad=(0,0), conv_type=1)
        new_data = project_data + conv2
        act = mx.symbol.Activation(data=new_data, act_type='relu')
        return act

def residual_net(data, n):
    #fisrt 2n layers
    for i in range(n):
        data = residual_factory(data=data, num_filter=16, dim_match=True)
    
    #second 2n layers
    for i in range(n):
        if i==0:
            data = residual_factory(data=data, num_filter=32, dim_match=False)
        else:
            data = residual_factory(data=data, num_filter=32, dim_match=True)
    
    #third 2n layers
    for i in range(n):
        if i==0:
            data = residual_factory(data=data, num_filter=64, dim_match=False)
        else:
            data = residual_factory(data=data, num_filter=64, dim_match=True)
     
    return data

def get_symbol(num_classes = 10):
    conv = conv_factory(data=mx.symbol.Variable(name='data'), num_filter=16, kernel=(3,3), stride=(1,1), pad=(1,1), act_type='relu', conv_type=0)
    n = 3 # set n = 3 means get a model with 3*6+2=20 layers, set n = 9 means 9*6+2=56 layers
    resnet = residual_net(conv, n) # 
    pool = mx.symbol.Pooling(data=resnet, kernel=(7,7), pool_type='avg')
    flatten = mx.symbol.Flatten(data=pool, name='flatten')
    fc = mx.symbol.FullyConnected(data=flatten, num_hidden=num_classes,  name='fc1')
    softmax = mx.symbol.SoftmaxOutput(data=fc, name='softmax')
    return softmax


