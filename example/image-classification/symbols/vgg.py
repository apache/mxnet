"""References:

Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for
large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/
"""
import mxnet as mx
import numpy as np
def vgg_11(input_data):
    # group 1
    conv1_1 = mx.sym.Convolution(data=input_data, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_1")
    relu1_1 = mx.sym.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    pool1 = mx.sym.Pooling(
        data=relu1_1, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool1")
    # group 2
    conv2_1 = mx.sym.Convolution(
        data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_1")
    relu2_1 = mx.sym.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    pool2 = mx.sym.Pooling(
        data=relu2_1, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool2")
    # group 3
    conv3_1 = mx.sym.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_1")
    relu3_1 = mx.sym.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.sym.Convolution(
        data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_2")
    relu3_2 = mx.sym.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    pool3 = mx.sym.Pooling(
        data=relu3_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool3")
    # group 4
    conv4_1 = mx.sym.Convolution(
        data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_1")
    relu4_1 = mx.sym.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.sym.Convolution(
        data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_2")
    relu4_2 = mx.sym.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    pool4 = mx.sym.Pooling(
        data=relu4_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool4")
    # group 5
    conv5_1 = mx.sym.Convolution(
        data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_1")
    relu5_1 = mx.sym.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.sym.Convolution(
        data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_2")
    relu5_2 = mx.sym.Activation(data=conv5_2, act_type="relu", name="relu5_2")
    pool5 = mx.sym.Pooling(
        data=relu5_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool5")
    # return    
    return pool5

def vgg_16(input_data):
     # group 1
    conv1_1 = mx.sym.Convolution(data=input_data, kernel=(3, 3), pad=(1, 1), num_filter=64,
                 name="conv1_1")
    relu1_1 = mx.sym.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    conv1_2 = mx.sym.Convolution(data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=64,
                 name="conv1_2")            
    relu1_2 = mx.sym.Activation(data=conv1_2, act_type="relu", name="relu1_2")
    pool1 = mx.sym.Pooling(data=relu1_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool1")
    # group 2
    conv2_1 = mx.sym.Convolution(data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128,
                 name="conv2_1")            
    relu2_1 = mx.sym.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    conv2_2 = mx.sym.Convolution(data=relu2_1, kernel=(3, 3), pad=(1, 1), num_filter=128,
                 name="conv2_2")              
    relu2_2 = mx.sym.Activation(data=conv2_2, act_type="relu", name="relu2_2")
    pool2 = mx.sym.Pooling(data=relu2_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool2")
    # group 3
    conv3_1 = mx.sym.Convolution(data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256,
                 name="conv3_1")
    relu3_1 = mx.sym.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.sym.Convolution(data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256,
                 name="conv3_2")
    relu3_2 = mx.sym.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    conv3_3 = mx.sym.Convolution(data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=256,
                 name="conv3_3") 
    relu3_3 = mx.sym.Activation(data=conv3_3, act_type="relu", name="relu3_3")
    pool3 = mx.sym.Pooling(data=relu3_3, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool3")
    # group 4
    conv4_1 = mx.sym.Convolution(data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512,
                 name="conv4_1")
    relu4_1 = mx.sym.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.sym.Convolution(data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512,
                 name="conv4_2")
    relu4_2 = mx.sym.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    conv4_3 = mx.sym.Convolution(data=relu4_2, kernel=(3, 3), pad=(1, 1), num_filter=512,
                 name="conv4_3")
    relu4_3 = mx.sym.Activation(data=conv4_3, act_type="relu", name="relu4_3")
    pool4 = mx.sym.Pooling(data=relu4_3, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool4")
    # group 5
    conv5_1 = mx.sym.Convolution(data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512,
                 name="conv5_1")
    relu5_1 = mx.sym.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.sym.Convolution(data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512,
                 name="conv5_2")
    relu5_2 = mx.sym.Activation(data=conv5_2, act_type="relu", name="relu5_2")
    conv5_3 = mx.sym.Convolution(data=relu5_2, kernel=(3, 3), pad=(1, 1), num_filter=512,
                 name="conv5_3")
    relu5_3 = mx.sym.Activation(data=conv5_3, act_type="relu", name="relu5_3")
    pool5 = mx.sym.Pooling(data=relu5_3, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool5")
    # return
    return pool5

def vgg_19(input_data):
    # group 1
    conv1_1 = mx.sym.Convolution(data=input_data, kernel=(3, 3), pad=(1, 1), num_filter=64,
                 name="conv1_1")
    relu1_1 = mx.sym.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    conv1_2 = mx.sym.Convolution(data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=64,
                 name="conv1_2")            
    relu1_2 = mx.sym.Activation(data=conv1_2, act_type="relu", name="relu1_2")
    pool1 = mx.sym.Pooling(data=relu1_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool1")
    # group 2
    conv2_1 = mx.sym.Convolution(data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128,
                 name="conv2_1")            
    relu2_1 = mx.sym.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    conv2_2 = mx.sym.Convolution(data=relu2_1, kernel=(3, 3), pad=(1, 1), num_filter=128,
                 name="conv2_2")              
    relu2_2 = mx.sym.Activation(data=conv2_2, act_type="relu", name="relu2_2")
    pool2 = mx.sym.Pooling(data=relu2_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool2")
    # group 3
    conv3_1 = mx.sym.Convolution(data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256,
                 name="conv3_1")
    relu3_1 = mx.sym.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.sym.Convolution(data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256,
                 name="conv3_2")
    relu3_2 = mx.sym.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    conv3_3 = mx.sym.Convolution(data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=256,
                 name="conv3_3") 
    relu3_3 = mx.sym.Activation(data=conv3_3, act_type="relu", name="relu3_3")
    conv3_4 = mx.sym.Convolution(data=relu3_3, kernel=(3, 3), pad=(1, 1), num_filter=256,
                 name="conv3_4") 
    relu3_4 = mx.sym.Activation(data=conv3_4, act_type="relu", name="relu3_4")
    pool3 = mx.sym.Pooling(data=relu3_4, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool3")
    # group 4
    conv4_1 = mx.sym.Convolution(data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512,
                 name="conv4_1")
    relu4_1 = mx.sym.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.sym.Convolution(data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512,
                 name="conv4_2")
    relu4_2 = mx.sym.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    conv4_3 = mx.sym.Convolution(data=relu4_2, kernel=(3, 3), pad=(1, 1), num_filter=512,
                 name="conv4_3")
    relu4_3 = mx.sym.Activation(data=conv4_3, act_type="relu", name="relu4_3")
    conv4_4 = mx.sym.Convolution(data=relu4_3, kernel=(3, 3), pad=(1, 1), num_filter=512,
                 name="conv4_4")
    relu4_4 = mx.sym.Activation(data=conv4_4, act_type="relu", name="relu4_4")
    pool4 = mx.sym.Pooling(data=relu4_4, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool4")
    # group 5
    conv5_1 = mx.sym.Convolution(data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512,
                 name="conv5_1")
    relu5_1 = mx.sym.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.sym.Convolution(data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512,
                 name="conv5_2")
    relu5_2 = mx.sym.Activation(data=conv5_2, act_type="relu", name="relu5_2")
    conv5_3 = mx.sym.Convolution(data=relu5_2, kernel=(3, 3), pad=(1, 1), num_filter=512,
                 name="conv5_3")
    relu5_3 = mx.sym.Activation(data=conv5_3, act_type="relu", name="relu5_3")
    conv5_4 = mx.sym.Convolution(data=relu5_3, kernel=(3, 3), pad=(1, 1), num_filter=512,
                 name="conv5_4")
    relu5_4 = mx.sym.Activation(data=conv5_4, act_type="relu", name="relu5_4")
    pool5 = mx.sym.Pooling(data=relu5_4, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool5")
    # return
    return pool5 

def fully_connected(input_data, num_classes, dtype='float32'):
    # group 6
    flatten = mx.sym.Flatten(data=input_data, name="flatten")
    fc6 = mx.sym.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
    relu6 = mx.sym.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.sym.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.sym.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
    relu7 = mx.sym.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.sym.Dropout(data=relu7, p=0.5, name="drop7")
    # output
    fc8 = mx.sym.FullyConnected(data=drop7, num_hidden=num_classes, name="fc8")
    if dtype == 'float16':
        fc8 = mx.sym.Cast(data=fc8, dtype=np.float32)  
    softmax = mx.sym.SoftmaxOutput(data=fc8, name='softmax')
    # return
    return softmax      

def get_symbol(num_classes, num_layers, dtype='float32', **kwargs):
    ## define VGG
    data = mx.sym.Variable(name="data")
    import pdb
    pdb.set_trace()
    if dtype == 'float16':
        data = mx.sym.Cast(data=data, dtype=np.float16)

    if num_layers == 11:
        output = vgg_11(data)
    elif num_layers == 16:
        output = vgg_16(data)
    elif num_layers == 19:
        output = vgg_19(data)
    else:
        raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))

    output = fully_connected(output, num_layers, dtype)

    return output
