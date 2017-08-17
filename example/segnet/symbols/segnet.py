import mxnet as mx
def vgg_conv(input_data, fix_gamma, eps, bn_mom):
    # group 1
    conv1_1 = mx.symbol.Convolution(data=input_data, kernel=(3, 3), pad=(1, 1), num_filter=64,
                 name="conv1_1")
    bn_conv1_1 = mx.symbol.BatchNorm(data=conv1_1, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom, 
                 name='bn_conv1_1')
    relu1_1 = mx.symbol.Activation(data=bn_conv1_1, act_type="relu", name="relu1_1")
    conv1_2 = mx.symbol.Convolution(data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=64,
                 name="conv1_2")
    bn_conv1_2 = mx.symbol.BatchNorm(data=conv1_2, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom, 
                 name='bn_conv1_2')             
    relu1_2 = mx.symbol.Activation(data=bn_conv1_2, act_type="relu", name="relu1_2")
    pool1 = mx.symbol.Pooling(data=relu1_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool1")
    # group 2
    conv2_1 = mx.symbol.Convolution(data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128,
                 name="conv2_1")
    bn_conv2_1 = mx.symbol.BatchNorm(data=conv2_1, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom, 
                 name='bn_conv2_1')             
    relu2_1 = mx.symbol.Activation(data=bn_conv2_1, act_type="relu", name="relu2_1")
    conv2_2 = mx.symbol.Convolution(data=relu2_1, kernel=(3, 3), pad=(1, 1), num_filter=128,
                 name="conv2_2")  
    bn_conv2_2 = mx.symbol.BatchNorm(data=conv2_2, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom, 
                 name='bn_conv2_2')            
    relu2_2 = mx.symbol.Activation(data=bn_conv2_2, act_type="relu", name="relu2_2")
    pool2 = mx.symbol.Pooling(data=relu2_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool2")
    # group 3
    conv3_1 = mx.symbol.Convolution(data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256,
                 name="conv3_1")
    bn_conv3_1 = mx.symbol.BatchNorm(data=conv3_1, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom, 
                 name='bn_conv3_1') 
    relu3_1 = mx.symbol.Activation(data=bn_conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.symbol.Convolution(data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256,
                 name="conv3_2")
    bn_conv3_2 = mx.symbol.BatchNorm(data=conv3_2, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom, 
                 name='bn_conv3_2') 
    relu3_2 = mx.symbol.Activation(data=bn_conv3_2, act_type="relu", name="relu3_2")
    conv3_3 = mx.symbol.Convolution(data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=256,
                 name="conv3_3")
    bn_conv3_3 = mx.symbol.BatchNorm(data=conv3_3, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom, 
                 name='bn_conv3_3') 
    relu3_3 = mx.symbol.Activation(data=bn_conv3_3, act_type="relu", name="relu3_3")
    pool3 = mx.symbol.Pooling(data=relu3_3, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool3")
    # group 4
    conv4_1 = mx.symbol.Convolution(data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512,
                 name="conv4_1")
    bn_conv4_1 = mx.symbol.BatchNorm(data=conv4_1, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom, 
                 name='bn_conv4_1') 
    relu4_1 = mx.symbol.Activation(data=bn_conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.symbol.Convolution(data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512,
                 name="conv4_2")
    bn_conv4_2 = mx.symbol.BatchNorm(data=conv4_2, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom, 
                 name='bn_conv4_2') 
    relu4_2 = mx.symbol.Activation(data=bn_conv4_2, act_type="relu", name="relu4_2")
    conv4_3 = mx.symbol.Convolution(data=relu4_2, kernel=(3, 3), pad=(1, 1), num_filter=512,
                 name="conv4_3")
    bn_conv4_3 = mx.symbol.BatchNorm(data=conv4_3, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom, 
                 name='bn_conv4_3') 
    relu4_3 = mx.symbol.Activation(data=bn_conv4_3, act_type="relu", name="relu4_3")
    pool4 = mx.symbol.Pooling(data=relu4_3, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool4")
    # group 5
    conv5_1 = mx.symbol.Convolution(data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512,
                 name="conv5_1")
    bn_conv5_1 = mx.symbol.BatchNorm(data=conv5_1, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom, 
                 name='bn_conv5_1') 
    relu5_1 = mx.symbol.Activation(data=bn_conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.symbol.Convolution(data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512,
                 name="conv5_2")
    bn_conv5_2 = mx.symbol.BatchNorm(data=conv5_2, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom, 
                 name='bn_conv5_2') 
    relu5_2 = mx.symbol.Activation(data=bn_conv5_2, act_type="relu", name="relu5_2")
    conv5_3 = mx.symbol.Convolution(data=relu5_2, kernel=(3, 3), pad=(1, 1), num_filter=512,
                 name="conv5_3")
    bn_conv5_3 = mx.symbol.BatchNorm(data=conv5_3, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom, 
                 name='bn_conv5_3') 
    relu5_3 = mx.symbol.Activation(data=bn_conv5_3, act_type="relu", name="relu5_3")
    pool5 = mx.symbol.Pooling(data=relu5_3, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool5")

    return pool5

def vgg_deconv(input_data, fix_gamma, eps, bn_mom):
    # group 5
    pool5_D = mx.symbol.UpSampling(input_data, scale=2, sample_type='nearest', name="pool5_D")
    conv5_3_D = mx.symbol.Convolution(data=pool5_D, kernel=(3, 3), pad=(1, 1), num_filter=512,
                 name="conv5_3_D")
    bn_conv5_3_D = mx.symbol.BatchNorm(data=conv5_3_D, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom, 
                 name='bn_conv5_3_D') 
    relu5_3_D = mx.symbol.Activation(data=bn_conv5_3_D, act_type="relu", name="relu5_3_D")
    conv5_2_D = mx.symbol.Convolution(data=relu5_3_D, kernel=(3, 3), pad=(1, 1), num_filter=512,
                 name="conv5_2_D")
    bn_conv5_2_D = mx.symbol.BatchNorm(data=conv5_2_D, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom, 
                 name='bn_conv5_2_D') 
    relu5_2_D = mx.symbol.Activation(data=bn_conv5_2_D, act_type="relu", name="relu5_2_D")
    conv5_1_D = mx.symbol.Convolution(data=relu5_2_D, kernel=(3, 3), pad=(1, 1), num_filter=512,
                 name="conv5_1_D")
    bn_conv5_1_D = mx.symbol.BatchNorm(data=conv5_1_D, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom, 
                 name='bn_conv5_1_D') 
    relu5_1_D = mx.symbol.Activation(data=bn_conv5_1_D, act_type="relu", name="relu5_1_D")
    # group 4
    pool4_D = mx.symbol.UpSampling(relu5_1_D, scale=2, sample_type='nearest', name="pool4_D")
    pad4_D = mx.symbol.Pad(data = pool4_D, mode='constant', constant_value=0.0, pad_width=(0, 0, 0, 0, 0, 1, 0, 0,), name="pad4_D")
    conv4_3_D = mx.symbol.Convolution(data=pad4_D, kernel=(3, 3), pad=(1, 1), num_filter=512,
                 name="conv4_3_D")
    bn_conv4_3_D = mx.symbol.BatchNorm(data=conv4_3_D, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom, 
                 name='bn_conv4_3_D') 
    relu4_3_D = mx.symbol.Activation(data=bn_conv4_3_D, act_type="relu", name="relu4_3_D")
    conv4_2_D = mx.symbol.Convolution(data=relu4_3_D, kernel=(3, 3), pad=(1, 1), num_filter=512,
                 name="conv4_2_D")
    bn_conv4_2_D = mx.symbol.BatchNorm(data=conv4_2_D, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom, 
                 name='bn_conv4_2_D') 
    relu4_2_D = mx.symbol.Activation(data=bn_conv4_2_D, act_type="relu", name="relu4_2_D")
    conv4_1_D = mx.symbol.Convolution(data=relu4_2_D, kernel=(3, 3), pad=(1, 1), num_filter=512,
                 name="conv4_1_D")
    bn_conv4_1_D = mx.symbol.BatchNorm(data=conv4_1_D, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom, 
                 name='bn_conv4_1_D') 
    relu4_1_D = mx.symbol.Activation(data=bn_conv4_1_D, act_type="relu", name="relu4_1_D")
    # group 3
    pool3_D = mx.symbol.UpSampling(relu4_1_D, scale=2, sample_type='nearest', name="pool3_D")
    conv3_3_D = mx.symbol.Convolution(data=pool3_D, kernel=(3, 3), pad=(1, 1), num_filter=256,
                 name="conv3_3_D")
    bn_conv3_3_D = mx.symbol.BatchNorm(data=conv3_3_D, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom, 
                 name='bn_conv3_3_D') 
    relu3_3_D = mx.symbol.Activation(data=bn_conv3_3_D, act_type="relu", name="relu3_3_D")
    conv3_2_D = mx.symbol.Convolution(data=relu3_3_D, kernel=(3, 3), pad=(1, 1), num_filter=256,
                 name="conv3_2_D")
    bn_conv3_2_D = mx.symbol.BatchNorm(data=conv3_2_D, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom, 
                 name='bn_conv3_2_D') 
    relu3_2_D = mx.symbol.Activation(data=bn_conv3_2_D, act_type="relu", name="relu3_2_D")
    conv3_1_D = mx.symbol.Convolution(data=relu3_2_D, kernel=(3, 3), pad=(1, 1), num_filter=256,
                 name="conv3_1_D")
    bn_conv3_1_D = mx.symbol.BatchNorm(data=conv3_1_D, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom, 
                 name='bn_conv3_1_D') 
    relu3_1_D = mx.symbol.Activation(data=conv3_1_D, act_type="relu", name="relu3_1_D")
    # group 2
    pool2_D = mx.symbol.UpSampling(relu3_1_D, scale=2, sample_type='nearest', name="pool2_D")
    conv2_2_D = mx.symbol.Convolution(data=pool2_D, kernel=(3, 3), pad=(1, 1), num_filter=128,
                 name="conv2_2_D")  
    bn_conv2_2_D = mx.symbol.BatchNorm(data=conv2_2_D, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom, 
                 name='bn_conv2_2_D')            
    relu2_2_D = mx.symbol.Activation(data=bn_conv2_2_D, act_type="relu", name="relu2_2_D")
    conv2_1_D = mx.symbol.Convolution(data=relu2_2_D, kernel=(3, 3), pad=(1, 1), num_filter=128,
                 name="conv2_1_D")
    bn_conv2_1_D = mx.symbol.BatchNorm(data=conv2_1_D, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom, 
                 name='bn_conv2_1_D')             
    relu2_1_D = mx.symbol.Activation(data=bn_conv2_1_D, act_type="relu", name="relu2_1_D")
    # group 1
    pool1_D = mx.symbol.UpSampling(relu2_1_D, scale=2, sample_type='nearest', name="pool1_D")
    conv1_2_D = mx.symbol.Convolution(data=pool1_D, kernel=(3, 3), pad=(1, 1), num_filter=64,
                 name="conv1_2_D")
    bn_conv1_2_D = mx.symbol.BatchNorm(data=conv1_2_D, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom, 
                 name='bn_conv1_2_D')             
    relu1_2_D = mx.symbol.Activation(data=bn_conv1_2_D, act_type="relu", name="relu1_2_D")
    drop = mx.symbol.Dropout(data=relu1_2_D, p=0.5, name="drop")
    conv1_1_D = mx.symbol.Convolution(data=drop, kernel=(3, 3), pad=(1, 1), num_filter=11,
                 name="conv1_1_D")
    return conv1_1_D

def segnet_symbol(num_classes, **kwargs):
    fix_gamma = True
    eps = 0.001
    bn_mom = 0.9
    data = mx.symbol.Variable(name="data") 
    encode = vgg_conv(data, fix_gamma, eps, bn_mom)   
    decode = vgg_deconv(encode, fix_gamma, eps, bn_mom) 
    softmax = mx.symbol.SoftmaxOutput(data=decode, multi_output=True, ignore_label=11, name='softmax', num_labels = num_classes)
    return softmax
