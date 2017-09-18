# In[1]:

import mxnet as mx
import numpy as np
import time


# In[2]:

# Basic Info
dev = mx.cpu()
batch_size = 256
image_shape = (3,224,224)
dshape = [('data', (batch_size,)+image_shape)] #(batch_size, 3, 224, 224)
# lshape = (batch_size)
num_epoch = 100

# Mock data iterator
# tmp_data = np.random.uniform(-1, 1, dshape).astype("float32")

# train_iter = mx.io.NDArrayIter(data=tmp_data,  batch_size=batch_size, shuffle=False, last_batch_handle='pad')



# In[5]:
def get_symbol(num_classes, **kwargs):
    input_data = mx.symbol.Variable(name="data")
    # stage 1
    conv1 = mx.symbol.Convolution(name='conv1',
        data=input_data, kernel=(5, 5), stride=(1, 1), pad=(2, 2), num_filter=32)
    pool1 = mx.symbol.Pooling(
        data=conv1, pool_type="max", kernel=(3, 3), stride=(2,2), pad=(1, 1))
    relu1 = mx.symbol.Activation(data=pool1, act_type="relu")
    # lrn1 = mx.symbol.LRN(data=relu1, alpha=0.0001, beta=0.75, knorm=2, nsize=5)
    lrn1 = mx.symbol.LRN(data=relu1, alpha=0.0001, beta=0.75, knorm=2, nsize=3)
    # stage 2
    conv2 = mx.symbol.Convolution(name='conv2',
        data=lrn1, kernel=(5, 5), pad=(2, 2), num_filter=32)
    relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    pool2 = mx.symbol.Pooling(data=relu2, kernel=(3, 3), stride=(2, 2), pool_type="max")
    lrn2 = mx.symbol.LRN(data=pool2, alpha=0.0001, beta=0.75, knorm=2, nsize=3)
    # stage 3
    conv3 = mx.symbol.Convolution(name='conv3',
        data=lrn2, kernel=(5, 5), pad=(2, 2), num_filter=64)
    relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    pool3 = mx.symbol.Pooling(data=relu3, kernel=(3, 3), stride=(2, 2), pool_type="max")
    # conv4 = mx.symbol.Convolution(name='conv4',
    #     data=relu3, kernel=(3, 3), pad=(1, 1), num_filter=384)
    # relu4 = mx.symbol.Activation(data=conv4, act_type="relu")
    # conv5 = mx.symbol.Convolution(name='conv5',
    #     data=relu4, kernel=(3, 3), pad=(1, 1), num_filter=256)
    # relu5 = mx.symbol.Activation(data=conv5, act_type="relu")
    # pool3 = mx.symbol.Pooling(data=relu5, kernel=(3, 3), stride=(2, 2), pool_type="max")
    # stage 4
    flatten = mx.symbol.Flatten(data=pool3)
    # fc1 = mx.symbol.FullyConnected(name='fc1', data=flatten, num_hidden=384)
    # relu6 = mx.symbol.Activation(data=fc1, act_type="relu")
    # # dropout1 = mx.symbol.Dropout(data=relu6, p=0.5)
    # # stage 5
    # fc2 = mx.symbol.FullyConnected(name='fc2', data=relu6, num_hidden=192)
    # relu7 = mx.symbol.Activation(data=fc2, act_type="relu")
    # dropout2 = mx.symbol.Dropout(data=relu7, p=0.5)
    # stage 6
    fc3 = mx.symbol.FullyConnected(name='fc3', data=flatten, num_hidden=num_classes)
    softmax = mx.symbol.SoftmaxOutput(data=fc3, name='softmax')
    return softmax

def get_alexnet_symbol():
    ## define alexnet
    input_data = mx.symbol.Variable(name="data")
    # stage 1
    conv1 = mx.symbol.Convolution(
        data=input_data, kernel=(11, 11), stride=(4, 4), num_filter=64)
    relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    pool1 = mx.symbol.Pooling(
        data=relu1, pool_type="max", kernel=(3, 3), stride=(2,2))
#    lrn1 = mx.symbol.LRN(data=pool1, alpha=0.0001, beta=0.75, knorm=1, nsize=5)
    # stage 2
    conv2 = mx.symbol.Convolution(
        data=pool1, kernel=(5, 5), pad=(2, 2), num_filter=192)
    relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    pool2 = mx.symbol.Pooling(data=relu2, kernel=(3, 3), stride=(2, 2), pool_type="max")
#    lrn2 = mx.symbol.LRN(data=pool2, alpha=0.0001, beta=0.75, knorm=1, nsize=5)
    # stage 3
    conv3 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=384)
    relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    conv4 = mx.symbol.Convolution(
        data=relu3, kernel=(3, 3), pad=(1, 1), num_filter=256)
    relu4 = mx.symbol.Activation(data=conv4, act_type="relu")
    conv5 = mx.symbol.Convolution(
        data=relu4, kernel=(3, 3), pad=(1, 1), num_filter=256)
    relu5 = mx.symbol.Activation(data=conv5, act_type="relu")
    pool3 = mx.symbol.Pooling(data=relu5, kernel=(3, 3), stride=(2, 2), pool_type="max")
    # stage 4
    flatten = mx.symbol.Flatten(data=pool3)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096)
    relu6 = mx.symbol.Activation(data=fc1, act_type="relu")
    # stage 5
    fc2 = mx.symbol.FullyConnected(data=relu6, num_hidden=4096)
    relu7 = mx.symbol.Activation(data=fc2, act_type="relu")
    # stage 6
    fc3 = mx.symbol.FullyConnected(data=relu7, num_hidden=1000)
    softmax = mx.symbol.SoftmaxOutput(data=fc3, name='softmax')

    return softmax

# In[6]:

# bind to get executor
# This is what happened behind mx.model.Feedforward
softmax = get_alexnet_symbol()

mod_inf = mx.mod.Module(symbol=softmax, context=dev)
mod_inf.bind(for_training     = False,
             inputs_need_grad = False,
             data_shapes      = dshape)
mod_inf.init_params(initializer=mx.init.Xavier(magnitude=2.))
mod_inf.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1), ))
# fc3 = get_symbol(10)
mod = mx.mod.Module(symbol=softmax, context=dev)
mod.bind(for_training     = True,
             inputs_need_grad = False,
             data_shapes      = dshape)
mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1), ))
# alex_exec = fc3.simple_bind(ctx=dev, grad_req="write", data=dshape)
# print("Temp space: ", alex_exec.debug_str().split('\n')[-3])
# Find where to set data


# In[7]:

# some useful structure
# data structues 
# arg_names = fc3.list_arguments()
# arg_map = dict(zip(arg_names, alex_exec.arg_arrays))
# grad_map = dict(zip(arg_names, alex_exec.grad_arrays))


# param_blocks = [(i, arg_map[arg_names[i]], grad_map[arg_names[i]]) for i in range(len(arg_names)) if grad_map[arg_names[i]] != None]
# input_ndarray = arg_map["data"]
# grad = mx.nd.zeros((batch_size, 1000), ctx=mx.cpu())
# param_len = len(param_blocks)


# # In[8]:

# #init
# for i in range(param_len):
#     param_blocks[i][1][:] = mx.rnd.uniform(-0.01, 0.01, param_blocks[i][1].shape)
#     param_blocks[i][2][:] = 0.
# # Set data
# train_iter.reset()
# dbatch = train_iter.next()
# dbatch.data[0].copyto(input_ndarray)
# # block all async all
# mx.nd.waitall()


# In[12]:

# Test forward
def test_forward(mod, epoch):
    data = [mx.random.uniform(-1.0, 1.0, shape=shape, ctx=dev) for _, shape in mod.data_shapes]
    batch = mx.io.DataBatch(data, []) # empty label

    # dry run
    for i in range(5):
        # model.forward(is_train=False)
        mod.forward(batch, is_train=False)
        for output in mod.get_outputs():
            output.wait_to_read()
    mx.nd.waitall()

    tic = time.time()
    for i in range(epoch):
        mod.forward(batch, is_train=False)
        for output in mod.get_outputs():
            output.wait_to_read()
        # Note: This command will force thread engine block, which hurts performance a lot
        # Remove it will bring parallelism bias
        # model.outputs[0].wait_to_read()
    # model.outputs[0].wait_to_read()
    toc = time.time()
    return (toc - tic) / epoch

print("Avg forward images per sec: ", batch_size/test_forward(mod_inf, num_epoch))


# In[13]:

# Test full path
def test_full(mod, epoch):
    data = [mx.random.uniform(-1.0, 1.0, shape=shape, ctx=dev) for _, shape in mod.data_shapes]
    batch = mx.io.DataBatch(data, []) # empty label

    # dry run
    for i in range(5):
        mod.forward(batch, is_train=True)
        mod.backward()
        mod.update()
        for output in mod.get_outputs():
            output.wait_to_read()
    mx.nd.waitall()
    
    tic = time.time()
    for i in range(epoch):
        mod.forward(batch, is_train=True)
        mod.backward()
        mod.update()
        for output in mod.get_outputs():
            output.wait_to_read()
        # model.update()
        #model.outputs[0].wait_to_read()
        # mx.nd.waitall()
        # mock update
        # for i in range(param_len):
        #     param_blocks[i][1][:] -= 0.0 * param_blocks[i][2][:]
    # Note: This command will force thread engine block, which hurts performance a lot
    mx.nd.waitall()
    toc = time.time()
    return (toc - tic) / epoch

print("Avg fullpath images per sec: ", batch_size/test_full(mod, num_epoch))


# In[ ]: