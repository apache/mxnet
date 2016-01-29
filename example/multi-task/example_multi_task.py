# pylint: skip-file
import sys
sys.path.insert(0, "../../python/")
from data import mnist_iterator
import mxnet as mx
import numpy as np
import logging
import time

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def build_network():
    data = mx.symbol.Variable('data')
    fc1 = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
    act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
    fc2 = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
    act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
    fc3 = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10)
    fc4 = mx.symbol.FullyConnected(data = act2, name='fc4', num_hidden=10)
    #two tasks    
    sm1 = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax1')
    sm2 = mx.symbol.SoftmaxOutput(data = fc4, name = 'softmax2')

    softmax = mx.symbol.Group([sm1, sm2])

    return softmax

        
def Init_param(args, init=mx.initializer.Uniform(0.07)):
    """initialize the parameters of the network."""
    for key, arr in args.items():
        if key != 'data' and not key.endswith('_label'):
            init(key, arr)

batch_size=100
num_round=100
device = mx.cpu()

train_iter, val_iter = mnist_iterator(batch_size=batch_size, input_shape = (784,))

net = build_network()
executor = net.simple_bind(ctx=device, data=(batch_size,784), grad_req='write')
optimizer = mx.optimizer.create("sgd", learning_rate=0.01, momentum=0.9, wd=0.00005, clip_gradient=10)
optimizer.rescale_grad = 1.0/batch_size
updater = mx.optimizer.get_updater(optimizer)
metric1 = mx.metric.Accuracy()
metric2 = mx.metric.Accuracy()

arg_arrays = executor.arg_arrays
grad_arrays = executor.grad_arrays
aux_arrays = executor.aux_arrays
output_arrays = executor.outputs

args = dict(zip(net.list_arguments(), arg_arrays))
grads = dict(zip(net.list_arguments(), grad_arrays))
aux_states = dict(zip(net.list_auxiliary_states(), aux_arrays))
outputs = dict(zip(net.list_outputs(), output_arrays))

Init_param(args)

pred_prob1 = mx.nd.zeros(executor.outputs[0].shape)
pred_prob2 = mx.nd.zeros(executor.outputs[1].shape)
logging.info("start training")
for i in range(num_round):
    train_iter.reset()
    val_iter.reset()
    start = time.time()
    # train
    for dbatch in train_iter:
        data = dbatch.data[0]
        label = dbatch.label[0]
        # copy data and two labels into args
        args["data"][:] = data 
        args["softmax1_label"][:] = label
        args["softmax2_label"][:] = label
        executor.forward(is_train=True)
        pred_prob1[:] = executor.outputs[0]
        pred_prob2[:] = executor.outputs[1]
        executor.backward()
        for k, v in args.items():
            if grads[k]:
                updater(k, grads[k], v)
        
        metric1.update([label], [pred_prob1])
        metric2.update([label], [pred_prob2])
   
    logging.info("Finish training iteration %d" % i)
    train_acc1 = metric1.get()
    train_acc2 = metric2.get()
    metric1.reset()
    metric2.reset()
    # eval
    for dbatch in val_iter:
        data = dbatch.data[0]
        label = dbatch.label[0]
        args["data"][:] = data
        executor.forward(is_train=False)
        pred_prob1[:] = executor.outputs[0]
        pred_prob2[:] = executor.outputs[1]
        metric1.update([label], [pred_prob1])
        metric2.update([label], [pred_prob2])

    val_acc1 = metric1.get()
    val_acc2 = metric2.get()
    metric1.reset()
    logging.info("Train1 Acc: %.4f" % train_acc1[1])
    logging.info("Val1 Acc: %.4f" % val_acc1[1])
    logging.info("Train2 Acc: %.4f" % train_acc2[1])
    logging.info("Val2 Acc: %.4f" % val_acc2[1])
    logging.info("Time used: %.5f" % (time.time()-start))
