import mxnet as mx
import mxnet.gluon.model_zoo.vision as models
import time
import logging
import argparse
import subprocess
import os
import errno

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser(description='Gluon modelzoo-based CNN perf')

parser.add_argument('--model', type=str, default='all', 
                               choices=['all', 'alexnet', 'densenet121', 'densenet161', 
                                        'densenet169', 'densenet201', 'inceptionv3', 'mobilenet0.25',
                                        'mobilenet0.5', 'mobilenet0.75', 'mobilenet1.0', 'mobilenetv2_0.25',
                                        'mobilenetv2_0.5', 'mobilenetv2_0.75', 'mobilenetv2_1.0', 'resnet101_v1',
                                        'resnet101_v2', 'resnet152_v1', 'resnet152_v2', 'resnet18_v1', 
                                        'resnet18_v2', 'resnet34_v1', 'resnet34_v2', 'resnet50_v1', 
                                        'resnet50_v2', 'squeezenet1.0', 'squeezenet1.1', 'vgg11', 
                                        'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 
                                        'vgg19', 'vgg19_bn'])
parser.add_argument('--batch-size', type=int, default=0)
parser.add_argument('--type', type=str, default='inf', choices=['all', 'train', 'inf'])

opt = parser.parse_args()

num_batches = 100
dry_run = 10  # use 10 iterations to warm up
batch_inf = [1, 16, 32, 64, 128, 256]
batch_train = [1, 2, 4, 8, 16, 32, 64, 126, 256]
image_shapes = [(3, 224, 224), (3, 299, 299)]

def get_gpus():
    """
    return a list of GPUs
    """
    try:
        re = subprocess.check_output(["nvidia-smi", "-L"], universal_newlines=True)
    except OSError:
        return []
    return range(len([i for i in re.split('\n') if 'GPU' in i]))

def score(network, batch_size, ctx):
    net = models.get_model(network)
    if 'inceptionv3' == network:
        data_shape = [('data', (batch_size,) + image_shapes[1])]
    else:
        data_shape = [('data', (batch_size,) + image_shapes[0])]

    net.hybridize()
    data = mx.sym.var('data')
    out = net(data)
    softmax = mx.sym.SoftmaxOutput(out, name='softmax')
    mod = mx.mod.Module(softmax, context=ctx)
    mod.bind(for_training     = False,
             inputs_need_grad = False,
             data_shapes      = data_shape)
    mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
    data = [mx.random.uniform(-1.0, 1.0, shape=shape, ctx=ctx) for _, shape in mod.data_shapes]
    batch = mx.io.DataBatch(data, [])
    for i in range(dry_run + num_batches):
        if i == dry_run:
            tic = time.time()
        mod.forward(batch, is_train=False)
        for output in mod.get_outputs():
            output.wait_to_read()
    fwd = time.time() - tic
    return fwd


def train(network, batch_size, ctx):
    net = models.get_model(network)
    if 'inceptionv3' == network:
        data_shape = [('data', (batch_size,) + image_shapes[1])]
    else:
        data_shape = [('data', (batch_size,) + image_shapes[0])]

    net.hybridize()
    data = mx.sym.var('data')
    out = net(data)
    softmax = mx.sym.SoftmaxOutput(out, name='softmax')
    mod = mx.mod.Module(softmax, context=ctx)
    mod.bind(for_training     = True,
             inputs_need_grad = False,
             data_shapes      = data_shape)
    mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
    mod.init_optimizer(kvstore='local', optimizer='sgd')
    data = [mx.random.uniform(-1.0, 1.0, shape=shape, ctx=ctx) for _, shape in mod.data_shapes]
    batch = mx.io.DataBatch(data, [])
    for i in range(dry_run + num_batches):
        if i == dry_run:
            tic = time.time()
        mod.forward(batch, is_train=True)
        for output in mod.get_outputs():
            output.wait_to_read()
        mod.backward()
        mod.update()
    bwd = time.time() - tic
    return bwd

if __name__ == '__main__':
    runtype = opt.type
    bs = opt.batch_size

    if opt.model == 'all':
        networks = ['alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201',
                    'inceptionv3', 'mobilenet0.25', 'mobilenet0.5', 'mobilenet0.75',
                    'mobilenet1.0', 'mobilenetv2_0.25', 'mobilenetv2_0.5', 'mobilenetv2_0.75',
                    'mobilenetv2_1.0', 'resnet101_v1', 'resnet101_v2', 'resnet152_v1', 'resnet152_v2',
                    'resnet18_v1', 'resnet18_v2', 'resnet34_v1', 'resnet34_v2', 'resnet50_v1', 
                    'resnet50_v2', 'squeezenet1.0', 'squeezenet1.1', 'vgg11', 'vgg11_bn', 'vgg13', 
                    'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']
        logging.info('It may take some time to run all models, '
                     'set --network to run a specific one')
    else:
        networks = [opt.model]

    devs = [mx.gpu(0)] if len(get_gpus()) > 0 else []
    # Enable USE_MKLDNN for better CPU performance
    devs.append(mx.cpu())

    for network in networks:
        logging.info('network: %s', network)
        for d in devs:
            logging.info('device: %s', d)
            if runtype == 'inf' or runtype == 'all':
                if bs != 0:
                    fwd_time = score(network, bs, d)
                    fps = (bs*num_batches)/fwd_time
                    logging.info(network + ' inference perf for BS %d is %f img/s', bs, fps)
                else:
                    for batch_size in batch_inf:
                        fwd_time = score(network, batch_size, d)
                        fps = (batch_size * num_batches) / fwd_time
                        logging.info(network + ' inference perf for BS %d is %f img/s', batch_size, fps)
            if runtype == 'train' or runtype == 'all':
                if bs != 0:
                    bwd_time = train(network, bs, d)
                    fps = (bs*num_batches)/bwd_time
                    logging.info(network + ' training perf for BS %d is %f img/s', bs, fps)
                else:
                    for batch_size in batch_train:
                        bwd_time = train(network, batch_size, d)
                        fps = (batch_size * num_batches) / bwd_time
                        logging.info(network + ' training perf for BS %d is %f img/s', batch_size, fps)


