import os, sys
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(curr_path, "../../python"))
sys.path.insert(0, os.path.join(curr_path, "../../example/image-classification"))
import mxnet as mx
import logging
import argparse
import time
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="command for benchmark kv-store")
    parser.add_argument('--network', type=str, default="resnet",
                        help='the neural network to test')
    parser.add_argument('--gpus', type=str, default='0,1',
                        help='the gpus to be used, e.g "0,1,2,3"')
    parser.add_argument('--depth', type=int, default=152,
                        help='the depth of network, only valid for resnet')
    parser.add_argument('--kv-store', type=str, default='device',
                        help='the kvstore type')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size. should not affect the results')
    parser.add_argument('--num-batches', type=int, default=5,
                        help='number of batches to run')
    parser.add_argument('--disp-batches', type=int, default=1,
                        help='show averaged results for every n batches')
    parser.add_argument('--test-results', type=int, default=1,
                        help='if or not evalute the results correctness')
    parser.add_argument('--data-shape', type=str, default='128,3,224,224',
                        help='input data shape')
    parser.add_argument('--num-classes', type=int, default=1000,
                        help='number of classes')
    parser.add_argument('--optimizer', type=str, default='None',
                        help='the optimizer set to kvstore. None means no optimizer')
    args = parser.parse_args()
    logging.info(args)
    return args

def get_resnet(args):
    resnet_path = os.path.join(curr_path, "./ResNet")
    if not os.path.isdir(resnet_path):
        os.system("git clone https://github.com/tornadomeet/ResNet")
    sys.path.insert(0, resnet_path)
    from symbol_resnet import resnet
    if args.depth == 18:
        units = [2, 2, 2, 2]
    elif args.depth == 34:
        units = [3, 4, 6, 3]
    elif args.depth == 50:
        units = [3, 4, 6, 3]
    elif args.depth == 101:
        units = [3, 4, 23, 3]
    elif args.depth == 152:
        units = [3, 8, 36, 3]
    elif args.depth == 200:
        units = [3, 24, 36, 3]
    else:
        raise ValueError("no experiments done on detph {}, you can do it youself".format(args.depth))

    filter_list=[64, 256, 512, 1024, 2048] if args.depth >=50 else [64, 64, 128, 256, 512]
    bottle_neck = True if args.depth >= 50 else False
    symbol = resnet(units=units, num_stage=4, filter_list=filter_list,
                    num_class=args.num_classes, data_type="imagenet", bottle_neck=bottle_neck, bn_mom=.9, workspace=512)
    return symbol

def get_shapes(symbol, data_shape):
    arg_name = symbol.list_arguments()
    arg_shape, _, _ = symbol.infer_shape(data=data_shape)
    shapes = [s for n,s in zip(arg_name, arg_shape) if 'weight' in n or 'bias' in n]
    return shapes

def diff(a, b):
    return np.sum(np.abs(a.asnumpy() - b.asnumpy()))

def error(gpu_res, cpu_res):
    res = sum([sum([diff(a, b) for a in w]) for w, b in zip(gpu_res, cpu_res)])
    res /= sum([np.sum(np.abs(g.asnumpy())) for g in cpu_res])
    return res

def run():
    args = parse_args();
    # create kvstore and optimizer
    devs = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    kv = mx.kv.create(args.kv_store)
    if args.optimizer == 'None':
        optimizer = None
    else:
        optimizer = mx.optimizer.Optimizer.create_optimizer(args.optimizer)
        updater = mx.optimizer.get_updater(mx.optimizer.Optimizer.create_optimizer(args.optimizer))
        kv.set_optimizer(optimizer)

    # create network
    if args.network == 'resnet':
        symbol = get_resnet(args)
    else:
        import importlib
        symbol = importlib.import_module('symbol_' + args.network).get_symbol(args.num_classes)
    data_shape = tuple([int(s) for s in args.data_shape.split(',')])
    shapes = get_shapes(symbol, data_shape)

    size = float(sum([reduce(lambda x,y : x*y, s, 1) for s in shapes])) * 4 / 1e6
    logging.info('num of arrays = %d, total size = %f MB' % (len(shapes), size))

    for i, s in enumerate(shapes):
        kv.init(i, mx.nd.zeros(s))

    grads_val = [[mx.random.uniform(-1,1,shape=s) for d in devs] for s in shapes]
    grads = [[g.as_in_context(d) for g, d in zip(gs, devs)] for gs in grads_val]
    weights = [[mx.nd.zeros(s, d) for d in devs] for s in shapes]

    cpu_grads = [mx.nd.array(sum([g.asnumpy() for g in gs]))*kv.num_workers for gs in grads_val]
    cpu_weights = [mx.nd.zeros(s) for s in shapes]
    toc = 0
    for b in range(0, args.num_batches+1):
        tic = time.time()
        for i,g in enumerate(grads):
            kv.push(i, g, i)

        for i,w in enumerate(weights):
            kv.pull(i, w, i)
        for ws in weights:
            for w in ws:
                w.wait_to_read()
        toc += time.time() - tic
        if args.test_results:
            if optimizer == None:
                err = error(weights, cpu_grads)
            else:
                for i, wg in enumerate(zip(cpu_weights, cpu_grads)):
                    updater(i, wg[1], wg[0])
                err = error(weights, cpu_weights)
        else:
            err = -1

        if b % args.disp_batches == 0:
            toc /= args.disp_batches
            if b != 0:
                # 0 is used for warmup, ignored
                logging.info('iter %d, %f sec, %f GB/sec per gpu, error %f' % (
                    b, toc, size*2*(len(devs)-1)/len(devs)/toc/1e3, err))
            toc = 0

if __name__ == "__main__":
    run()
