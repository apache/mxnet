import argparse
import mxnet as mx
import time
import os
import logging
from common import modelzoo

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser(description='score a model on a dataset')

parser.add_argument('--model', type=str, required=True,
                    help = 'the model name.')
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--rgb-mean', type=str, default='0,0,0')
parser.add_argument('--data-val', type=str, required=True)
parser.add_argument('--image-shape', type=str, default='3,224,224')
parser.add_argument('--data-nthreads', type=int, default=4,
                    help='number of threads for data decoding')
args = parser.parse_args()

# create data iterator
rgb_mean = [float(i) for i in args.rgb_mean.split(',')]
data_shape = tuple([int(i) for i in args.image_shape.split(',')])
data = mx.io.ImageRecordIter(
    path_imgrec        = args.data_val,
    label_width        = 1,
    mean_r             = rgb_mean[0],
    mean_g             = rgb_mean[1],
    mean_b             = rgb_mean[2],
    preprocess_threads = args.data_nthreads,
    batch_size         = args.batch_size,
    data_shape         = data_shape,
    rand_crop          = False,
    rand_mirror        = False)

# download model
dir_path = os.path.dirname(os.path.realpath(__file__))
(prefix, epoch) = modelzoo.download_model(
    args.model, os.path.join(dir_path, 'model'))

# create module
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
if args.gpus == '':
    devs = mx.cpu()
else:
    devs = [mx.gpu(int(i)) for i in args.gpus.split(',')]

mod = mx.mod.Module(symbol=sym, context=devs)
mod.bind(for_training=False,
         data_shapes=data.provide_data,
         label_shapes=data.provide_label)
mod.set_params(arg_params, aux_params)

metrics = [mx.metric.create('acc'),
           mx.metric.create('top_k_accuracy', top_k = 5)]

tic = time.time()
num = 0
for batch in data:
    mod.forward(batch, is_train=False)
    for m in metrics:
        mod.update_metric(m, batch.label)
    num += args.batch_size
logging.info('Finished with %f images per second', num / (time.time() - tic))
for m in metrics:
    logging.info(m.get())
