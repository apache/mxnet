import find_mxnet
import mxnet as mx
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)
import argparse
from collections import namedtuple
from skimage import io, transform
from skimage.restoration import denoise_tv_chambolle

parser = argparse.ArgumentParser(description='neural style')

parser.add_argument('--model', type=str, default='vgg19',
                    choices = ['vgg'],
                    help = 'the pretrained model to use')
parser.add_argument('--content-image', type=str, default='input/IMG_4343.jpg',
                    help='the content image')
parser.add_argument('--style-image', type=str, default='input/starry_night.jpg',
                    help='the style image')
parser.add_argument('--stop-eps', type=float, default=.005,
                    help='stop if the relative chanage is less than eps')
parser.add_argument('--content-weight', type=float, default=10,
                    help='the weight for the content image')
parser.add_argument('--style-weight', type=float, default=1,
                    help='the weight for the style image')
parser.add_argument('--max-num-epochs', type=int, default=1000,
                    help='the maximal number of training epochs')
parser.add_argument('--max-long-edge', type=int, default=600,
                    help='resize the content image')
parser.add_argument('--lr', type=float, default=.1,
                    help='the initial learning rate')
parser.add_argument('--gpu', type=int, default=0,
                    help='which gpu card to use, -1 means using cpu')
parser.add_argument('--output', type=str, default='output/out.jpg',
                    help='the output image')
parser.add_argument('--save-epochs', type=int, default=50,
                    help='save the output every n epochs')
parser.add_argument('--remove-noise', type=float, default=.2,
                    help='the magtitute to remove noise')

args = parser.parse_args()

def PreprocessContentImage(path, long_edge):
    img = io.imread(path)
    logging.info("load the content image, size = %s", img.shape[:2])
    factor = float(long_edge) / max(img.shape[:2])
    new_size = (int(img.shape[0] * factor), int(img.shape[1] * factor))
    resized_img = transform.resize(img, new_size)
    sample = np.asarray(resized_img) * 256
    # swap axes to make image from (224, 224, 3) to (3, 224, 224)
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)
    # sub mean
    sample[0, :] -= 123.68
    sample[1, :] -= 116.779
    sample[2, :] -= 103.939
    logging.info("resize the content image to %s", new_size)
    return np.resize(sample, (1, 3, sample.shape[1], sample.shape[2]))

def PreprocessStyleImage(path, shape):
    img = io.imread(path)
    resized_img = transform.resize(img, (shape[2], shape[3]))
    sample = np.asarray(resized_img) * 256
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)

    sample[0, :] -= 123.68
    sample[1, :] -= 116.779
    sample[2, :] -= 103.939
    return np.resize(sample, (1, 3, sample.shape[1], sample.shape[2]))

def PostprocessImage(img):
    img = np.resize(img, (3, img.shape[2], img.shape[3]))
    img[0, :] += 123.68
    img[1, :] += 116.779
    img[2, :] += 103.939
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 0, 1)
    img = np.clip(img, 0, 255)
    return img.astype('uint8')

def SaveImage(img, filename):
    logging.info('save output to %s', filename)
    out = PostprocessImage(img)
    out = denoise_tv_chambolle(out, weight=args.remove_noise, multichannel=True)
    io.imsave(filename, out)

# input
dev = mx.gpu(args.gpu) if args.gpu >= 0 else mx.cpu()
content_np = PreprocessContentImage(args.content_image, args.max_long_edge)
style_np = PreprocessStyleImage(args.style_image, shape=content_np.shape)
size = content_np.shape[2:]

# model
Executor = namedtuple('Executor', ['executor', 'data', 'data_grad'])

def StyleGramExecutor(input_shape, ctx):
    # symbol
    data = mx.sym.Variable("conv")
    rs_data = mx.sym.Reshape(data=data, target_shape=(int(input_shape[1]), int(np.prod(input_shape[2:]))))
    weight = mx.sym.Variable("weight")
    rs_weight = mx.sym.Reshape(data=weight, target_shape=(int(input_shape[1]), int(np.prod(input_shape[2:]))))
    fc = mx.sym.FullyConnected(data=rs_data, weight=rs_weight, no_bias=True, num_hidden=input_shape[1])
    # executor
    conv = mx.nd.zeros(input_shape, ctx=ctx)
    grad = mx.nd.zeros(input_shape, ctx=ctx)
    args = {"conv" : conv, "weight" : conv}
    grad = {"conv" : grad}
    reqs = {"conv" : "write", "weight" : "null"}
    executor = fc.bind(ctx=ctx, args=args, args_grad=grad, grad_req=reqs)
    return Executor(executor=executor, data=conv, data_grad=grad["conv"])


import importlib
model_executor = importlib.import_module('model_' + args.model).get_model(size, dev)
gram_executor = [StyleGramExecutor(arr.shape, dev) for arr in model_executor.style]


# get style representation
style_array = [mx.nd.zeros(gram.executor.outputs[0].shape, ctx=dev) for gram in gram_executor]
model_executor.data[:] = style_np
model_executor.executor.forward()

for i in range(len(model_executor.style)):
    model_executor.style[i].copyto(gram_executor[i].data)
    gram_executor[i].executor.forward()
    gram_executor[i].executor.outputs[0].copyto(style_array[i])

# get content representation
content_array = mx.nd.zeros(model_executor.content.shape, ctx=dev)
content_grad  = mx.nd.zeros(model_executor.content.shape, ctx=dev)
model_executor.data[:] = content_np
model_executor.executor.forward()
model_executor.content.copyto(content_array)

# train
img = mx.nd.zeros(content_np.shape, ctx=dev)
img[:] = mx.rnd.uniform(-0.1, 0.1, img.shape)

lr = mx.lr_scheduler.FactorScheduler(step=10, factor=.9)

optimizer = mx.optimizer.SGD(
    learning_rate = args.lr,
    momentum = 0.9,
    wd = 0.005,
    lr_scheduler = lr,
    clip_gradient = 10)
optim_state = optimizer.create_state(0, img)

logging.info('start training arguments %s', args)
old_img = img.asnumpy()
for e in range(args.max_num_epochs):
    img.copyto(model_executor.data)
    model_executor.executor.forward()

    # style gradient
    for i in range(len(model_executor.style)):
        model_executor.style[i].copyto(gram_executor[i].data)
        gram_executor[i].executor.forward()
        gram_executor[i].executor.backward([gram_executor[i].executor.outputs[0] - style_array[i]])
        gram_executor[i].data_grad[:] /= (gram_executor[i].data.shape[1] **2) * (float(np.prod(gram_executor[i].data.shape[2:])))
        gram_executor[i].data_grad[:] *= args.style_weight

    # content gradient
    content_grad[:] = (model_executor.content - content_array) * args.content_weight

    # image gradient
    grad_array = [gram_executor[i].data_grad for i in range(len(gram_executor))] + [content_grad]
    model_executor.executor.backward(grad_array)

    optimizer.update(0, img, model_executor.data_grad, optim_state)

    new_img = img.asnumpy()
    eps = np.linalg.norm(old_img - new_img) / np.linalg.norm(new_img)
    old_img = new_img
    logging.info('epoch %d, relative change %f', e, eps)

    if eps < args.stop_eps:
        logging.info('eps < args.stop_eps, training finished')
        break

    if (e+1) % args.save_epochs == 0:
        SaveImage(new_img, 'output/tmp_'+str(e+1)+'.jpg')

SaveImage(new_img, args.output)



# In[ ]:
