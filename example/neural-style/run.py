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
parser.add_argument('--tv-weight', type=float, default=1e-2,
                    help='the magtitute on TV loss')
parser.add_argument('--max-num-epochs', type=int, default=1000,
                    help='the maximal number of training epochs')
parser.add_argument('--max-long-edge', type=int, default=600,
                    help='resize the content image')
parser.add_argument('--lr', type=float, default=.001,
                    help='the initial learning rate')
parser.add_argument('--gpu', type=int, default=0,
                    help='which gpu card to use, -1 means using cpu')
parser.add_argument('--output', type=str, default='output/out.jpg',
                    help='the output image')
parser.add_argument('--save-epochs', type=int, default=50,
                    help='save the output every n epochs')
parser.add_argument('--remove-noise', type=float, default=.02,
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
    img = np.swapaxes(img, 1, 2)
    img = np.swapaxes(img, 0, 2)
    img = np.clip(img, 0, 255)
    return img.astype('uint8')

def SaveImage(img, filename):
    logging.info('save output to %s', filename)
    out = PostprocessImage(img)
    if args.remove_noise != 0.0:
        out = denoise_tv_chambolle(out, weight=args.remove_noise, multichannel=True)
    io.imsave(filename, out)

# input
dev = mx.gpu(args.gpu) if args.gpu >= 0 else mx.cpu()
content_np = PreprocessContentImage(args.content_image, args.max_long_edge)
style_np = PreprocessStyleImage(args.style_image, shape=content_np.shape)
size = content_np.shape[2:]

# model
Executor = namedtuple('Executor', ['executor', 'data', 'data_grad'])

def style_gram_symbol(input_size, style):
    _, output_shapes, _ = style.infer_shape(data=(1, 3, input_size[0], input_size[1]))
    gram_list = []
    grad_scale = []
    for i in range(len(style.list_outputs())):
        shape = output_shapes[i]
        x = mx.sym.Reshape(style[i], target_shape=(int(shape[1]), int(np.prod(shape[2:]))))
        # use fully connected to quickly do dot(x, x^T)
        gram = mx.sym.FullyConnected(x, x, no_bias=True, num_hidden=shape[1])
        gram_list.append(gram)
        grad_scale.append(np.prod(shape[1:]) * shape[1])
    return mx.sym.Group(gram_list), grad_scale


def get_loss(gram, content):
    gram_loss = []
    for i in range(len(gram.list_outputs())):
        gvar = mx.sym.Variable("target_gram_%d" % i)
        gram_loss.append(mx.sym.sum(mx.sym.square(gvar - gram[i])))
    cvar = mx.sym.Variable("target_content")
    content_loss = mx.sym.sum(mx.sym.square(cvar - content))
    return mx.sym.Group(gram_loss), content_loss

def get_tv_grad_executor(img, ctx, tv_weight):
    """create TV gradient executor with input binded on img
    """
    if tv_weight <= 0.0:
        return None
    nchannel = img.shape[1]
    simg = mx.sym.Variable("img")
    skernel = mx.sym.Variable("kernel")
    channels = mx.sym.SliceChannel(simg, num_outputs=nchannel)
    out = mx.sym.Concat(*[
        mx.sym.Convolution(data=channels[i], weight=skernel,
                           num_filter=1,
                           kernel=(3, 3), pad=(1,1),
                           no_bias=True, stride=(1,1))
        for i in range(nchannel)])
    kernel = mx.nd.array(np.array([[0, -1, 0],
                                   [-1, 4, -1],
                                   [0, -1, 0]])
                         .reshape((1, 1, 3, 3)),
                         ctx) / 8.0
    out = out * tv_weight
    return out.bind(ctx, args={"img": img,
                               "kernel": kernel})


import importlib
model_module =  importlib.import_module('model_' + args.model)
style, content = model_module.get_symbol()
gram, gscale = style_gram_symbol(size, style)
model_executor = model_module.get_executor(gram, content, size, dev)
model_executor.data[:] = style_np
model_executor.executor.forward()
style_array = []
for i in range(len(model_executor.style)):
    style_array.append(model_executor.style[i].copyto(mx.cpu()))

model_executor.data[:] = content_np
model_executor.executor.forward()
content_array = model_executor.content.copyto(mx.cpu())

# delete the executor
del model_executor

style_loss, content_loss = get_loss(gram, content)
model_executor = model_module.get_executor(
    style_loss, content_loss, size, dev)

grad_array = []
for i in range(len(style_array)):
    style_array[i].copyto(model_executor.arg_dict["target_gram_%d" % i])
    grad_array.append(mx.nd.ones((1,), dev) * (float(args.style_weight) / gscale[i]))
grad_array.append(mx.nd.ones((1,), dev) * (float(args.content_weight)))

print([x.asscalar() for x in grad_array])
content_array.copyto(model_executor.arg_dict["target_content"])

# train
img = mx.nd.zeros(content_np.shape, ctx=dev)
img[:] = mx.rnd.uniform(-0.1, 0.1, img.shape)

lr = mx.lr_scheduler.FactorScheduler(step=80, factor=.9)

optimizer = mx.optimizer.SGD(
    learning_rate = args.lr,
    wd = 0.0005,
    momentum=0.9,
    lr_scheduler = lr)
optim_state = optimizer.create_state(0, img)

logging.info('start training arguments %s', args)
old_img = img.copyto(dev)
clip_norm = 1 * np.prod(img.shape)
tv_grad_executor = get_tv_grad_executor(img, dev, args.tv_weight)

for e in range(args.max_num_epochs):
    img.copyto(model_executor.data)
    model_executor.executor.forward()
    model_executor.executor.backward(grad_array)
    gnorm = mx.nd.norm(model_executor.data_grad).asscalar()
    if gnorm > clip_norm:
        model_executor.data_grad[:] *= clip_norm / gnorm

    if tv_grad_executor is not None:
        tv_grad_executor.forward()
        optimizer.update(0, img,
                         model_executor.data_grad + tv_grad_executor.outputs[0],
                         optim_state)
    else:
        optimizer.update(0, img, model_executor.data_grad, optim_state)
    new_img = img
    eps = (mx.nd.norm(old_img - new_img) / mx.nd.norm(new_img)).asscalar()

    old_img = new_img.copyto(dev)
    logging.info('epoch %d, relative change %f', e, eps)
    if eps < args.stop_eps:
        logging.info('eps < args.stop_eps, training finished')
        break
    if (e+1) % args.save_epochs == 0:
        SaveImage(new_img.asnumpy(), 'output/tmp_'+str(e+1)+'.jpg')

SaveImage(new_img.asnumpy(), args.output)



# In[ ]:
