import sys
sys.path.insert(0, "../../mxnet/python")

import mxnet as mx
import numpy as np

import basic
import data_processing
import gen_v3
import gen_v4

# params
vgg_params = mx.nd.load("./vgg19.params")
style_weight = 1.2
content_weight = 10
dshape = (1, 3, 384, 384)
clip_norm = 0.05 * np.prod(dshape)
model_prefix = "v3"
ctx = mx.gpu(0)

# init style
style_np = data_processing.PreprocessStyleImage("../starry_night.jpg", shape=dshape)
style_mod = basic.get_style_module("style", dshape, ctx, vgg_params)
style_mod.forward(mx.io.DataBatch([mx.nd.array(style_np)], [0]), is_train=False)
style_array = [arr.copyto(mx.cpu()) for arr in style_mod.get_outputs()]
del style_mod

# content
content_mod = basic.get_content_module("content", dshape, ctx, vgg_params)

# loss
loss, gscale = basic.get_loss_module("loss", dshape, ctx, vgg_params)
extra_args = {"target_gram_%d" % i : style_array[i] for i in range(len(style_array))}
loss.set_params(extra_args, {}, True, True)
grad_array = []
for i in range(len(style_array)):
    grad_array.append(mx.nd.ones((1,), ctx) * (float(style_weight) / gscale[i]))
grad_array.append(mx.nd.ones((1,), ctx) * (float(content_weight)))

# generator
gens = [gen_v4.get_module("g0", dshape, ctx),
        gen_v3.get_module("g1", dshape, ctx),
        gen_v3.get_module("g2", dshape, ctx),
        gen_v4.get_module("g3", dshape, ctx)]
for gen in gens:
    gen.init_optimizer(
        optimizer='sgd',
        optimizer_params={
            'learning_rate': 1e-4,
            'momentum' : 0.9,
            'wd': 5e-3,
            'clip_gradient' : 5.0
        })


# tv-loss
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
tv_weight = 1e-2

start_epoch = 0
end_epoch = 3


# data
import os
import random
import logging

data_root = "../data/"
file_list = os.listdir(data_root)
num_image = len(file_list)
logging.info("Dataset size: %d" % num_image)


# train

for i in range(start_epoch, end_epoch):
    random.shuffle(file_list)
    for idx in range(num_image):
        loss_grad_array = []
        data_array = []
        path = data_root + file_list[idx]
        content_np = data_processing.PreprocessContentImage(path, min(dshape[2:]), dshape)
        data = mx.nd.array(content_np)
        data_array.append(data)
        # get content
        content_mod.forward(mx.io.DataBatch([data], [0]), is_train=False)
        content_array = content_mod.get_outputs()[0].copyto(mx.cpu())
        # set target content
        loss.set_params({"target_content" : content_array}, {}, True, True)
        # gen_forward
        for k in range(len(gens)):
            gens[k].forward(mx.io.DataBatch([data_array[-1]], [0]), is_train=True)
            data_array.append(gens[k].get_outputs()[0].copyto(mx.cpu()))
            # loss forward
            loss.forward(mx.io.DataBatch([data_array[-1]], [0]), is_train=True)
            loss.backward(grad_array)
            grad = loss.get_input_grads()[0]
            loss_grad_array.append(grad.copyto(mx.cpu()))
        grad = mx.nd.zeros(data.shape)
        for k in range(len(gens) - 1, -1, -1):
            tv_grad_executor = get_tv_grad_executor(gens[k].get_outputs()[0],
                    ctx, tv_weight)
            tv_grad_executor.forward()

            grad[:] += loss_grad_array[k] + tv_grad_executor.outputs[0].copyto(mx.cpu())
            gnorm = mx.nd.norm(grad).asscalar()
            if gnorm > clip_norm:
                grad[:] *= clip_norm / gnorm

            gens[k].backward([grad])
            gens[k].update()
        if idx % 20 == 0:
            logging.info("Epoch %d: Image %d" % (i, idx))
            for k in range(len(gens)):
                logging.info("Data Norm :%.5f" %\
                        (mx.nd.norm(gens[k].get_input_grads()[0]).asscalar() / np.prod(dshape)))
        if idx % 1000 == 0:
            for k in range(len(gens)):
                gens[k].save_params("./model/%d/%s_%04d-%07d.params" % (k, model_prefix, i, idx))





