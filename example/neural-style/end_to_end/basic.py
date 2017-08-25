import sys
sys.path.insert(0, "../../mxnet/python/")

import mxnet as mx
import numpy as np
import model_vgg19 as vgg

class PretrainedInit(mx.init.Initializer):
    def __init__(self, prefix, params, verbose=False):
        self.prefix_len = len(prefix) + 1
        self.verbose = verbose
        self.arg_params = {k : v for k, v in params.items() if k.startswith("arg:")}
        self.aux_params = {k : v for k, v in params.items() if k.startswith("aux:")}
        self.arg_names = set([k[4:] for k in self.arg_params.keys()])
        self.aux_names = set([k[4:] for k in self.aux_params.keys()])

    def __call__(self, name, arr):
        key = name[self.prefix_len:]
        if key in self.arg_names:
            if self.verbose:
                print("Init %s" % name)
            self.arg_params["arg:" + key].copyto(arr)
        elif key in self.aux_params:
            if self.verbose:
                print("Init %s" % name)
            self.aux_params["aux:" + key].copyto(arr)
        else:
            print("Unknown params: %s, init with 0" % name)
            arr[:] = 0.


def style_gram_symbol(input_shape, style):
    _, output_shapes, _ = style.infer_shape(**input_shape)
    gram_list = []
    grad_scale = []
    for i in range(len(style.list_outputs())):
        shape = output_shapes[i]
        x = mx.sym.Reshape(style[i], shape=(int(shape[1]), int(np.prod(shape[2:]))))
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

def get_content_module(prefix, dshape, ctx, params):
    sym = vgg.get_vgg_symbol(prefix, True)
    init = PretrainedInit(prefix, params)
    mod = mx.mod.Module(symbol=sym,
                        data_names=("%s_data" % prefix,),
                        label_names=None,
                        context=ctx)
    mod.bind(data_shapes=[("%s_data" % prefix, dshape)], for_training=False)
    mod.init_params(init)
    return mod

def get_style_module(prefix, dshape, ctx, params):
    input_shape = {"%s_data" % prefix : dshape}
    style, content = vgg.get_vgg_symbol(prefix)
    gram, gscale = style_gram_symbol(input_shape, style)
    init = PretrainedInit(prefix, params)
    mod = mx.mod.Module(symbol=gram,
                        data_names=("%s_data" % prefix,),
                        label_names=None,
                        context=ctx)
    mod.bind(data_shapes=[("%s_data" % prefix, dshape)], for_training=False)
    mod.init_params(init)
    return mod


def get_loss_module(prefix, dshape, ctx, params):
    input_shape = {"%s_data" % prefix : dshape}
    style, content = vgg.get_vgg_symbol(prefix)
    gram, gscale = style_gram_symbol(input_shape, style)
    style_loss, content_loss = get_loss(gram, content)
    sym = mx.sym.Group([style_loss, content_loss])
    init = PretrainedInit(prefix, params)
    gram_size = len(gram.list_outputs())
    mod = mx.mod.Module(symbol=sym,
                        data_names=("%s_data" % prefix,),
                        label_names=None,
                        context=ctx)
    mod.bind(data_shapes=[("%s_data" % prefix, dshape)],
             for_training=True, inputs_need_grad=True)
    mod.init_params(init)
    return mod, gscale



if __name__ == "__main__":
    from data_processing import PreprocessContentImage, PreprocessStyleImage
    from data_processing import PostprocessImage, SaveImage
    vgg_params = mx.nd.load("./model/vgg19.params")
    style_weight = 2
    content_weight = 10
    long_edge = 384
    content_np = PreprocessContentImage("./input/IMG_4343.jpg", long_edge)
    style_np = PreprocessStyleImage("./input/starry_night.jpg", shape=content_np.shape)
    dshape = content_np.shape
    ctx = mx.gpu()
    # style
    style_mod = get_style_module("style", dshape, ctx, vgg_params)
    style_mod.forward(mx.io.DataBatch([mx.nd.array(style_np)], [0]), is_train=False)
    style_array = [arr.copyto(mx.cpu()) for arr in style_mod.get_outputs()]
    del style_mod
    # content
    content_mod = get_content_module("content", dshape, ctx, vgg_params)
    content_mod.forward(mx.io.DataBatch([mx.nd.array(content_np)], [0]), is_train=False)
    content_array = content_mod.get_outputs()[0].copyto(mx.cpu())
    del content_mod
    # loss
    mod, gscale = get_loss_module("loss", dshape, ctx, vgg_params)
    extra_args = {"target_gram_%d" % i : style_array[i] for i in range(len(style_array))}
    extra_args["target_content"] = content_array
    mod.set_params(extra_args, {}, True, True)
    grad_array = []
    for i in range(len(style_array)):
        grad_array.append(mx.nd.ones((1,), ctx) * (float(style_weight) / gscale[i]))
    grad_array.append(mx.nd.ones((1,), ctx) * (float(content_weight)))
    # train
    img = mx.nd.zeros(content_np.shape, ctx=ctx)
    img[:] = mx.rnd.uniform(-0.1, 0.1, img.shape)
    lr = mx.lr_scheduler.FactorScheduler(step=80, factor=.9)
    optimizer = mx.optimizer.SGD(
            learning_rate = 0.001,
            wd = 0.0005,
            momentum=0.9,
            lr_scheduler = lr)
    optim_state = optimizer.create_state(0, img)

    old_img = img.copyto(ctx)
    clip_norm = 1 * np.prod(img.shape)

    import logging
    for e in range(800):
        mod.forward(mx.io.DataBatch([img], [0]), is_train=True)
        mod.backward(grad_array)
        data_grad = mod.get_input_grads()[0]
        gnorm = mx.nd.norm(data_grad).asscalar()
        if gnorm > clip_norm:
            print("Data Grad: ", gnorm / clip_norm)
            data_grad[:] *= clip_norm / gnorm

        optimizer.update(0, img, data_grad, optim_state)
        new_img = img
        eps = (mx.nd.norm(old_img - new_img) / mx.nd.norm(new_img)).asscalar()
        old_img = new_img.copyto(ctx)
        logging.info('epoch %d, relative change %f', e, eps)
        if (e+1) % 50 == 0:
            SaveImage(new_img.asnumpy(), 'output/tmp_'+str(e+1)+'.jpg')

    SaveImage(new_img.asnumpy(), "./output/out.jpg")

