# pylint: skip-file
import mxnet as mx
import numpy as np

# make a bilinear interpolation kernel, return a numpy.ndarray
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

def init_fcn32s_params(ctx, fcn32s_symbol, vgg16fc_arg_params, vgg16fc_aux_params, retrain):
    fcn32s_arg_params = vgg16fc_arg_params.copy()
    fcn32s_aux_params = vgg16fc_aux_params.copy()
    if not retrain:
        for k,v in fcn32s_arg_params.items():
            if(v.context != ctx):
                fcn32s_arg_params[k] = mx.nd.zeros(v.shape, ctx)
                v.copyto(fcn32s_arg_params[k])
        for k,v in fcn32s_aux_params.items():
            if(v.context != ctx):
                fcn32s_aux_params[k] = mx.nd.zeros(v.shape, ctx)
                v.copyto(fcn32s_aux_params[k])
        data_shape=(1,3,500,500)
        arg_names = fcn32s_symbol.list_arguments()
        arg_shapes, _, _ = fcn32s_symbol.infer_shape(data=data_shape)
        rest_params = dict([(x[0], mx.nd.zeros(x[1], ctx)) for x in zip(arg_names, arg_shapes)
            if x[0] in ['fc8_weight', 'fc8_bias']])
        fcn32s_arg_params.update(rest_params)
        deconv_params = dict([(x[0], x[1]) for x in zip(arg_names, arg_shapes) if x[0] \
            in ["deconv8_weight"]])
        for k, v in deconv_params.items():
            filt = upsample_filt(v[3])
            initw = np.zeros(v)
            initw[range(v[0]), range(v[1]), :, :] = filt  # becareful here is the slice assing
            fcn32s_arg_params[k] = mx.nd.array(initw, ctx)
    else:
        print "it is retrain, so will use the model weight trained before."
    return fcn32s_arg_params, fcn32s_aux_params

def init_fcn16s_params(ctx, fcn16s_symbol, vgg16fc_arg_params, vgg16fc_aux_params):
    fcn16s_arg_params = vgg16fc_arg_params.copy()
    fcn16s_aux_params = vgg16fc_aux_params.copy()
    del fcn16s_arg_params["fc8_weight"]
    del fcn16s_arg_params["fc8_bias"]
    for k,v in fcn16s_arg_params.items():
        if(v.context != ctx):
            fcn16s_arg_params[k] = mx.nd.zeros(v.shape, ctx)
            v.copyto(fcn8s_arg_params[k])
    for k,v in fcn16s_aux_params.items():
        if(v.context != ctx):
            fcn16s_aux_params[k] = mx.nd.zeros(v.shape, ctx)
            v.copyto(fcn8s_aux_params[k])
    data_shape=(1,3,500,500)
    arg_names = fcn16s_symbol.list_arguments()
    arg_shapes, _, _ = fcn16s_symbol.infer_shape(data=data_shape)
    rest_params = dict([(x[0], mx.nd.zeros(x[1], ctx)) for x in zip(arg_names, arg_shapes)
        if x[0] in ['score_weight', 'score_bias',
                    'score_pool4_weight', 'score_pool4_bias']])
    fcn16s_arg_params.update(rest_params)
    deconv_params = dict([(x[0], x[1]) for x in zip(arg_names, arg_shapes) if x[0] \
        in ['score2_weight', 'bigscore_weight']])
    for k, v in deconv_params.items():
        filt = upsample_filt(v[3])
        initw = np.zeros(v)
        initw[range(v[0]), range(v[1]), :, :] = filt  # becareful here is the slice assing
        fcn16s_arg_params[k] = mx.nd.array(initw, ctx)

    # print "fcn16s_arg_params[conv1_1_weight]=", fcn16s_arg_params["conv1_1_weight"].asnumpy()
    # print "fcn16s_arg_params[conv1_2_weight]=", fcn16s_arg_params["conv1_2_weight"].asnumpy()
    return fcn16s_arg_params, fcn16s_aux_params

def init_fcn16s_params_from_fcn32s(ctx, fcn16s_symbol, fcn32s_arg_params, fcn32s_aux_params):
    fcn16s_arg_params = fcn32s_arg_params.copy()
    fcn16s_aux_params = fcn32s_aux_params.copy()
    fcn16s_arg_params["score_weight"] = fcn16s_arg_params["fc8_weight"]
    fcn16s_arg_params["score_bias"] = fcn16s_arg_params["fc8_bias"]
    del fcn16s_arg_params["deconv8_weight"]
    for k,v in fcn16s_arg_params.items():
        if(v.context != ctx):
            fcn16s_arg_params[k] = mx.nd.zeros(v.shape, ctx)
            v.copyto(fcn16s_arg_params[k])
    for k,v in fcn16s_aux_params.items():
        if(v.context != ctx):
            fcn16s_aux_params[k] = mx.nd.zeros(v.shape, ctx)
            v.copyto(fcn16s_aux_params[k])
    data_shape=(1,3,500,500)
    arg_names = fcn16s_symbol.list_arguments()
    arg_shapes, _, _ = fcn16s_symbol.infer_shape(data=data_shape)
    rest_params = dict([(x[0], mx.nd.zeros(x[1], ctx)) for x in zip(arg_names, arg_shapes)
        if x[0] in ['score_pool4_weight', 'score_pool4_bias']])
    fcn16s_arg_params.update(rest_params)
    deconv_params = dict([(x[0], x[1]) for x in zip(arg_names, arg_shapes) if x[0] \
        in ['score2_weight', 'bigscore_weight']])
    for k, v in deconv_params.items():
        filt = upsample_filt(v[3])
        initw = np.zeros(v)
        initw[range(v[0]), range(v[1]), :, :] = filt  # becareful here is the slice assing
        fcn16s_arg_params[k] = mx.nd.array(initw, ctx)

    return fcn16s_arg_params, fcn16s_aux_params

def init_fcn8s_params(ctx, fcn8s_symbol, vgg16fc_arg_params, vgg16fc_aux_params):
    fcn8s_arg_params = vgg16fc_arg_params.copy()
    fcn8s_aux_params = vgg16fc_aux_params.copy()
    del fcn8s_arg_params["fc8_weight"]
    del fcn8s_arg_params["fc8_bias"]
    for k,v in fcn8s_arg_params.items():
        if(v.context != ctx):
            fcn8s_arg_params[k] = mx.nd.zeros(v.shape, ctx)
            v.copyto(fcn8s_arg_params[k])
    for k,v in fcn8s_aux_params.items():
        if(v.context != ctx):
            fcn8s_aux_params[k] = mx.nd.zeros(v.shape, ctx)
            v.copyto(fcn8s_aux_params[k])
    data_shape=(1,3,500,500)
    arg_names = fcn8s_symbol.list_arguments()
    arg_shapes, _, _ = fcn8s_symbol.infer_shape(data=data_shape)
    rest_params = dict([(x[0], mx.nd.zeros(x[1], ctx)) for x in zip(arg_names, arg_shapes)
        if x[0] in ['score_weight', 'score_bias',
                    'score_pool4_weight', 'score_pool4_bias', \
                    'score_pool3_bias', 'score_pool3_weight']])
    fcn8s_arg_params.update(rest_params)
    deconv_params = dict([(x[0], x[1]) for x in zip(arg_names, arg_shapes) if x[0] \
        in ['score2_weight', 'score4_weight', 'bigscore_weight']])
    for k, v in deconv_params.items():
        filt = upsample_filt(v[3])
        initw = np.zeros(v)
        initw[range(v[0]), range(v[1]), :, :] = filt  # becareful here is the slice assing
        fcn8s_arg_params[k] = mx.nd.array(initw, ctx)

    # print "fcn8s_arg_params[conv1_1_weight]=", fcn8s_arg_params["conv1_1_weight"].asnumpy()
    # print "fcn8s_arg_params[conv1_2_weight]=", fcn8s_arg_params["conv1_2_weight"].asnumpy()
    return fcn8s_arg_params, fcn8s_aux_params

def init_fcn8s_params_from_fcn16s(ctx, fcn8s_symbol, fcn16s_arg_params, fcn16s_aux_params):
    fcn8s_arg_params = fcn16s_arg_params.copy()
    fcn8s_aux_params = fcn16s_aux_params.copy()
    for k,v in fcn8s_arg_params.items():
        if(v.context != ctx):
            fcn8s_arg_params[k] = mx.nd.zeros(v.shape, ctx)
            v.copyto(fcn8s_arg_params[k])
    for k,v in fcn8s_aux_params.items():
        if(v.context != ctx):
            fcn8s_aux_params[k] = mx.nd.zeros(v.shape, ctx)
            v.copyto(fcn8s_aux_params[k])
    data_shape=(1,3,500,500)
    arg_names = fcn8s_symbol.list_arguments()
    arg_shapes, _, _ = fcn8s_symbol.infer_shape(data=data_shape)
    rest_params = dict([(x[0], mx.nd.zeros(x[1], ctx)) for x in zip(arg_names, arg_shapes)
        if x[0] in ['score_pool3_bias', 'score_pool3_weight']])
    fcn8s_arg_params.update(rest_params)
    deconv_params = dict([(x[0], x[1]) for x in zip(arg_names, arg_shapes) if x[0] \
        in ['score4_weight', 'bigscore_weight']])
    for k, v in deconv_params.items():
        filt = upsample_filt(v[3])
        initw = np.zeros(v)
        initw[range(v[0]), range(v[1]), :, :] = filt  # becareful here is the slice assing
        fcn8s_arg_params[k] = mx.nd.array(initw, ctx)

    return fcn8s_arg_params, fcn8s_aux_params
