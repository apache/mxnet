# pylint: skip-file
import mxnet as mx

def filter_map(kernel=1, stride=1, pad=0):
    # why not return (stride, (kernel-stride)/2-pad)??
    return (stride, (kernel-1)/2-pad)

def compose_fp(fp_first, fp_second):
    return (fp_first[0]*fp_second[0], fp_first[0]*fp_second[1]+fp_first[1])

def compose_fp_list(fp_list):
    fp_out = (1.0, 0.0)
    for fp in fp_list:
        fp_out = compose_fp(fp_out, fp)
    return fp_out

def inv_fp(fp_in):
    return (1.0/fp_in[0], -1.0*fp_in[1]/fp_in[0])

def offset():
    conv1_1_fp = filter_map(kernel=3, pad=100)
    conv1_2_fp = conv2_1_fp = conv2_2_fp = conv3_1_fp = conv3_2_fp = conv3_3_fp \
               = conv4_1_fp = conv4_2_fp = conv4_3_fp = conv5_1_fp = conv5_2_fp \
               = conv5_3_fp = filter_map(kernel=3, pad=1)
    pool1_fp = pool2_fp = pool3_fp = pool4_fp = pool5_fp = filter_map(kernel=2, stride=2)
    fc6_fp = filter_map(kernel=7)
    fc7_fp = score_fp = score_pool4_fp = score_pool3_fp = filter_map()
    # for fcn-32s
    fcn32s_upscore_fp = inv_fp(filter_map(kernel=64, stride=32))
    fcn32s_upscore_list = [conv1_1_fp, conv1_2_fp, pool1_fp, conv2_1_fp, conv2_2_fp,
                           pool2_fp, conv3_1_fp, conv3_2_fp, conv3_3_fp, pool3_fp,
                           conv4_1_fp, conv4_2_fp, conv4_3_fp, pool4_fp, conv5_1_fp,
                           conv5_2_fp, conv5_3_fp, pool5_fp, fc6_fp, fc7_fp, score_fp,
                           fcn32s_upscore_fp]
    crop = {}
    crop["fcn32s_upscore"] = (-int(round(compose_fp_list(fcn32s_upscore_list)[1])),
                              -int(round(compose_fp_list(fcn32s_upscore_list)[1])))
    # for fcn-16s
    score2_fp = inv_fp(filter_map(kernel=4, stride=2))
    fcn16s_upscore_fp = inv_fp(filter_map(kernel=32, stride=16))
    score_pool4c_fp_list = [inv_fp(score2_fp), inv_fp(score_fp), inv_fp(fc7_fp), inv_fp(fc6_fp),
                            inv_fp(pool5_fp), inv_fp(conv5_3_fp), inv_fp(conv5_2_fp),
                            inv_fp(conv5_1_fp), score_pool4_fp]
    crop["score_pool4c"] = (-int(round(compose_fp_list(score_pool4c_fp_list)[1])),
                            -int(round(compose_fp_list(score_pool4c_fp_list)[1])))
    fcn16s_upscore_list =  [conv1_1_fp, conv1_2_fp, pool1_fp, conv2_1_fp, conv2_2_fp,
                            pool2_fp, conv3_1_fp, conv3_2_fp, conv3_3_fp, pool3_fp,
                            conv4_1_fp, conv4_2_fp, conv4_3_fp, pool4_fp, score_pool4_fp,
                            inv_fp((1, -crop["score_pool4c"][0])), fcn16s_upscore_fp]
    crop["fcn16s_upscore"] = (-int(round(compose_fp_list(fcn16s_upscore_list)[1])),
                              -int(round(compose_fp_list(fcn16s_upscore_list)[1])))
    # for fcn-8s
    score4_fp = inv_fp(filter_map(kernel=4, stride=2))
    fcn8s_upscore_fp = inv_fp(filter_map(kernel=16, stride=8))
    score_pool3c_fp_list = [inv_fp(score4_fp), (1, -crop["score_pool4c"][0]), inv_fp(score_pool4_fp),
                            inv_fp(pool4_fp), inv_fp(conv4_3_fp), inv_fp(conv4_2_fp),
                            inv_fp(conv4_1_fp), score_pool3_fp, score_pool3_fp]
    crop["score_pool3c"] = (-int(round(compose_fp_list(score_pool3c_fp_list)[1])),
                            -int(round(compose_fp_list(score_pool3c_fp_list)[1])))
    fcn8s_upscore_list =  [conv1_1_fp, conv1_2_fp, pool1_fp, conv2_1_fp, conv2_2_fp, pool2_fp,
                           conv3_1_fp, conv3_2_fp, conv3_3_fp, pool3_fp, score_pool3_fp,
                           inv_fp((1, -crop["score_pool3c"][0])), fcn8s_upscore_fp]
    crop["fcn8s_upscore"] = (-int(round(compose_fp_list(fcn8s_upscore_list)[1])),
                             -int(round(compose_fp_list(fcn8s_upscore_list)[1])))
    return crop

def vgg16_pool3(input, workspace_default=1024):
    # group 1
    conv1_1 = mx.symbol.Convolution(data=input, kernel=(3, 3), pad=(100, 100), num_filter=64,
                workspace=workspace_default, name="conv1_1")
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    conv1_2 = mx.symbol.Convolution(data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=64,
                workspace=workspace_default, name="conv1_2")
    relu1_2 = mx.symbol.Activation(data=conv1_2, act_type="relu", name="relu1_2")
    pool1 = mx.symbol.Pooling(data=relu1_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool1")
    # group 2
    conv2_1 = mx.symbol.Convolution(data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128,
                workspace=workspace_default, name="conv2_1")
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    conv2_2 = mx.symbol.Convolution(data=relu2_1, kernel=(3, 3), pad=(1, 1), num_filter=128,
                workspace=workspace_default, name="conv2_2")
    relu2_2 = mx.symbol.Activation(data=conv2_2, act_type="relu", name="relu2_2")
    pool2 = mx.symbol.Pooling(data=relu2_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool2")
    # group 3
    conv3_1 = mx.symbol.Convolution(data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256,
                workspace=workspace_default, name="conv3_1")
    relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.symbol.Convolution(data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256,
                workspace=workspace_default, name="conv3_2")
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    conv3_3 = mx.symbol.Convolution(data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=256,
                workspace=workspace_default, name="conv3_3")
    relu3_3 = mx.symbol.Activation(data=conv3_3, act_type="relu", name="relu3_3")
    pool3 = mx.symbol.Pooling(data=relu3_3, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool3")
    return pool3

def vgg16_pool4(input, workspace_default=1024):
    # group 4
    conv4_1 = mx.symbol.Convolution(data=input, kernel=(3, 3), pad=(1, 1), num_filter=512,
                workspace=workspace_default, name="conv4_1")
    relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.symbol.Convolution(data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512,
                workspace=workspace_default, name="conv4_2")
    relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    conv4_3 = mx.symbol.Convolution(data=relu4_2, kernel=(3, 3), pad=(1, 1), num_filter=512,
                workspace=workspace_default, name="conv4_3")
    relu4_3 = mx.symbol.Activation(data=conv4_3, act_type="relu", name="relu4_3")
    pool4 = mx.symbol.Pooling(data=relu4_3, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool4")
    return pool4

def vgg16_score(input, numclass, workspace_default=1024):
    # group 5
    conv5_1 = mx.symbol.Convolution(data=input, kernel=(3, 3), pad=(1, 1), num_filter=512,
                workspace=workspace_default, name="conv5_1")
    relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.symbol.Convolution(data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512,
                workspace=workspace_default, name="conv5_2")
    relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="conv1_2")
    conv5_3 = mx.symbol.Convolution(data=relu5_2, kernel=(3, 3), pad=(1, 1), num_filter=512,
                workspace=workspace_default, name="conv5_3")
    relu5_3 = mx.symbol.Activation(data=conv5_3, act_type="relu", name="relu5_3")
    pool5 = mx.symbol.Pooling(data=relu5_3, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool5")
    # group 6
    fc6 = mx.symbol.Convolution(data=pool5, kernel=(7, 7), num_filter=4096,
                workspace=workspace_default, name="fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.symbol.Convolution(data=drop6, kernel=(1, 1), num_filter=4096,
                workspace=workspace_default, name="fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
    # group 8
    score = mx.symbol.Convolution(data=drop7, kernel=(1, 1), num_filter=numclass,
                workspace=workspace_default, name="score")
    return score

def fcnxs_score(input, crop, offset, kernel=(64,64), stride=(32,32), numclass=21, workspace_default=1024):
    # score out
    bigscore = mx.symbol.Deconvolution(data=input, kernel=kernel, stride=stride, num_filter=numclass,
                workspace=workspace_default, name="bigscore")
    upscore = mx.symbol.Crop(*[bigscore, crop], offset=offset, name="upscore")
    softmax = mx.symbol.SoftmaxOutput(data=upscore, multi_output=True, use_ignore=True, ignore_label=255, name="softmax")
    return softmax

def get_fcn32s_symbol(numclass=21, workspace_default=1024):
    data = mx.symbol.Variable(name="data")
    pool3 = vgg16_pool3(data, workspace_default)
    pool4 = vgg16_pool4(pool3, workspace_default)
    score = vgg16_score(pool4, numclass, workspace_default)
    softmax = fcnxs_score(score, data, offset()["fcn32s_upscore"], (64,64), (32,32), numclass, workspace_default)
    return softmax

def get_fcn16s_symbol(numclass=21, workspace_default=1024):
    data = mx.symbol.Variable(name="data")
    pool3 = vgg16_pool3(data, workspace_default)
    pool4 = vgg16_pool4(pool3, workspace_default)
    score = vgg16_score(pool4, numclass, workspace_default)
    # score 2X
    score2 = mx.symbol.Deconvolution(data=score, kernel=(4, 4), stride=(2, 2), num_filter=numclass,
                workspace=workspace_default, name="score2")  # 2X
    score_pool4 = mx.symbol.Convolution(data=pool4, kernel=(1, 1), num_filter=numclass,
                workspace=workspace_default, name="score_pool4")
    score_pool4c = mx.symbol.Crop(*[score_pool4, score2], offset=offset()["score_pool4c"], name="score_pool4c")
    score_fused = score2 + score_pool4c
    softmax = fcnxs_score(score_fused, data, offset()["fcn16s_upscore"], (32, 32), (16, 16), numclass, workspace_default)
    return softmax

def get_fcn8s_symbol(numclass=21, workspace_default=1024):
    data = mx.symbol.Variable(name="data")
    pool3 = vgg16_pool3(data, workspace_default)
    pool4 = vgg16_pool4(pool3, workspace_default)
    score = vgg16_score(pool4, numclass, workspace_default)
    # score 2X
    score2 = mx.symbol.Deconvolution(data=score, kernel=(4, 4), stride=(2, 2),num_filter=numclass,
                workspace=workspace_default, name="score2")  # 2X
    score_pool4 = mx.symbol.Convolution(data=pool4, kernel=(1, 1), num_filter=numclass,
                workspace=workspace_default, name="score_pool4")
    score_pool4c = mx.symbol.Crop(*[score_pool4, score2], offset=offset()["score_pool4c"], name="score_pool4c")
    score_fused = score2 + score_pool4c
    # score 4X
    score4 = mx.symbol.Deconvolution(data=score_fused, kernel=(4, 4), stride=(2, 2),num_filter=numclass,
                workspace=workspace_default, name="score4") # 4X
    score_pool3 = mx.symbol.Convolution(data=pool3, kernel=(1, 1), num_filter=numclass,
                workspace=workspace_default, name="score_pool3")
    score_pool3c = mx.symbol.Crop(*[score_pool3, score4], offset=offset()["score_pool3c"], name="score_pool3c")
    score_final = score4 + score_pool3c
    softmax = fcnxs_score(score_final, data, offset()["fcn8s_upscore"], (16, 16), (8, 8), numclass, workspace_default)
    return softmax
