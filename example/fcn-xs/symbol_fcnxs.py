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

def coord_map_fcn32s():
    conv1_1_fp = filter_map(kernel=3, pad=100)
    conv1_2_fp = filter_map(kernel=3, pad=1)
    pool1_fp = filter_map(kernel=2, stride=2)
    conv2_1_fp = filter_map(kernel=3, pad=1)
    conv2_2_fp = filter_map(kernel=3, pad=1)
    pool2_fp = filter_map(kernel=2, stride=2)
    conv3_1_fp = filter_map(kernel=3, pad=1)
    conv3_2_fp = filter_map(kernel=3, pad=1)
    conv3_3_fp = filter_map(kernel=3, pad=1)
    pool3_fp = filter_map(kernel=2, stride=2)
    conv4_1_fp = filter_map(kernel=3, pad=1)
    conv4_2_fp = filter_map(kernel=3, pad=1)
    conv4_3_fp = filter_map(kernel=3, pad=1)
    pool4_fp = filter_map(kernel=2, stride=2)
    conv5_1_fp = filter_map(kernel=3, pad=1)
    conv5_2_fp = filter_map(kernel=3, pad=1)
    conv5_3_fp = filter_map(kernel=3, pad=1)
    pool5_fp = filter_map(kernel=2, stride=2)
    fc6_fp = filter_map(kernel=7)
    fc7_fp = filter_map()
    fc8_fp = filter_map()
    deconv8_fp = inv_fp(filter_map(kernel=64, stride=32))
    fp_list = [conv1_1_fp, conv1_2_fp, pool1_fp,
               conv2_1_fp, conv2_2_fp, pool2_fp,
               conv3_1_fp, conv3_2_fp, conv3_3_fp, pool3_fp,
               conv4_1_fp, conv4_2_fp, conv4_3_fp, pool4_fp,
               conv5_1_fp, conv5_2_fp, conv5_3_fp, pool5_fp,
               fc6_fp, fc7_fp, fc8_fp, deconv8_fp]
    crop = {}
    crop["crop8"] = (-int(round(compose_fp_list(fp_list)[1])),
        -int(round(compose_fp_list(fp_list)[1])))
    # print  "crop8=", crop["crop8"]
    return crop

def coord_map_fcn16s():
    conv1_1_fp = filter_map(kernel=3, pad=100)
    conv1_2_fp = filter_map(kernel=3, pad=1)
    pool1_fp = filter_map(kernel=2, stride=2)
    conv2_1_fp = filter_map(kernel=3, pad=1)
    conv2_2_fp = filter_map(kernel=3, pad=1)
    pool2_fp = filter_map(kernel=2, stride=2)
    conv3_1_fp = filter_map(kernel=3, pad=1)
    conv3_2_fp = filter_map(kernel=3, pad=1)
    conv3_3_fp = filter_map(kernel=3, pad=1)
    pool3_fp = filter_map(kernel=2, stride=2)
    conv4_1_fp = filter_map(kernel=3, pad=1)
    conv4_2_fp = filter_map(kernel=3, pad=1)
    conv4_3_fp = filter_map(kernel=3, pad=1)
    pool4_fp = filter_map(kernel=2, stride=2)
    conv5_1_fp = filter_map(kernel=3, pad=1)
    conv5_2_fp = filter_map(kernel=3, pad=1)
    conv5_3_fp = filter_map(kernel=3, pad=1)
    pool5_fp = filter_map(kernel=2, stride=2)
    fc6_fp = filter_map(kernel=7)
    fc7_fp = filter_map()
    score_fp = filter_map()
    score2_fp = inv_fp(filter_map(kernel=4, stride=2))
    score_pool4_fp = filter_map()
    bigscore_fp = inv_fp(filter_map(kernel=32, stride=16))
    crop = {}
    score_pool4c_fp_list = [inv_fp(score2_fp), inv_fp(score_fp),
                            inv_fp(fc7_fp), inv_fp(fc6_fp),
                            inv_fp(pool5_fp), inv_fp(conv5_3_fp),
                            inv_fp(conv5_2_fp), inv_fp(conv5_1_fp),
                            score_pool4_fp]
    crop["score_pool4c"] = (-int(round(compose_fp_list(score_pool4c_fp_list)[1])),
                            -int(round(compose_fp_list(score_pool4c_fp_list)[1])))
    upscore_fp_list =  [conv1_1_fp, conv1_2_fp, pool1_fp,
                        conv2_1_fp, conv2_2_fp, pool2_fp,
                        conv3_1_fp, conv3_2_fp, conv3_3_fp, pool3_fp,
                        conv4_1_fp, conv4_2_fp, conv4_3_fp, pool4_fp,
                        score_pool4_fp, inv_fp((1, -crop["score_pool4c"][0])),
                        bigscore_fp]
    crop["upscore"] = (-int(round(compose_fp_list(upscore_fp_list)[1])),
                       -int(round(compose_fp_list(upscore_fp_list)[1])))
    # print  "score_pool4c=", crop["score_pool4c"]
    # print  "upscore=", crop["upscore"]
    return crop

def coord_map_fcn8s():
    conv1_1_fp = filter_map(kernel=3, pad=100)
    conv1_2_fp = filter_map(kernel=3, pad=1)
    pool1_fp = filter_map(kernel=2, stride=2)
    conv2_1_fp = filter_map(kernel=3, pad=1)
    conv2_2_fp = filter_map(kernel=3, pad=1)
    pool2_fp = filter_map(kernel=2, stride=2)
    conv3_1_fp = filter_map(kernel=3, pad=1)
    conv3_2_fp = filter_map(kernel=3, pad=1)
    conv3_3_fp = filter_map(kernel=3, pad=1)
    pool3_fp = filter_map(kernel=2, stride=2)
    conv4_1_fp = filter_map(kernel=3, pad=1)
    conv4_2_fp = filter_map(kernel=3, pad=1)
    conv4_3_fp = filter_map(kernel=3, pad=1)
    pool4_fp = filter_map(kernel=2, stride=2)
    conv5_1_fp = filter_map(kernel=3, pad=1)
    conv5_2_fp = filter_map(kernel=3, pad=1)
    conv5_3_fp = filter_map(kernel=3, pad=1)
    pool5_fp = filter_map(kernel=2, stride=2)
    fc6_fp = filter_map(kernel=7)
    fc7_fp = filter_map()
    score_fp = filter_map()
    score2_fp = inv_fp(filter_map(kernel=4, stride=2))
    score_pool4_fp = filter_map()
    score4_fp = inv_fp(filter_map(kernel=4, stride=2))
    score_pool3_fp = filter_map()
    bigscore_fp = inv_fp(filter_map(kernel=16, stride=8))
    crop = {}
    score_pool4c_fp_list = [inv_fp(score2_fp), inv_fp(score_fp),
                            inv_fp(fc7_fp), inv_fp(fc6_fp),
                            inv_fp(pool5_fp), inv_fp(conv5_3_fp),
                            inv_fp(conv5_2_fp), inv_fp(conv5_1_fp),
                            score_pool4_fp]
    crop["score_pool4c"] = (-int(round(compose_fp_list(score_pool4c_fp_list)[1])),
                            -int(round(compose_fp_list(score_pool4c_fp_list)[1])))
    score_pool3c_fp_list = [inv_fp(score4_fp), (1, -crop["score_pool4c"][0]),
                            inv_fp(score_pool4_fp), inv_fp(pool4_fp),
                            inv_fp(conv4_3_fp), inv_fp(conv4_2_fp),
                            inv_fp(conv4_1_fp), score_pool3_fp,
                            score_pool3_fp]
    crop["score_pool3c"] = (-int(round(compose_fp_list(score_pool3c_fp_list)[1])),
                            -int(round(compose_fp_list(score_pool3c_fp_list)[1])))
    upscore_fp_list =  [conv1_1_fp, conv1_2_fp, pool1_fp,
                        conv2_1_fp, conv2_2_fp, pool2_fp,
                        conv3_1_fp, conv3_2_fp, conv3_3_fp,
                        pool3_fp, score_pool3_fp, inv_fp((1, -crop["score_pool3c"][0])),
                        bigscore_fp]
    crop["upscore"] = (-int(round(compose_fp_list(upscore_fp_list)[1])),
                       -int(round(compose_fp_list(upscore_fp_list)[1])))
    # print  "score_pool4c=", crop["score_pool4c"]
    # print  "score_pool3c=", crop["score_pool3c"]
    # print  "upscore=", crop["upscore"]
    return crop

def get_fcn32s_symbol(numclass=21, workspace_default=1024):
    data = mx.symbol.Variable(name="data")
    # group 1
    conv1_1 = mx.symbol.Convolution(
        data=data, kernel=(3, 3), pad=(100, 100), num_filter=64, name="conv1_1",
        workspace=workspace_default)  # coord_map()
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    conv1_2 = mx.symbol.Convolution(
        data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_2",
        workspace=workspace_default)
    relu1_2 = mx.symbol.Activation(data=conv1_2, act_type="relu", name="relu1_2")
    pool1 = mx.symbol.Pooling(
        data=relu1_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool1")
    # group 2
    conv2_1 = mx.symbol.Convolution(
        data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_1",
        workspace=workspace_default)
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    conv2_2 = mx.symbol.Convolution(
        data=relu2_1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_2",
        workspace=workspace_default)
    relu2_2 = mx.symbol.Activation(data=conv2_2, act_type="relu", name="relu2_2")
    pool2 = mx.symbol.Pooling(
        data=relu2_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool2")
    # group 3
    conv3_1 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_1",
        workspace=workspace_default)
    relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.symbol.Convolution(
        data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_2",
        workspace=workspace_default)
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    conv3_3 = mx.symbol.Convolution(
        data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_3",
        workspace=workspace_default)
    relu3_3 = mx.symbol.Activation(data=conv3_3, act_type="relu", name="relu3_3")
    pool3 = mx.symbol.Pooling(
        data=relu3_3, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool3")
    # group 4
    conv4_1 = mx.symbol.Convolution(
        data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_1",
        workspace=workspace_default)
    relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.symbol.Convolution(
        data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_2",
        workspace=workspace_default)
    relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    conv4_3 = mx.symbol.Convolution(
        data=relu4_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_3",
        workspace=workspace_default)
    relu4_3 = mx.symbol.Activation(data=conv4_3, act_type="relu", name="relu4_3")
    pool4 = mx.symbol.Pooling(
        data=relu4_3, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool4")
    # group 5
    conv5_1 = mx.symbol.Convolution(
        data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_1",
        workspace=workspace_default)
    relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.symbol.Convolution(
        data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_2",
        workspace=workspace_default)
    relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="conv1_2")
    conv5_3 = mx.symbol.Convolution(
        data=relu5_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_3",
        workspace=workspace_default)
    relu5_3 = mx.symbol.Activation(data=conv5_3, act_type="relu", name="relu5_3")
    pool5 = mx.symbol.Pooling(
        data=relu5_3, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool5")
    # group 6
    fc6 = mx.symbol.Convolution(
        data=pool5, kernel=(7, 7), num_filter=4096, name="fc6",
        workspace=workspace_default)
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.symbol.Convolution(
        data=drop6, kernel=(1, 1), num_filter=4096, name="fc7",
        workspace=workspace_default)
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
    # group 8
    fc8 = mx.symbol.Convolution(
        data=drop7, kernel=(1, 1), num_filter=21, name="fc8",
        workspace=workspace_default)
    deconv8 = mx.symbol.Deconvolution(
        data=fc8, kernel=(64, 64), stride=(32, 32),
        num_filter=21, name="deconv8",
        workspace=workspace_default)
    crop8 = mx.symbol.Crop(
        data=deconv8, crop_like=data,
        offset=coord_map_fcn32s()["crop8"], name="crop8")
    softmax = mx.symbol.SoftmaxOutput(
         data=crop8, multi_output=True, ignore_label=255, name="softmax")
    return softmax

def get_fcn16s_symbol(numclass=21, workspace_default=1024):
    data = mx.symbol.Variable(name="data")
    # group 1
    conv1_1 = mx.symbol.Convolution(
        data=data, kernel=(3, 3), pad=(100, 100), num_filter=64, name="conv1_1",
        workspace=workspace_default)  # coord_map()
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    conv1_2 = mx.symbol.Convolution(
        data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_2",
        workspace=workspace_default)
    relu1_2 = mx.symbol.Activation(data=conv1_2, act_type="relu", name="relu1_2")
    pool1 = mx.symbol.Pooling(
        data=relu1_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool1")
    # group 2
    conv2_1 = mx.symbol.Convolution(
        data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_1",
        workspace=workspace_default)
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    conv2_2 = mx.symbol.Convolution(
        data=relu2_1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_2",
        workspace=workspace_default)
    relu2_2 = mx.symbol.Activation(data=conv2_2, act_type="relu", name="relu2_2")
    pool2 = mx.symbol.Pooling(
        data=relu2_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool2")
    # group 3
    conv3_1 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_1",
        workspace=workspace_default)
    relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.symbol.Convolution(
        data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_2",
        workspace=workspace_default)
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    conv3_3 = mx.symbol.Convolution(
        data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_3",
        workspace=workspace_default)
    relu3_3 = mx.symbol.Activation(data=conv3_3, act_type="relu", name="relu3_3")
    pool3 = mx.symbol.Pooling(
        data=relu3_3, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool3")
    # group 4
    conv4_1 = mx.symbol.Convolution(
        data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_1",
        workspace=workspace_default)
    relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.symbol.Convolution(
        data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_2",
        workspace=workspace_default)
    relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    conv4_3 = mx.symbol.Convolution(
        data=relu4_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_3",
        workspace=workspace_default)
    relu4_3 = mx.symbol.Activation(data=conv4_3, act_type="relu", name="relu4_3")
    pool4 = mx.symbol.Pooling(
        data=relu4_3, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool4")
    # group 5
    conv5_1 = mx.symbol.Convolution(
        data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_1",
        workspace=workspace_default)
    relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.symbol.Convolution(
        data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_2",
        workspace=workspace_default)
    relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="conv1_2")
    conv5_3 = mx.symbol.Convolution(
        data=relu5_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_3",
        workspace=workspace_default)
    relu5_3 = mx.symbol.Activation(data=conv5_3, act_type="relu", name="relu5_3")
    pool5 = mx.symbol.Pooling(
        data=relu5_3, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool5")
    # group 6
    fc6 = mx.symbol.Convolution(
        data=pool5, kernel=(7, 7), num_filter=4096, name="fc6",
        workspace=workspace_default)
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.symbol.Convolution(
        data=drop6, kernel=(1, 1), num_filter=4096, name="fc7",
        workspace=workspace_default)
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
    # group 8
    score = mx.symbol.Convolution(
        data=drop7, kernel=(1, 1), num_filter=21, name="score",
        workspace=workspace_default)
    # score 2X
    score2 = mx.symbol.Deconvolution(
        data=score, kernel=(4, 4), stride=(2, 2),
        num_filter=21, name="score2",
        workspace=workspace_default)  # 2X
    score_pool4 = mx.symbol.Convolution(
        data=pool4, kernel=(1, 1), num_filter=21, name="score_pool4",
        workspace=workspace_default)
    score_pool4c = mx.symbol.Crop(
        data=score_pool4, crop_like=score2,
        offset=coord_map_fcn16s()["score_pool4c"], name="score_pool4c") # TODO
    score_fused = mx.symbol.ElementWiseSum(*[score2, score_pool4c], name='score_fused')
    # score out
    bigscore = mx.symbol.Deconvolution(
        data=score_fused, kernel=(32, 32), stride=(16, 16),
        num_filter=21, name="bigscore",
        workspace=workspace_default) # 16X TODO
    upscore = mx.symbol.Crop(
        data=bigscore, crop_like=data,
        offset=coord_map_fcn16s()["upscore"], name="upscore")  # TODO
    softmax = mx.symbol.SoftmaxOutput(
         data=upscore, multi_output=True, ignore_label=255, name="softmax")
    return softmax

def get_fcn8s_symbol(numclass=21, workspace_default=1024):
    data = mx.symbol.Variable(name="data")
    # group 1
    conv1_1 = mx.symbol.Convolution(
        data=data, kernel=(3, 3), pad=(100, 100), num_filter=64, name="conv1_1",
        workspace=workspace_default)  # coord_map()
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    conv1_2 = mx.symbol.Convolution(
        data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_2",
        workspace=workspace_default)
    relu1_2 = mx.symbol.Activation(data=conv1_2, act_type="relu", name="relu1_2")
    pool1 = mx.symbol.Pooling(
        data=relu1_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool1")
    # group 2
    conv2_1 = mx.symbol.Convolution(
        data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_1",
        workspace=workspace_default)
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    conv2_2 = mx.symbol.Convolution(
        data=relu2_1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_2",
        workspace=workspace_default)
    relu2_2 = mx.symbol.Activation(data=conv2_2, act_type="relu", name="relu2_2")
    pool2 = mx.symbol.Pooling(
        data=relu2_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool2")
    # group 3
    conv3_1 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_1",
        workspace=workspace_default)
    relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.symbol.Convolution(
        data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_2",
        workspace=workspace_default)
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    conv3_3 = mx.symbol.Convolution(
        data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_3",
        workspace=workspace_default)
    relu3_3 = mx.symbol.Activation(data=conv3_3, act_type="relu", name="relu3_3")
    pool3 = mx.symbol.Pooling(
        data=relu3_3, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool3")
    # group 4
    conv4_1 = mx.symbol.Convolution(
        data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_1",
        workspace=workspace_default)
    relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.symbol.Convolution(
        data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_2",
        workspace=workspace_default)
    relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    conv4_3 = mx.symbol.Convolution(
        data=relu4_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_3",
        workspace=workspace_default)
    relu4_3 = mx.symbol.Activation(data=conv4_3, act_type="relu", name="relu4_3")
    pool4 = mx.symbol.Pooling(
        data=relu4_3, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool4")
    # group 5
    conv5_1 = mx.symbol.Convolution(
        data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_1",
        workspace=workspace_default)
    relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.symbol.Convolution(
        data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_2",
        workspace=workspace_default)
    relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="conv1_2")
    conv5_3 = mx.symbol.Convolution(
        data=relu5_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_3",
        workspace=workspace_default)
    relu5_3 = mx.symbol.Activation(data=conv5_3, act_type="relu", name="relu5_3")
    pool5 = mx.symbol.Pooling(
        data=relu5_3, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool5")
    # group 6
    fc6 = mx.symbol.Convolution(
        data=pool5, kernel=(7, 7), num_filter=4096, name="fc6",
        workspace=workspace_default)
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.symbol.Convolution(
        data=drop6, kernel=(1, 1), num_filter=4096, name="fc7",
        workspace=workspace_default)
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
    # group 8
    score = mx.symbol.Convolution(
        data=drop7, kernel=(1, 1), num_filter=21, name="score",
        workspace=workspace_default)
    # score 2X
    score2 = mx.symbol.Deconvolution(
        data=score, kernel=(4, 4), stride=(2, 2),
        num_filter=21, name="score2",
        workspace=workspace_default)  # 2X
    score_pool4 = mx.symbol.Convolution(
        data=pool4, kernel=(1, 1), num_filter=21, name="score_pool4",
        workspace=workspace_default)
    score_pool4c = mx.symbol.Crop(
        data=score_pool4, crop_like=score2,
        offset=coord_map_fcn8s()["score_pool4c"], name="score_pool4c") # TODO
    score_fused = mx.symbol.ElementWiseSum(*[score2, score_pool4c], name='score_fused')
    # score 4X
    score4 = mx.symbol.Deconvolution(
        data=score_fused, kernel=(4, 4), stride=(2, 2),
        num_filter=21, name="score4",
        workspace=workspace_default) # 2X
    score_pool3 = mx.symbol.Convolution(
        data=pool3, kernel=(1, 1), num_filter=21, name="score_pool3",
        workspace=workspace_default)
    score_pool3c = mx.symbol.Crop(
        data=score_pool3, crop_like=score4,
        offset=coord_map_fcn8s()["score_pool3c"], name="score_pool3c") # TODO
    score_final = mx.symbol.ElementWiseSum(*[score4, score_pool3c], name='score_final')
    # score out
    bigscore = mx.symbol.Deconvolution(
        data=score_final, kernel=(16, 16), stride=(8, 8),
        num_filter=21, name="bigscore",
        workspace=workspace_default) # 8X
    upscore = mx.symbol.Crop(
        data=bigscore, crop_like=data,
        offset=coord_map_fcn8s()["upscore"], name="upscore")  # TODO
    softmax = mx.symbol.SoftmaxOutput(
         data=upscore, multi_output=True, ignore_label=255, name="softmax")
    return softmax
