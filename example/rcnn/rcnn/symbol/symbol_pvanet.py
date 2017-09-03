import mxnet as mx
import proposal
import proposal_target
from rcnn.config import config

def get_pvanet_conv(data):
    conv1_1 = crelu(data=data, num_filter=16, kernel=(7, 7), pad=(3, 3), stride=(2, 2), name='conv1_1')  # (1056*640)x3/(528*320)x32
    pool1_1 = mx.sym.Pooling(data=conv1_1, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='pool1_1')  # (528*320)x32/(264*160)x32
    conv2_1 = res_crelu(data=pool1_1, middle_filter=[24, 24], num_filter=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1), bsr=False, proj=True, name='conv2_1')  # (264*160)x32/(264*160)x64
    conv2_2 = res_crelu(data=conv2_1, middle_filter=[24, 24], num_filter=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1), bsr=True, proj=False, name='conv2_2')  # (264*160)x64/(264*160)x64
    conv2_3 = res_crelu(data=conv2_2, middle_filter=[24, 24], num_filter=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1), bsr=True, proj=False, name='conv2_3')  # (264*160)x64/(264*160)x64
    scale3_1 = bn_scale_relu(data=conv2_3, name='bsr3_1', suffix='')
    conv3_1 = res_crelu(data=scale3_1, middle_filter=[48, 48], num_filter=128, kernel=(3, 3), stride=(2, 2), pad=(1, 1), bsr=False, proj=True, name='conv3_1')  # (264*160)x64/(132*80)x128
    conv3_2 = res_crelu(data=conv3_1, middle_filter=[48, 48], num_filter=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1), bsr=True, proj=False, name='conv3_2')  # (132*80)x64/(132*80)x128
    conv3_3 = res_crelu(data=conv3_2, middle_filter=[48, 48], num_filter=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1), bsr=True, proj=False, name='conv3_3')  # (132*80)x64/(132*80)x128
    conv3_4 = res_crelu(data=conv3_3, middle_filter=[48, 48], num_filter=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1), bsr=True, proj=False, name='conv3_4')  # (132*80)x64/(132*80)x128
    downscale = mx.sym.Pooling(data=conv3_4, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='downscole')  # (132*80)x64/(66*40)x128
    conv4_1 = inception(data=conv3_4, middle_filter=[64, [48, 128], [24, 48, 48], 128], num_filter=256, kernel=(3, 3), stride=(2, 2), proj=True, name='conv4_1', suffix='')  # (132*80)x128/(66*40)x256
    conv4_2 = inception(data=conv4_1, middle_filter=[64, [64, 128], [24, 48, 48]], num_filter=256, kernel=(3, 3), stride=(1, 1), proj=False, name='conv4_2', suffix='')  # (66*40)x256/(66*40)x256
    conv4_3 = inception(data=conv4_2, middle_filter=[64, [64, 128], [24, 48, 48]], num_filter=256, kernel=(3, 3), stride=(1, 1), proj=False, name='conv4_3', suffix='')  # (66*40)x256/(66*40)x256
    conv4_4 = inception(data=conv4_3, middle_filter=[64, [64, 128], [24, 48, 48]], num_filter=256, kernel=(3, 3), stride=(1, 1), proj=False, name='conv4_4', suffix='')  # (66*40)x256/(66*40)x256
    conv5_1 = inception(data=conv4_4, middle_filter=[64, [96, 192], [32, 64, 64], 128], num_filter=384, kernel=(3, 3), stride=(2, 2), proj=True, name='conv5_1', suffix='')  # (66*40)x256/(33*20)x384
    conv5_2 = inception(data=conv5_1, middle_filter=[64, [96, 192], [32, 64, 64]], num_filter=384, kernel=(3, 3), stride=(1, 1), proj=False, name='conv5_2', suffix='')  # (33*20)x384/(33*20)x384
    conv5_3 = inception(data=conv5_2, middle_filter=[64, [96, 192], [32, 64, 64]], num_filter=384, kernel=(3, 3), stride=(1, 1), proj=False, name='conv5_3', suffix='')  # (33*20)x384/(33*20)x384
    conv5_4 = inception_last(data=conv5_3, middle_filter=[64, [96, 192], [32, 64, 64]], num_filter=384, kernel=(3, 3), stride=(1, 1), proj=False, name='conv5_4', suffix='')  # (33*20)x384/(33*20)x384
    bsr = bn_scale_relu(data=conv5_4, name='bsr', suffix='last')
    upscale = mx.sym.Deconvolution(data=bsr, num_filter=384, kernel=(4, 4), stride=(2, 2), pad=(1, 1), name='deconv')  # (33*20)x384/(66*40)x384
    crop = mx.sym.Crop(upscale, downscale, name='crop')
    concat = mx.sym.concat(downscale, conv4_4, crop, name='concat')  # (66*40)x768/(66*40)x768
    convf = mx.sym.Convolution(data=concat, num_filter=512, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name='convf')  # (66*40)x768/(66*40)x512
    return convf

def crelu(data, num_filter, kernel, stride, pad, name=None, suffix=''):
    conv1=mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, name='%s%s_conv2d' %(name, suffix))
    bn=mx.symbol.BatchNorm(data=conv1, name='%s%s_batchnorm' %(name, suffix), fix_gamma=True)
    negative=mx.symbol.negative(data=bn, name='negative')
    concat=mx.symbol.concat(bn, negative)
    net=scale_and_shift(concat, num_filter)
    act=mx.symbol.Activation(data=net, act_type='relu', name='%s%s_relu' %(name, suffix))
    return act

# use linalg_gemm instead
def scale_and_shift(data, name=None, suffix=''):
    alpha = mx.symbol.Variable(name='%s%s_alpha' %(name, suffix), shape=(1), dtype='float32', init=mx.initializer.One())
    beta = mx.symbol.Variable(name='%s%s_beta' %(name, suffix), shape=(1), dtype='float32', init = mx.initializer.Zero())
    multi = mx.symbol.broadcast_mul(data, alpha)
    add = mx.symbol.broadcast_add(multi, beta)
    return add

def res_crelu(data, middle_filter, num_filter, kernel, stride, pad, bsr, proj, name=None, suffix=''):
    if bsr:
        input = bn_scale_relu(data=data, name=name, suffix='bsr')
    else:
        input = data
    if proj:
        shortcut = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride, name='%s%s_proj_con2d' %(name, suffix))
    else:
        shortcut = data
    conv1 = mx.sym.Convolution(data=input, num_filter=middle_filter[0], kernel=(1, 1), stride=stride, pad=(0, 0), name='%s%s_1_con2d' %(name, suffix))
    bsr = bn_scale_relu(data=conv1, name=name, suffix='group')
    conv2 = mx.sym.Convolution(data=bsr, num_filter=middle_filter[1], kernel=kernel, stride=(1, 1), pad=pad, name='%s%s_2_con2d' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv2, name='%s%s_batchnorm' %(name, suffix), fix_gamma=True)
    negative = mx.sym.negative(data=bn, name='%s%s_negative' %(name, suffix))
    concat = mx.sym.concat(bn, negative)
    scale = scale_and_shift(data=concat, name=name, suffix=suffix)
    relu = mx.sym.Activation(data=scale, act_type='relu', name='%s%s_relu' %(name, suffix))
    conv3 = mx.sym.Convolution(data=relu, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name='%s%s_3_con2d' %(name, suffix))
    act = conv3+shortcut
    return act

def bn_scale_relu(data, name=None, suffix=''):
    bn = mx.sym.BatchNorm(data=data, name='%s%s_batchnorm' % (name, suffix), fix_gamma=True)
    scale = scale_and_shift(data=bn, name=name, suffix=suffix)
    act = mx.sym.Activation(data=scale, act_type='relu', name='%s%s_relu' % (name, suffix))
    return act

def inception(data, middle_filter, num_filter, kernel, stride, proj, name, suffix):
    if proj:
        shortcut = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride, name='%s_proj' %(name))
    else:
        shortcut = data
    bsr = bn_scale_relu(data=data, name=name, suffix='bsr')
    conv_a = Conv(data=bsr, num_filter=middle_filter[0], kernel=(1, 1), stride=stride, pad=(0, 0), name=name, suffix='a')

    conv_b1 = Conv(data=bsr, num_filter=middle_filter[1][0], kernel=(1, 1), stride=stride, pad=(0, 0), name=name, suffix='b1')
    conv_b2 = Conv(data=conv_b1, num_filter=middle_filter[1][1], kernel=kernel, stride=(1, 1), pad=(1, 1), name=name, suffix='b2')

    conv_c1 = Conv(data=bsr, num_filter=middle_filter[2][0], kernel=(1, 1), stride=stride, pad=(0, 0), name=name, suffix='c1')
    conv_c2 = Conv(data=conv_c1, num_filter=middle_filter[2][1], kernel=kernel, stride=(1, 1), pad=(1, 1), name=name, suffix='c2')
    conv_c3 = Conv(data=conv_c2, num_filter=middle_filter[2][2], kernel=kernel, stride=(1, 1), pad=(1, 1), name=name, suffix='c3')

    if stride[1] > 1:
        pool_d = mx.sym.Pooling(data=bsr, kernel=kernel, stride=stride, pad=(1, 1), pool_type='max', name=name+'pool')
        conv_d = Conv(data=pool_d, num_filter=middle_filter[3], kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=name, suffix='proj_conv')
        conv_concat = mx.sym.concat(conv_a, conv_b2, conv_c3, conv_d)
    else:
        conv_concat = mx.sym.concat(conv_a, conv_b2, conv_c3)
    conv = mx.sym.Convolution(data=conv_concat, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name='%s_conv' %(name))
    output = conv+shortcut
    return output
def inception_last(data, middle_filter, num_filter, kernel, stride, proj, name, suffix):
    if proj:
        shortcut = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride, name='%s_proj' %(name))
    else:
        shortcut = data
    bsr = bn_scale_relu(data=data, name=name, suffix='bsr')
    conv_a = Conv(data=bsr, num_filter=middle_filter[0], kernel=(1, 1), stride=stride, pad=(0, 0), name=name, suffix='a')

    conv_b1 = Conv(data=bsr, num_filter=middle_filter[1][0], kernel=(1, 1), stride=stride, pad=(0, 0), name=name, suffix='b1')
    conv_b2 = Conv(data=conv_b1, num_filter=middle_filter[1][1], kernel=kernel, stride=(1, 1), pad=(1, 1), name=name, suffix='b2')

    conv_c1 = Conv(data=bsr, num_filter=middle_filter[2][0], kernel=(1, 1), stride=stride, pad=(0, 0), name=name, suffix='c1')
    conv_c2 = Conv(data=conv_c1, num_filter=middle_filter[2][1], kernel=kernel, stride=(1, 1), pad=(1, 1), name=name, suffix='c2')
    conv_c3 = Conv(data=conv_c2, num_filter=middle_filter[2][2], kernel=kernel, stride=(1, 1), pad=(1, 1), name=name, suffix='c3')

    if stride[1] > 1:
        pool_d = mx.sym.Pooling(data=bsr, kernel=kernel, stride=stride, pad=(1, 1), pool_type='max', name=name+'pool')
        conv_d = Conv(data=pool_d, num_filter=middle_filter[3], kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=name, suffix='proj_conv')
        conv_concat = mx.sym.concat(conv_a, conv_b2, conv_c3, conv_d)
    else:
        conv_concat = mx.sym.concat(conv_a, conv_b2, conv_c3)
    conv = mx.sym.Convolution(data=conv_concat, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name='%s_conv' %(name))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' % (name, suffix), fix_gamma=True)
    scale = scale_and_shift(data=bn, name=name, suffix='last')
    output = scale+shortcut
    return output

def Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=True)
    scale = scale_and_shift(bn, name='%s%s_scale_and_shift' %(name, suffix))
    act = mx.sym.Activation(data=scale, act_type='relu', name='%s%s_relu' %(name, suffix))
    return act

def get_pvanet_test(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
    """
    Faster R-CNN test with pvanet conv layers
    :param num_classes: used to determine output size
    :param num_anchors: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")

    # shared convolutional layers
    convf = get_pvanet_conv(data)
    slice = mx.sym.slice_axis(data=convf, axis=1, begin=0, end=128, name='slice')
    # RPN
    rpn_conv = mx.symbol.Convolution(
        data=slice, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
    rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
    rpn_cls_score = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
    rpn_bbox_pred = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")

    # ROI Proposal
    rpn_cls_score_reshape = mx.symbol.Reshape(
        data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
    rpn_cls_prob = mx.symbol.SoftmaxActivation(
        data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
    rpn_cls_prob_reshape = mx.symbol.Reshape(
        data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
    if config.TEST.CXX_PROPOSAL:
        rois = mx.contrib.symbol.Proposal(
            cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
            feature_stride=config.RPN_FEAT_STRIDE, scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TEST.RPN_POST_NMS_TOP_N,
            threshold=config.TEST.RPN_NMS_THRESH, rpn_min_size=config.TEST.RPN_MIN_SIZE)
    else:
        rois = mx.symbol.Custom(
            cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
            op_type='proposal', feat_stride=config.RPN_FEAT_STRIDE,
            scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TEST.RPN_POST_NMS_TOP_N,
            threshold=config.TEST.RPN_NMS_THRESH, rpn_min_size=config.TEST.RPN_MIN_SIZE)

    # Fast R-CNN
    pool5 = mx.symbol.ROIPooling(
        name='roi_pool5', data=convf, rois=rois, pooled_size=(7, 7), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
    # group 6
    flatten = mx.symbol.Flatten(data=pool5, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=drop7, num_hidden=num_classes)
    cls_prob = mx.symbol.softmax(name='cls_prob', data=cls_score)
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=drop7, num_hidden=num_classes * 4)

    # reshape output
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TEST.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
    bbox_pred = mx.symbol.Reshape(data=bbox_pred, shape=(config.TEST.BATCH_IMAGES, -1, 4 * num_classes), name='bbox_pred_reshape')

    # group output
    group = mx.symbol.Group([rois, cls_prob, bbox_pred])
    return group


def get_pvanet_train(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
    """
    Faster R-CNN end-to-end with pvanet conv layers
    :param num_classes: used to determine output size
    :param num_anchors: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")
    gt_boxes = mx.symbol.Variable(name="gt_boxes")
    rpn_label = mx.symbol.Variable(name='label')
    rpn_bbox_target = mx.symbol.Variable(name='bbox_target')
    rpn_bbox_weight = mx.symbol.Variable(name='bbox_weight')

    # shared convolutional layers
    relu5_3 = get_pvanet_conv(data)

    slice = mx.sym.slice_axis(data=relu5_3, axis=1, begin=0, end=128, name='slice')

    # RPN layers
    rpn_conv = mx.symbol.Convolution(
        data=slice, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
    rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
    rpn_cls_score = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
    rpn_bbox_pred = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")

    # prepare rpn data
    rpn_cls_score_reshape = mx.symbol.Reshape(
        data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")

    # classification
    rpn_cls_prob = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
                                           normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob")
    # bounding box regression
    rpn_bbox_loss_ = rpn_bbox_weight * mx.symbol.smooth_l1(name='rpn_bbox_loss_', scalar=3.0, data=(rpn_bbox_pred - rpn_bbox_target))
    rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_, grad_scale=1.0 / config.TRAIN.RPN_BATCH_SIZE)

    # ROI proposal
    rpn_cls_act = mx.symbol.SoftmaxActivation(
        data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_act")
    rpn_cls_act_reshape = mx.symbol.Reshape(
        data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape')
    if config.TRAIN.CXX_PROPOSAL:
        rois = mx.contrib.symbol.Proposal(
            cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
            feature_stride=config.RPN_FEAT_STRIDE, scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TRAIN.RPN_POST_NMS_TOP_N,
            threshold=config.TRAIN.RPN_NMS_THRESH, rpn_min_size=config.TRAIN.RPN_MIN_SIZE)
    else:
        rois = mx.symbol.Custom(
            cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
            op_type='proposal', feat_stride=config.RPN_FEAT_STRIDE,
            scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TRAIN.RPN_POST_NMS_TOP_N,
            threshold=config.TRAIN.RPN_NMS_THRESH, rpn_min_size=config.TRAIN.RPN_MIN_SIZE)

    # ROI proposal target
    gt_boxes_reshape = mx.symbol.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')
    group = mx.symbol.Custom(rois=rois, gt_boxes=gt_boxes_reshape, op_type='proposal_target',
                             num_classes=num_classes, batch_images=config.TRAIN.BATCH_IMAGES,
                             batch_rois=config.TRAIN.BATCH_ROIS, fg_fraction=config.TRAIN.FG_FRACTION)
    rois = group[0]
    label = group[1]
    bbox_target = group[2]
    bbox_weight = group[3]

    # Fast R-CNN
    pool5 = mx.symbol.ROIPooling(
        name='roi_pool5', data=relu5_3, rois=rois, pooled_size=(7, 7), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
    # group 6
    flatten = mx.symbol.Flatten(data=pool5, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=drop7, num_hidden=num_classes)
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='batch')
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=drop7, num_hidden=num_classes * 4)
    bbox_loss_ = bbox_weight * mx.symbol.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
    bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / config.TRAIN.BATCH_ROIS)

    # reshape output
    label = mx.symbol.Reshape(data=label, shape=(config.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TRAIN.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
    bbox_loss = mx.symbol.Reshape(data=bbox_loss, shape=(config.TRAIN.BATCH_IMAGES, -1, 4 * num_classes), name='bbox_loss_reshape')

    group = mx.symbol.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.symbol.BlockGrad(label)])
    return group
