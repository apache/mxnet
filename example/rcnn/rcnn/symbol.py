import mxnet as mx
import rpn.proposal, rpn.proposal_target
from config import config


def get_vgg_conv(data):
    """
    shared convolutional layers
    :param data: Symbol
    :return: Symbol
    """
    # group 1
    conv1_1 = mx.symbol.Convolution(
        data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_1")
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    conv1_2 = mx.symbol.Convolution(
        data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_2")
    relu1_2 = mx.symbol.Activation(data=conv1_2, act_type="relu", name="relu1_2")
    pool1 = mx.symbol.Pooling(
        data=relu1_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool1")
    # group 2
    conv2_1 = mx.symbol.Convolution(
        data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_1")
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    conv2_2 = mx.symbol.Convolution(
        data=relu2_1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_2")
    relu2_2 = mx.symbol.Activation(data=conv2_2, act_type="relu", name="relu2_2")
    pool2 = mx.symbol.Pooling(
        data=relu2_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool2")
    # group 3
    conv3_1 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_1")
    relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.symbol.Convolution(
        data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_2")
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    conv3_3 = mx.symbol.Convolution(
        data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_3")
    relu3_3 = mx.symbol.Activation(data=conv3_3, act_type="relu", name="relu3_3")
    pool3 = mx.symbol.Pooling(
        data=relu3_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool3")
    # group 4
    conv4_1 = mx.symbol.Convolution(
        data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_1")
    relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.symbol.Convolution(
        data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_2")
    relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    conv4_3 = mx.symbol.Convolution(
        data=relu4_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_3")
    relu4_3 = mx.symbol.Activation(data=conv4_3, act_type="relu", name="relu4_3")
    pool4 = mx.symbol.Pooling(
        data=relu4_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool4")
    # group 5
    conv5_1 = mx.symbol.Convolution(
        data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_1")
    relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.symbol.Convolution(
        data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_2")
    relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="relu5_2")
    conv5_3 = mx.symbol.Convolution(
        data=relu5_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_3")
    relu5_3 = mx.symbol.Activation(data=conv5_3, act_type="relu", name="relu5_3")

    return relu5_3


def get_vgg_rcnn(num_classes=21):
    """
    Fast R-CNN with VGG 16 conv layers
    :param num_classes: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    rois = mx.symbol.Variable(name='rois')
    label = mx.symbol.Variable(name='label')
    bbox_target = mx.symbol.Variable(name='bbox_target')
    bbox_inside_weight = mx.symbol.Variable(name='bbox_inside_weight')
    bbox_outside_weight = mx.symbol.Variable(name='bbox_outside_weight')

    # reshape input
    rois = mx.symbol.Reshape(data=rois, shape=(-1, 5), name='rois_reshape')
    label = mx.symbol.Reshape(data=label, shape=(-1, ), name='label_reshape')
    bbox_target = mx.symbol.Reshape(data=bbox_target, shape=(-1, 4 * num_classes), name='bbox_target_reshape')
    bbox_inside_weight = mx.symbol.Reshape(data=bbox_inside_weight, shape=(-1, 4 * num_classes), name='bbox_inside_weight_reshape')
    bbox_outside_weight = mx.symbol.Reshape(data=bbox_outside_weight, shape=(-1, 4 * num_classes), name='bbox_outside_weight_reshape')

    # shared convolutional layers
    relu5_3 = get_vgg_conv(data)

    # Fast R-CNN
    pool5 = mx.symbol.ROIPooling(
        name='roi_pool5', data=relu5_3, rois=rois, pooled_size=(7, 7), spatial_scale=0.0625)
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
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score, label=label)
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=drop7, num_hidden=num_classes * 4)
    bbox_loss_ = bbox_outside_weight * \
                 mx.symbol.smooth_l1(name='bbox_loss_', scalar=1.0,
                                     data=bbox_inside_weight * (bbox_pred - bbox_target))
    bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_)

    # reshape output
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TRAIN.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
    bbox_loss = mx.symbol.Reshape(data=bbox_loss, shape=(config.TRAIN.BATCH_IMAGES, -1, 4 * num_classes), name='bbox_loss_reshape')

    # group output
    group = mx.symbol.Group([cls_prob, bbox_loss])
    return group


def get_vgg_rcnn_test(num_classes=21):
    """
    Fast R-CNN Network with VGG
    :param num_classes: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    rois = mx.symbol.Variable(name='rois')

    # reshape rois
    rois = mx.symbol.Reshape(data=rois, shape=(-1, 5), name='rois_reshape')

    # shared convolutional layer
    relu5_3 = get_vgg_conv(data)
    
    # Fast R-CNN
    pool5 = mx.symbol.ROIPooling(
        name='roi_pool5', data=relu5_3, rois=rois, pooled_size=(7, 7), spatial_scale=0.0625)
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
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score)
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=drop7, num_hidden=num_classes * 4)

    # reshape output
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TEST.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
    bbox_pred = mx.symbol.Reshape(data=bbox_pred, shape=(config.TEST.BATCH_IMAGES, -1, 4 * num_classes), name='bbox_pred_reshape')

    # group output
    group = mx.symbol.Group([cls_prob, bbox_pred])
    return group


def get_vgg_rpn(num_classes=21, num_anchors=9):
    """
    Region Proposal Network with VGG
    :param num_classes: used to determine output size
    :param num_anchors: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    label = mx.symbol.Variable(name='label')
    bbox_target = mx.symbol.Variable(name='bbox_target')
    bbox_inside_weight = mx.symbol.Variable(name='bbox_inside_weight')
    bbox_outside_weight = mx.symbol.Variable(name='bbox_outside_weight')

    # shared convolutional layers
    relu5_3 = get_vgg_conv(data)

    # RPN
    rpn_conv = mx.symbol.Convolution(
        data=relu5_3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
    rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
    rpn_cls_score = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
    rpn_bbox_pred = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")

    # prepare rpn data
    rpn_cls_score_reshape = mx.symbol.Reshape(
        data=rpn_cls_score, shape=(0, 2, -1), name="rpn_cls_score_reshape")

    # classification
    cls_prob = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape, label=label, multi_output=True,
                                       normalization='valid', use_ignore=True, ignore_label=-1, name="cls_prob")
    # bounding box regression
    bbox_loss_ = bbox_outside_weight * \
                 mx.symbol.smooth_l1(name='bbox_loss_', scalar=3.0,
                                     data=bbox_inside_weight * (rpn_bbox_pred - bbox_target))
    bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_)
    # group output
    group = mx.symbol.Group([cls_prob, bbox_loss])
    return group


def get_vgg_rpn_test(num_classes=21, num_anchors=9):
    """
    Region Proposal Network with VGG
    :param num_classes: used to determine output size
    :param num_anchors: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")

    # shared convolutional layers
    relu5_3 = get_vgg_conv(data)

    # RPN
    rpn_conv = mx.symbol.Convolution(
        data=relu5_3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
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
    group = mx.symbol.Custom(
        cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
        op_type='proposal', feat_stride=16, scales=(8, 16, 32), ratios=(0.5, 1, 2), output_score=True)
    # rois = group[0]
    # score = group[1]

    return group


def get_vgg_test(num_classes=21, num_anchors=9):
    """
    Faster R-CNN test with VGG 16 conv layers
    :param num_classes: used to determine output size
    :param num_anchors: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")

    # shared convolutional layers
    relu5_3 = get_vgg_conv(data)

    # RPN
    rpn_conv = mx.symbol.Convolution(
        data=relu5_3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
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
    rois = mx.symbol.Custom(
        cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
        op_type='proposal', feat_stride=16, scales=(8, 16, 32), ratios=(0.5, 1, 2))

    # Fast R-CNN
    pool5 = mx.symbol.ROIPooling(
        name='roi_pool5', data=relu5_3, rois=rois, pooled_size=(7, 7), spatial_scale=0.0625)
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
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score)
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=drop7, num_hidden=num_classes * 4)

    # reshape output
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TEST.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
    bbox_pred = mx.symbol.Reshape(data=bbox_pred, shape=(config.TEST.BATCH_IMAGES, -1, 4 * num_classes), name='bbox_pred_reshape')

    # group output
    group = mx.symbol.Group([rois, cls_prob, bbox_pred])
    return group

def get_faster_rcnn(num_classes=21, num_anchors=9):
    """
    Faster R-CNN with VGG 16 conv layers
    :param num_classes: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")
    # label
    gt_boxes = mx.symbol.Variable(name="gt_boxes")
    label = mx.symbol.Variable(name='label')
    bbox_target = mx.symbol.Variable(name='bbox_target')
    bbox_inside_weight = mx.symbol.Variable(name='bbox_inside_weight')
    bbox_outside_weight = mx.symbol.Variable(name='bbox_outside_weight')

    gt_boxes = mx.symbol.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')

    relu5_3 = get_vgg_conv(data)

    ## RPN
    rpn_conv = mx.symbol.Convolution(
        data=relu5_3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
    rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
    rpn_cls_score = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
    rpn_bbox_pred = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")
    # prepare rpn data
    rpn_cls_score_reshape = mx.symbol.Reshape(
        data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")

    # classification
    rpn_cls_loss = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape, label=label, multi_output=True,
                                       normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_loss")
    # bounding box regression
    rpn_bbox_loss_ = bbox_outside_weight * \
                 mx.symbol.smooth_l1(name='rpn_bbox_loss_', scalar=3.0,
                                     data=bbox_inside_weight * (rpn_bbox_pred - bbox_target))
    rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_)

    rpn_cls_prob = mx.symbol.SoftmaxActivation(
        data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
    rpn_cls_prob_reshape = mx.symbol.Reshape(
        data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')

    rpn_roi = mx.symbol.Custom(
        cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rpn_rois',
        op_type='proposal', feat_stride=16, scales=(8, 16, 32), ratios=(0.5, 1, 2), is_train=True)  # TODO(be careful of cls_prob)
    rois = mx.symbol.Custom(
        rpn_roi=rpn_roi, gt_boxes=gt_boxes, name='rois', op_type='proposal_target',
        num_classes=num_classes, is_train=True)  #

    # R-CNN
    pool5 = mx.symbol.ROIPooling(
        name='roi_pool5', data=relu5_3, rois=rois[0], pooled_size=(7, 7), spatial_scale=0.0625)

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


    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score, label=rois[1], normalization='batch')
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=drop7, num_hidden=num_classes * 4)
    bbox_loss_ = rois[4] * \
                 mx.symbol.smooth_l1(name='bbox_loss_', scalar=1.0,
                                     data=rois[3] * (bbox_pred - rois[2]))
    bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / config.TRAIN.BATCH_SIZE)

    # reshape output
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TRAIN.IMS_PER_BATCH, -1, num_classes), name='cls_prob_reshape')
    bbox_pred = mx.symbol.Reshape(data=bbox_loss, shape=(config.TRAIN.IMS_PER_BATCH, -1, 4 * num_classes), name='bbox_pred_reshape')

    # group output
    group = mx.symbol.Group([rois[1], rpn_cls_loss, rpn_bbox_loss, cls_prob, bbox_pred])  # rois[1] is used for evaluation

    return group