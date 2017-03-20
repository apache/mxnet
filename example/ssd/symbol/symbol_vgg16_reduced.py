import mxnet as mx
from common import conv_act_layer
from common import multibox_layer

def get_symbol_train(num_classes=20):
    """
    Single-shot multi-box detection with VGG 16 layers ConvNet
    This is a modified version, with fc6/fc7 layers replaced by conv layers
    And the network is slightly smaller than original VGG 16 network
    This is a training network with losses

    Parameters:
    ----------
    num_classes: int
        number of object classes not including background

    Returns:
    ----------
    mx.Symbol
    """
    data = mx.symbol.Variable(name="data")
    label = mx.symbol.Variable(name="label")

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
        data=relu3_3, pool_type="max", kernel=(2, 2), stride=(2, 2), \
        pooling_convention="full", name="pool3")
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
    pool5 = mx.symbol.Pooling(
        data=relu5_3, pool_type="max", kernel=(3, 3), stride=(1, 1),
        pad=(1,1), name="pool5")
    # group 6
    conv6 = mx.symbol.Convolution(
        data=pool5, kernel=(3, 3), pad=(6, 6), dilate=(6, 6),
        num_filter=1024, name="conv6")
    relu6 = mx.symbol.Activation(data=conv6, act_type="relu", name="relu6")
    # drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    conv7 = mx.symbol.Convolution(
        data=relu6, kernel=(1, 1), pad=(0, 0), num_filter=1024, name="conv7")
    relu7 = mx.symbol.Activation(data=conv7, act_type="relu", name="relu7")
    # drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")

    ### ssd extra layers ###
    conv8_1, relu8_1 = conv_act_layer(relu7, "8_1", 256, kernel=(1,1), pad=(0,0), \
        stride=(1,1), act_type="relu", use_batchnorm=False)
    conv8_2, relu8_2 = conv_act_layer(relu8_1, "8_2", 512, kernel=(3,3), pad=(1,1), \
        stride=(2,2), act_type="relu", use_batchnorm=False)
    conv9_1, relu9_1 = conv_act_layer(relu8_2, "9_1", 128, kernel=(1,1), pad=(0,0), \
        stride=(1,1), act_type="relu", use_batchnorm=False)
    conv9_2, relu9_2 = conv_act_layer(relu9_1, "9_2", 256, kernel=(3,3), pad=(1,1), \
        stride=(2,2), act_type="relu", use_batchnorm=False)
    conv10_1, relu10_1 = conv_act_layer(relu9_2, "10_1", 128, kernel=(1,1), pad=(0,0), \
        stride=(1,1), act_type="relu", use_batchnorm=False)
    conv10_2, relu10_2 = conv_act_layer(relu10_1, "10_2", 256, kernel=(3,3), pad=(1,1), \
        stride=(2,2), act_type="relu", use_batchnorm=False)
    # global Pooling
    pool10 = mx.symbol.Pooling(data=relu10_2, pool_type="avg",
        global_pool=True, kernel=(1,1), name='pool10')

    # specific parameters for VGG16 network
    from_layers = [relu4_3, relu7, relu8_2, relu9_2, relu10_2, pool10]
    sizes = [[.1], [.2,.276], [.38, .461], [.56, .644], [.74, .825], [.92, 1.01]]
    ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
        [1,2,.5,3,1./3], [1,2,.5,3,1./3]]
    normalizations = [20, -1, -1, -1, -1, -1]
    num_channels = [512]

    loc_preds, cls_preds, anchor_boxes = multibox_layer(from_layers, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_channels, clip=True, interm_layer=0)

    tmp = mx.symbol.MultiBoxTarget(
        *[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
        ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=0, \
        negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
        name="multibox_target")
    loc_target = tmp[0]
    loc_target_mask = tmp[1]
    cls_target = tmp[2]

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=3., multi_output=True, \
        normalization='valid', name="cls_prob")
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", \
        data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1., \
        normalization='valid', name="loc_loss")

    # monitoring training status
    cls_label = mx.symbol.MakeLoss(data=cls_target, grad_scale=0, name="cls_label")

    # group output
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label])
    return out

def get_symbol(num_classes=20, nms_thresh=0.5, force_suppress=True):
    """
    Single-shot multi-box detection with VGG 16 layers ConvNet
    This is a modified version, with fc6/fc7 layers replaced by conv layers
    And the network is slightly smaller than original VGG 16 network
    This is the detection network

    Parameters:
    ----------
    num_classes: int
        number of object classes not including background
    nms_thresh : float
        threshold of overlap for non-maximum suppression

    Returns:
    ----------
    mx.Symbol
    """
    net = get_symbol_train(num_classes)
    # print net.get_internals().list_outputs()
    cls_preds = net.get_internals()["multibox_cls_pred_output"]
    loc_preds = net.get_internals()["multibox_loc_pred_output"]
    anchor_boxes = net.get_internals()["multibox_anchors_output"]

    cls_prob = mx.symbol.SoftmaxActivation(data=cls_preds, mode='channel', \
        name='cls_prob')
    # group output
    # out = mx.symbol.Group([loc_preds, cls_preds, anchor_boxes])
    out = mx.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2))
    return out
