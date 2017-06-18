"""
SSD module based on Inception V3
Input size 331x331
"""
import mxnet as mx
from inceptionv3 import get_symbol as get_body
from common import conv_act_layer, multibox_layer

def get_symbol_train(num_classes=20, nms_thresh=0.5, force_suppress=False,
                     nms_topk=400, **kwargs):
    label = mx.sym.Variable('label')
    body = get_body(num_classes)
    mixed_2 = body.get_internals()['ch_concat_mixed_2_chconcat_output']  # 288x39x39
    mixed_7 = body.get_internals()['ch_concat_mixed_7_chconcat_output']  # 768x19x19
    mixed_10 = body.get_internals()['ch_concat_mixed_10_chconcat_output']  # 2048x9x9

    # adjust layers
    _, in_1 = conv_act_layer(mixed_2, "in_1", 512, kernel=(3, 3), pad=(1, 1), \
        stride=(1,1), act_type="relu", use_batchnorm=False)
    _, in_2 = conv_act_layer(mixed_7, "in_2", 512, kernel=(3, 3), pad=(1, 1), \
        stride=(1,1), act_type="relu", use_batchnorm=False)
    _, in_3 = conv_act_layer(mixed_10, "in_3", 512, kernel=(3, 3), pad=(1, 1), \
        stride=(1,1), act_type="relu", use_batchnorm=False)

    # extend network
    _, in_4_1 = conv_act_layer(in_3, "in_4_1", 256, kernel=(1,1), pad=(0,0), \
        stride=(1,1), act_type="relu", use_batchnorm=False)
    _, in_4_2 = conv_act_layer(in_4_1, "in_4_2", 512, kernel=(3,3), pad=(1,1), \
        stride=(2,2), act_type="relu", use_batchnorm=False)
    _, in_5_1 = conv_act_layer(in_4_2, "in_5_1", 128, kernel=(1,1), pad=(0,0), \
        stride=(1,1), act_type="relu", use_batchnorm=False)
    _, in_5_2 = conv_act_layer(in_5_1, "in_5_2", 256, kernel=(3,3), pad=(1,1), \
        stride=(2,2), act_type="relu", use_batchnorm=False)
    _, in_6_1 = conv_act_layer(in_5_2, "in_6_1", 128, kernel=(1,1), pad=(0,0), \
        stride=(1,1), act_type="relu", use_batchnorm=False)
    _, in_6_2 = conv_act_layer(in_6_1, "in_6_2", 256, kernel=(3,3), pad=(0,0), \
        stride=(1,1), act_type="relu", use_batchnorm=False)

    # specific parameters for VGG16 network
    from_layers = [in_1, in_2, in_3, in_4_2, in_5_2, in_6_2]
    sizes = [[.1, .141], [.2,.272], [.37, .447], [.54, .619], [.71, .79], [.88, .961]]
    ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
        [1,2,.5], [1,2,.5]]
    normalizations = [-1, -1, -1, -1, -1, -1]
    steps = [ x / 299.0 for x in [8, 16, 32, 64, 100, 300]]

    loc_preds, cls_preds, anchor_boxes = multibox_layer(from_layers, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=[], clip=False, interm_layer=0, steps=steps)

    tmp = mx.contrib.symbol.MultiBoxTarget(
        *[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
        ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=0, \
        negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
        name="multibox_target")
    loc_target = tmp[0]
    loc_target_mask = tmp[1]
    cls_target = tmp[2]

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=1., multi_output=True, \
        normalization='valid', name="cls_prob")
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", \
        data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1., \
        normalization='valid', name="loc_loss")

    # monitoring training status
    cls_label = mx.symbol.MakeLoss(data=cls_target, grad_scale=0, name="cls_label")
    det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    det = mx.symbol.MakeLoss(data=det, grad_scale=0, name="det_out")

    # group output
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label, det])
    return out

def get_symbol(num_classes=20, nms_thresh=0.5, force_suppress=False, nms_topk=400):
    """
    Single-shot multi-box detection with inception-v3 layers ConvNet
    This is the detection network

    Parameters:
    ----------
    num_classes: int
        number of object classes not including background
    nms_thresh : float
        threshold of overlap for non-maximum suppression
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections

    Returns:
    ----------
    mx.Symbol
    """
    net = get_symbol_train(num_classes)
    cls_preds = net.get_internals()["multibox_cls_pred_output"]
    loc_preds = net.get_internals()["multibox_loc_pred_output"]
    anchor_boxes = net.get_internals()["multibox_anchors_output"]

    cls_prob = mx.symbol.SoftmaxActivation(data=cls_preds, mode='channel', \
        name='cls_prob')
    out = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    return out
