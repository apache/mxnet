import mxnet as mx
from mxnet.model import save_checkpoint
from rcnn.config import config
import numpy as np

def load_checkpoint(prefix, epoch):
    """
    Load model checkpoint from file.
    :param prefix: Prefix of model name.
    :param epoch: Epoch number of model we would like to load.
    :return: (arg_params, aux_params)
    arg_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    aux_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.
    """
    save_dict = mx.nd.load('%s-%04d.params' % (prefix, epoch))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params, aux_params


def do_checkpoint(prefix):
    """Callback to checkpoint the model to prefix every epoch.

    Parameters
    ----------
    prefix : str
        The file prefix to checkpoint to

    Returns
    -------
    callback : function
        The callback function that can be passed as iter_end_callback to fit.
    """
    def _callback(iter_no, sym, arg, aux):
        if config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
            print "save model with mean/std"
            num_classes = len(arg['bbox_pred_bias'].asnumpy()) / 4
            means = np.tile(np.array(config.TRAIN.BBOX_MEANS), (1, num_classes))
            stds = np.tile(np.array(config.TRAIN.BBOX_STDS), (1, num_classes))
            arg['bbox_pred_weight'] = (arg['bbox_pred_weight'].T * mx.nd.array(stds)).T
            arg['bbox_pred_bias'] = arg['bbox_pred_bias'] * mx.nd.array(np.squeeze(stds)) + \
                                           mx.nd.array(np.squeeze(means))
        """The checkpoint function."""
        save_checkpoint(prefix, iter_no + 1, sym, arg, aux)
    return _callback


def convert_context(params, ctx):
    """
    :param params: dict of str to NDArray
    :param ctx: the context to convert to
    :return: dict of str of NDArray with context ctx
    """
    new_params = dict()
    for k, v in params.items():
        new_params[k] = v.as_in_context(ctx)
    return new_params


def load_param(prefix, epoch, convert=False, ctx=None):
    """
    wrapper for load checkpoint
    :param prefix: Prefix of model name.
    :param epoch: Epoch number of model we would like to load.
    :param convert: reference model should be converted to GPU NDArray first
    :param ctx: if convert then ctx must be designated.
    :return: (arg_params, aux_params)
    """
    arg_params, aux_params = load_checkpoint(prefix, epoch)
    num_classes = 1000
    if "bbox_pred_bias" in arg_params.keys():
        num_classes = len(arg_params['bbox_pred_bias'].asnumpy()) / 4
    if config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED and "bbox_pred_bias" in arg_params.keys():
        print "lode model with mean/std"
        means = np.tile(np.array(config.TRAIN.BBOX_MEANS_INV), (1, num_classes))
        stds = np.tile(np.array(config.TRAIN.BBOX_STDS_INV), (1, num_classes))
        arg_params['bbox_pred_weight'] = (arg_params['bbox_pred_weight'].T * mx.nd.array(stds)).T
        arg_params['bbox_pred_bias'] = (arg_params['bbox_pred_bias'] - mx.nd.array(np.squeeze(means))) * \
                                       mx.nd.array(np.squeeze(stds))

    if convert:
        if ctx is None:
            ctx = mx.cpu()
        arg_params = convert_context(arg_params, ctx)
        aux_params = convert_context(aux_params, ctx)
    return arg_params, aux_params, num_classes

