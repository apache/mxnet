import mxnet as mx


def load_model(model_name, epoch_num, data_shapes, label_shapes, label_names, gpus=''):
    """

    :param model_name:
    :param epoch_num:
    :param data_shapes:
    :param label_shapes:
    :param label_names:
    :param gpus:
    :return:
    """
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_name, epoch_num)
    if gpus == '':
        devices = mx.cpu()
    else:
        devices = [mx.gpu(int(i)) for i in gpus.split(',')]
    mod = mx.mod.Module(
        symbol=sym,
        context=devices,
        label_names=label_names
    )
    mod.bind(
        for_training=False,
        data_shapes=data_shapes,
        label_shapes=label_shapes
    )
    mod.set_params(
        arg_params=arg_params,
        aux_params=aux_params,
        allow_missing=True
    )
    return mod


