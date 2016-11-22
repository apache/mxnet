from __future__ import absolute_import, division, print_function

import os
import numpy
import json
import sys
import re
import scipy.signal
import logging
import ast
import inspect
import collections
import numbers
try:
    import cPickle as pickle
except:
    import pickle
from collections import namedtuple, OrderedDict
import time
import mxnet as mx
import mxnet.ndarray as nd


_ctx = mx.cpu()
_numpy_rng = numpy.random.RandomState(123456)


def get_default_ctx():
    return _ctx


def get_numpy_rng():
    return _numpy_rng


def get_saving_path(prefix="", epoch=None):
    sym_saving_path = os.path.join('%s-symbol.json' % prefix)
    if epoch is not None:
        param_saving_path = os.path.join('%s-%05d.params' % (prefix, epoch))
    else:
        param_saving_path = os.path.join('%s.params' % prefix)
    misc_saving_path = os.path.join('%s-misc.json' % prefix)
    return sym_saving_path, param_saving_path, misc_saving_path


def logging_config(name=None, level=logging.DEBUG, console_level=logging.DEBUG):
    if name is None:
        name = inspect.stack()[1][1].split('.')[0]
    folder = os.path.join(os.getcwd(), name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    logpath = os.path.join(folder, name + ".log")
    print("All Logs will be saved to %s"  %logpath)
    logging.root.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logfile = logging.FileHandler(logpath)
    logfile.setLevel(level)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)
    #TODO Update logging patterns in other files
    logconsole = logging.StreamHandler()
    logconsole.setLevel(console_level)
    logconsole.setFormatter(formatter)
    logging.root.addHandler(logconsole)
    return folder


def save_params(dir_path=os.curdir, epoch=None, name="", params=None, aux_states=None,
                ctx=mx.cpu()):
    prefix = os.path.join(dir_path, name)
    _, param_saving_path, _ = get_saving_path(prefix, epoch)
    if not os.path.isdir(dir_path) and not (dir_path == ""):
        os.makedirs(dir_path)
    save_dict = {('arg:%s' % k): v.copyto(ctx) for k, v in params.items()}
    save_dict.update({('aux:%s' % k): v.copyto(ctx) for k, v in aux_states.items()})
    nd.save(param_saving_path, save_dict)
    return param_saving_path


def save_misc(dir_path=os.curdir, epoch=None, name="", content=None):
    prefix = os.path.join(dir_path, name)
    _, _, misc_saving_path = get_saving_path(prefix, epoch)
    with open(misc_saving_path, 'w') as fp:
        json.dump(content, fp)
    return misc_saving_path


def quick_save_json(dir_path=os.curdir, file_name="", content=None):
    file_path = os.path.join(dir_path, file_name)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    with open(file_path, 'w') as fp:
        json.dump(content, fp)
    logging.info('Save json into %s' % file_path)


def safe_eval(expr):
    if type(expr) is str:
        return ast.literal_eval(expr)
    else:
        return expr


def norm_clipping(params_grad, threshold):
    assert isinstance(params_grad, dict)
    norm_val = numpy.sqrt(sum([nd.norm(grad).asnumpy()[0]**2 for grad in params_grad.values()]))
    # print('grad norm: %g' % norm_val)
    ratio = 1.0
    if norm_val > threshold:
        ratio = threshold / norm_val
        for grad in params_grad.values():
            grad *= ratio
    return norm_val


def sample_categorical(prob, rng):
    """Sample from independent categorical distributions

    Each batch is an independent categorical distribution.

    Parameters
    ----------
    prob : numpy.ndarray
      Probability of the categorical distribution. Shape --> (batch_num, category_num)
    rng : numpy.random.RandomState

    Returns
    -------
    ret : numpy.ndarray
      Sampling result. Shape --> (batch_num,)
    """
    ret = numpy.empty(prob.shape[0], dtype=numpy.float32)
    for ind in range(prob.shape[0]):
        ret[ind] = numpy.searchsorted(numpy.cumsum(prob[ind]), rng.rand()).clip(min=0.0,
                                                                                max=prob.shape[
                                                                                        1] - 0.5)
    return ret


def sample_normal(mean, var, rng):
    """Sample from independent normal distributions

    Each element is an independent normal distribution.

    Parameters
    ----------
    mean : numpy.ndarray
      Means of the normal distribution. Shape --> (batch_num, sample_dim)
    var : numpy.ndarray
      Variance of the normal distribution. Shape --> (batch_num, sample_dim)
    rng : numpy.random.RandomState

    Returns
    -------
    ret : numpy.ndarray
       The sampling result. Shape --> (batch_num, sample_dim)
    """
    ret = numpy.sqrt(var) * rng.randn(*mean.shape) + mean
    return ret


def sample_mog(prob, mean, var, rng):
    """Sample from independent mixture of gaussian (MoG) distributions

    Each batch is an independent MoG distribution.

    Parameters
    ----------
    prob : numpy.ndarray
      mixture probability of each gaussian. Shape --> (batch_num, center_num)
    mean : numpy.ndarray
      mean of each gaussian. Shape --> (batch_num, center_num, sample_dim)
    var : numpy.ndarray
      variance of each gaussian. Shape --> (batch_num, center_num, sample_dim)
    rng : numpy.random.RandomState

    Returns
    -------
    ret : numpy.ndarray
      sampling result. Shape --> (batch_num, sample_dim)
    """
    gaussian_inds = sample_categorical(prob, rng).astype(numpy.int32)
    mean = mean[numpy.arange(mean.shape[0]), gaussian_inds, :]
    var = var[numpy.arange(mean.shape[0]), gaussian_inds, :]
    ret = sample_normal(mean=mean, var=var, rng=rng)
    return ret


def npy_softmax(x, axis=1):
    e_x = numpy.exp(x - numpy.max(x, axis=axis, keepdims=True))
    out = e_x / e_x.sum(axis=axis, keepdims=True)
    return out


def npy_sigmoid(x):
    return 1/(1 + numpy.exp(-x))


def npy_onehot(x, num):
    ret = numpy.zeros(shape=(x.size, num))
    ret[numpy.arange(x.size), x.ravel()] = 1
    ret = ret.reshape(x.shape + (num,))
    return ret

def npy_binary_entropy(prediction, target):
    assert prediction.shape == target.shape
    return - (numpy.log(prediction + 1E-9) * target +
              numpy.log(1 - prediction + 1E-9) * (1 - target)).sum()


def block_all(sym_list):
    return [mx.symbol.BlockGrad(sym) for sym in sym_list]


def load_params(dir_path="", epoch=None, name=""):
    prefix = os.path.join(dir_path, name)
    _, param_loading_path, _ = get_saving_path(prefix, epoch)
    while not os.path.isfile(param_loading_path):
        logging.info("in load_param, %s Not Found!" % param_loading_path)
        time.sleep(60)
    save_dict = nd.load(param_loading_path)
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params, aux_params, param_loading_path


def load_misc(dir_path="", epoch=None, name=""):
    prefix = os.path.join(dir_path, name)
    _, _, misc_saving_path = get_saving_path(prefix, epoch)
    with open(misc_saving_path, 'r') as fp:
        misc = json.load(fp)
    return misc


def load_npz(path):
    with numpy.load(path) as data:
        ret = {k: data[k] for k in data.keys()}
        return ret


def discount_cumsum(x, discount):
    # See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering
    # Here, we have y[t] - discount*y[t+1] = x[t]
    # or rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]
    return scipy.signal.lfilter([1], [1, -discount], x[::-1], axis=0)[::-1]


def discount_return(x, discount):
    return numpy.sum(x * (discount ** numpy.arange(len(x))))


def update_on_kvstore(kv, params, params_grad):
    for ind, k in enumerate(params.keys()):
        kv.push(ind, params_grad[k], priority=-ind)
        kv.pull(ind, params[k], priority=-ind)


def parse_ctx(ctx_args):
    ctx = re.findall('([a-z]+)(\d*)', ctx_args)
    ctx = [(device, int(num)) if len(num) > 0 else (device, 0) for device, num in ctx]
    return ctx


def get_npy_list(ndarray_list):
    """Get a numpy-array list from a ndarray list
    Parameters
    ----------
    ndarray_list : list of NDArray

    Returns
    -------
    ret : list of numpy.ndarray
    """
    ret = [v.asnumpy() for v in ndarray_list]
    return ret


def get_sym_list(syms, default_names=None, default_shapes=None):
    if syms is None and default_names is not None:
        if default_shapes is not None:
            return [mx.sym.Variable(name=name, shape=shape) for (name, shape)
                    in zip(default_names, default_shapes)]
        else:
            return [mx.sym.Variable(name=name) for name in default_names]
    assert isinstance(syms, (list, tuple, mx.symbol.Symbol))
    if isinstance(syms, (list, tuple)):
        if default_names is not None and len(syms) != len(default_names):
            raise ValueError("Size of symbols do not match expectation. Received %d, Expected %d. "
                             "syms=%s, names=%s" %(len(syms), len(default_names),
                                                   str(list(sym.name for sym in syms)),
                                                   str(default_names)))
        return list(syms)
    else:
        if default_names is not None and len(default_names) != 1:
            raise ValueError("Size of symbols do not match expectation. Received 1, Expected %d. "
                             "syms=%s, names=%s"
                             % (len(default_names), str([syms.name]), str(default_names)))
        return [syms]


def get_numeric_list(values, typ, expected_len=None):
    if isinstance(values, numbers.Number):
        if expected_len is not None:
            return [typ(values)] * expected_len
        else:
            return [typ(values)]
    elif isinstance(values, (list, tuple)):
        if expected_len is not None:
            assert len(values) == expected_len
        try:
            ret = [typ(value) for value in values]
            return ret
        except(ValueError):
            print("Need iterable with numeric elements, received: %s" %str(values))
            sys.exit(1)
    else:
        raise ValueError("Unaccepted value type, values=%s" %str(values))


def get_int_list(values, expected_len=None):
    return get_numeric_list(values, numpy.int32, expected_len)


def get_float_list(values, expected_len=None):
    return get_numeric_list(values, numpy.float32, expected_len)


def get_bucket_key(bucket_kwargs):
    assert isinstance(bucket_kwargs, dict)
    return tuple(bucket_kwargs.items())
