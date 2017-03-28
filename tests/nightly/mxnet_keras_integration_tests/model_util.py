import os
from keras import backend as K
from keras.models import Model
from keras.layers import Input, merge
from keras.layers.core import Lambda

# Before running the integration tests, users are expected to set these
# environment variables.
IS_GPU = (os.environ['MXNET_KERAS_TEST_MACHINE'] == 'GPU')
GPU_NUM = int(os.environ['GPU_NUM']) if IS_GPU else 0
KERAS_BACKEND = os.environ['KERAS_BACKEND']

def slice_batch(x, n_gpus, part):
    sh = K.shape(x)
    L = sh[0] / n_gpus
    if part == n_gpus - 1:
        return x[part*L:]
    return x[part*L:(part+1)*L]

def prepare_gpu_model(model, **kwargs):
    gpu_list = []
    for i in range(GPU_NUM):
        gpu_list.append('gpu(%d)' % i)
    if KERAS_BACKEND == 'mxnet':
        kwargs['context'] = gpu_list
        model.compile(**kwargs)
    else:
        model.compile(**kwargs)

def prepare_cpu_model(model, **kwargs):
    model.compile(**kwargs)

def make_model(model, **kwargs):
    """
        Compiles the Keras Model object for given backend type and machine type.
        Use this function to write one Keras code and run it across different machine type.

        If environment variable - MXNET_KERAS_TEST_MACHINE is set to CPU, then Compiles
        Keras Model for running on CPU.

        If environment variable - MXNET_KERAS_TEST_MACHINE is set to GPU, then Compiles
        Keras Model running on GPU using number of GPUs equal to number specified in
        GPU_NUM environment variable.

        Currently supports only MXNet as Keras backend.
    """
    if(IS_GPU):
        prepare_gpu_model(model, **kwargs)
    else:
        prepare_cpu_model(model, **kwargs)
    return model
