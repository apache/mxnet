# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import sys
import ctypes
import mxnet as mx
from mxnet.base import SymbolHandle, check_call, _LIB, mx_uint, c_str_array, c_str, mx_real_t
from mxnet.symbol import Symbol
import numpy as np
from mxnet.test_utils import assert_almost_equal
from mxnet.numpy_extension import get_cuda_compute_capability
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import nd
from mxnet.gluon.model_zoo import vision

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from common import setup_module, with_seed, teardown

####################################
######### FP32/FP16 tests ##########
####################################

# Using RN50 to test TRT integration
def get_model(batch_shape, gluon_model=False):
    if not gluon_model:
        path = 'resnet50_v2'
        if not os.path.exists(path):
            model = vision.resnet50_v2(pretrained=True)
            model.hybridize()
            model.forward(mx.nd.zeros(batch_shape))
            model.export(path)
        sym, arg_params, aux_params = mx.model.load_checkpoint(path, 0)
        return sym, arg_params, aux_params
    else:
        model = vision.resnet50_v2(pretrained=True)
        model.hybridize()
        return model


def get_default_executor(input_data):
     sym, arg_params, aux_params = get_model(batch_shape=input_data.shape)
     executor = sym.simple_bind(ctx=mx.gpu(0), data=input_data.shape, grad_req='null', force_rebind=True)
     executor.copy_params_from(arg_params, aux_params)
     return executor    

def get_baseline(input_data):
    executor = get_default_executor(input_data) 
    output = executor.forward(is_train=False, data=input_data)
    return output


def check_tensorrt_symbol(baseline, input_data, fp16_mode, rtol=None, atol=None):
    sym, arg_params, aux_params = get_model(batch_shape=input_data.shape)
    trt_sym = sym.optimize_for('TensorRT', args=arg_params, aux=aux_params, ctx=mx.gpu(0),
                               precision='fp16' if fp16_mode else 'fp32')
    
    executor = trt_sym.simple_bind(ctx=mx.gpu(), data=input_data.shape,
                                   grad_req='null', force_rebind=True)

    output = executor.forward(is_train=False, data=input_data)
    assert_almost_equal(output[0], baseline[0], rtol=rtol, atol=atol)

@with_seed()
def test_tensorrt_symbol():
    batch_shape = (32, 3, 224, 224)
    input_data = mx.nd.random.uniform(shape=(batch_shape), ctx=mx.gpu(0))
    baseline = get_baseline(input_data)
    print("Testing resnet50 with TensorRT backend numerical accuracy...")
    print("FP32")
    check_tensorrt_symbol(baseline, input_data, fp16_mode=False)
    print("FP16")
    check_tensorrt_symbol(baseline, input_data, fp16_mode=True, rtol=1e-2, atol=1e-1)

##############################
######### INT8 tests ##########
##############################

def get_dali_iter():
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
    from nvidia.dali.plugin.mxnet import DALIGluonIterator as GluonIterator

    val_rec='val_256_q90.rec'
    val_idx='val_256_q90.idx'

    class RecordIOPipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id):
            super(RecordIOPipeline, self).__init__(batch_size,
                                            num_threads,
                                            device_id)
            self.input = ops.MXNetReader(path = val_rec, index_path = val_idx)

            self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
            self.uniform = ops.Uniform(range = (0.0, 1.0))
            self.res = ops.Resize(device="gpu",
                                resize_shorter=224,
                                interp_type=types.INTERP_TRIANGULAR)
            self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                                dtype=types.FLOAT,
                                                output_layout=types.NCHW,
                                                crop=(224, 224),
                                                mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                                std=[0.229 * 255,0.224 * 255,0.225 * 255])
            self.iter = 0


        def define_graph(self):
            inputs, labels = self.input(name="Reader")
            images = self.decode(inputs)
            res = self.res(images)
            output = self.cmnp(res)
            return (output, labels)

        def iter_setup(self):
            pass
    pipe = RecordIOPipeline(1, 4, 0)
    pipe.build()
    return GluonIterator(pipe, pipe.epoch_size("Reader"), fill_last_batch=True)

def get_top1(logits):
    prob = logits.squeeze()
    sorted_prob = mx.nd.argsort(prob)
    return sorted_prob[-1]


def test_tensorrt_symbol_int8():
    ctx = mx.gpu(0)

    # INT8 engine output are not lossless, so we don't expect numerical uniformity,
    # but we have to compare the TOP1 metric

    batch_shape=(1,3,224,224)
    sym, arg_params, aux_params = get_model(batch_shape=batch_shape)
    calibration_iters = 700
    trt_sym = sym.optimize_for('TensorRT', args=arg_params, aux=aux_params, ctx=ctx,
                               precision='int8',
                               calibration_iters=calibration_iters)
    
    executor = trt_sym.simple_bind(ctx=ctx, data=batch_shape,
                               grad_req='null', force_rebind=True)
    
    dali_val_iter = get_dali_iter()

    # Calibration phase
    for i,it in enumerate(dali_val_iter):
        data, _ = it[0] # gpu 0
        if i == calibration_iters:
            break
        y_gen = executor.forward(is_train=False, data=data)

        y_gen[0].wait_to_read()

    executor_fp32 = get_default_executor(data) 

    top1_accuracy_similarity = 0
    top1_accuracy_default = 0
    top1_accuracy_int8 = 0

    iters = 1000
    for i,it in enumerate(dali_val_iter):
        if i == iters:
            break
        input_data, label = it[0] # gpu 0

        output = executor.forward(is_train=False, data=input_data)
        baseline = executor_fp32.forward(is_train=False, data=input_data)

        top1_output = get_top1(output[0])
        top1_baseline = get_top1(baseline[0])

        label = label.squeeze().as_in_context(top1_baseline.context)
        top1_accuracy_similarity += (top1_output == top1_baseline).asscalar()
        top1_accuracy_default += (top1_baseline == label).asscalar()
        top1_accuracy_int8 += (top1_output == label).asscalar()


    top1_accuracy_similarity = (top1_accuracy_similarity / iters)
    
    top1_accuracy_default = (top1_accuracy_default / iters)
    top1_accuracy_int8 = (top1_accuracy_int8 / iters)
    delta_top1_accuracy = abs(top1_accuracy_default - top1_accuracy_int8)

    # These values are provided by the TensorRT team, and reflects the expected accuracy loss when using
    expected_max_delta_top1_accuracy = 0.02 # this is the accuracy gap measure with TRT7, TRT7.1 can be at 0.01
    expected_min_similarity = 0.92
    print('Delta between FP32 and INT8 TOP1 accuracies: {}'.format(delta_top1_accuracy))
    print('TOP1 similarity accuracy (when top1_fp32 == top1_int8): {}'.format(expected_min_similarity))
    assert(delta_top1_accuracy < expected_max_delta_top1_accuracy)
    assert(top1_accuracy_similarity > expected_min_similarity)

if __name__ == '__main__':
    import nose
    nose.runmodule()
