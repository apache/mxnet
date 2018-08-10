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
import gluoncv
import mxnet as mx

from tests.python.unittest.common import assertRaises


def test_training_without_trt():
    run_resnet(is_train=True, use_tensorrt=False)


def test_inference_without_trt():
    run_resnet(is_train=False, use_tensorrt=False)


def test_training_with_trt():
    assertRaises(RuntimeError, run_resnet, is_train=True, use_tensorrt=True)


def test_inference_with_trt():
    run_resnet(is_train=False, use_tensorrt=True)


def run_resnet(is_train, use_tensorrt):
    original_trt_value = mx.contrib.tensorrt.get_use_tensorrt()
    try:
        mx.contrib.tensorrt.set_use_tensorrt(use_tensorrt)
        ctx = mx.gpu(0)
        batch_size = 1
        h = 32
        w = 32
        model_name = 'cifar_resnet20_v1'
        resnet = gluoncv.model_zoo.get_model(model_name, pretrained=True)
        data = mx.sym.var('data')
        out = resnet(data)
        softmax = mx.sym.SoftmaxOutput(out, name='softmax')
        if is_train:
            grad_req = 'write'
        else:
            grad_req = 'null'
        if use_tensorrt:
            all_params = dict([(k, v.data()) for k, v in resnet.collect_params().items()])
            mx.contrib.tensorrt.tensorrt_bind(softmax, ctx=ctx, all_params=all_params,
                                              data=(batch_size, 3, h, w), softmax_label=(batch_size,),
                                              force_rebind=True, grad_req=grad_req)
        else:
            softmax.simple_bind(ctx=ctx, data=(batch_size, 3, h, w), softmax_label=(batch_size,),
                                force_rebind=True, grad_req=grad_req)
    finally:
        mx.contrib.tensorrt.set_use_tensorrt(original_trt_value)


if __name__ == '__main__':
    import nose
    nose.runmodule()
