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

import gc
import gluoncv
import mxnet as mx
import numpy as np

from mxnet import gluon
from time import time

from mxnet.gluon.data.vision import transforms


def get_classif_model(model_name, use_tensorrt, ctx=mx.gpu(0), batch_size=128):
    mx.contrib.tensorrt.set_use_fp16(False)
    h, w = 32, 32
    net = gluoncv.model_zoo.get_model(model_name, pretrained=True)
    net.hybridize()
    net.forward(mx.nd.zeros((batch_size, 3, h, w)))
    net.export(model_name)
    _sym, arg_params, aux_params = mx.model.load_checkpoint(model_name, 0)
    if use_tensorrt:
        sym = _sym.get_backend_symbol('TensorRT') 
        mx.contrib.tensorrt.init_tensorrt_params(sym, arg_params, aux_params)
    else:
        sym = _sym
    executor = sym.simple_bind(ctx=ctx, data=(batch_size, 3, h, w),
                               softmax_label=(batch_size,),
			       grad_req='null', force_rebind=True)
    executor.copy_params_from(arg_params, aux_params)
    return executor


def cifar10_infer(model_name, use_tensorrt, num_workers, ctx=mx.gpu(0), batch_size=128):
    executor = get_classif_model(model_name, use_tensorrt, ctx, batch_size)

    num_ex = 10000
    all_preds = np.zeros([num_ex, 10])

    all_label_test = np.zeros(num_ex)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    data_loader = lambda: gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=False).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers=num_workers)

    val_data = data_loader()

    for idx, (data, label) in enumerate(val_data):
        # Skip last batch if it's undersized.
        if data.shape[0] < batch_size:
            continue
        offset = idx * batch_size
        all_label_test[offset:offset + batch_size] = label.asnumpy()

        # warm-up, but don't use result
        executor.forward(is_train=False, data=data)
        executor.outputs[0].wait_to_read()

    gc.collect()
    val_data = data_loader()
    example_ct = 0
    start = time()

    # if use_tensorrt:
    for idx, (data, label) in enumerate(val_data):
        # Skip last batch if it's undersized.
        if data.shape[0] < batch_size:
            continue
        executor.forward(is_train=False, data=data)
        preds = executor.outputs[0].asnumpy()
        offset = idx * batch_size
        all_preds[offset:offset + batch_size, :] = preds[:batch_size]
        example_ct += batch_size

    all_preds = np.argmax(all_preds, axis=1)
    matches = (all_preds[:example_ct] == all_label_test[:example_ct]).sum()
    duration = time() - start

    return duration, 100.0 * matches / example_ct


def run_experiment_for(model_name, batch_size, num_workers):
    print("\n===========================================")
    print("Model: %s" % model_name)
    print("===========================================")
    print("*** Running inference using pure MXNet ***\n")
    mx_duration, mx_pct = cifar10_infer(model_name=model_name, batch_size=batch_size,
                                        num_workers=num_workers, use_tensorrt=False)
    print("\nMXNet: time elapsed: %.3fs, accuracy: %.2f%%" % (mx_duration, mx_pct))
    print("\n*** Running inference using MXNet + TensorRT ***\n")
    trt_duration, trt_pct = cifar10_infer(model_name=model_name, batch_size=batch_size,
                                          num_workers=num_workers, use_tensorrt=True)
    print("TensorRT: time elapsed: %.3fs, accuracy: %.2f%%" % (trt_duration, trt_pct))
    speedup = mx_duration / trt_duration
    print("TensorRT speed-up (not counting compilation): %.2fx" % speedup)

    acc_diff = abs(mx_pct - trt_pct)
    print("Absolute accuracy difference: %f" % acc_diff)
    return speedup, acc_diff


def test_tensorrt_on_cifar_resnets(batch_size=32, tolerance=0.1, num_workers=1):
    original_use_fp16 = mx.contrib.tensorrt.get_use_fp16()
    try:
        models = [
            'cifar_resnet20_v1',
            'cifar_resnet56_v1',
            'cifar_resnet110_v1',
            'cifar_resnet20_v2',
            'cifar_resnet56_v2',
            'cifar_resnet110_v2',
            'cifar_wideresnet16_10',
            'cifar_wideresnet28_10',
            'cifar_wideresnet40_8',
            'cifar_resnext29_16x64d'
        ]

        num_models = len(models)

        speedups = np.zeros(num_models, dtype=np.float32)
        acc_diffs = np.zeros(num_models, dtype=np.float32)

        test_start = time()

        for idx, model in enumerate(models):
            speedup, acc_diff = run_experiment_for(model, batch_size, num_workers)
            speedups[idx] = speedup
            acc_diffs[idx] = acc_diff
            assert acc_diff < tolerance, "Accuracy difference between MXNet and TensorRT > %.2f%% for model %s" % (
                tolerance, model)

        print("Perf and correctness checks run on the following models:")
        print(models)
        mean_speedup = np.mean(speedups)
        std_speedup = np.std(speedups)
        print("\nSpeedups:")
        print(speedups)
        print("Speedup range: [%.2f, %.2f]" % (np.min(speedups), np.max(speedups)))
        print("Mean speedup: %.2f" % mean_speedup)
        print("St. dev. of speedups: %.2f" % std_speedup)
        print("\nAcc. differences: %s" % str(acc_diffs))

        test_duration = time() - test_start

        print("Test duration: %.2f seconds" % test_duration)
    finally:
        mx.contrib.tensorrt.set_use_fp16(original_use_fp16)


if __name__ == '__main__':
    import nose

    nose.runmodule()
