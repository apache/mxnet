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

# pylint: skip-file

import logging
import numpy as np
import mxnet as mx

class Softmax(mx.operator.CustomOp):
    def __init__(self):
        super(Softmax,self).__init__()
        # Each thread processes a row (a sample in the batch).
        fwd_src = r"""
            template<class DType>
            __global__ void fwd(const DType* x, DType* y, const int row_size, const int req) {
                const int offset = row_size * threadIdx.x;
                DType max = x[offset];
                for(int i = 1; i < row_size; ++i) {
                    if(max < x[offset + i]) {
                        max = x[offset + i];
                    }
                }
                DType sum = 0;
                for(int i = 0; i < row_size; ++i) {
                    sum += exp(x[offset + i] - max);
                }
                switch(req) {
                    case 1:
                        for(int i = 0; i < row_size; ++i) {
                            y[offset + i] = exp(x[offset + i] - max) / sum;
                        }
                        break;
                    case 2:
                        for(int i = 0; i < row_size; ++i) {
                            y[offset + i] += exp(x[offset + i] - max) / sum;
                        }
                        break;
                }
            }
        """

        # Each block processes a row and each thread in a block calculate an element of `dx`.
        bwd_src = r"""
            template<class DType>
            __global__ void bwd(const DType* l, const DType* y, DType* dx, const int req) {
                const int z = static_cast<int>(l[blockIdx.x]);
                const int i = threadIdx.x + blockDim.x * blockIdx.x;
                if(req == 1) {
                    dx[i]  = threadIdx.x == z ? y[i] - 1 : y[i];
                } else {
                    dx[i] += threadIdx.x == z ? y[i] - 1 : y[i];
                }
            }
        """
        fwd_kernel_mod = mx.rtc.CudaModule(fwd_src, exports=["fwd<float>", "fwd<double>"])
        bwd_kernel_mod = mx.rtc.CudaModule(bwd_src, exports=["bwd<float>", "bwd<double>"])

        fwd_kernel_float_signature = "const float*, const float*, const int, const int"
        self.fwd_float_kernel = fwd_kernel_mod.get_kernel("fwd<float>", fwd_kernel_float_signature)

        bwd_kernel_float_signature = "const float*, const float*, float*, const int"
        self.bwd_float_kernel = bwd_kernel_mod.get_kernel("bwd<float>", bwd_kernel_float_signature)

        fwd_kernel_double_signature = "const double*, const double*, const int, const int"
        self.fwd_double_kernel = fwd_kernel_mod.get_kernel("fwd<double>", fwd_kernel_double_signature)

        bwd_kernel_double_signature = "const double*, const double*, double*, const int"
        self.bwd_double_kernel = bwd_kernel_mod.get_kernel("bwd<double>", bwd_kernel_double_signature)

    def forward(self, is_train, req, in_data, out_data, aux):
        if req[0] == "null":
            return
        x = in_data[0]  # input
        y = out_data[0] # output

        if y.dtype == np.float64:
            # args, ctx, grid_shape, block_shape, shared_mem = 0
            self.fwd_double_kernel.launch((x, y, x.shape[1], self._reqCode(req[0])), mx.gpu(0), (1, 1, 1), (x.shape[0], 1, 1))
        else:
            # args, ctx, grid_shape, block_shape, shared_mem = 0
            self.fwd_float_kernel.launch((x, y, x.shape[1], self._reqCode(req[0])), mx.gpu(0), (1, 1, 1), (x.shape[0], 1, 1))            

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        if req[0] == "null":
            return
        l = in_data[1]  # label
        y = out_data[0] # output from the forward pass
        dx = in_grad[0] # the storage for the gradient

        if dx.dtype == np.float64:
            # args, ctx, grid_shape, block_shape, shared_mem = 0
            self.bwd_double_kernel.launch((l, y, dx, self._reqCode(req[0])), mx.gpu(0), (y.shape[0], 1, 1), (y.shape[1], 1, 1))
        else:
            # args, ctx, grid_shape, block_shape, shared_mem = 0
            self.bwd_float_kernel.launch((l, y, dx, self._reqCode(req[0])), mx.gpu(0), (y.shape[0], 1, 1), (y.shape[1], 1, 1))

    def _reqCode(self, req):
        if(req == "write"):
            return 1
        elif(req == "add"):
            return 2
        elif(req == "null"):
            return 0
        else:
            raise ValueError("Invalid value of `req`: {}".format(req))


@mx.operator.register("softmax")
class SoftmaxProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(SoftmaxProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [data_shape, label_shape], [output_shape], []

    def infer_type(self, in_type):
        return in_type, [in_type[0]], []

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return Softmax()

# define mlp

data = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(data=data, name='fc1', num_hidden=128)
act1 = mx.symbol.Activation(data=fc1, name='relu1', act_type="relu")
fc2 = mx.symbol.FullyConnected(data=act1, name='fc2', num_hidden=64)
act2 = mx.symbol.Activation(data=fc2, name='relu2', act_type="relu")
fc3 = mx.symbol.FullyConnected(data=act2, name='fc3', num_hidden=10)
#mlp = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')
mlp = mx.symbol.Custom(data=fc3, name='softmax', op_type='softmax')

# data

train, val = mx.test_utils.get_mnist_iterator(batch_size=100, input_shape=(784,))

# train

logging.basicConfig(level=logging.DEBUG)

context = mx.gpu(0)
mod = mx.mod.Module(mlp, context=context)
mod.fit(
    train_data=train,
    eval_data=val,
    optimizer='sgd',
    optimizer_params={'learning_rate':0.1, 'momentum': 0.9, 'wd': 0.00001},
    num_epoch=10,
    batch_end_callback=mx.callback.Speedometer(100, 100)
)

