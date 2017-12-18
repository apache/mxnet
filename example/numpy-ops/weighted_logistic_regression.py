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

import numpy as np
import mxnet as mx

class WeightedLogisticRegression(mx.operator.CustomOp):
    def __init__(self, pos_grad_scale, neg_grad_scale):
        self.pos_grad_scale = float(pos_grad_scale)
        self.neg_grad_scale = float(neg_grad_scale)
    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], mx.nd.divide(1, (1 + mx.nd.exp(- in_data[0]))))
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        in_grad[0][:] = ((out_data[0] - 1) * in_data[1] * self.pos_grad_scale
                         + out_data[0] * (1 - in_data[1]) * self.neg_grad_scale) / out_data[0].shape[1]

@mx.operator.register("weighted_logistic_regression")
class WeightedLogisticRegressionProp(mx.operator.CustomOpProp):
    def __init__(self, pos_grad_scale, neg_grad_scale):
        self.pos_grad_scale = pos_grad_scale
        self.neg_grad_scale = neg_grad_scale
        super(WeightedLogisticRegressionProp, self).__init__(False)
    def list_arguments(self):
        return ['data', 'label']
    def list_outputs(self):
        return ['output']
    def infer_shape(self, in_shape):
        shape = in_shape[0]
        return [shape, shape], [shape]
    def create_operator(self, ctx, shapes, dtypes):
        return WeightedLogisticRegression(self.pos_grad_scale, self.neg_grad_scale)

if __name__ == '__main__':
    m, n = 2, 5
    pos, neg = 1, 0.1
    data = mx.sym.Variable('data')

    wlr = mx.sym.Custom(data, pos_grad_scale=pos, neg_grad_scale=neg, name='wlr',
                        op_type='weighted_logistic_regression')
    lr = mx.sym.LogisticRegressionOutput(data, name='lr')

    # MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
    context = mx.cpu()
    # Uncomment this line to compute on GPU
    # context=mx.gpu(0)

    exe1 = wlr.simple_bind(ctx=context, data=(2 * m, n))
    exe2 = lr.simple_bind(ctx=context, data=(2 * m, n))

    exe1.arg_dict['data'][:] = np.ones([2 * m, n])
    exe2.arg_dict['data'][:] = np.ones([2 * m, n])

    exe1.arg_dict['wlr_label'][:] = np.vstack([np.ones([m, n]), np.zeros([m, n])])
    exe2.arg_dict['lr_label'][:] = np.vstack([np.ones([m, n]), np.zeros([m, n])])

    exe1.forward(is_train=True)
    exe2.forward(is_train=True)

    print('Weighted Logistic Regression output:')
    print(exe1.outputs[0].asnumpy())

    print('Logistic Regression output:')
    print(exe2.outputs[0].asnumpy())

    exe1.backward()
    exe2.backward()

    print('Weighted Logistic Regression gradients:')
    print(exe1.grad_dict['data'].asnumpy())

    print('Logistic Regression gradients:')
    print(exe2.grad_dict['data'].asnumpy())
