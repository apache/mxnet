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
import mxnet as mx
import numpy as np

class Sqr(mx.operator.CustomOp):
    '''Example of how to use custom op with sparse ndarrays
    '''
    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], mx.nd.sparse.square(in_data[0]))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 2 * in_data[0] * out_grad[0])

@mx.operator.register("sqr")
class SqrProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(SqrProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return in_shape, [in_shape[0]], []

    def infer_type(self, in_type):
        return in_type, [in_type[0]], []

    def infer_storage_type(self, in_stype):
        '''Infer storage type logic for the forward pass
        Takes a list of storage types for inputs
        Returns three lists lists, one for input storage types inferred,
        second for output storage types inferred and third for aux storage
        types inferred
        The in_stype is the list containing storage type for inputs
        If the input is a dense ndarray then we infer the input
        and output to be dense. If input is csr then input and output
        are inferred as csr.
        '''
        if in_stype[0] == 'default':
            return ['default'], ['default'], []
        return ['csr'], ['csr'], []

    def infer_storage_type_backward(self, ograd_stype, in_stype, out_stype, aux_stype):
        '''Infer storage type logic for the backward pass
        Takes storage type of output gradients(ograd_stype), inputs(in_stype),
        outputs(out_stype) and aux(aux_stype).
        Returns a inferred storage type in the following order:
        ograd_stype, in_stype, out_stype, igrad_stype (Storage type for input gradients)
        and aux_stype
        '''
        if in_stype[0] == 'default':
            return ['default'], ['default'], ['default'], ['default'], []
        return ['default'], ['csr'], ['csr'], ['csr'], []

    def create_operator(self, ctx, shapes, dtypes):
        return Sqr()

x = mx.nd.array(np.random.uniform(1, 10, size=(4,10)))
x = x.tostype('csr')
x.attach_grad()
with mx.contrib.autograd.train_section():
    y = mx.nd.Custom(x, op_type='sqr')
    y.backward()
mx.nd.waitall()
print("Original ndarray")
print("--------------")
print(x.asnumpy())
print("Squared ndarray")
print("--------------")
print(y.asnumpy())
print("stype of input is {}".format(x.stype))
print("stype of output is {}".format(y.stype))
