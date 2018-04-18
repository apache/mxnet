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

import mxnet as mx


class WeightedSoftmaxCrossEntropyLoss(mx.operator.CustomOp):
    """ softmax cross entropy weighted loss, where the loss is adjusted by \
((1 + label * pos_cls_weight) / pos_cls_weight)

    """

    def __init__(self, positive_cls_weight):
        self.positive_cls_weight = float(positive_cls_weight)

    def forward(self, is_train, req, in_data, out_data, aux):
        """Implements forward computation.

        is_train : bool, whether forwarding for training or testing.
        req : list of {'null', 'write', 'inplace', 'add'}, how to assign to out_data. 'null' means skip assignment, etc.
        in_data : list of NDArray, input data.
        out_data : list of NDArray, pre-allocated output buffers.
        aux : list of NDArray, mutable auxiliary states. Usually not used.
        """
        data = in_data[0]
        label = in_data[1]
        pred = mx.nd.SoftmaxOutput(data, label)
        self.assign(out_data[0], req[0], pred)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        """Implements backward computation

        req : list of {'null', 'write', 'inplace', 'add'}, how to assign to in_grad
        out_grad : list of NDArray, gradient w.r.t. output data.
        in_grad : list of NDArray, gradient w.r.t. input data. This is the output buffer.
        """
        label = in_data[1]
        pred = out_data[0]
        dx = pred - mx.nd.one_hot(label, 2)
        pos_cls_weight = self.positive_cls_weight
        scale_factor = ((1 + label * pos_cls_weight) / pos_cls_weight).reshape((pred.shape[0],1))
        rescaled_dx = scale_factor * dx
        self.assign(in_grad[0], req[0], rescaled_dx)

@mx.operator.register("weighted_softmax_ce_loss")
class WeightedSoftmaxCrossEntropyLossProp(mx.operator.CustomOpProp):
    def __init__(self, positive_cls_weight):
        super(WeightedSoftmaxCrossEntropyLossProp, self).__init__(True)
        self.positive_cls_weight = positive_cls_weight
        assert(float(positive_cls_weight) > 0)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shapes):
        """Calculate output shapes from input shapes. This can be
        omited if all your inputs and outputs have the same shape.

        in_shapes : list of shape. Shape is described by a tuple of int.
        """
        data_shape = in_shapes[0]
        output_shape = data_shape
        # return 3 lists representing inputs shapes, outputs shapes, and aux data shapes.
        return (in_shapes), (output_shape,), ()

    def create_operator(self, ctx, in_shapes, in_dtypes):
        #  create and return the CustomOp class.
        return WeightedSoftmaxCrossEntropyLoss(self.positive_cls_weight)
