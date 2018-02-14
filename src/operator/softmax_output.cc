/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2015 by Contributors
 * \file softmax_output.cc
 * \brief
 * \author Bing Xu
*/
#include "./softmax_output-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(SoftmaxOutputParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new SoftmaxOutputOp<cpu, DType>(param);
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *SoftmaxOutputProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(SoftmaxOutputParam);

MXNET_REGISTER_OP_PROPERTY(SoftmaxOutput, SoftmaxOutputProp)
.describe(R"code(Computes the gradient of cross entropy loss with respect to softmax output.

- This operator computes the gradient in two steps.
  The cross entropy loss does not actually need to be computed.

  - Applies softmax function on the input array.
  - Computes and returns the gradient of cross entropy loss w.r.t. the softmax output.

- The softmax function, cross entropy loss and gradient is given by:

  - Softmax Function:

    .. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}

  - Cross Entropy Function:

    .. math:: \text{CE(label, output)} = - \sum_i \text{label}_i \log(\text{output}_i)

  - The gradient of cross entropy loss w.r.t softmax output:

    .. math:: \text{gradient} = \text{output} - \text{label}

- During forward propagation, the softmax function is computed for each instance in the input array.

  For general *N*-D input arrays with shape :math:`(d_1, d_2, ..., d_n)`. The size is
  :math:`s=d_1 \cdot d_2 \cdot \cdot \cdot d_n`. We can use the parameters `preserve_shape`
  and `multi_output` to specify the way to compute softmax:

  - By default, `preserve_shape` is ``false``. This operator will reshape the input array
    into a 2-D array with shape :math:`(d_1, \frac{s}{d_1})` and then compute the softmax function for
    each row in the reshaped array, and afterwards reshape it back to the original shape
    :math:`(d_1, d_2, ..., d_n)`.
  - If `preserve_shape` is ``true``, the softmax function will be computed along
    the last axis (`axis` = ``-1``).
  - If `multi_output` is ``true``, the softmax function will be computed along
    the second axis (`axis` = ``1``).

- During backward propagation, the gradient of cross-entropy loss w.r.t softmax output array is computed.
  The provided label can be a one-hot label array or a probability label array.

  - If the parameter `use_ignore` is ``true``, `ignore_label` can specify input instances
    with a particular label to be ignored during backward propagation. **This has no effect when
    softmax `output` has same shape as `label`**.

    Example::

      data = [[1,2,3,4],[2,2,2,2],[3,3,3,3],[4,4,4,4]]
      label = [1,0,2,3]
      ignore_label = 1
      SoftmaxOutput(data=data, label = label,\
                    multi_output=true, use_ignore=true,\
                    ignore_label=ignore_label)
      ## forward softmax output
      [[ 0.0320586   0.08714432  0.23688284  0.64391428]
       [ 0.25        0.25        0.25        0.25      ]
       [ 0.25        0.25        0.25        0.25      ]
       [ 0.25        0.25        0.25        0.25      ]]
      ## backward gradient output
      [[ 0.    0.    0.    0.  ]
       [-0.75  0.25  0.25  0.25]
       [ 0.25  0.25 -0.75  0.25]
       [ 0.25  0.25  0.25 -0.75]]
      ## notice that the first row is all 0 because label[0] is 1, which is equal to ignore_label.

  - The parameter `grad_scale` can be used to rescale the gradient, which is often used to
    give each loss function different weights.

  - This operator also supports various ways to normalize the gradient by `normalization`,
    The `normalization` is applied if softmax output has different shape than the labels.
    The `normalization` mode can be set to the followings:

    - ``'null'``: do nothing.
    - ``'batch'``: divide the gradient by the batch size.
    - ``'valid'``: divide the gradient by the number of instances which are not ignored.

)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input array.")
.add_argument("label", "NDArray-or-Symbol", "Ground truth label.")
.add_arguments(SoftmaxOutputParam::__FIELDS__());


MXNET_REGISTER_OP_PROPERTY(Softmax, DeprecatedSoftmaxProp)
.describe(R"code(Please use `SoftmaxOutput`.

.. note::

  This operator has been renamed to `SoftmaxOutput`, which
  computes the gradient of cross-entropy loss w.r.t softmax output.
  To just compute softmax output, use the `softmax` operator.

)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input array.")
.add_arguments(SoftmaxOutputParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
