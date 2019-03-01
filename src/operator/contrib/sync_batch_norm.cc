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
 * Copyright (c) 2018 by Contributors
 * \file sync_batch_norm.cc
 * \brief Synchronized BatchNorm modified from BatchNormV1
 * \author Hang Zhang
*/

#include "sync_batch_norm-inl.h"
#include <nnvm/op_attr_types.h>

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(SyncBatchNormParam param, int dtype) {
  return new SyncBatchNorm<cpu>(param);
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *SyncBatchNormProp::CreateOperatorEx(Context ctx, mxnet::ShapeVector *in_shape,
    std::vector<int> *in_type) const {
    mxnet::ShapeVector out_shape, aux_shape;
    std::vector<int> out_type, aux_type;
    CHECK(InferType(in_type, &out_type, &aux_type));
    CHECK(InferShape(in_shape, &out_shape, &aux_shape));
    DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(SyncBatchNormParam);

MXNET_REGISTER_OP_PROPERTY(_contrib_SyncBatchNorm, SyncBatchNormProp)
.describe(R"code(Batch normalization.

Normalizes a data batch by mean and variance, and applies a scale ``gamma`` as
well as offset ``beta``.
Standard BN [1]_ implementation only normalize the data within each device.
SyncBN normalizes the input within the whole mini-batch.
We follow the sync-onece implmentation described in the paper [2]_.

Assume the input has more than one dimension and we normalize along axis 1.
We first compute the mean and variance along this axis:

.. math::

  data\_mean[i] = mean(data[:,i,:,...]) \\
  data\_var[i] = var(data[:,i,:,...])

Then compute the normalized output, which has the same shape as input, as following:

.. math::

  out[:,i,:,...] = \frac{data[:,i,:,...] - data\_mean[i]}{\sqrt{data\_var[i]+\epsilon}} * gamma[i] + beta[i]

Both *mean* and *var* returns a scalar by treating the input as a vector.

Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both ``data_mean`` and
``data_var`` as well, which are needed for the backward pass.

Besides the inputs and the outputs, this operator accepts two auxiliary
states, ``moving_mean`` and ``moving_var``, which are *k*-length
vectors. They are global statistics for the whole dataset, which are updated
by::

  moving_mean = moving_mean * momentum + data_mean * (1 - momentum)
  moving_var = moving_var * momentum + data_var * (1 - momentum)

If ``use_global_stats`` is set to be true, then ``moving_mean`` and
``moving_var`` are used instead of ``data_mean`` and ``data_var`` to compute
the output. It is often used during inference.

Both ``gamma`` and ``beta`` are learnable parameters. But if ``fix_gamma`` is true,
then set ``gamma`` to 1 and its gradient to 0.

Reference:
  .. [1] Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating \
    deep network training by reducing internal covariate shift." *ICML 2015*
  .. [2] Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, \
    Ambrish Tyagi, and Amit Agrawal. "Context Encoding for Semantic Segmentation." *CVPR 2018*
)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input data to batch normalization")
.add_argument("gamma", "NDArray-or-Symbol", "gamma array")
.add_argument("beta", "NDArray-or-Symbol", "beta array")
.add_argument("moving_mean", "NDArray-or-Symbol", "running mean of input")
.add_argument("moving_var", "NDArray-or-Symbol", "running variance of input")
.add_arguments(SyncBatchNormParam::__FIELDS__());

NNVM_REGISTER_OP(_contrib_SyncBatchNorm)
.set_attr<nnvm::FSetInputVarAttrOnCompose>("FSetInputVarAttrOnCompose",
    [](const nnvm::NodeAttrs& attrs, nnvm::NodePtr var, const int index) {
      if (var->attrs.dict.find("__init__") != var->attrs.dict.end()) return;
      if (index == 3) {
        var->attrs.dict["__init__"] = "[\"zero\", {}]";
      } else if (index == 4) {
        var->attrs.dict["__init__"] = "[\"one\", {}]";
      }
    });

}  // namespace op
}  // namespace mxnet
