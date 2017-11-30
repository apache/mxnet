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
 * Copyright (c) 2017 by Contributors
 * \file softmax.cc
 * \brief CPU Implementation of softmax
 */
#include "./softmax-inl.h"
#include "../tensor/elemwise_unary_op.h"
#include "../tensor/elemwise_binary_op.h"
#include "mkldnn/mkldnn_base-inl.h"
#include "mkldnn/mkldnn_ops-inl.h"

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(SoftmaxParam);

static void SoftmaxCompute_CPU(const nnvm::NodeAttrs& attrs,
    const OpContext& ctx, const std::vector<NDArray>& inputs,
    const std::vector<OpReqType>& req, const std::vector<NDArray>& outputs) {
#if MXNET_USE_MKLDNN == 1
  const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
  // It seems MKLDNN softmax doesn't support training.
  // and it only supports non-negative axis.
  if (SupportMKLDNN(inputs[0]) && !ctx.is_train && param.axis >= 0) {
    MKLDNNSoftmax_Forward(attrs, ctx, inputs[0], req[0], outputs[0]);
    return;
  }
#endif
  std::vector<TBlob> in_blobs(inputs.size());
  for (size_t i = 0; i < in_blobs.size(); i++)
    in_blobs[i] = inputs[i].data();
  std::vector<TBlob> out_blobs(outputs.size());
  for (size_t i = 0; i < out_blobs.size(); i++)
    out_blobs[i] = outputs[i].data();
  SoftmaxCompute<cpu, mxnet_op::softmax_fwd>(attrs, ctx, in_blobs, req, out_blobs);
}

inline static bool SoftmaxStorageType(const nnvm::NodeAttrs& attrs,
                                 const int dev_mask,
                                 DispatchMode* dispatch_mode,
                                 std::vector<int> *in_attrs,
                                 std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 1);

#if MXNET_USE_MKLDNN == 1
  if (dev_mask == mshadow::cpu::kDevMask) {
    *dispatch_mode = DispatchMode::kFComputeEx;
    (*out_attrs)[0] = kMKLDNNStorage;
    return true;
  }
#endif
  *dispatch_mode = DispatchMode::kFCompute;
  (*out_attrs)[0] = kDefaultStorage;
  return true;
}

MXNET_OPERATOR_REGISTER_UNARY(softmax)
.describe(R"code(Applies the softmax function.

The resulting array contains elements in the range (0,1) and the elements along the given axis sum up to 1.

.. math::
   softmax(\mathbf{z})_j = \frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}}

for :math:`j = 1, ..., K`

Example::

  x = [[ 1.  1.  1.]
       [ 1.  1.  1.]]

  softmax(x,axis=0) = [[ 0.5  0.5  0.5]
                       [ 0.5  0.5  0.5]]

  softmax(x,axis=1) = [[ 0.33333334,  0.33333334,  0.33333334],
                       [ 0.33333334,  0.33333334,  0.33333334]]

)code" ADD_FILELINE)
.set_attr_parser(ParamParser<SoftmaxParam>)
.set_attr<FCompute>("FCompute<cpu>", SoftmaxCompute<cpu, mxnet_op::softmax_fwd>)
.set_attr<FComputeEx>("FComputeEx<cpu>", SoftmaxCompute_CPU)
.set_attr<FInferStorageType>("FInferStorageType", SoftmaxStorageType)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_softmax"})
.add_arguments(SoftmaxParam::__FIELDS__());

MXNET_OPERATOR_REGISTER_BINARY(_backward_softmax)
.set_attr_parser(ParamParser<SoftmaxParam>)
.set_attr<FCompute>("FCompute<cpu>", SoftmaxGradCompute<cpu, op::mshadow_op::mul,
                                                        mxnet_op::softmax_bwd>);

MXNET_OPERATOR_REGISTER_UNARY(log_softmax)
.describe(R"code(Computes the log softmax of the input.
This is equivalent to computing softmax followed by log.

Examples::

  >>> x = mx.nd.array([1, 2, .1])
  >>> mx.nd.log_softmax(x).asnumpy()
  array([-1.41702998, -0.41702995, -2.31702995], dtype=float32)

  >>> x = mx.nd.array( [[1, 2, .1],[.1, 2, 1]] )
  >>> mx.nd.log_softmax(x, axis=0).asnumpy()
  array([[-0.34115392, -0.69314718, -1.24115396],
         [-1.24115396, -0.69314718, -0.34115392]], dtype=float32)


)code")
.set_attr_parser(ParamParser<SoftmaxParam>)
.set_attr<FCompute>("FCompute<cpu>", SoftmaxCompute<cpu, mxnet_op::log_softmax_fwd>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_log_softmax"})
.add_arguments(SoftmaxParam::__FIELDS__());

MXNET_OPERATOR_REGISTER_BINARY(_backward_log_softmax)
.set_attr_parser(ParamParser<SoftmaxParam>)
.set_attr<FCompute>("FCompute<cpu>", SoftmaxGradCompute<cpu, mshadow_op::left,
                                                        mxnet_op::log_softmax_bwd>);

}  // namespace op
}  // namespace mxnet
