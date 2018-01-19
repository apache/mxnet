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
 * \file lrn.cc
 * \brief
 * \author Bing Xu, Patric Zhao (patric.zhao@intel.com)
*/

#include "./lrn-inl.h"
#include "../operator_common.h"
#if MXNET_USE_MKLDNN == 1
#include "./mkldnn/mkldnn_lrn-inl.h"
#endif

namespace mxnet {
namespace op {

bool LRNShape(const nnvm::NodeAttrs& attrs,
              std::vector<TShape> *in_shape,
              std::vector<TShape> *out_shape) {
  using namespace mshadow;
  CHECK_EQ(in_shape->size(), 1U) << "Input:[data]";
  const TShape &dshape = in_shape->at(0);
  if (dshape.ndim() == 0) return false;
  out_shape->clear();
  out_shape->push_back(dshape);
  out_shape->push_back(dshape);
  return true;
}

inline std::vector<std::string> ListArguments() {
  return {"data"};
}

bool LRNType(const nnvm::NodeAttrs& attrs,
             std::vector<int> *in_type,
             std::vector<int> *out_type) {
  CHECK_GE(in_type->size(), 1U);
  int dtype = (*in_type)[0];
  CHECK_NE(dtype, -1) << "First input must have specified type";
  for (index_t i = 0; i < in_type->size(); ++i) {
    if ((*in_type)[i] == -1) {
      (*in_type)[i] = dtype;
    } else {
      UNIFORM_TYPE_CHECK((*in_type)[i], dtype, ListArguments()[i]);
    }
  }
  int n_out = 2;
  out_type->clear();
  for (int i = 0; i < n_out; ++i ) out_type->push_back(dtype);
  return true;
}

struct LRNGrad {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                const std::vector<nnvm::NodeEntry>& ograds) const {
    std::vector<nnvm::NodeEntry> heads;
    heads.push_back(ograds[0]);  // out_grad
    heads.push_back(n->inputs[lrn_enum::kData]);
    heads.emplace_back(nnvm::NodeEntry{n, lrn_enum::kTmpNorm, 0});
    return MakeGradNode(op_name, n, heads, n->attrs.dict);
  }
};

bool LRNForwardInferStorageType(const nnvm::NodeAttrs& attrs,
                                const int dev_mask,
                                DispatchMode* dispatch_mode,
                                std::vector<int> *in_attrs,
                                std::vector<int> *out_attrs) {
  CHECK(!in_attrs->empty());
#if MXNET_USE_MKLDNN == 1
  if (dev_mask == mshadow::cpu::kDevMask) {
    storage_type_assign(out_attrs, mxnet::kDefaultStorage,
                        dispatch_mode, DispatchMode::kFComputeEx);
    return true;
  }
#endif
  storage_type_assign(out_attrs, mxnet::kDefaultStorage,
                      dispatch_mode, DispatchMode::kFCompute);
  return true;
}

bool LRNBackwardInferStorageType(const nnvm::NodeAttrs& attrs,
                                 const int dev_mask,
                                 DispatchMode* dispatch_mode,
                                 std::vector<int> *in_attrs,
                                 std::vector<int> *out_attrs) {
  CHECK(!in_attrs->empty());
#if MXNET_USE_MKLDNN == 1
  if (dev_mask == mshadow::cpu::kDevMask) {
    storage_type_assign(out_attrs, mxnet::kDefaultStorage,
                        dispatch_mode, DispatchMode::kFComputeEx);
    return true;
  }
#endif
  storage_type_assign(out_attrs, mxnet::kDefaultStorage,
                      dispatch_mode, DispatchMode::kFCompute);
  return true;
}

#if MXNET_USE_MKLDNN == 1
void LRNComputeExCPU(const nnvm::NodeAttrs &attrs,
                     const OpContext &ctx,
                     const std::vector<NDArray> &inputs,
                     const std::vector<OpReqType> &req,
                     const std::vector<NDArray> &outputs) {
  const LRNParam &param = nnvm::get<LRNParam>(attrs.parsed);
  if (SupportMKLDNN(inputs[0])) {
    MKLDNNLRNForward(ctx, param, inputs[0], req[0], outputs[0]);
    return;
  }
  FallBackCompute(LRNCompute<cpu>, attrs, ctx, inputs, req, outputs);
}

void LRNGradComputeExCPU(const nnvm::NodeAttrs &attrs,
                         const OpContext &ctx,
                         const std::vector<NDArray> &inputs,
                         const std::vector<OpReqType> &req,
                         const std::vector<NDArray> &outputs) {
  const LRNParam &param = nnvm::get<LRNParam>(attrs.parsed);
  const NDArray &out_grad = inputs[0];
  const NDArray &in_data = inputs[1];
  const NDArray &in_grad = outputs[0];

  if (SupportMKLDNN(inputs[0])) {
    MKLDNNLRNBackward(ctx, param, out_grad, in_data, req[0], in_grad);
    return;
  }
  FallBackCompute(LRNGradCompute<cpu>, attrs, ctx, inputs, req, outputs);
}
#endif

DMLC_REGISTER_PARAMETER(LRNParam);

NNVM_REGISTER_OP(LRN)
.describe(R"code(Applies local response normalization to the input.

The local response normalization layer performs "lateral inhibition" by normalizing
over local input regions.

If :math:`a_{x,y}^{i}` is the activity of a neuron computed by applying kernel :math:`i` at position
:math:`(x, y)` and then applying the ReLU nonlinearity, the response-normalized
activity :math:`b_{x,y}^{i}` is given by the expression:

.. math::
   b_{x,y}^{i} = \frac{a_{x,y}^{i}}{\Bigg({k + \alpha \sum_{j=max(0, i-\frac{n}{2})}^{min(N-1, i+\frac{n}{2})} (a_{x,y}^{j})^{2}}\Bigg)^{\beta}}

where the sum runs over :math:`n` "adjacent" kernel maps at the same spatial position, and :math:`N` is the total
number of kernels in the layer.

)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs",
                                    [](const NodeAttrs& attrs) { return 1; })
.set_attr_parser(ParamParser<LRNParam>)
.set_attr<nnvm::FInferShape>("FInferShape", LRNShape)
.set_attr<nnvm::FInferType>("FInferType", LRNType)
.set_attr<FInferStorageType>("FInferStorageType", LRNForwardInferStorageType)
.set_attr<FCompute>("FCompute<cpu>", LRNCompute<cpu>)
#if MXNET_USE_MKLDNN == 1
.set_attr<FComputeEx>("FComputeEx<cpu>", LRNComputeExCPU)
#endif
.set_attr<nnvm::FGradient>("FGradient", LRNGrad{"_backward_LRN"})
.add_argument("data", "NDArray-or-Symbol", "Input data to LRN")
.add_arguments(LRNParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_LRN)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LRNParam>)
.set_attr<FInferStorageType>("FInferStorageType", LRNBackwardInferStorageType)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
#if MXNET_USE_MKLDNN == 1
.set_attr<FComputeEx>("FComputeEx<cpu>", LRNGradComputeExCPU)
#endif
.set_attr<FCompute>("FCompute<cpu>", LRNGradCompute<cpu>);

}  // namespace op
}  // namespace mxnet
