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
 * \file dropout.cc
 * \brief
 * \author Bing Xu, Da Zheng, Hang Zhang
*/

#include "./dropout-inl.h"
#include "../operator_common.h"
#include "mxnet/op_attr_types.h"



namespace mxnet {
namespace op {

OpStatePtr CreateDropoutState(const nnvm::NodeAttrs &attrs,
                                     const Context ctx,
                                     const mxnet::ShapeVector &in_shapes,
                                     const std::vector<int> &in_types) {
  const auto& param = nnvm::get<DropoutParam>(attrs.parsed);
  OpStatePtr state;
  MSHADOW_REAL_TYPE_SWITCH(in_types[dropout::kData], DType, {
    if (ctx.dev_type == kGPU) {
      state = OpStatePtr::Create<DropoutOp<mxnet::gpu, DType>>(param, ctx);
    } else {
      state = OpStatePtr::Create<DropoutOp<mxnet::cpu, DType>>(param, ctx);
    }
    return state;
  });
  LOG(FATAL) << "should never reach here";
  return OpStatePtr();  // should never reach here
}

struct DropoutGrad {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    std::vector<nnvm::NodeEntry> heads;
    heads.push_back(ograds[0]);
    heads.emplace_back(n, dropout::kMask, 0);
    return MakeGradNode(op_name, n, heads, n->attrs.dict);
  }
};

DMLC_REGISTER_PARAMETER(DropoutParam);

NNVM_REGISTER_OP(Dropout)
.add_alias("_npx_dropout")
.describe(R"(Applies dropout operation to input array.

- During training, each element of the input is set to zero with probability p.
  The whole array is rescaled by :math:`1/(1-p)` to keep the expected
  sum of the input unchanged.

- During testing, this operator does not change the input if mode is 'training'.
  If mode is 'always', the same computaion as during training will be applied.

Example::

  random.seed(998)
  input_array = array([[3., 0.5,  -0.5,  2., 7.],
                      [2., -0.4,   7.,  3., 0.2]])
  a = symbol.Variable('a')
  dropout = symbol.Dropout(a, p = 0.2)
  executor = dropout.simple_bind(a = input_array.shape)

  ## If training
  executor.forward(is_train = True, a = input_array)
  executor.outputs
  [[ 3.75   0.625 -0.     2.5    8.75 ]
   [ 2.5   -0.5    8.75   3.75   0.   ]]

  ## If testing
  executor.forward(is_train = False, a = input_array)
  executor.outputs
  [[ 3.     0.5   -0.5    2.     7.   ]
   [ 2.    -0.4    7.     3.     0.2  ]]
)" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr_parser(ParamParser<DropoutParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"data"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output", "mask"};
})
.set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs",
    [](const NodeAttrs& attrs) {
  return 1;
})
.set_attr<mxnet::FInferShape>("FInferShape", [](const nnvm::NodeAttrs& attrs,
      mxnet::ShapeVector *in_shape, mxnet::ShapeVector *out_shape){
  using namespace mshadow;
  CHECK_EQ(in_shape->size(), 1U);
  const DropoutParam& param = nnvm::get<DropoutParam>(attrs.parsed);
  mxnet::TShape dshape(in_shape->at(0));
  if (!mxnet::ndim_is_known(dshape)) return false;
  out_shape->clear();
  out_shape->push_back(dshape);
  for (int i = 0; i < param.axes.ndim(); ++i) {
    dshape[param.axes[i]] = 1;
  }
  out_shape->push_back(dshape);
  return true;
})
.set_attr<nnvm::FInferType>("FInferType", [](const nnvm::NodeAttrs& attrs,
      std::vector<int> *in_type, std::vector<int> *out_type) {
  CHECK_EQ(in_type->size(), 1U);
  int dtype = in_type->at(0);

  if (dtype == -1) {
    LOG(FATAL) << "input type to dropout is not specified.";
    return false;
  }

  size_t nout = 2;
  out_type->clear();
  for (size_t i = 0; i < nout; ++i) out_type->push_back(dtype);
  return true;
})
.set_attr<FCreateOpState>("FCreateOpState", CreateDropoutState)
.set_attr<FStatefulCompute>("FStatefulCompute<cpu>", DropoutCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", DropoutGrad{"_backward_Dropout"})
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs){
  return std::vector<std::pair<int, int> >{{0, 0}};
})
.set_attr<FResourceRequestEx>("FResourceRequestEx",
  [](const NodeAttrs& attrs, const int dev_mask, const DispatchMode dispatch_mode) {
    std::vector<ResourceRequest> request;
    const DropoutParam& param = nnvm::get<DropoutParam>(attrs.parsed);
    if (param.p == 0) return request;
    if (dev_mask == kGPU) {
#if MXNET_USE_CUDNN_DROPOUT
      // if cudnn is used, parallel random is not needed.
      if (1.0f - param.p > 0
          && !(param.cudnn_off && param.cudnn_off.value())
          && param.axes.ndim() == 0) {
        request.emplace_back(ResourceRequest::kCuDNNDropoutDesc);
        return request;
      }
#endif
    }
    request.emplace_back(ResourceRequest::kParallelRandom);
    return request;
  })
.add_argument("data", "NDArray-or-Symbol", "Input array to which dropout will be applied.")
.add_arguments(DropoutParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_Dropout)
.set_num_outputs(1)
.set_attr<bool>("TIsLayerOpBackward", true)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr_parser(ParamParser<DropoutParam>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs){
  return std::vector<std::pair<int, int> >{{0, 0}};
})
.set_attr<FStatefulCompute>("FStatefulCompute<cpu>", DropoutGradCompute<cpu>);

}  // namespace op
}  // namespace mxnet
