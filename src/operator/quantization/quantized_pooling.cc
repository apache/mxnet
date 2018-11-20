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
 * \file quantized_pooling.cc
*/
#include <mxnet/op_attr_types.h>
#include "../nn/pooling-inl.h"
#if MXNET_USE_MKLDNN == 1
#include "../nn/mkldnn/mkldnn_pooling-inl.h"
#endif

namespace mxnet {
namespace op {

bool QuantizedPoolingShape(const nnvm::NodeAttrs& attrs,
                           std::vector<TShape> *in_shape,
                           std::vector<TShape> *out_shape) {
  const PoolingParam& param = nnvm::get<PoolingParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 3U);
  if (shape_is_none(in_shape->at(0))) return false;
  const TShape &dshape = (*in_shape)[0];
  CHECK_EQ(dshape.ndim(), 4U)
      << "quantized_pooling: Input data should be 4D in "
      << "(batch, channel, y, x)";
  // NCHW layout
  const int N = 0, H = 2, W = 3, C = 1;
  TShape oshape(4);
  CHECK_EQ(param.kernel.ndim(), 2) << "QuantizedPoolingOp only supports 2D pooling for now";
  CHECK(param.kernel[0] <= dshape[H] + 2 * param.pad[0])
      << "kernel size (" << param.kernel[0]
      << ") exceeds input (" << dshape[H]
      << " padded to " << (dshape[H] + 2*param.pad[0]) << ")";
  CHECK(param.kernel[1] <= dshape[W] + 2 * param.pad[1])
      << "kernel size (" << param.kernel[1]
      << ") exceeds input (" << dshape[W]
      << " padded to " << (dshape[W] + 2*param.pad[1]) << ")";
  // only support valid convention
  oshape[N] = dshape[N];
  oshape[C] = dshape[C];
  if (param.global_pool) {
    oshape[H] = 1;
    oshape[W] = 1;
  } else {
    oshape[H] = 1 + (dshape[H] + 2 * param.pad[0] - param.kernel[0]) /
        param.stride[0];
    oshape[W] = 1 + (dshape[W] + 2 * param.pad[1] - param.kernel[1]) /
        param.stride[1];
  }

  SHAPE_ASSIGN_CHECK(*in_shape, 1, TShape{1});
  SHAPE_ASSIGN_CHECK(*in_shape, 2, TShape{1});

  out_shape->clear();
  out_shape->push_back(oshape);
  out_shape->push_back(TShape{1});
  out_shape->push_back(TShape{1});
  return true;
}

bool QuantizedPoolingType(const nnvm::NodeAttrs& attrs,
                          std::vector<int> *in_type,
                          std::vector<int> *out_type) {
  const PoolingParam& param = nnvm::get<PoolingParam>(attrs.parsed);
  CHECK_EQ(in_type->size(), 3U);
  CHECK_EQ(out_type->size(), 3U);
  if (param.pool_type == pool_enum::kMaxPooling || param.pool_type == pool_enum::kAvgPooling) {
#if MXNET_USE_MKLDNN  == 1
    TYPE_ASSIGN_CHECK(*out_type, 0, (*in_type)[0]);
#else
    TYPE_ASSIGN_CHECK(*in_type, 0, mshadow::kInt8);
    TYPE_ASSIGN_CHECK(*out_type, 0, mshadow::kInt8);
#endif
  } else {
    LOG(FATAL) << "QuantizedPoolingOp only supports pool_type=max/avg for now";
  }
  TYPE_ASSIGN_CHECK(*in_type, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*in_type, 2, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_type, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_type, 2, mshadow::kFloat32);
  return true;
}

inline static bool QuantizedPoolingStorageType(const nnvm::NodeAttrs &attrs,
                                               const int dev_mask,
                                               DispatchMode *dispatch_mode,
                                               std::vector<int> *in_attrs,
                                               std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 3);

  *dispatch_mode = DispatchMode::kFCompute;
#if MXNET_USE_MKLDNN == 1
  const PoolingParam &param = nnvm::get<PoolingParam>(attrs.parsed);
  if (dev_mask == mshadow::cpu::kDevMask && SupportMKLDNNPooling(param)) {
    *dispatch_mode = DispatchMode::kFComputeEx;
  }
#else
  CHECK_EQ(out_attrs->size(), 3);
#endif
  for (int& out_attr : *out_attrs)
    out_attr = kDefaultStorage;
  return true;
}

NNVM_REGISTER_OP(_contrib_quantized_pooling)
.describe(R"code(Pooling operator for input and output data type of int8.
The input and output data comes with min and max thresholds for quantizing
the float32 data into int8.

.. Note::
    This operator only supports forward propogation. DO NOT use it in training.
    This operator only supports `pool_type` of `avg` or `max`.)code" ADD_FILELINE)
.set_num_inputs(3)
.set_num_outputs(3)
.set_attr_parser(PoolingParamParser)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "min_data", "max_data"};
  })
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"output", "min_output", "max_output"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", QuantizedPoolingShape)
.set_attr<nnvm::FInferType>("FInferType", QuantizedPoolingType)
.set_attr<FInferStorageType>("FInferStorageType", QuantizedPoolingStorageType)
.set_attr<FNeedRequantize>("FNeedRequantize",
  [](const NodeAttrs& attrs) {
    const PoolingParam& param = nnvm::get<PoolingParam>(attrs.parsed);
    CHECK(param.pool_type == pool_enum::kMaxPooling || param.pool_type == pool_enum::kAvgPooling)
      << "QuantizedPoolingOp only supports pool_type=max/avg for now";
    return false;
  })
.add_argument("data", "NDArray-or-Symbol", "Input data.")
.add_argument("min_data", "NDArray-or-Symbol", "Minimum value of data.")
.add_argument("max_data", "NDArray-or-Symbol", "Maximum value of data.")
.add_arguments(PoolingParam::__FIELDS__());

NNVM_REGISTER_OP(Pooling)
.set_attr<FQuantizedOp>("FQuantizedOp", [](const NodeAttrs& attrs) {
    PoolingParam param;
    param.Init(attrs.dict);
    // TODO(junwu): Uncomment the following line and remove the above lines
    // after pooling op is refactored
    // const PoolingParam& param = nnvm::get<PoolingParam>(attrs.parsed);
    nnvm::NodePtr node = nnvm::Node::Create();
    if (param.pool_type == pool_enum::kMaxPooling || param.pool_type == pool_enum::kAvgPooling) {
      node->attrs.op = Op::Get("_contrib_quantized_pooling");
      node->attrs.name = "quantized_" + attrs.name;
    } else {
      node->attrs.op = Op::Get("Pooling");
      node->attrs.name = attrs.name;
    }
    node->attrs.dict = attrs.dict;
    if (node->op()->attr_parser != nullptr) {
      node->op()->attr_parser(&(node->attrs));
    }
    return node;
  });

}  // namespace op
}  // namespace mxnet
