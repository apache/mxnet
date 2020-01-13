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
 * \file dynamic_shape_ops.cc
*/

#include "./dynamic_shape_ops-inl.h"
#include "../tensor/elemwise_binary_op.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

inline bool DynamicReshapeType(const nnvm::NodeAttrs& attrs,
                               std::vector<index_t> *in_attrs,
                               std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, (*in_attrs)[0]);
  TYPE_ASSIGN_CHECK(*in_attrs, 0, (*out_attrs)[0]);
  return true;
}

bool DynamicReshapeStorageType(const nnvm::NodeAttrs& attrs,
                               const int dev_mask,
                               DispatchMode* dispatch_mode,
                               std::vector<int> *in_attrs,
                               std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2);
  CHECK_EQ(out_attrs->size(), 1);
  for (size_t i = 0; i < in_attrs->size(); ++i) {
    STORAGE_TYPE_ASSIGN_CHECK(*in_attrs, i, kDefaultStorage);
  }
  for (size_t i = 0; i < out_attrs->size(); ++i) {
    STORAGE_TYPE_ASSIGN_CHECK(*out_attrs, i, kDefaultStorage);
  }
  DISPATCH_MODE_ASSIGN_CHECK(dispatch_mode, 0, DispatchMode::kFComputeEx);
  return true;
}

bool DynamicReshapeBackwardStorageType(const nnvm::NodeAttrs& attrs,
                                       const int dev_mask,
                                       DispatchMode* dispatch_mode,
                                       std::vector<int> *in_attrs,
                                       std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 2);
  for (size_t i = 0; i < in_attrs->size(); ++i) {
    STORAGE_TYPE_ASSIGN_CHECK(*in_attrs, i, kDefaultStorage);
  }
  for (size_t i = 0; i < out_attrs->size(); ++i) {
    STORAGE_TYPE_ASSIGN_CHECK(*out_attrs, i, kDefaultStorage);
  }
  DISPATCH_MODE_ASSIGN_CHECK(dispatch_mode, 0, DispatchMode::kFComputeEx);
  return true;
}

NNVM_REGISTER_OP(_contrib_dynamic_reshape)
.describe(R"code(
Experimental CPU-only support for reshape operator with dynamic shape.

Accepts 2 inputs - data and shape.
The output data has the
Example::
   data = mx.nd.array(np.random.normal(0,1,(2,3,5,5)))
   shape = mx.nd.array((0,-1))
   out = mx.sym.contrib.dynamic_reshape(data = data, shape = shape)
   // out will be of shape (2,75)
)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "shape"};
  })
.set_attr<nnvm::FInferType>("FInferType", DynamicReshapeType)
.set_attr<FInferStorageType>("FInferStorageType", DynamicReshapeStorageType)
.set_attr<FComputeEx>("FComputeEx<cpu>", DynamicReshapeForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_contrib_dynamic_reshape"})
.add_argument("data", "NDArray-or-Symbol", "Data")
.add_argument("shape", "NDArray-or-Symbol", "Shape");


NNVM_REGISTER_OP(_backward_contrib_dynamic_reshape)
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FInferStorageType>("FInferStorageType", DynamicReshapeBackwardStorageType)
.set_attr<FComputeEx>("FComputeEx<cpu>", DynamicReshapeBackward<cpu>);

}  // namespace op
}  // namespace mxnet
