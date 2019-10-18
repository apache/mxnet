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
 * Copyright (c) 2019 by Contributors
 * \file np_memory_op.cc
 */

#include "./np_memory_op.h"

namespace mxnet {
namespace op {

inline bool NumpyShareMemoryType(const nnvm::NodeAttrs& attrs,
                                 std::vector<int> *in_attrs,
                                 std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kBool);
  return out_attrs->at(0) != -1;
}

inline bool NumpyShareMemoryShape(const nnvm::NodeAttrs& attrs,
                                  mxnet::ShapeVector *in_attrs,
                                  mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, mxnet::TShape(0, -1));
  return true;
}

NNVM_REGISTER_OP(_npi_share_memory)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a", "b"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", NumpyShareMemoryShape)
.set_attr<nnvm::FInferType>("FInferType", NumpyShareMemoryType)
.set_attr<FCompute>("FCompute<cpu>", NumpyShareMemoryCompute<cpu>)
.add_argument("a", "NDArray-or-Symbol", "First input")
.add_argument("b", "NDArray-or-Symbol", "Second input");

}  // namespace op
}  // namespace mxnet
