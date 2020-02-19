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
 * \file expr.cc
 * \brief The expression AST nodes for the common IR infra.
 */
// Acknowledgement: This file originates from incubator-tvm

#include <mxnet/ir/expr.h>

namespace mxnet {

IntImm::IntImm(MXNetDataType dtype, int64_t value) {
  CHECK(dtype.is_scalar())
      << "ValueError: IntImm can only take scalar.";
  CHECK(dtype.is_int() || dtype.is_uint())
      << "ValueError: IntImm can only take scalar.";
  if (dtype.is_uint()) {
    CHECK_GE(value, 0U);
  }
  runtime::ObjectPtr<IntImmNode> node = make_object<IntImmNode>();
  node->dtype = dtype;
  node->value = value;
  data_ = std::move(node);
}

FloatImm::FloatImm(MXNetDataType dtype, double value) {
  CHECK_EQ(dtype.lanes(), 1)
      << "ValueError: FloatImm can only take scalar.";
  runtime::ObjectPtr<FloatImmNode> node = make_object<FloatImmNode>();
  node->dtype = dtype;
  node->value = value;
  data_ = std::move(node);
}

}  // namespace mxnet
