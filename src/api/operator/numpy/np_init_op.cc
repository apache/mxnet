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
 * \file np_init_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy/np_init_op.cc
 */
#include "../utils.h"
#include "../../../operator/tensor/init_op.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.zeros")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_zeros");
  nnvm::NodeAttrs attrs;
  op::InitOpParam param;
  if (args[0].type_code() == kDLInt) {
    param.shape = TShape(1, args[0].operator int64_t());
  } else {
    param.shape = TShape(args[0].operator ObjectRef());
  }
  if (args[1].type_code() == kNull) {
    param.dtype = mshadow::kFloat32;
  } else {
    param.dtype = String2MXNetTypeWithBool(args[1].operator std::string());
  }
  attrs.parsed = std::move(param);
  attrs.op = op;
  if (args[2].type_code() != kNull) {
    attrs.dict["ctx"] = args[2].operator std::string();
  }
  int num_outputs = 0;
  auto ndoutputs = Invoke<op::InitOpParam>(op, &attrs, 0, nullptr, &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

}  // namespace mxnet
