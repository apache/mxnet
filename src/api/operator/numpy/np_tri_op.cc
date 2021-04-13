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
 * \file np_tri_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy/np_diff.cc
 */
#include <mxnet/api_registry.h>
#include "../utils.h"
#include "../../../operator/numpy/np_tri_op-inl.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.tri")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_tri");
  nnvm::NodeAttrs attrs;
  op::TriParam param;
  param.N = args[0].operator nnvm::dim_t();
  if (args[1].type_code() == kNull) {
    param.M = dmlc::nullopt;
  } else {
    param.M = args[1].operator nnvm::dim_t();
  }
  param.k = args[2].operator int();
  param.dtype = args[3].type_code() == kNull ? mshadow::kFloat32 :
                String2MXNetTypeWithBool(args[3].operator std::string());
  if (args[4].type_code() != kNull) {
    attrs.dict["ctx"] = args[4].operator std::string();
  }

  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::TriParam>(&attrs);

  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, 0, nullptr, &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

}  // namespace mxnet
