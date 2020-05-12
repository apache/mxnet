
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
 * \file np_pad_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy/np_pad_op.cc
 */
#include <dmlc/optional.h>
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../../../operator/numpy/np_pad_op-inl.h"

namespace mxnet {

inline int String2MXNetPadType(const std::string& s) {
  using namespace op;
  if (s == "constant") {
    return pad_enum::kConstant;
  } else if (s == "edge") {
    return pad_enum::kEdge;
  } else if (s == "reflect") {
    return pad_enum::kReflect;
  } else if (s == "symmetric") {
    return pad_enum::kSymmetric;
  } else if (s == "maximum") {
    return pad_enum::kMaximum;
  } else if (s == "minimum") {
    return pad_enum::kMinimum;
  } else {
    LOG(FATAL) << "unknown type " << s;
  }
  LOG(FATAL) << "should not reach here ";
  return 0;
}

MXNET_REGISTER_API("_npi.pad")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_pad");
  nnvm::NodeAttrs attrs;
  op::NumpyPadParam param;
  ADT adt = Downcast<ADT, ObjectRef>(args[1].operator ObjectRef());
  int ndim = adt.size();
  std::vector<mxnet::Tuple<int>> temp;
  int counter = 0;
  for (counter = 0; counter < ndim; counter++) {
    temp.emplace_back(mxnet::Tuple<int>(adt[counter]));
  }
  param.pad_width = Tuple<Tuple<int>>(temp.begin(), temp.end());
  param.mode = String2MXNetPadType(args[2].operator std::string());
  if (args[3].type_code() != kNull) {
    param.constant_values = args[3].operator double();
  }
  if (args[4].type_code() != kNull) {
    param.reflect_type = args[4].operator std::string();
  }
  attrs.op = op;
  attrs.parsed = std::move(param);
  SetAttrDict<op::NumpyPadParam>(&attrs);
  int num_inputs = 1;
  int num_outputs = 0;
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  *ret = reinterpret_cast<mxnet::NDArray*>(ndoutputs[0]);
});

}  // namespace mxnet
