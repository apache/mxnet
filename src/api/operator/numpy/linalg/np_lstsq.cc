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
 * \file np_lstsq.cc
 * \brief Implementation of the API of functions in src/operator/numpy/linalg/np_lstsq.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../../utils.h"
#include "../../../../operator/numpy/linalg/np_lstsq-inl.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.lstsq")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_lstsq");
  nnvm::NodeAttrs attrs;
  op::LstsqParam param;
  if (args[2].type_code() == kNull) {
    param.rcond = static_cast<double>(1);
  } else if (args[2].type_code() == kStr) {
    const std::string rcond_str = args[2].operator std::string();
    if (rcond_str == "warn") {
      param.rcond = static_cast<double>(-1);
    } else {
      CHECK(false) << "ValueError: wrong parameter rcond = " << rcond_str;
    }
  } else {
    param.rcond = args[2].operator double();
  }
  param.finfoEps32 = args[3].operator double();
  param.finfoEps64 = args[4].operator double();
  param.new_default = args[2].type_code() == kNull ? true : false;
  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::LstsqParam>(&attrs);
  int num_inputs = 2;
  int num_outputs = 0;
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*(), args[1].operator mxnet::NDArray*()};
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  *ret = ADT(0, {NDArrayHandle(ndoutputs[0]),
                 NDArrayHandle(ndoutputs[1]),
                 NDArrayHandle(ndoutputs[2]),
                 NDArrayHandle(ndoutputs[3])});
});

}  // namespace mxnet
