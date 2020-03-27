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
 * \file np_moments_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy/np_moments_op.cc
 */

#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../../../operator/numpy/np_broadcast_reduce_op.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.std")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_std");
  op::NumpyMomentsParam param;
  nnvm::NodeAttrs attrs;
  attrs.op = op;

  // parse axis
  if (args[1].type_code() == kNull) {
    param.axis = dmlc::nullopt;
  } else {
    if (args[1].type_code() == kDLInt) {
      param.axis = Tuple<int>(1, args[1].operator int64_t());
    } else {
      param.axis = Tuple<int>(args[1].operator ObjectRef());
    }
  }

  // parse dtype
  if (args[2].type_code() == kNull) {
    param.dtype = dmlc::nullopt;
  } else {
    param.dtype = String2MXNetTypeWithBool(args[2].operator std::string());
  }

  // parse ddof
  param.ddof = args[3].operator int();

  // parse keepdims
  if (args[4].type_code() == kNull) {
    param.keepdims = false;
  } else {
    param.keepdims = args[4].operator bool();
  }

  attrs.parsed = std::move(param);

  SetAttrDict<op::NumpyMomentsParam>(&attrs);

  NDArray* inputs[] = {args[0].operator NDArray*()};
  int num_inputs = 1;

  NDArray* outputs[] = {args[5].operator NDArray*()};
  NDArray** out = (outputs[0] == nullptr) ? nullptr : outputs;
  int num_outputs = (outputs[0] != nullptr);
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, out);

  if (out) {
    *ret = PythonArg(5);
  } else {
    *ret = reinterpret_cast<mxnet::NDArray*>(ndoutputs[0]);
  }
});

MXNET_REGISTER_API("_npi.var")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_var");
  op::NumpyMomentsParam param;
  nnvm::NodeAttrs attrs;
  attrs.op = op;

  // parse axis
  if (args[1].type_code() == kNull) {
    param.axis = dmlc::nullopt;
  } else {
    if (args[1].type_code() == kDLInt) {
      param.axis = Tuple<int>(1, args[1].operator int64_t());
    } else {
      param.axis = Tuple<int>(args[1].operator ObjectRef());
    }
  }

  // parse dtype
  if (args[2].type_code() == kNull) {
    param.dtype = dmlc::nullopt;
  } else {
    param.dtype = String2MXNetTypeWithBool(args[2].operator std::string());
  }

  // parse ddof
  param.ddof = args[3].operator int();

  // parse keepdims
  if (args[4].type_code() == kNull) {
    param.keepdims = false;
  } else {
    param.keepdims = args[4].operator bool();
  }

  attrs.parsed = std::move(param);

  SetAttrDict<op::NumpyMomentsParam>(&attrs);

  NDArray* inputs[] = {args[0].operator NDArray*()};
  int num_inputs = 1;

  NDArray* outputs[] = {args[5].operator NDArray*()};
  NDArray** out = (outputs[0] == nullptr) ? nullptr : outputs;
  int num_outputs = (outputs[0] != nullptr);
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, out);

  if (out) {
    *ret = PythonArg(5);
  } else {
    *ret = reinterpret_cast<mxnet::NDArray*>(ndoutputs[0]);
  }
});

MXNET_REGISTER_API("_npi.average")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_average");
  op::NumpyWeightedAverageParam param;
  nnvm::NodeAttrs attrs;
  attrs.op = op;

  // parse axis
  if (args[2].type_code() == kNull) {
    param.axis = dmlc::nullopt;
  } else {
    if (args[2].type_code() == kDLInt) {
      param.axis = Tuple<int>(1, args[2].operator int64_t());
    } else {
      param.axis = Tuple<int>(args[2].operator ObjectRef());
    }
  }

  // parse returned
  CHECK_NE(args[3].type_code(), kNull)
    << "returned cannot be None";
  param.returned = args[3].operator bool();

  // parse weighted
  CHECK_NE(args[4].type_code(), kNull)
    << "weighted cannot be None";
  param.weighted = args[4].operator bool();

  attrs.parsed = std::move(param);

  SetAttrDict<op::NumpyWeightedAverageParam>(&attrs);

  int num_inputs = param.weighted ? 2 : 1;
  NDArray* outputs[] = {args[5].operator NDArray*()};
  NDArray** out = (outputs[0] == nullptr) ? nullptr : outputs;
  int num_outputs = (outputs[0] != nullptr);

  if (param.weighted) {
    NDArray* inputs[] = {args[0].operator NDArray*(), args[1].operator NDArray*()};
    auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, out);
    if (out) {
      *ret = PythonArg(5);
    } else {
      if (param.returned) {
        *ret = ADT(0, {NDArrayHandle(ndoutputs[0]),
                       NDArrayHandle(ndoutputs[1])});
      } else {
        *ret = reinterpret_cast<mxnet::NDArray*>(ndoutputs[0]);
      }
    }
  } else {
    NDArray* inputs[] = {args[0].operator NDArray*()};
    auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, out);
    if (out) {
      *ret = PythonArg(5);
    } else {
      if (param.returned) {
        *ret = ADT(0, {NDArrayHandle(ndoutputs[0]),
                       NDArrayHandle(ndoutputs[1])});
      } else {
        *ret = reinterpret_cast<mxnet::NDArray*>(ndoutputs[0]);
      }
    }
  }
});

};  // namespace mxnet
