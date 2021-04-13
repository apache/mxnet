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
 * \file ufunc_helper.cc
 * \brief ufunc helper
 */
#include "ufunc_helper.h"
#include "utils.h"
#include "../../imperative/imperative_utils.h"
#include "../../operator/tensor/elemwise_binary_scalar_op.h"

namespace mxnet {

template<>
void SetAttrDict<double>(nnvm::NodeAttrs* attrs) {
  if (Imperative::Get()->is_recording()) {
    attrs->dict["scalar"] = std::to_string(::dmlc::get<double>(attrs->parsed));
  }
}

void UFuncHelper(NDArray* lhs, NDArray* rhs, NDArray* out,
                 runtime::MXNetRetValue* ret, const nnvm::Op* op) {
  using namespace runtime;
  nnvm::NodeAttrs attrs;
  attrs.op = op;
  NDArray* inputs[] = {lhs, rhs};
  int num_inputs = 2;
  NDArray** outputs = out == nullptr ? nullptr : &out;
  int num_outputs = out != nullptr;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, outputs);
  if (outputs) {
    *ret = PythonArg(2);
  } else {
    *ret = reinterpret_cast<NDArray*>(ndoutputs[0]);
  }
}

void UFuncHelper(NDArray* lhs, int64_t rhs, NDArray* out,
                 runtime::MXNetRetValue* ret, const nnvm::Op* op) {
  using namespace runtime;
  nnvm::NodeAttrs attrs;
  op::NumpyBinaryScalarParam param;
  param.scalar = rhs;
  param.is_int = true;
  attrs.op = op;
  attrs.parsed = param;
  SetAttrDict<op::NumpyBinaryScalarParam>(&attrs);
  NDArray** inputs = &lhs;
  int num_inputs = 1;
  NDArray** outputs = out == nullptr ? nullptr : &out;
  int num_outputs = out != nullptr;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, outputs);
  if (outputs) {
    *ret = PythonArg(2);
  } else {
    *ret = reinterpret_cast<NDArray*>(ndoutputs[0]);
  }
}

void UFuncHelper(NDArray* lhs, double rhs, NDArray* out,
                 runtime::MXNetRetValue* ret, const nnvm::Op* op) {
  using namespace runtime;
  nnvm::NodeAttrs attrs;
  op::NumpyBinaryScalarParam param;
  param.scalar = rhs;
  param.is_int = false;
  attrs.op = op;
  attrs.parsed = param;
  SetAttrDict<op::NumpyBinaryScalarParam>(&attrs);
  NDArray** inputs = &lhs;
  int num_inputs = 1;
  NDArray** outputs = out == nullptr ? nullptr : &out;
  int num_outputs = out != nullptr;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, outputs);
  if (outputs) {
    *ret = PythonArg(2);
  } else {
    *ret = reinterpret_cast<NDArray*>(ndoutputs[0]);
  }
}

void UFuncHelper(int64_t lhs, NDArray* rhs, NDArray* out,
                 runtime::MXNetRetValue* ret, const nnvm::Op* op) {
  using namespace runtime;
  nnvm::NodeAttrs attrs;
  op::NumpyBinaryScalarParam param;
  param.scalar = lhs;
  param.is_int = true;
  attrs.op = op;
  attrs.parsed = param;
  SetAttrDict<op::NumpyBinaryScalarParam>(&attrs);
  NDArray** inputs = &rhs;
  int num_inputs = 1;
  NDArray** outputs = out == nullptr ? nullptr : &out;
  int num_outputs = out != nullptr;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, outputs);
  if (outputs) {
    *ret = PythonArg(2);
  } else {
    *ret = reinterpret_cast<NDArray*>(ndoutputs[0]);
  }
}

void UFuncHelper(double lhs, NDArray* rhs, NDArray* out,
                 runtime::MXNetRetValue* ret, const nnvm::Op* op) {
  using namespace runtime;
  nnvm::NodeAttrs attrs;
  op::NumpyBinaryScalarParam param;
  param.scalar = lhs;
  param.is_int = false;
  attrs.op = op;
  attrs.parsed = param;
  SetAttrDict<op::NumpyBinaryScalarParam>(&attrs);
  NDArray** inputs = &rhs;
  int num_inputs = 1;
  NDArray** outputs = out == nullptr ? nullptr : &out;
  int num_outputs = out != nullptr;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, outputs);
  if (outputs) {
    *ret = PythonArg(2);
  } else {
    *ret = reinterpret_cast<NDArray*>(ndoutputs[0]);
  }
}

void UFuncHelper(runtime::MXNetArgs args,
                 runtime::MXNetRetValue* ret,
                 const nnvm::Op* fn_array,
                 const nnvm::Op* lfn_scalar,
                 const nnvm::Op* rfn_scalar) {
  using namespace runtime;
  NDArray* out = args[2].operator NDArray*();
  if (args[0].type_code() == kNDArrayHandle) {
    if (args[1].type_code() == kNDArrayHandle) {
      UFuncHelper(args[0].operator NDArray*(), args[1].operator NDArray*(), out, ret, fn_array);
    } else if (args[1].type_code() == kDLInt) {
      UFuncHelper(args[0].operator NDArray*(), args[1].operator int64_t(), out, ret, lfn_scalar);
    } else {
      UFuncHelper(args[0].operator NDArray*(), args[1].operator double(), out, ret, lfn_scalar);
    }
  } else if (args[0].type_code() == kDLInt) {
    UFuncHelper(args[0].operator int64_t(), args[1].operator NDArray*(), out, ret,
                rfn_scalar ? rfn_scalar : lfn_scalar);
  } else {
    UFuncHelper(args[0].operator double(), args[1].operator NDArray*(), out, ret,
                rfn_scalar ? rfn_scalar : lfn_scalar);
  }
}

void UFuncHelper(runtime::MXNetArgs args,
                 runtime::MXNetRetValue* ret,
                 const nnvm::Op* op) {
  using namespace runtime;
  nnvm::NodeAttrs attrs;
  attrs.op = op;
  NDArray* inputs[] = {args[0].operator NDArray*()};
  NDArray* out = args[1].operator NDArray*();
  NDArray** outputs = out == nullptr ? nullptr : &out;
  int num_inputs = 1;
  int num_outputs = out != nullptr;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, outputs);
  if (outputs) {
    *ret = PythonArg(1);
  } else {
    *ret = ndoutputs[0];
  }
}

}  // namespace mxnet
