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
 * \file api_npi.cc
 * \brief Implementation of API _npi functions
 */
#include <mxnet/runtime/ffi_helper.h>
#include <mxnet/runtime/container.h>
#include <mxnet/runtime/packed_func.h>
#include <mxnet/api_registry.h>
#include <mxnet/base.h>
#include <nnvm/c_api.h>
#include <iostream>

#include "../operator/tensor/init_op.h"
#include "../operator/numpy/np_tensordot_op-inl.h"
#include "../imperative/imperative_utils.h"

namespace mxnet {

inline void SetInOut(std::vector<NDArray*>* ndinputs,
                     std::vector<NDArray*>* ndoutputs,
                     int num_inputs,
                     NDArray** inputs,
                     int *num_outputs,
                     int infered_num_outputs,
                     int num_visible_outputs,
                     NDArray** out_array) {
  ndinputs->clear();
  ndinputs->reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    NDArray* inp = reinterpret_cast<NDArray*>(inputs[i]);
    if (!features::is_enabled(features::INT64_TENSOR_SIZE)) {
      CHECK_LT(inp->shape().Size(), (int64_t{1} << 31) - 1) <<
                "[SetNDInputsOutputs] Size of tensor you are trying to allocate is larger than "
                "2^31 elements. Please build with flag USE_INT64_TENSOR_SIZE=1";
    }
    ndinputs->emplace_back(inp);
  }

  ndoutputs->clear();
  ndoutputs->reserve(infered_num_outputs);
  if (out_array == nullptr) {
    for (int i = 0; i < infered_num_outputs; ++i) {
      ndoutputs->emplace_back(new NDArray());
    }
    *num_outputs = num_visible_outputs;
  } else {
    CHECK(*num_outputs == infered_num_outputs || *num_outputs == num_visible_outputs)
      << "Operator expects " << infered_num_outputs << " (all) or "
      << num_visible_outputs << " (visible only) outputs, but got "
      << *num_outputs << " instead.";
    for (int i = 0; i < *num_outputs; ++i) {
      ndoutputs->emplace_back(out_array[i]);
    }
    for (int i = *num_outputs; i < infered_num_outputs; ++i) {
      ndoutputs->emplace_back(new NDArray());
    }
  }
}

template<typename T>
inline std::vector<NDArray*> Invoke(const nnvm::Op* op,
                                    nnvm::NodeAttrs* attrs,
                                    int num_inputs,
                                    NDArray** inputs,
                                    int* num_outputs,
                                    NDArray** outputs) {
  int infered_num_outputs;
  int num_visible_outputs;
  imperative::SetNumOutputs(op, *attrs, num_inputs, &infered_num_outputs, &num_visible_outputs);

  std::vector<NDArray*> ndinputs, ndoutputs;
  SetInOut(&ndinputs, &ndoutputs, num_inputs, inputs,
      num_outputs, infered_num_outputs, num_visible_outputs, outputs);

  auto state = Imperative::Get()->Invoke(Context::CPU(), *attrs, ndinputs, ndoutputs);
  if (Imperative::Get()->is_recording()) {
    ::dmlc::get<T>(attrs->parsed).SetAttrDict(&(attrs->dict));
    Imperative::Get()->RecordOp(std::move(*attrs), ndinputs, ndoutputs, state);
  }
  for (int i = *num_outputs; i < infered_num_outputs; ++i) delete ndoutputs[i];
  return ndoutputs;
}

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

inline static void _npi_tensordot_int_axes(runtime::MXNetArgs args,
                                           runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_tensordot_int_axes");
  op::TensordotIntAxesParam param;
  nnvm::NodeAttrs attrs;
  attrs.op = op;
  param.axes = args[2].operator int();
  // we directly copy TensordotIntAxesParam, which is trivially-copyable
  attrs.parsed = param;
  int num_outputs = 0;
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*(), args[1].operator mxnet::NDArray*()};
  auto ndoutputs = Invoke<op::TensordotIntAxesParam>(op, &attrs, 2, inputs, &num_outputs, nullptr);
  *ret = reinterpret_cast<mxnet::NDArray*>(ndoutputs[0]);
}

inline static void _npi_tensordot(runtime::MXNetArgs args,
                                  runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_tensordot");
  op::TensordotParam param;
  nnvm::NodeAttrs attrs;
  attrs.op = op;
  const ObjectRef ref = args[2].operator ObjectRef();
  if (const ADTObj* obj = ref.as<ADTObj>()) {
    if (const IntegerObj* lop = (*obj)[0].as<IntegerObj>()) {
      param.a_axes_summed = Tuple<int>(1, lop->value);
      param.b_axes_summed = Tuple<int>(1, Downcast<Integer, ObjectRef>((*obj)[1])->value);
    } else {
      param.a_axes_summed = Tuple<int>((*obj)[0]);
      param.b_axes_summed = Tuple<int>((*obj)[1]);
    }
  } else {
    Array<ObjectRef> arr = Downcast<Array<ObjectRef>, ObjectRef>(ref);
    if (const IntImmNode* lop = arr[0].as<IntImmNode>()) {
      param.a_axes_summed = Tuple<int>(1, lop->value);
      param.b_axes_summed = Tuple<int>(1, Downcast<IntImm, ObjectRef>(arr[1])->value);
    } else {
      param.a_axes_summed = Tuple<int>(arr[0]);
      param.b_axes_summed = Tuple<int>(arr[1]);
    }
  }
  attrs.parsed = std::move(param);
  int num_outputs = 0;
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*(), args[1].operator mxnet::NDArray*()};
  auto ndoutputs = Invoke<op::TensordotParam>(op, &attrs, 2, inputs, &num_outputs, nullptr);
  *ret = reinterpret_cast<mxnet::NDArray*>(ndoutputs[0]);
}

MXNET_REGISTER_API("_npi.tensordot")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  if (args[2].type_code() == kDLInt) {
    _npi_tensordot_int_axes(args, ret);
  } else {
    _npi_tensordot(args, ret);
  }
});

MXNET_REGISTER_API("_npi.nop")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
});

}  // namespace mxnet
