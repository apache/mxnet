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
    Imperative::Get()->RecordOp(std::move(*attrs), ndinputs, ndoutputs, state);
  }
  for (int i = *num_outputs; i < infered_num_outputs; ++i) delete ndoutputs[i];
  return ndoutputs;
}

MXNET_REGISTER_API("_npi.zeros")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  const nnvm::Op* op = Op::Get("_npi_zeros");
  nnvm::NodeAttrs attrs;
  op::InitOpParam param;
  param.shape = args[0].operator TShape();
  if (args[1].type_code() == kNull) {
    param.dtype = mshadow::kFloat32;
  } else {
    param.dtype = runtime::String2MXNetTypeWithBool(args[1].operator std::string());
  }
  attrs.parsed = std::move(param);
  attrs.op = op;
  if (args[2].type_code() != kNull) {
    attrs.dict["ctx"] = args[2].operator std::string();
  }
  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, 0, nullptr, &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

MXNET_REGISTER_API("_npi.tensordot")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  bool isscalar = args[2].type_code() == kDLInt;
  const nnvm::Op* op = Op::Get(isscalar ?
                               "_npi_tensordot_int_axes" :
                               "_npi_tensordot");
  nnvm::NodeAttrs attrs;
  attrs.op = op;
  if (isscalar) {
    mxnet::op::TensordotIntAxesParam param;
    param.axes = args[2].operator int();
    attrs.parsed = std::move(param);
  } else {
    mxnet::op::TensordotParam param;
    const runtime::ObjectRef ref = args[2].operator runtime::ObjectRef();
    const runtime::ADTObj* obj = ref.as<runtime::ADTObj>();
    if (obj->operator[](0).get()->IsInstance<::mxnet::runtime::IntegerObj>()) {
      param.a_axes_summed = Tuple<int>(1,
        obj->operator[](0).as<::mxnet::runtime::IntegerObj>()->value);
      param.b_axes_summed = Tuple<int>(1,
        obj->operator[](1).as<::mxnet::runtime::IntegerObj>()->value);
    } else {
      const runtime::ADTObj* a_axes_summed = obj->operator[](0).as<runtime::ADTObj>();
      const runtime::ADTObj* b_axes_summed = obj->operator[](1).as<runtime::ADTObj>();
      param.a_axes_summed = Tuple<int>(a_axes_summed->size, 0);
      param.b_axes_summed = Tuple<int>(b_axes_summed->size, 0);
      for (uint32_t i = 0; i < a_axes_summed->size; ++i) {
        param.a_axes_summed[i] =
          a_axes_summed->operator[](i).as<::mxnet::runtime::IntegerObj>()->value;
        param.b_axes_summed[i] =
          b_axes_summed->operator[](i).as<::mxnet::runtime::IntegerObj>()->value;
      }
    }
    attrs.parsed = std::move(param);
  }
  int num_outputs = 0;
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*(), args[1].operator mxnet::NDArray*()};
  auto ndoutputs = Invoke(op, &attrs, 2, inputs, &num_outputs, nullptr);
  *ret = reinterpret_cast<mxnet::NDArray*>(ndoutputs[0]);
});

MXNET_REGISTER_API("_npi.nop")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
});

}  // namespace mxnet
