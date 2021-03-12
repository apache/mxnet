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
 * \file np_registered_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy/np_registered_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include <vector>
#include "../utils.h"

namespace mxnet {

MXNET_REGISTER_GLOBAL("ndarray._imperative_invoke")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;

  int args_size = args.size();
  AtomicSymbolCreator creator = static_cast<AtomicSymbolCreator>(args[0].value().v_handle);
  const nnvm::Op* op = static_cast<nnvm::Op*>(creator);
  
  int num_inputs = args[1];
  std::vector<NDArray*> ndinputs;
  ndinputs.reserve(num_inputs);
  for (int i = 2; i < num_inputs + 2; ++i) {
    ndinputs.push_back(static_cast<NDArray*>(args[i]));
  }
  
  int num_params = args[num_inputs + 2];
  nnvm::NodeAttrs attrs;
  attrs.dict.reserve(num_params + 1);
  int params_type_code = args[num_inputs + 3].type_code();
  if (params_type_code == kParams) {
    std::string *params = static_cast<std::string*>(args[num_inputs + 3].value().v_handle);
    for (int i = 0; i < num_params*2; i+=2) {
      attrs.dict.emplace(params[i], params[i+1]);
    }
  } else if (params_type_code == kHandle) {
    const char **params = static_cast<const char **>(args[num_inputs + 3].value().v_handle);
    for (int i = 0; i < num_params*2; i+=2) {
      attrs.dict.emplace(params[i], params[i+1]);
    }
  } else {
    MXNET_CHECK_TYPE_CODE(params_type_code, kNull);
  }
  static auto& num_args = nnvm::Op::GetAttr<std::string>("key_var_num_args");
  attrs.op = op;
  if (num_args.count(op)) {
    attrs.dict.emplace(num_args[op], std::to_string(num_inputs));
  }
  if (op->attr_parser != nullptr) {
    op->attr_parser(&attrs);
  }
  if (attrs.op) {
    attrs.name = attrs.op->name;
  }

  NDArray** outputs;
  int num_outputs;
  int out_type_code = args[args_size - 1].type_code();
  if (out_type_code == kNull) {
    outputs = nullptr;
    num_outputs = 0;
  } else if (out_type_code == kNDArrayHandle) {
    NDArray* out = static_cast<NDArray*>(args[args_size - 1].value().v_handle);
    outputs = &out;
    num_outputs = 1;
  } else {
    MXNET_CHECK_TYPE_CODE(out_type_code, kObjectHandle);
    ADT adt = Downcast<ADT, ObjectRef>(args[args_size - 1].operator ObjectRef());
    num_outputs = adt.size();
    std::vector<NDArray*> outputs_vec;
    outputs_vec.reserve(num_outputs);
    for (int i = 0; i < num_outputs; ++i) {
      const auto& temp_handle = Downcast<NDArrayHandle>(adt[i]);
      outputs_vec.push_back(temp_handle.getArray());
    }
    outputs = outputs_vec.data();
  }

  auto ndoutputs = Invoke(op, &attrs, num_inputs, ndinputs.data(), &num_outputs, outputs);
  
  if (out_type_code == kNull) {
    if (num_outputs == 1) {
      *ret = reinterpret_cast<NDArray*>(ndoutputs[0]);
    } else {
      std::vector<ObjectRef> outputs_obj;
      outputs_obj.reserve(num_outputs);
      for (int i = 0; i < num_outputs; ++i) {
        ObjectRef out = NDArrayHandle(ndoutputs[i]);
        outputs_obj.push_back(out);
        delete ndoutputs[i];
      }
      *ret = runtime::ADT(0, outputs_obj.begin(), outputs_obj.end());
    }
  } else {
    *ret = PythonArg(args_size - 1);
  }

});

}  // namespace mxnet
