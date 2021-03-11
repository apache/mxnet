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
  
  int num_inputs = args_size - 3;
  std::vector<NDArray*> ndinputs;
  ndinputs.reserve(num_inputs);
  for (int i = 1; i < num_inputs + 1; ++i) {
    ndinputs.push_back(static_cast<NDArray*>(args[i]));
  }
  
  int num_params;
  nnvm::NodeAttrs attrs;
  if (args[args_size - 2].type_code() == kNull) {
    num_params = 0;
    attrs.dict.reserve(1);
  } else {
    MXNET_CHECK_TYPE_CODE(args[args_size - 2].type_code(), kObjectHandle);
    Object* flags_ptr = static_cast<Object*>(args[args_size - 2].value().v_handle);
    auto* n = static_cast<const MapObj*>(flags_ptr);
    num_params = static_cast<int>(n->size());
    attrs.dict.reserve(num_params + 1);
    for (const auto& kv : *n) {
      attrs.dict.emplace(std::string(runtime::Downcast<runtime::String>(kv.first)),
                        std::string(runtime::Downcast<runtime::String>(kv.second)));
    }
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
