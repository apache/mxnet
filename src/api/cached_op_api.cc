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
 * \file cached_op_api.cc
 * \brief The API of function to invoke CachedOp in src/imperative/cached_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../imperative/cached_op.h"
#include "../imperative/cached_op_threadsafe.h"

namespace mxnet {

MXNET_REGISTER_GLOBAL("_api._cached_op_invoke")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  MXAPIThreadLocalEntry<> *local_ret = MXAPIThreadLocalStore<>::Get();
  CachedOpHandle handle = static_cast<CachedOpHandle>(static_cast<void*>(args[0]));
  int num_inputs = args[1];
  NDArrayHandle **inputs = static_cast<NDArrayHandle**>(static_cast<void*>(args[2]));
  int default_dev_type = args[3];
  int default_dev_id = args[4];
  int *num_outputs = static_cast<int*>(static_cast<void*>(args[5]));
  NDArrayHandle ***outputs = static_cast<NDArrayHandle ***>(static_cast<void*>(args[6]));
  const int **out_stypes = static_cast<const int**>(static_cast<void*>(args[7]));

  CachedOpPtr op_shared = *static_cast<CachedOpPtr*>(handle);
  // CachedOp* points to CachedOpThreadSafe object if CreateCachedOpEX
  // was called with thread_safe=true
  CachedOp* op = dynamic_cast<CachedOp*>(op_shared.get());
  std::vector<NDArray*> ndinputs;
  ndinputs.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    ndinputs.push_back(reinterpret_cast<NDArray*>(inputs[i]));
  }

  std::vector<NDArray*> ndoutputs;
  ndoutputs.reserve(op->num_outputs());
  if (*outputs == nullptr) {
    *num_outputs = op->num_outputs();
    for (int i = 0; i < *num_outputs; ++i) ndoutputs.push_back(new NDArray());
  } else {
    CHECK_EQ(*num_outputs, op->num_outputs())
        << "CachedOp expects " << op->num_outputs() << " outputs, but "
        << *num_outputs << " was given.";
    for (int i = 0; i < *num_outputs; ++i) {
      ndoutputs.push_back(reinterpret_cast<NDArray*>((*outputs)[i]));
    }
  }
  // construct default context
  Context ctx = Context::Create(static_cast<Context::DeviceType>(default_dev_type),
                                default_dev_id);
  op->Forward(op_shared, ndinputs, ndoutputs, ctx);

  if (*outputs == nullptr) {
    local_ret->ret_handles.clear();
    local_ret->ret_handles.reserve(*num_outputs);
    for (int i = 0; i < *num_outputs; ++i) {
      local_ret->ret_handles.push_back(ndoutputs[i]);
    }
    *outputs = reinterpret_cast<NDArrayHandle**>(dmlc::BeginPtr(local_ret->ret_handles));
  }

  NDArray** out_array = reinterpret_cast<NDArray**>(*outputs);
  local_ret->out_types.clear();
  local_ret->out_types.reserve(*num_outputs);
  for (int i = 0; i < *num_outputs; ++i) {
    local_ret->out_types.emplace_back(out_array[i]->storage_type());
  }
  *out_stypes = dmlc::BeginPtr(local_ret->out_types);
});

}
