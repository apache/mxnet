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
#include <mxnet/runtime/container_ext.h>
#include "../imperative/cached_op.h"
#include "../imperative/cached_op_threadsafe.h"

namespace mxnet {

MXNET_REGISTER_GLOBAL("_api.MapVSum")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  Object* ptr = static_cast<Object*>(args[0].value().v_handle);
  auto* n = static_cast<const runtime::MapObj*>(ptr);
  int all = 0;
  for (const auto& kv : *n) {
    runtime::Integer value = Downcast<runtime::Integer, ObjectRef>(kv.second);
    all = all + value->value;
  }
  *ret = all;
});

MXNET_REGISTER_GLOBAL("cached_op.invoke")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  CachedOpPtr op_shared = *static_cast<CachedOpPtr*>(
    static_cast<CachedOpHandle>(static_cast<void*>(args[0])));
  // CachedOp* points to CachedOpThreadSafe object if CreateCachedOpEX
  // was called with thread_safe=true
  CachedOp* op = dynamic_cast<CachedOp*>(op_shared.get());

  ObjectRef inputs_obj = args[1];
  const auto& adt_inputs = Downcast<runtime::ADT>(inputs_obj);
  int num_inputs = static_cast<int>(adt_inputs.size());
  std::vector<NDArray*> ndinputs;
  ndinputs.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    const auto& temp_handle = Downcast<NDArrayHandle>(adt_inputs[i]);
    ndinputs.push_back(temp_handle.getArray());
  }

  ObjectRef outputs_obj = args[2];
  std::vector<NDArray*> ndoutputs;
  ndoutputs.reserve(op->num_outputs());
  if (args[2].type_code() == kNull) {
    for (int i = 0; i < op->num_outputs(); ++i) ndoutputs.push_back(new NDArray());
  } else {
    const auto& adt_outputs = Downcast<runtime::ADT>(outputs_obj);
    int array_size = static_cast<int>(adt_outputs.size());
    CHECK_EQ(array_size, op->num_outputs())
        << "CachedOp expects " << op->num_outputs() << " outputs, but "
        << array_size << " was given.";
    for (int i = 0; i < array_size; ++i) {
      const auto& temp_handle = Downcast<NDArrayHandle>(adt_outputs[i]);
      ndoutputs.push_back(temp_handle.getArray());
    }
  }

  int default_dev_type;
  int default_dev_id;
  if (args[3].type_code() != kNull) {
    default_dev_type = args[3];
    default_dev_id = args[4];
  } else {
    const Context &ctx = ndinputs[0]->ctx();
    default_dev_type = ctx.dev_type;
    default_dev_id = ctx.dev_id;
  }

  // construct default context
  Context ctx = Context::Create(static_cast<Context::DeviceType>(default_dev_type),
                                default_dev_id);
  op->Forward(op_shared, ndinputs, ndoutputs, ctx);

  std::vector<ObjectRef> outputs;
  outputs.reserve(op->num_outputs());
  for (int i = 0; i < op->num_outputs(); ++i) {
    ObjectRef out = NDArrayHandle(ndoutputs[i]);
    outputs.push_back(out);
  }
  *ret = runtime::ADT(0, outputs.begin(), outputs.end());
});

MXNET_REGISTER_GLOBAL("cached_op.create")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  nnvm::Symbol* sym = static_cast<nnvm::Symbol*>(static_cast<void*>(args[0]));
  int num_flags = args[1];
  Object* flags_ptr = static_cast<Object*>(args[2].value().v_handle);
  auto* n = static_cast<const runtime::MapObj*>(flags_ptr);
  bool thread_safe = args[3];
  std::vector<std::pair<std::string, std::string> > flags;
  flags.reserve(num_flags);
  for (const auto& kv : *n) {
    flags.emplace_back(std::string(runtime::Downcast<runtime::String>(kv.first)),
                       std::string(runtime::Downcast<runtime::String>(kv.second)));
  }
  mxnet::CachedOpPtr *out;
  if (!thread_safe) {
    out = new CachedOpPtr(new CachedOp(*sym, flags));
  } else {
    out = new CachedOpPtr(new CachedOpThreadSafe(*sym, flags));
  }
  *ret = static_cast<void*>(out);
});

MXNET_REGISTER_GLOBAL("cached_op.free")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  CachedOpPtr* g = static_cast<CachedOpPtr*>(static_cast<void*>(args[0]));
  delete g;
});

MXNET_REGISTER_GLOBAL("cached_op.get_optimized_symbol")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  auto s = new nnvm::Symbol();
  CachedOpPtr op = *static_cast<CachedOpPtr*>(static_cast<void*>(args[0]));
  *s = op->GetOptimizedSymbol();
  *ret = static_cast<void*>(static_cast<SymbolHandle>(s));
});

}  // namespace mxnet
