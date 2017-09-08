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
#include <mxnet/operator.h>
#include <mxnet/executor.h>
#include <mxnet/imperative_runtime.h>
#include <nnvm/pass_functions.h>
#include <unordered_set>
#include <iostream>
#include "../executor/graph_executor.h"
#include "../executor/exec_pass.h"
#include "../c_api/c_api_common.h"
#include "../common/utils.h"

namespace mxnet {
#if DMLC_CXX11_THREAD_LOCAL
thread_local bool ImperativeRuntime::is_train_ = false;
thread_local bool ImperativeRuntime::is_recording_ = false;
#else
MX_THREAD_LOCAL bool ImperativeRuntime::is_train_ = false;
MX_THREAD_LOCAL bool ImperativeRuntime::is_recording_ = false;
#endif

ImperativeRuntime* ImperativeRuntime::Get() {
  static ImperativeRuntime inst;
  return &inst;
}

namespace imperative {

Context GetContext(const nnvm::NodeAttrs& attrs,
                const std::vector<NDArray*>& inputs,
                const std::vector<NDArray*>& outputs,
                const Context& default_ctx) {
  Context ctx;
  if (inputs.size()) {
    ctx = inputs[0]->ctx();
    for (size_t i = 1; i < inputs.size(); ++i) {
      CHECK_EQ(inputs[i]->ctx().dev_mask(), ctx.dev_mask())
          << "Operator " << attrs.op->name
          << " require all inputs live on the same context. "
          << "But the first argument is on "
          << ctx << " while the " << i+1 << "-th argument is on "
          << inputs[i]->ctx();
    }
  } else if (outputs.size() && !outputs[0]->is_none()) {
    ctx = outputs[0]->ctx();
  } else if (attrs.dict.find("ctx") != attrs.dict.end()) {
    ctx = Context::FromString(attrs.dict.at("ctx"));
  } else {
    ctx = default_ctx;
  }
  // Pinned context doesn't propagate
  if (ctx.dev_type == Context::kCPUPinned) {
    ctx = Context::CPU();
  }
#if !MXNET_USE_CUDA
  if (ctx.dev_mask() == gpu::kDevMask) {
    LOG(INFO) << "GPU support is disabled. Compile MXNet with "
              << "USE_CUDA=1 to enable GPU support.";
  }
#endif  // MXNET_USE_CUDA
  return ctx;
}

// Set the shape, dtype and storage type
void SetShapeType(const Context& ctx,
                  const nnvm::NodeAttrs& attrs,
                  const std::vector<NDArray*>& inputs,
                  const std::vector<NDArray*>& outputs) {
  static auto& infershape = nnvm::Op::GetAttr<nnvm::FInferShape>("FInferShape");
  static auto& infertype = nnvm::Op::GetAttr<nnvm::FInferType>("FInferType");
  static auto& inferstorage = nnvm::Op::GetAttr<FInferStorageType>("FInferStorageType");
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  // infer shape
  std::vector<TShape>& in_shapes  = ret->arg_shapes;
  in_shapes.clear();
  in_shapes.reserve(inputs.size());
  for (auto& i : inputs) {
    in_shapes.push_back(i->shape());
  }
  std::vector<TShape>& out_shapes = ret->out_shapes;
  out_shapes.clear();
  out_shapes.reserve(outputs.size());
  for (auto& i : outputs) {
    out_shapes.push_back(i->shape());
  }
  CHECK(infershape.count(attrs.op))
    << "Operator " << attrs.op->name << " is missing FInferShape attribute";
  CHECK(infershape[attrs.op](attrs, &in_shapes, &out_shapes));
  CHECK_EQ(out_shapes.size(), outputs.size());

  // infer type
  std::vector<int>& in_types = ret->arg_types;
  in_types.clear();
  in_types.reserve(inputs.size());
  for (auto& i : inputs) {
    in_types.push_back(i->dtype());
  }
  std::vector<int>& out_types = ret->out_types;
  out_types.clear();
  out_types.reserve(outputs.size());
  for (auto& i : outputs) {
    out_types.push_back(i->dtype());
  }
  CHECK(infertype.count(attrs.op))
    << "Operator " << attrs.op->name << " is missing FInferType attribute";
  CHECK(infertype[attrs.op](attrs, &in_types, &out_types));
  CHECK_EQ(out_types.size(), outputs.size());

  // infer storage type
  auto& in_storage_types = ret->arg_storage_types;
  in_storage_types.clear();
  in_storage_types.reserve(inputs.size());
  for (auto& i : inputs) {
    in_storage_types.push_back(i->storage_type());
  }
  auto& out_storage_types = ret->out_storage_types;
  out_storage_types.clear();
  out_storage_types.reserve(outputs.size());
  for (auto& i : outputs) {
    out_storage_types.push_back(i->storage_type());
  }
  if (inferstorage.count(attrs.op)) {
    CHECK(inferstorage[attrs.op](attrs, ctx, &in_storage_types, &out_storage_types));
    CHECK_EQ(out_storage_types.size(), outputs.size());
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    NDArrayStorageType storage_type = static_cast<NDArrayStorageType>(out_storage_types[i]);
    if (outputs[i]->is_none()) {
      // if failed to infer the storage type, assume the output storage is dense
      if (storage_type == kDefaultStorage || out_storage_types[i] == kUndefinedStorage) {
        *outputs[i] = NDArray(out_shapes[i], ctx, true, out_types[i]);
      } else {
        *outputs[i] = NDArray(storage_type, out_shapes[i], ctx, true, out_types[i]);
      }
    } else {
      CHECK_EQ(outputs[i]->shape(), out_shapes[i])
        << i << "-th output has invalid shape. "
        << "Expecting " << out_shapes[i] << " got "
        << outputs[i]->shape() << " in operator " << attrs.op->name;
      CHECK_EQ(outputs[i]->dtype(), out_types[i])
        << i << "-th output has invalid shape. "
        << "Expecting " << out_types[i] << " got "
        << outputs[i]->dtype()  << " in operator " << attrs.op->name;
    }
  }
}

void SetDependency(const nnvm::NodeAttrs& attrs,
                   const Context& ctx,
                   const std::vector<NDArray*>& inputs,
                   const std::vector<NDArray*>& outputs,
                   std::vector<engine::VarHandle> *p_read_vars,
                   std::vector<engine::VarHandle> *p_write_vars,
                   std::vector<Resource> *p_requested,
                   std::vector<uint32_t> *p_mutate_idx) {
  static auto& fmutate = nnvm::Op::GetAttr<nnvm::FMutateInputs>("FMutateInputs");
  static auto& ftmp_resource = nnvm::Op::GetAttr<FResourceRequest>("FResourceRequest");

  std::vector<engine::VarHandle>& read_vars  = *p_read_vars;
  std::vector<engine::VarHandle>& write_vars = *p_write_vars;
  std::vector<Resource>& requested = *p_requested;
  std::vector<uint32_t>& mutate_idx = *p_mutate_idx;

  if (ftmp_resource.count(attrs.op)) {
    int ntmp = 0;
    auto resource_reqs = ftmp_resource[attrs.op](attrs);
    for (const auto& req : resource_reqs) {
      switch (req.type) {
       case ResourceRequest::kTempSpace:
        ++ntmp;
       case ResourceRequest::kRandom:
        requested.push_back(ResourceManager::Get()->Request(ctx, req));
        write_vars.push_back(requested.back().var);
        break;
       default:
        LOG(FATAL) << "resource type not yet supported";
      }
    }
    CHECK_LE(ntmp, 1) << "Only support 1 temp space request";
  }

  read_vars.reserve(inputs.size());
  for (auto& i : inputs) {
    read_vars.push_back(i->var());
  }
  write_vars.reserve(outputs.size());
  for (auto& i : outputs) {
    write_vars.push_back(i->var());
  }
  if (fmutate.count(attrs.op)) {
    mutate_idx = fmutate[attrs.op](attrs);
    for (auto & i : mutate_idx) {
      write_vars.push_back(inputs[i]->var());
    }
  }
  Engine::Get()->DeduplicateVarHandle(&read_vars, &write_vars);
}

inline void SetWriteInplaceReq(const std::vector<NDArray*>& inputs,
                               const std::vector<NDArray*>& outputs,
                               std::vector<OpReqType> *req) {
  std::unordered_set<engine::VarHandle> in_vars;
  in_vars.reserve(inputs.size());
  for (auto &i : inputs) {
    in_vars.insert(i->var());
  }
  req->clear();
  req->resize(outputs.size(), kWriteTo);
  for (size_t i = 0; i < outputs.size(); i++) {
    // output NDArray shares the memory with the input NDArray
    if (in_vars.find(outputs[i]->var()) != in_vars.end()) {
      req->at(i) = kWriteInplace;
    }
  }
}

inline void DerefInputOutput(const std::vector<NDArray*>& inputs,
                             const std::vector<NDArray*>& outputs,
                             std::vector<NDArray>* p_inputs,
                             std::vector<NDArray>* p_outputs) {
  p_inputs->reserve(inputs.size());
  p_outputs->reserve(outputs.size());
  for (NDArray* i : inputs) p_inputs->emplace_back(*i);
  for (NDArray* i : outputs) p_outputs->emplace_back(*i);
}

void PushFCompute(const FCompute& fn,
                  const nnvm::Op* op,
                  const nnvm::NodeAttrs& attrs,
                  const Context& ctx,
                  const std::vector<engine::VarHandle>& read_vars,
                  const std::vector<engine::VarHandle>& write_vars,
                  const std::vector<Resource>& requested,
                  const std::vector<NDArray*>& p_inputs,
                  const std::vector<NDArray*>& p_outputs,
                  const std::vector<uint32_t>& mutate_idx,
                  const std::vector<OpReqType>& req) {
  using namespace common;
  bool is_train = ImperativeRuntime::Get()->is_training();
  std::vector<NDArray> inputs, outputs;
  DerefInputOutput(p_inputs, p_outputs, &inputs, &outputs);
  Engine::Get()->PushAsync(
    [ctx, attrs, fn, inputs, outputs, requested, is_train, mutate_idx, req](
        RunContext rctx,
        engine::CallbackOnComplete on_complete) {
      std::vector<TBlob> input_blobs, output_blobs;
      // pre-fcompute and post-fcompute storage fallback src NDArrays and dst NDArrays
      std::vector<NDArray> pre_temp_src, pre_temp_dst, post_temp_dst, post_temp_src;
      // mapping from index in input_blobs to index in pre_temp_dst
      std::unordered_map<uint32_t, uint32_t> in_temp_idx_map;
      // populate input blobs and output blobs
      SetupDefaultBlobs(inputs, &input_blobs, &pre_temp_src, &pre_temp_dst, &in_temp_idx_map);
      SetupDefaultBlobs(outputs, &output_blobs, &post_temp_dst, &post_temp_src);
      // add mutable inputs to post temp list
      for (const auto idx : mutate_idx) {
        auto map_iter = in_temp_idx_map.find(idx);
        if (map_iter != in_temp_idx_map.end()) {
          post_temp_src.push_back(pre_temp_dst[map_iter->second]);
          post_temp_dst.push_back(inputs[idx]);
        }
      }
      OpContext opctx{is_train, rctx,
                      engine::CallbackOnComplete(),
                      requested};
      if (ctx.dev_mask() == gpu::kDevMask) {
#if MXNET_USE_CUDA
        CastNonDefaultStorage<gpu>(pre_temp_src, pre_temp_dst, opctx);
        fn(attrs, opctx, input_blobs, req, output_blobs);
        // cast to original storage type, if necessary
        CastNonDefaultStorage<gpu>(post_temp_src, post_temp_dst, opctx);
        rctx.get_stream<gpu>()->Wait();
#else
        LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
#endif
      } else {
        CastNonDefaultStorage<cpu>(pre_temp_src, pre_temp_dst, opctx);
        fn(attrs, opctx, input_blobs, req, output_blobs);
        // cast to original storage type, if necessary
        CastNonDefaultStorage<cpu>(post_temp_src, post_temp_dst, opctx);
      }
      on_complete();
    }, ctx, read_vars, write_vars, FnProperty::kNormal,
    0, PROFILER_MESSAGE(op->name.c_str()));
}

void PushFComputeEx(const FComputeEx& fn,
                    const nnvm::Op* op,
                    const nnvm::NodeAttrs& attrs,
                    const Context& ctx,
                    const std::vector<engine::VarHandle>& read_vars,
                    const std::vector<engine::VarHandle>& write_vars,
                    const std::vector<Resource>& requested,
                    const std::vector<NDArray*>& p_inputs,
                    const std::vector<NDArray*>& p_outputs,
                    const std::vector<OpReqType>& req) {
  bool is_train = ImperativeRuntime::Get()->is_training();
  std::vector<NDArray> inputs, outputs;
  DerefInputOutput(p_inputs, p_outputs, &inputs, &outputs);
  Engine::Get()->PushAsync([ctx, is_train, attrs, fn, inputs, outputs, requested, req](
        RunContext rctx,
        engine::CallbackOnComplete on_complete) {
      std::vector<TBlob> input_blobs, output_blobs;
      OpContext opctx{is_train, rctx,
                      engine::CallbackOnComplete(),
                      requested};
      fn(attrs, opctx, inputs, req, outputs);
      if (ctx.dev_mask() == gpu::kDevMask) {
        rctx.get_stream<gpu>()->Wait();
      }
      on_complete();
    }, ctx, read_vars, write_vars, FnProperty::kNormal,
    0, PROFILER_MESSAGE(op->name.c_str()));
}

void PushOperator(const OpStatePtr& state,
                  const nnvm::Op* op,
                  const nnvm::NodeAttrs& attrs,
                  const Context& ctx,
                  const std::vector<engine::VarHandle>& read_vars,
                  const std::vector<engine::VarHandle>& write_vars,
                  const std::vector<Resource>& requested,
                  const std::vector<NDArray*>& p_inputs,
                  const std::vector<NDArray*>& p_outputs,
                  const std::vector<uint32_t>& mutate_idx,
                  const std::vector<OpReqType>& req) {
  using namespace common;
  static auto& fexec_type = nnvm::Op::GetAttr<FExecType>("FExecType");

  bool is_train = ImperativeRuntime::Get()->is_training();
  ExecType exec_type = ExecType::kSync;
  if (fexec_type.count(op)) {
    exec_type = fexec_type[op](attrs);
  }
  std::vector<NDArray> inputs, outputs;
  DerefInputOutput(p_inputs, p_outputs, &inputs, &outputs);

  auto fcompute = common::GetFCompute<FStatefulCompute>(op, "FStatefulCompute", ctx);
  if (fcompute != nullptr) {
    CHECK(exec_type == ExecType::kSync || exec_type == ExecType::kAsync);
    Engine::Get()->PushAsync(
      [state, fcompute, inputs, outputs, requested, is_train, exec_type, mutate_idx, req](
          RunContext rctx,
          engine::CallbackOnComplete on_complete) {
        OpContext opctx{is_train, rctx, on_complete, requested};

        std::vector<TBlob> input_blobs, output_blobs;
        // pre-fcompute and post-fcompute storage fallback src NDArrays and dst NDArrays
        std::vector<NDArray> pre_temp_src, pre_temp_dst, post_temp_dst, post_temp_src;
        // mapping from index in input_blobs to index in pre_temp_dst
        std::unordered_map<uint32_t, uint32_t> in_temp_idx_map;
        // populate input blobs and output blobs
        SetupDefaultBlobs(inputs, &input_blobs, &pre_temp_src, &pre_temp_dst, &in_temp_idx_map);
        SetupDefaultBlobs(outputs, &output_blobs, &post_temp_dst, &post_temp_src);
        // add mutable inputs to post temp list
        for (const auto idx : mutate_idx) {
          if (in_temp_idx_map.find(idx) != in_temp_idx_map.end()) {
            post_temp_src.push_back(pre_temp_dst[in_temp_idx_map[idx]]);
            post_temp_dst.push_back(inputs[idx]);
          }
        }
        if (rctx.get_ctx().dev_mask() == gpu::kDevMask) {
#if MXNET_USE_CUDA
          CastNonDefaultStorage<gpu>(pre_temp_src, pre_temp_dst, opctx);
          fcompute(state, opctx, input_blobs, req, output_blobs);
          CastNonDefaultStorage<gpu>(post_temp_src, post_temp_dst, opctx);
#else
          LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
#endif
        } else {
          CastNonDefaultStorage<cpu>(pre_temp_src, pre_temp_dst, opctx);
          fcompute(state, opctx, input_blobs, req, output_blobs);
          CastNonDefaultStorage<cpu>(post_temp_src, post_temp_dst, opctx);
        }
        if (exec_type == ExecType::kSync) {
          if (rctx.get_ctx().dev_mask() == gpu::kDevMask) {
            rctx.get_stream<gpu>()->Wait();
          }
          on_complete();
        }
      }, ctx, read_vars, write_vars, FnProperty::kNormal,
      0, PROFILER_MESSAGE(op->name.c_str()));
  } else {
    auto fcompute_ex = common::GetFCompute<FStatefulComputeEx>(
        op, "FStatefulComputeEx", ctx);
    CHECK(fcompute_ex != nullptr)
        << "One of FStatefulCompute and FStatefulComputeEx must be registered "
        << "for stateful operator " << op->name;
    const auto& run = [state, fcompute_ex, inputs, outputs, requested, is_train,
                       exec_type, req](
          RunContext rctx,
          engine::CallbackOnComplete on_complete) {
        OpContext opctx{is_train, rctx, on_complete, requested};
        fcompute_ex(state, opctx, inputs, req, outputs);
        if (exec_type == ExecType::kSync) {
          if (rctx.get_ctx().dev_mask() == gpu::kDevMask) {
            rctx.get_stream<gpu>()->Wait();
          }
          on_complete();
        }
      };
    if (exec_type == ExecType::kLocal) {
      run(RunContext{ctx, nullptr}, engine::CallbackOnComplete());
    } else {
      Engine::Get()->PushAsync(run, ctx, read_vars, write_vars, FnProperty::kNormal,
                               0, PROFILER_MESSAGE(op->name.c_str()));
    }
  }
}

nnvm::Graph CheckAndInferShape(nnvm::Graph&& g, nnvm::ShapeVector&& shapes,
                               bool use_inputs,
                               std::pair<size_t, size_t> node_range = {0, 0},
                               std::pair<size_t, size_t> entry_range = {0, 0}) {
  using namespace nnvm;
  if (use_inputs) {
    if (g.attrs.count("shape_inputs") &&
        g.GetAttr<ShapeVector>("shape_inputs") == shapes) return g;
  } else if (g.attrs.count("shape")) {
    const auto& prev_shapes = g.GetAttr<ShapeVector>("shape");
    CHECK_EQ(prev_shapes.size(), shapes.size());
    bool match = true;
    for (size_t i = 0; i < shapes.size(); ++i) {
      if (i == entry_range.first) {
        i = entry_range.second;
        if (i >= shapes.size()) break;
      }
      if (shapes[i] == prev_shapes[i]) continue;
      match = false;
      break;
    }
    if (match) return g;
  }
  g.attrs.erase("shape");
  g.attrs.erase("shape_inputs");
  if (node_range.second > node_range.first) {
    g.attrs["node_range"] = std::make_shared<dmlc::any>(node_range);
  }
  if (node_range.second > node_range.first) {
    g.attrs["node_range"] = std::make_shared<dmlc::any>(node_range);
  }
  if (use_inputs) {
    g = exec::InferShape(std::move(g), std::move(shapes));
  } else {
    g.attrs["shape"] = std::make_shared<dmlc::any>(std::move(shapes));
    g = exec::InferShape(std::move(g));
  }
  CHECK_EQ(g.GetAttr<size_t>("shape_num_unknown_nodes"), 0U);

  return g;
}


nnvm::Graph CheckAndInferType(nnvm::Graph&& g, nnvm::DTypeVector&& dtypes,
                              bool use_inputs,
                              std::pair<size_t, size_t> node_range = {0, 0},
                              std::pair<size_t, size_t> entry_range = {0, 0}) {
  using namespace nnvm;
  if (use_inputs) {
    if (g.attrs.count("dtype_inputs") &&
        g.GetAttr<DTypeVector>("dtype_inputs") == dtypes) return g;
  } else if (g.attrs.count("dtype")) {
    const auto& prev_dtypes = g.GetAttr<DTypeVector>("dtype");
    CHECK_EQ(prev_dtypes.size(), dtypes.size());
    bool match = true;
    for (size_t i = 0; i < dtypes.size(); ++i) {
      if (i == entry_range.first) {
        i = entry_range.second;
        if (i >= dtypes.size()) break;
      }
      if (dtypes[i] == prev_dtypes[i]) continue;
      match = false;
      break;
    }
    if (match) return g;
  }
  g.attrs.erase("dtype");
  g.attrs.erase("dtype_inputs");
  if (node_range.second > node_range.first) {
    g.attrs["node_range"] = std::make_shared<dmlc::any>(node_range);
  }
  if (node_range.second > node_range.first) {
    g.attrs["node_range"] = std::make_shared<dmlc::any>(node_range);
  }
  if (use_inputs) {
    g = exec::InferType(std::move(g), std::move(dtypes));
  } else {
    g.attrs["dtype"] = std::make_shared<dmlc::any>(std::move(dtypes));
    g = exec::InferType(std::move(g));
  }
  CHECK_EQ(g.GetAttr<size_t>("dtype_num_unknown_nodes"), 0U);

  return g;
}


nnvm::Graph CheckAndInferStorageType(const Context& ctx, nnvm::Graph&& g,
                                     StorageTypeVector&& storage_types, bool use_inputs,
                                     std::pair<size_t, size_t> node_range = {0, 0},
                                     std::pair<size_t, size_t> entry_range = {0, 0}) {
  using namespace nnvm;
  bool ctx_match = false;
  if (g.attrs.count("context")) {
    const auto& prev_vctx = g.GetAttr<exec::ContextVector>("context");
    if (prev_vctx.size() && prev_vctx[0].dev_mask() == ctx.dev_mask()) ctx_match = true;
  }
  if (!ctx_match) {
    exec::ContextVector vctx(g.indexed_graph().num_nodes(), ctx);
    g.attrs["context"] = std::make_shared<dmlc::any>(std::move(vctx));
  }

  if (ctx_match && use_inputs) {
    if (g.attrs.count("storage_type_inputs") &&
        g.GetAttr<StorageTypeVector>("storage_type_inputs") == storage_types) return g;
  } else if (ctx_match && g.attrs.count("storage_type")) {
    const auto& prev_storage_types = g.GetAttr<StorageTypeVector>("storage_type");
    CHECK_EQ(prev_storage_types.size(), storage_types.size());
    bool match = true;
    for (size_t i = 0; i < storage_types.size(); ++i) {
      if (i == entry_range.first) {
        i = entry_range.second;
        if (i >= storage_types.size()) break;
      }
      if (storage_types[i] == prev_storage_types[i]) continue;
      match = false;
      break;
    }
    if (match) return g;
  }
  g.attrs.erase("storage_type");
  g.attrs.erase("storage_type_inputs");
  if (node_range.second > node_range.first) {
    g.attrs["node_range"] = std::make_shared<dmlc::any>(node_range);
  }
  if (node_range.second > node_range.first) {
    g.attrs["node_range"] = std::make_shared<dmlc::any>(node_range);
  }
  if (use_inputs) {
    g = exec::InferStorageType(std::move(g), std::move(storage_types));
  } else {
    g.attrs["storage_type"] = std::make_shared<dmlc::any>(std::move(storage_types));
    g = exec::InferStorageType(std::move(g));
  }
  CHECK_EQ(g.GetAttr<size_t>("storage_type_num_unknown_nodes"), 0U);

  return g;
}

}  // namespace imperative

OpStatePtr ImperativeRuntime::InvokeOp(
    const Context& ctx,
    const nnvm::NodeAttrs& attrs,
    const std::vector<NDArray*>& inputs,
    const std::vector<NDArray*>& outputs,
    const std::vector<OpReqType>& req,
    OpStatePtr state) {
  using namespace imperative;
  static auto& createop = nnvm::Op::GetAttr<FCreateOpState>("FCreateOpState");
  static auto& is_layer_backward = Op::GetAttr<bool>("TIsLayerOpBackward");
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();

  const nnvm::Op *op = attrs.op;

  std::vector<engine::VarHandle> read_vars, write_vars;
  std::vector<Resource> requested;
  std::vector<uint32_t> mutate_idx;
  SetDependency(attrs, ctx, inputs, outputs,
      &read_vars, &write_vars, &requested, &mutate_idx);

  FCompute fn = common::GetFCompute<FCompute>(op, "FCompute", ctx);
  FComputeEx fn_ex = common::GetFCompute<FComputeEx>(op, "FComputeEx", ctx);

  bool has_non_default = common::ContainsNonDefaultStorage(inputs) ||
                         common::ContainsNonDefaultStorage(outputs);
  if (fn && (!has_non_default || !fn_ex)) {
    PushFCompute(fn, op, attrs, ctx, read_vars, write_vars,
        requested, inputs, outputs, mutate_idx, req);
  } else if (fn_ex) {
    PushFComputeEx(fn_ex, op, attrs, ctx, read_vars, write_vars,
        requested, inputs, outputs, req);
  } else if (createop.count(op) || is_layer_backward.get(op, false)) {
    if (!state) {
      state = createop[op](attrs, ctx, ret->arg_shapes, ret->arg_types);
    }
    write_vars.push_back(state.get_var());
    PushOperator(state, op, attrs, ctx, read_vars, write_vars,
        requested, inputs, outputs, mutate_idx, req);
  } else {
    LOG(FATAL)
      << "Operator " << op->name << " is not implemented for "
      << (ctx.dev_mask() == gpu::kDevMask ? "GPU." : "CPU.");
  }

  return state;
}

OpStatePtr ImperativeRuntime::Invoke(
    const Context& default_ctx,
    const nnvm::NodeAttrs& attrs,
    const std::vector<NDArray*>& inputs,
    const std::vector<NDArray*>& outputs) {
  using namespace imperative;
  static auto& ndfunc = nnvm::Op::GetAttr<FNDArrayFunction>("FNDArrayFunction");

  if (ndfunc.count(attrs.op)) {
    std::vector<NDArray> p_inputs, p_outputs;
    DerefInputOutput(inputs, outputs, &p_inputs, &p_outputs);
    ndfunc[attrs.op](attrs, p_inputs, &p_outputs);
    for (size_t i = 0; i < outputs.size(); ++i) *outputs[i] = std::move(p_outputs[i]);
    return OpStatePtr();
  }

  // TODO(piiswrong): infer ctx
  Context ctx = GetContext(attrs, inputs, outputs, default_ctx);
  SetShapeType(ctx, attrs, inputs, outputs);
  std::vector<OpReqType> req;
  SetWriteInplaceReq(inputs, outputs, &req);

  return InvokeOp(ctx, attrs, inputs, outputs, req);
}

void ImperativeRuntime::MarkVariables(
    const std::vector<NDArray*>& variables,
    const std::vector<mx_uint>& grad_reqs,
    const std::vector<NDArray*>& gradients) {
  for (uint32_t i = 0; i < variables.size(); ++i) {
    std::string str_c(std::to_string(variable_count_++));

    variables[i]->entry_ = nnvm::NodeEntry{
        nnvm::Symbol::CreateVariable("var" + str_c).outputs[0].node, 0, 0};
    AGInfo& info = AGInfo::Create(variables[i]->entry_.node);
    info.outputs.emplace_back(variables[i]->Detach());
    info.out_grads.emplace_back(gradients[i]->Detach());
    info.grad_req = static_cast<OpReqType>(grad_reqs[i]);

    gradients[i]->entry_ = nnvm::NodeEntry{
        nnvm::Symbol::CreateVariable("grad" + str_c).outputs[0].node, 0, 0};
    AGInfo& grad_info = AGInfo::Create(gradients[i]->entry_.node);
    grad_info.outputs.emplace_back(gradients[i]->Detach());
  }
}


void ImperativeRuntime::GetBackwardDependency(
    const nnvm::NodePtr& node,
    uint32_t num_inputs, uint32_t num_outputs,
    std::vector<bool> *p_save_inputs,
    std::vector<bool> *p_save_outputs) {
  static auto& fgradient = nnvm::Op::GetAttr<nnvm::FGradient>("FGradient");
  std::vector<bool>& save_inputs = *p_save_inputs;
  std::vector<bool>& save_outputs = *p_save_outputs;
  save_inputs.resize(num_inputs);
  save_outputs.resize(num_outputs);
  std::fill(save_inputs.begin(), save_inputs.end(), false);
  std::fill(save_outputs.begin(), save_outputs.end(), false);

  node->inputs.clear();
  node->inputs.reserve(num_inputs);
  for (uint32_t i = 0; i < num_inputs; ++i) {
    node->inputs.emplace_back(nnvm::NodeEntry{nullptr, i, 0});
  }

  if (fgradient.count(node->op())) {
    std::vector<nnvm::NodeEntry> ograd_entries;
    ograd_entries.reserve(num_outputs);
    for (uint32_t i = 0; i < num_outputs; ++i) {
      ograd_entries.emplace_back(nnvm::NodeEntry{nullptr, i, 1});
    }
    auto igrad_entries = fgradient[node->op()](node, ograd_entries);
    for (const auto& i : igrad_entries) {
      if (i.node == nullptr && i.version == 0) {
        save_inputs[i.index] = true;
      } else if (i.node == node) {
        save_outputs[i.index] = true;
      }
    }
    DFSVisit(igrad_entries, [&](const nnvm::NodePtr& gnode) {
        if (!gnode || gnode == node) return;
        for (const auto& i : gnode->inputs) {
          if (i.node == nullptr && i.version == 0) {
            save_inputs[i.index] = true;
          } else if (i.node == node) {
            save_outputs[i.index] = true;
          }
        }
      });
  }
}

void ImperativeRuntime::RecordOp(
    nnvm::NodeAttrs&& attrs,
    const std::vector<NDArray*>& inputs,
    const std::vector<NDArray*>& outputs,
    const OpStatePtr& state,
    std::vector<bool>* p_save_inputs,
    std::vector<bool>* p_save_outputs) {
  MXAPIThreadLocalEntry *local_buff = MXAPIThreadLocalStore::Get();

  for (uint32_t i = 0; i < outputs.size(); ++i) {
    CHECK(AGInfo::IsNone(*(outputs[i])))
      << "Assigning to NDArrays that are already in a computational graph "
      << "will cause undefined behavior when evaluating gradients. "
      << "Please call backward first to clear the graph or do this out side of "
      << "a record section. ";
  }

  bool need_grad = false;
  for (const auto& i : inputs) {
    if (AGInfo::IsNone(*i)) continue;
    need_grad = true;
    break;
  }
  if (!need_grad) return;

  nnvm::NodePtr node = nnvm::Node::Create();
  node->attrs = std::move(attrs);
  node->attrs.name = "node_" + std::to_string(node_count_++);
  AGInfo& info = AGInfo::Create(node);
  info.state = state;

  if (p_save_inputs == nullptr) {
    p_save_inputs = &(local_buff->save_inputs);
    p_save_outputs = &(local_buff->save_outputs);
    GetBackwardDependency(
        node, inputs.size(), outputs.size(), p_save_inputs, p_save_outputs);
  } else {
    node->inputs.resize(inputs.size());
  }

  std::vector<bool>& save_inputs = *p_save_inputs;
  std::vector<bool>& save_outputs = *p_save_outputs;

  for (size_t i = 0; i < inputs.size(); ++i) {
    if (AGInfo::IsNone(*(inputs[i]))) {
      nnvm::NodeEntry entry{nnvm::Symbol::CreateVariable(
          "null" + std::to_string(variable_count_++)).outputs[0].node, 0, 0};
      AGInfo& input_info = AGInfo::Create(entry.node);
      if (save_inputs[i]) {
        input_info.outputs.emplace_back(*inputs[i]);
      } else {
        // Put a dummy array here since it will not be used.
        input_info.outputs.emplace_back();
        input_info.outputs.back().shape_ = inputs[i]->shape();
        input_info.outputs.back().dtype_ = inputs[i]->dtype();
        input_info.outputs.back().storage_type_ = inputs[i]->storage_type();
      }
      inputs[i]->entry_ = std::move(entry);  // assign last to prevent cyclic reference
    } else if (save_inputs[i]) {
      AGInfo::Get(inputs[i]->entry_.node).outputs[inputs[i]->entry_.index] = inputs[i]->Detach();
    }
    node->inputs[i] = inputs[i]->entry_;
  }

  for (uint32_t i = 0; i < outputs.size(); ++i) {
    CHECK(AGInfo::IsNone(*(outputs[i])))
      << "Inplace operations (+=, -=, x[:]=, etc) are not supported when "
      << "recording with autograd.";
  }

  for (uint32_t i = 0; i < outputs.size(); ++i) {
    if (save_outputs[i]) {
      info.outputs.emplace_back(outputs[i]->Detach());
    } else {
      // Put a dummy array here since it will not be used.
      info.outputs.emplace_back();
      info.outputs.back().shape_ = outputs[i]->shape();
      info.outputs.back().dtype_ = outputs[i]->dtype();
      info.outputs.back().storage_type_ = outputs[i]->storage_type();
    }
    outputs[i]->entry_ = nnvm::NodeEntry{node, i, 0};
  }
}

void ImperativeRuntime::RunGraph(
    const Context& default_ctx,
    const nnvm::IndexedGraph& idx,
    const std::vector<NDArray*> arrays,
    size_t node_start, size_t node_end,
    std::vector<OpReqType>&& array_reqs,
    std::vector<int>&& ref_count,
    std::vector<OpStatePtr> *p_states) {
  using namespace nnvm;
  using namespace imperative;
  static auto& createop = nnvm::Op::GetAttr<FCreateOpState>("FCreateOpState");
  static auto& is_layer_backward = Op::GetAttr<bool>("TIsLayerOpBackward");
  static const auto bwd_cached_op = Op::Get("_backward_CachedOp");

  std::vector<OpStatePtr>& states = *p_states;
  bool recording = is_recording();

  std::vector<NDArray*> ndinputs, ndoutputs;
  ShapeVector arg_shapes;
  DTypeVector arg_dtypes;
  std::vector<OpReqType> req;

  for (size_t i = node_start; i < node_end; ++i) {
    const nnvm::IndexedGraph::Node& node = idx[i];
    if (node.source->op() == nullptr) continue;
    ndinputs.clear();
    ndinputs.reserve(node.inputs.size());
    for (const auto& j : node.inputs) {
      ndinputs.emplace_back(arrays[idx.entry_id(j)]);
      CHECK(!ndinputs.back()->is_none()) << idx[j.node_id].source->attrs.name << " " << j.index;
    }
    ndoutputs.clear();
    ndoutputs.reserve(node.source->num_outputs());
    req.clear();
    req.reserve(node.source->num_outputs());
    for (size_t j = 0; j < node.source->num_outputs(); ++j) {
      size_t eid = idx.entry_id(i, j);
      ndoutputs.emplace_back(arrays[eid]);
      req.push_back(array_reqs[eid]);
      CHECK(!ndoutputs.back()->is_none());
    }

    if (node.source->op() == bwd_cached_op) {
      CHECK(!recording)
          << "CachedOp does not support higher order gradients. "
          << "If you want to do backward with create_graph please "
          << "do not use hybridize.";
      const auto& cached_op = dmlc::get<CachedOpPtr>(node.source->attrs.parsed);
      nnvm::Node* fwd_node = node.source->control_deps[0].get();
      auto fwd_node_id = idx.node_id(fwd_node);
      cached_op->Backward(states[fwd_node_id], ndinputs, req, ndoutputs);
    } else if (createop.count(node.source->op())) {
      arg_shapes.clear();
      arg_dtypes.clear();
      arg_shapes.reserve(ndinputs.size());
      arg_dtypes.reserve(ndinputs.size());
      for (size_t i = 0; i < ndinputs.size(); ++i) {
        arg_shapes.emplace_back(ndinputs[i]->shape());
        arg_dtypes.emplace_back(ndinputs[i]->dtype());
      }
      states[i] = createop[node.source->op()](
          node.source->attrs, default_ctx, arg_shapes, arg_dtypes);
      InvokeOp(default_ctx, node.source->attrs, ndinputs, ndoutputs, req, states[i]);
      if (recording) RecordOp(NodeAttrs(node.source->attrs), ndinputs, ndoutputs, states[i]);
    } else if (is_layer_backward.get(node.source->op(), false)) {
      nnvm::Node* fwd_node = node.source->control_deps[0].get();
      auto fwd_node_id = idx.node_id(fwd_node);
      InvokeOp(default_ctx, node.source->attrs, ndinputs, ndoutputs, req, states[fwd_node_id]);
      if (recording) {
        RecordOp(NodeAttrs(node.source->attrs), ndinputs, ndoutputs, states[fwd_node_id]);
      }
    } else {
      InvokeOp(default_ctx, node.source->attrs, ndinputs, ndoutputs, req);
      if (recording) RecordOp(NodeAttrs(node.source->attrs), ndinputs, ndoutputs);
    }

    for (const auto& j : node.inputs) {
      size_t eid = idx.entry_id(j);
      --ref_count[eid];
      if (ref_count[eid] == 0) arrays[eid]->ptr_.reset();
    }
    for (size_t j = 0; j < ndoutputs.size(); ++j) {
      size_t eid = idx.entry_id(i, j);
      if (ref_count[eid] == 0) arrays[eid]->ptr_.reset();
    }
  }
}


std::vector<NDArray*> ImperativeRuntime::Backward(
    const std::vector<NDArray*>& outputs,
    const std::vector<NDArray*>& ograds,
    const std::vector<NDArray*>& variables,
    bool is_train, bool retain_graph,
    bool create_graph) {
  using namespace nnvm;
  using namespace imperative;
  static const std::vector<const Op*> zero_ops{Op::Get("zeros_like"), Op::Get("_zeros")};

  // Construct forward graph
  Graph graph;
  graph.outputs.reserve(outputs.size());
  for (const auto& i : outputs) {
    CHECK(!AGInfo::IsNone(*i))
      << "Cannot differentiate node because it is not in a computational graph. "
      << "You need to set is_recording to true or use autograd.record() to save "
      << "computational graphs for backward. If you want to differentiate the same "
      << "graph twice, you need to pass retain_graph=True to backward.";
    graph.outputs.emplace_back(i->entry_);
  }
  size_t num_forward_outputs = graph.outputs.size();

  // Prepare head gradients
  std::vector<NodeEntry> ograd_entries;
  ograd_entries.reserve(ograds.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    ograd_entries.emplace_back(NodeEntry{Node::Create(), 0, 0});
    AGInfo& info = AGInfo::Create(ograd_entries.back().node);
    if (ograds[i] != nullptr) {
      info.outputs.emplace_back(*ograds[i]);
    } else {
      info.outputs.emplace_back(outputs[i]->shape(), outputs[i]->ctx(),
                                true, outputs[i]->dtype());
      info.outputs.back() = static_cast<real_t>(1.0);
    }
  }

  // Get gradient graph
  Symbol sym;
  sym.outputs = graph.outputs;
  std::vector<NodeEntry> xs;
  std::vector<NDArray*> x_grads;
  std::vector<OpReqType> x_reqs;
  if (variables.size()) {
    xs.reserve(variables.size());
    x_grads.reserve(variables.size());
    x_reqs.reserve(variables.size());
    for (size_t i = 0; i < variables.size(); ++i) {
      CHECK(!AGInfo::IsNone(*variables[i]) &&
            AGInfo::IsVariable(variables[i]->entry_.node))
          << "Cannot differentiate with respect to the " << i+1 << "-th variable"
          << " because it does not require gradient.";
      xs.emplace_back(variables[i]->entry_);
      x_grads.push_back(new NDArray());
      x_reqs.push_back(kWriteTo);
    }
  } else {
    std::vector<NodePtr> args = sym.ListInputs(Symbol::kReadOnlyArgs);
    xs.reserve(args.size());
    x_grads.reserve(args.size());
    x_reqs.reserve(args.size());
    for (const auto& i : args) {
      AGInfo& info = AGInfo::Get(i);
      if (info.grad_req == kNullOp) continue;
      xs.emplace_back(NodeEntry{i, 0, 0});
      x_grads.push_back(&info.out_grads[0]);
      x_reqs.push_back(info.grad_req);
      info.fresh_out_grad = true;
    }
    CHECK_GT(xs.size(), 0)
        << "There are no inputs in computation graph that require gradients.";
  }

  Graph g_graph = pass::Gradient(
      graph, graph.outputs, xs, ograd_entries,
      exec::AggregateGradient, false, nullptr,
      zero_ops, "_copy");
  CHECK_EQ(g_graph.outputs.size(), xs.size());
  for (const auto &e : g_graph.outputs) {
    graph.outputs.push_back(e);
  }
  const auto& idx = graph.indexed_graph();
  // get number of nodes used in forward pass
  size_t num_forward_nodes = 0;
  size_t num_forward_entries = 0;
  for (size_t i = 0; i < num_forward_outputs; ++i) {
    num_forward_nodes = std::max(
        num_forward_nodes, static_cast<size_t>(idx.outputs()[i].node_id + 1));
    num_forward_entries = std::max(
        num_forward_entries, static_cast<size_t>(idx.entry_id(idx.outputs()[i])) + 1);
  }

  // Allocate buffer
  std::vector<NDArray> buff(idx.num_node_entries());
  std::vector<int> ref_count(buff.size(), 0);
  std::vector<OpStatePtr> states;
  std::vector<NDArray*> arrays;
  arrays.reserve(buff.size());
  for (size_t i = 0; i < buff.size(); ++i) arrays.push_back(&buff[i]);
  if (create_graph) {
    states.resize(num_forward_nodes);
    nnvm::DFSVisit(sym.outputs, [&](const nnvm::NodePtr& n) {
      AGInfo& info = AGInfo::Get(n);
      states[idx.node_id(n.get())] = info.state;
      for (uint32_t i = 0; i < info.outputs.size(); ++i) {
        CHECK(idx.exist(n.get()));
        size_t nid = idx.node_id(n.get());
        size_t eid = idx.entry_id(nid, i);
        buff[eid] = info.outputs[i];
        buff[eid].entry_ = NodeEntry{n, i, 0};
        ref_count[eid] = 1;
      }
    });
    for (size_t i = 0; i < ograd_entries.size(); ++i) {
      AGInfo& info = AGInfo::Get(ograd_entries[i].node);
      if (!idx.exist(ograd_entries[i].node.get())) continue;
      size_t eid = idx.entry_id(ograd_entries[i]);
      buff[eid] = info.outputs[0];
      buff[eid].entry_ = ograd_entries[i];
    }
  } else {
    states.reserve(num_forward_nodes);
    for (size_t i = 0; i < num_forward_nodes; ++i) {
      const AGInfo& info = dmlc::get<AGInfo>(idx[i].source->info);
      states.emplace_back(info.state);
      for (size_t j = 0; j < info.outputs.size(); ++j) {
        size_t eid = idx.entry_id(i, j);
        arrays[eid] = const_cast<NDArray*>(&(info.outputs[j]));
        if (retain_graph || info.grad_req != kNullOp) ref_count[eid] = 1;
      }
    }
    for (size_t i = 0; i < ograd_entries.size(); ++i) {
      if (!idx.exist(ograd_entries[i].node.get())) continue;
      AGInfo& info = AGInfo::Get(ograd_entries[i].node);
      arrays[idx.entry_id(ograd_entries[i])] = &info.outputs[0];
    }
  }
  for (size_t i = num_forward_outputs; i < graph.outputs.size(); ++i) {
    size_t eid = idx.entry_id(graph.outputs[i]);
    arrays[eid] = x_grads[i - num_forward_outputs];
    ref_count[eid] = 1;
  }

  // Assign context
  Context default_ctx = outputs[0]->ctx();

  // Infer shape type
  {
    std::pair<uint32_t, uint32_t> node_range, entry_range;
    node_range = {num_forward_nodes, idx.num_nodes()};
    entry_range = {num_forward_entries, idx.num_node_entries()};

    ShapeVector shapes;
    shapes.reserve(idx.num_node_entries());
    for (const auto& i : arrays) shapes.emplace_back(i->shape());
    graph = CheckAndInferShape(std::move(graph), std::move(shapes), false,
                               node_range, entry_range);

    DTypeVector dtypes;
    dtypes.reserve(idx.num_node_entries());
    for (const auto& i : arrays) dtypes.emplace_back(i->dtype());
    graph = CheckAndInferType(std::move(graph), std::move(dtypes), false,
                              node_range, entry_range);

    StorageTypeVector stypes;
    stypes.reserve(idx.num_node_entries());
    for (const auto& i : arrays) stypes.emplace_back(i->storage_type());
    graph = CheckAndInferStorageType(
        default_ctx, std::move(graph), std::move(stypes), false,
        node_range, entry_range);
  }

  const auto& shapes = graph.GetAttr<ShapeVector>("shape");
  const auto& dtypes = graph.GetAttr<DTypeVector>("dtype");
  const auto& stypes = graph.GetAttr<StorageTypeVector>("storage_type");
  for (size_t i = num_forward_entries; i < arrays.size(); ++i) {
    if (!arrays[i]->is_none()) continue;
    if (stypes[i] == kDefaultStorage) {
      *arrays[i] = NDArray(shapes[i], default_ctx, true, dtypes[i]);
    } else {
      *arrays[i] = NDArray(static_cast<NDArrayStorageType>(stypes[i]),
                           shapes[i], default_ctx, true, dtypes[i]);
    }
  }

  // Calculate ref count
  for (size_t i = num_forward_nodes; i < idx.num_nodes(); ++i) {
    for (const auto& j : idx[i].inputs) {
       ++ref_count[idx.entry_id(j)];
    }
  }

  // Assign reqs
  std::vector<OpReqType> array_reqs(arrays.size(), kWriteTo);
  for (size_t i = num_forward_entries; i < idx.num_node_entries(); ++i) {
    if (ref_count[i] == 0) array_reqs[i] = kNullOp;
  }
  for (size_t i = num_forward_outputs; i < idx.outputs().size(); ++i) {
    size_t eid = idx.entry_id(idx.outputs()[i]);
    array_reqs[eid] = x_reqs[i - num_forward_outputs];
  }

  // Execution

  bool prev_recording = set_is_recording(create_graph);
  bool prev_training = set_is_training(is_train);

  RunGraph(default_ctx, idx, arrays, num_forward_nodes, idx.num_nodes(),
           std::move(array_reqs), std::move(ref_count), &states);

  set_is_recording(prev_recording);
  set_is_training(prev_training);

  // Clear history
  if (!retain_graph) {
    nnvm::DFSVisit(sym.outputs, [&](const nnvm::NodePtr& n) {
      AGInfo::Clear(n);
      n->inputs.clear();
    });
  }

  if (variables.size()) {
    return x_grads;
  }
  return {};
}

ImperativeRuntime::CachedOp::CachedOp(const nnvm::Symbol& sym) {
  using namespace nnvm;
  using namespace imperative;
  static const std::vector<const Op*> zero_ops{Op::Get("zeros_like"), Op::Get("_zeros")};

  // construct forward graph
  {
    fwd_graph_.outputs = sym.outputs;
    const auto& idx = fwd_graph_.indexed_graph();
    CHECK_GE(idx.input_nodes().size(), 1) << "CachedOp requires at least 1 input";

    std::vector<int> ref_count(idx.num_node_entries(), 0);
    for (const auto& i : idx.input_nodes()) ++ref_count[idx.entry_id(i, 0)];
    for (const auto& i : idx.outputs()) ++ref_count[idx.entry_id(i)];
    for (size_t i = 0; i < idx.num_nodes(); ++i) {
      for (const auto& j : idx[i].inputs) ++ref_count[idx.entry_id(j)];
    }

    fwd_graph_.attrs["ref_count"] = std::make_shared<dmlc::any>(std::move(ref_count));
  }

  // construct backward graph
  std::vector<NodeEntry> ograd_entries;
  {
    ograd_entries.reserve(fwd_graph_.outputs.size());
    for (size_t i = 0; i < fwd_graph_.outputs.size(); ++i) {
      ograd_entries.emplace_back(NodeEntry{Node::Create(), 0, 0});
    }

    std::vector<NodeEntry> xs;
    std::vector<NodePtr> args = sym.ListInputs(Symbol::kReadOnlyArgs);
    xs.reserve(args.size());
    for (const auto& i : args) xs.emplace_back(NodeEntry{i, 0, 0});
    CHECK_GT(xs.size(), 0)
        << "There are no inputs in computation graph that require gradients.";

    grad_graph_ = pass::Gradient(
        fwd_graph_, fwd_graph_.outputs, xs, ograd_entries,
        exec::AggregateGradient, false, nullptr,
        zero_ops, "_copy");
  }

  // construct full graph
  {
    size_t num_forward_nodes = fwd_graph_.indexed_graph().num_nodes();
    size_t num_forward_entries = fwd_graph_.indexed_graph().num_node_entries();

    full_graph_.outputs = fwd_graph_.outputs;
    curr_grad_req_ = std::vector<bool>(grad_graph_.outputs.size(), true);
    for (const auto& i : grad_graph_.outputs) full_graph_.outputs.emplace_back(i);
    const auto& idx = full_graph_.indexed_graph();

    std::vector<int> ref_count(idx.num_node_entries(), 0);
    for (size_t i = num_forward_nodes; i < idx.num_nodes(); ++i) {
      for (const auto& j : idx[i].inputs) {
         ++ref_count[idx.entry_id(j)];
      }
    }

    auto ref_count_recording = fwd_graph_.GetAttr<std::vector<int> >("ref_count");
    for (size_t i = 0; i < num_forward_entries; ++i) ref_count_recording[i] += ref_count[i];
    fwd_graph_.attrs["ref_count_recording"] =
        std::make_shared<dmlc::any>(std::move(ref_count_recording));

    size_t num_forward_inputs = num_inputs();
    for (uint32_t i = 0; i < ograd_entries.size(); ++i) {
      if (!idx.exist(ograd_entries[i].node.get())) continue;
      auto eid = idx.entry_id(ograd_entries[i]);
      if (ref_count[eid] > 0) {
        bwd_ograd_dep_.push_back(i);
        bwd_input_eid_.push_back(eid);
      }
    }
    save_inputs_.resize(num_forward_inputs, false);
    for (uint32_t i = 0; i < num_forward_inputs; ++i) {
      auto eid = idx.entry_id(idx.input_nodes()[i], 0);
      if (ref_count[eid] > 0) {
        save_inputs_[i] = true;
        bwd_in_dep_.push_back(i);
        bwd_input_eid_.push_back(eid);
      }
    }
    save_outputs_.resize(idx.outputs().size(), false);
    for (uint32_t i = 0; i < idx.outputs().size(); ++i) {
      auto eid = idx.entry_id(idx.outputs()[i]);
      if (ref_count[eid] > 0) {
        save_outputs_[i] = true;
        bwd_out_dep_.push_back(i);
        bwd_input_eid_.push_back(eid);
      }
    }
  }
}

std::vector<nnvm::NodeEntry> ImperativeRuntime::CachedOp::Gradient(
    const nnvm::NodePtr& node,
    const std::vector<nnvm::NodeEntry>& ograds) {
  using namespace nnvm;
  static const auto _backward_CachedOp = Op::Get("_backward_CachedOp");
  static const auto _CachedOp_NoGrad = Op::Get("_CachedOp_NoGrad");

  auto p = Node::Create();
  p->attrs.op = _backward_CachedOp;
  p->attrs.name = node->attrs.name + "_backward";
  p->attrs.parsed = node->attrs.parsed;
  p->control_deps.push_back(node);
  p->inputs.reserve(bwd_ograd_dep_.size() + bwd_in_dep_.size() + bwd_out_dep_.size());
  for (auto i : bwd_ograd_dep_) p->inputs.push_back(ograds[i]);
  for (auto i : bwd_in_dep_) p->inputs.push_back(node->inputs[i]);
  for (auto i : bwd_out_dep_) p->inputs.emplace_back(NodeEntry{node, i, 0});
  std::vector<NodeEntry> ret;
  ret.reserve(num_inputs());
  const auto& auxs = mutable_input_nodes();
  if (auxs.size()) {
    auto nop = Node::Create();
    nop->attrs.op = _CachedOp_NoGrad;
    nop->attrs.parsed = static_cast<uint32_t>(auxs.size());
    nop->control_deps.push_back(node);
    uint32_t j = 0, k = 0;
    for (const auto& i : fwd_graph_.indexed_graph().input_nodes()) {
      if (auxs.count(i)) {
        ret.emplace_back(NodeEntry{nop, j++, 0});
      } else {
        ret.emplace_back(NodeEntry{p, k++, 0});
      }
    }
  } else {
    for (uint32_t i = 0; i < num_inputs(); ++i) ret.emplace_back(NodeEntry{p, i, 0});
  }
  return ret;
}

nnvm::Graph ImperativeRuntime::CachedOp::GetForwardGraph(
    const std::vector<NDArray*>& inputs) {
  using namespace nnvm;
  using namespace imperative;
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK_EQ(inputs.size(), num_inputs());
  nnvm::Graph& g = fwd_graph_;

  ShapeVector shape_inputs;
  DTypeVector dtype_inputs;
  StorageTypeVector storage_type_inputs;
  shape_inputs.reserve(inputs.size());
  dtype_inputs.reserve(inputs.size());
  storage_type_inputs.reserve(inputs.size());
  for (uint32_t i = 0; i < inputs.size(); ++i) {
    shape_inputs.emplace_back(inputs[i]->shape());
    dtype_inputs.emplace_back(inputs[i]->dtype());
    storage_type_inputs.emplace_back(inputs[i]->storage_type());
  }

  g = CheckAndInferShape(std::move(g), std::move(shape_inputs), true);
  g = CheckAndInferType(std::move(g), std::move(dtype_inputs), true);
  g = CheckAndInferStorageType(inputs[0]->ctx(), std::move(g),
                               std::move(storage_type_inputs), true);

  return g;
}

nnvm::Graph ImperativeRuntime::CachedOp::GetBackwardGraph(
    const OpStatePtr& op_state,
    const std::vector<OpReqType>& reqs,
    const std::vector<NDArray*>& inputs) {
  using namespace nnvm;
  using namespace imperative;
  std::lock_guard<std::mutex> lock(mutex_);
  nnvm::Graph& g = full_graph_;
  auto& state = op_state.get_state<CachedOpState>();
  bool req_match = true;
  for (size_t i = 0; i < reqs.size(); ++i) {
    if (curr_grad_req_[i] != (reqs[i] != kNullOp)) {
      curr_grad_req_[i] = reqs[i] != kNullOp;
      req_match = false;
    }
  }
  if (!req_match) {
    g = nnvm::Graph();
    g.outputs = fwd_graph_.outputs;
    for (size_t i = 0; i < grad_graph_.outputs.size(); ++i) {
      if (curr_grad_req_[i]) g.outputs.emplace_back(grad_graph_.outputs[i]);
    }
  }

  const auto& idx = g.indexed_graph();
  size_t num_forward_nodes = fwd_graph_.indexed_graph().num_nodes();
  size_t num_forward_entries = fwd_graph_.indexed_graph().num_node_entries();

  if (!g.attrs.count("ref_count")) {
    std::vector<int> ref_count(idx.num_node_entries(), 0);
    for (size_t i = num_forward_nodes; i < idx.num_nodes(); ++i) {
      for (const auto& j : idx[i].inputs) ++ref_count[idx.entry_id(j)];
    }
    for (size_t i = 0; i < inputs.size(); ++i) ++ref_count[bwd_input_eid_[i]];
    for (const auto& i : idx.outputs()) ++ref_count[idx.entry_id(i)];
    g.attrs["ref_count"] = std::make_shared<dmlc::any>(std::move(ref_count));
  }

  ShapeVector shapes(idx.num_node_entries(), TShape());
  DTypeVector dtypes(idx.num_node_entries(), -1);
  StorageTypeVector stypes(idx.num_node_entries(), -1);

  for (size_t i = 0; i < num_forward_entries; ++i) {
    shapes[i] = state.buff[i].shape();
    dtypes[i] = state.buff[i].dtype();
    stypes[i] = state.buff[i].storage_type();
  }

  std::pair<uint32_t, uint32_t> node_range, entry_range;
  node_range = {num_forward_nodes, idx.num_nodes()};
  entry_range = {num_forward_entries, idx.num_node_entries()};

  g = CheckAndInferShape(std::move(g), std::move(shapes), false,
                         node_range, entry_range);
  g = CheckAndInferType(std::move(g), std::move(dtypes), false,
                        node_range, entry_range);
  g = CheckAndInferStorageType(inputs[0]->ctx(), std::move(g), std::move(stypes),
                               false, node_range, entry_range);

  return g;
}

OpStatePtr ImperativeRuntime::CachedOp::Forward(const std::vector<NDArray*>& inputs,
                                                const std::vector<NDArray*>& outputs) {
  using namespace nnvm;
  // Initialize
  nnvm::Graph g = GetForwardGraph(inputs);
  const auto& idx = g.indexed_graph();
  size_t num_inputs = idx.input_nodes().size();

  auto op_state_ptr = OpStatePtr::Create<CachedOpState>();
  auto& cached_op_state = op_state_ptr.get_state<CachedOpState>();
  auto& buff = cached_op_state.buff;
  auto& states = cached_op_state.states;

  // Allocate entries
  states.resize(idx.num_nodes());
  buff.resize(idx.num_node_entries());
  states.reserve(idx.num_nodes());
  std::vector<NDArray*> arrays;
  arrays.reserve(buff.size());
  for (size_t i = 0; i < buff.size(); ++i) arrays.push_back(&buff[i]);
  for (size_t i = 0; i < num_inputs; ++i) {
    arrays[idx.entry_id(idx.input_nodes()[i], 0)] = inputs[i];
  }
  for (size_t i = 0; i < idx.outputs().size(); ++i) {
    arrays[idx.entry_id(idx.outputs()[i])] = outputs[i];
  }

  // Allocate NDArrays
  Context default_ctx = inputs[0]->ctx();
  const auto& shapes = g.GetAttr<ShapeVector>("shape");
  const auto& dtypes = g.GetAttr<DTypeVector>("dtype");
  const auto& stypes = g.GetAttr<StorageTypeVector>("storage_type");
  for (size_t i = 0; i < arrays.size(); ++i) {
    if (!arrays[i]->is_none()) continue;
    if (stypes[i] == kDefaultStorage) {
      *arrays[i] = NDArray(shapes[i], default_ctx, true, dtypes[i]);
    } else {
      *arrays[i] = NDArray(static_cast<NDArrayStorageType>(stypes[i]),
                           shapes[i], default_ctx, true, dtypes[i]);
    }
  }

  // Execution
  bool prev_recording = ImperativeRuntime::Get()->set_is_recording(false);

  std::vector<int> ref_count;
  if (prev_recording) {
    ref_count = g.GetAttr<std::vector<int> >("ref_count_recording");
  } else {
    ref_count = g.GetAttr<std::vector<int> >("ref_count");
  }

  std::vector<OpReqType> array_reqs(arrays.size(), kWriteTo);
  for (size_t i = 0; i < idx.num_node_entries(); ++i) {
    if (ref_count[i] == 0) array_reqs[i] = kNullOp;
  }

  ImperativeRuntime::Get()->RunGraph(
      default_ctx, idx, arrays, 0, idx.num_nodes(), std::move(array_reqs),
      std::move(ref_count), &states);

  for (size_t i = 0; i < idx.num_node_entries(); ++i) {
    if (arrays[i] == &buff[i]) continue;
    buff[i].shape_ = arrays[i]->shape_;
    buff[i].dtype_ = arrays[i]->dtype_;
    buff[i].storage_type_ = arrays[i]->storage_type_;
  }

  ImperativeRuntime::Get()->set_is_recording(prev_recording);

  return op_state_ptr;
}


void ImperativeRuntime::CachedOp::Backward(
    const OpStatePtr& state,
    const std::vector<NDArray*>& inputs,
    const std::vector<OpReqType>& reqs,
    const std::vector<NDArray*>& outputs) {
  using namespace nnvm;
  // Initialize
  nnvm::Graph g = GetBackwardGraph(state, reqs, inputs);
  const auto& idx = g.indexed_graph();

  auto& cached_op_state = state.get_state<CachedOpState>();
  auto& buff = cached_op_state.buff;
  auto& states = cached_op_state.states;

  size_t num_forward_outputs = fwd_graph_.outputs.size();
  size_t num_forward_nodes = fwd_graph_.indexed_graph().num_nodes();
  size_t num_forward_entries = fwd_graph_.indexed_graph().num_node_entries();
  buff.resize(idx.num_node_entries());
  std::vector<NDArray*> arrays;
  arrays.reserve(buff.size());
  for (size_t i = 0; i < buff.size(); ++i) arrays.push_back(&buff[i]);
  for (size_t i = 0; i < inputs.size(); ++i) {
    arrays[bwd_input_eid_[i]] = inputs[i];
  }
  for (size_t i = 0, j = num_forward_outputs; i < reqs.size(); ++i) {
    if (reqs[i] == kNullOp) continue;
    arrays[idx.entry_id(idx.outputs()[j++])] = outputs[i];
  }

  // Allocate NDArrays
  Context default_ctx = inputs[0]->ctx();
  const auto& shapes = g.GetAttr<ShapeVector>("shape");
  const auto& dtypes = g.GetAttr<DTypeVector>("dtype");
  const auto& stypes = g.GetAttr<StorageTypeVector>("storage_type");
  for (size_t i = num_forward_entries; i < arrays.size(); ++i) {
    if (!arrays[i]->is_none()) continue;
    if (stypes[i] == kDefaultStorage) {
      *arrays[i] = NDArray(shapes[i], default_ctx, true, dtypes[i]);
    } else {
      *arrays[i] = NDArray(static_cast<NDArrayStorageType>(stypes[i]),
                           shapes[i], default_ctx, true, dtypes[i]);
    }
  }

  // Execution
  bool prev_recording = ImperativeRuntime::Get()->set_is_recording(false);

  std::vector<int> ref_count = g.GetAttr<std::vector<int> >("ref_count");

  std::vector<OpReqType> array_reqs(arrays.size(), kWriteTo);
  for (size_t i = num_forward_nodes; i < idx.num_node_entries(); ++i) {
    if (ref_count[i] == 0) array_reqs[i] = kNullOp;
  }

  ImperativeRuntime::Get()->RunGraph(
      default_ctx, idx, arrays, num_forward_nodes, idx.num_nodes(),
      std::move(array_reqs), std::move(ref_count), &states);

  ImperativeRuntime::Get()->set_is_recording(prev_recording);
}

NNVM_REGISTER_OP(_CachedOp_NoGrad)
.set_num_inputs(0)
.set_num_outputs([](const NodeAttrs& attrs) {
    const uint32_t& nout = nnvm::get<uint32_t>(attrs.parsed);
    return nout;
  });

NNVM_REGISTER_OP(_CachedOp)
.set_num_inputs([](const NodeAttrs& attrs) {
    const CachedOpPtr& op = nnvm::get<CachedOpPtr>(attrs.parsed);
    return op->num_inputs();
  })
.set_num_outputs([](const NodeAttrs& attrs) {
    const CachedOpPtr& op = nnvm::get<CachedOpPtr>(attrs.parsed);
    return op->num_outputs();
  })
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    const CachedOpPtr& op = nnvm::get<CachedOpPtr>(n->attrs.parsed);
    return op->Gradient(n, ograds);
  });

NNVM_REGISTER_OP(_backward_CachedOp)
.set_num_inputs([](const NodeAttrs& attrs){
    const CachedOpPtr& op = nnvm::get<CachedOpPtr>(attrs.parsed);
    return op->num_backward_inputs();
  })
.set_num_outputs([](const NodeAttrs& attrs){
    const CachedOpPtr& op = nnvm::get<CachedOpPtr>(attrs.parsed);
    return op->num_inputs() - op->mutable_input_nodes().size();
  })
.set_attr<bool>("TIsLayerOpBackward", true)
.set_attr<bool>("TIsBackward", true);

}  // namespace mxnet
