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
    std::sort(mutate_idx.begin(), mutate_idx.end());
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

OpStatePtr ImperativeRuntime::InvokeOp(
    const Context& ctx,
    const nnvm::NodeAttrs& attrs,
    const std::vector<NDArray*>& inputs,
    const std::vector<NDArray*>& outputs,
    const std::vector<OpReqType>& req,
    OpStatePtr state) {
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
  if (!is_recording()) return;
  MXAPIThreadLocalEntry *local_buff = MXAPIThreadLocalStore::Get();

  for (uint32_t i = 0; i < outputs.size(); ++i) {
    CHECK(AGInfo::IsNone(*(outputs[i])))
      << "Inplace operations (+=, -=, x[:]=, etc) are not supported when "
      << "recording with autograd. "
      << "Assigning to NDArrays that are already in a computational graph "
      << "will cause undefined behavior when evaluating gradients. "
      << "Please call backward first to clear the graph or do this out side of "
      << "a record section. ";
  }
  bool need_grad = false;
  for (const auto& i : inputs) {
    if (!AGInfo::IsNone(*i)) {
      need_grad = true;
      break;
    }
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

std::vector<NDArray*> ImperativeRuntime::Backward(
    const std::vector<NDArray*>& outputs,
    const std::vector<NDArray*>& ograds,
    const std::vector<NDArray*>& variables,
    bool is_train, bool retain_graph,
    bool create_graph) {
  using namespace nnvm;
  static auto& fmutate_inputs = Op::GetAttr<FMutateInputs>("FMutateInputs");
  static auto& is_layer_backward = Op::GetAttr<bool>("TIsLayerOpBackward");

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

  std::vector<const Op*> zero_ops;
  zero_ops.push_back(Op::Get("zeros_like"));
  zero_ops.push_back(Op::Get("_zeros"));

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
  std::vector<NDArray*> arrays;
  arrays.reserve(buff.size());
  for (size_t i = 0; i < buff.size(); ++i) arrays.push_back(&buff[i]);
  if (create_graph) {
    nnvm::DFSVisit(sym.outputs, [&](const nnvm::NodePtr& n) {
      AGInfo& info = AGInfo::Get(n);
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
    for (size_t i = 0; i < num_forward_nodes; ++i) {
      const AGInfo& info = dmlc::get<AGInfo>(idx[i].source->info);
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
  exec::ContextVector vctx(idx.num_nodes(), default_ctx);
  graph.attrs["context"] = std::make_shared<dmlc::any>(std::move(vctx));

  // Infer shape type
  {
    ShapeVector shapes;
    shapes.reserve(idx.num_node_entries());
    for (const auto& i : arrays) shapes.emplace_back(i->shape());
    graph.attrs["shape"] = std::make_shared<dmlc::any>(std::move(shapes));
    graph = exec::InferShape(std::move(graph));
    CHECK_EQ(graph.GetAttr<size_t>("shape_num_unknown_nodes"), 0U);

    DTypeVector dtypes;
    dtypes.reserve(idx.num_node_entries());
    for (const auto& i : arrays) dtypes.emplace_back(i->dtype());
    graph.attrs["dtype"] = std::make_shared<dmlc::any>(std::move(dtypes));
    graph = exec::InferType(std::move(graph));
    CHECK_EQ(graph.GetAttr<size_t>("dtype_num_unknown_nodes"), 0U);

    StorageTypeVector stypes;
    stypes.reserve(idx.num_node_entries());
    for (const auto& i : arrays) stypes.emplace_back(i->storage_type());
    graph.attrs["storage_type"] = std::make_shared<dmlc::any>(std::move(stypes));
    graph = exec::InferStorageType(std::move(graph));
    CHECK_EQ(graph.GetAttr<size_t>("storage_type_num_unknown_nodes"), 0U);
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
  std::vector<NDArray*> ndinputs, ndoutputs;
  std::vector<OpReqType> req;

  bool prev_recording = set_is_recording(create_graph);
  bool prev_training = set_is_training(is_train);

  for (size_t i = num_forward_nodes; i < idx.num_nodes(); ++i) {
    const nnvm::IndexedGraph::Node& node = idx[i];
    if (node.source->attrs.op == nullptr) continue;
    ndinputs.clear();
    ndinputs.reserve(node.inputs.size());
    for (const auto& j : node.inputs) {
      ndinputs.emplace_back(arrays[idx.entry_id(j)]);
      CHECK(!ndinputs.back()->is_none());
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

    if (is_layer_backward.get(node.source->attrs.op, false)) {
      CHECK_GE(node.source->control_deps.size(), 1);
      auto& state = AGInfo::Get(node.source->control_deps[0]).state;
      InvokeOp(default_ctx, node.source->attrs, ndinputs, ndoutputs, req, state);
      RecordOp(NodeAttrs(node.source->attrs), ndinputs, ndoutputs, state);
    } else {
      InvokeOp(default_ctx, node.source->attrs, ndinputs, ndoutputs, req);
      RecordOp(NodeAttrs(node.source->attrs), ndinputs, ndoutputs);
    }

    for (const auto& j : node.inputs) {
      size_t eid = idx.entry_id(j);
      --ref_count[eid];
      if (ref_count[eid] == 0) *arrays[eid] = NDArray();
    }
    for (size_t j = 0; j < ndoutputs.size(); ++j) {
      size_t eid = idx.entry_id(i, j);
      if (ref_count[eid] == 0) *arrays[eid] = NDArray();
    }
  }

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

}  // namespace mxnet
