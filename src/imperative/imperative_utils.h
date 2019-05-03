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
#include <mxnet/imperative.h>
#include <nnvm/pass_functions.h>
#include <utility>
#include <algorithm>
#include <vector>
#include <map>
#include <string>
#include "../executor/graph_executor.h"
#include "../executor/exec_pass.h"
#include "../c_api/c_api_common.h"
#include "../common/utils.h"
#include "../common/exec_utils.h"
#include "../operator/nn/mkldnn/mkldnn_base-inl.h"
#include "../operator/operator_common.h"

#ifndef MXNET_IMPERATIVE_IMPERATIVE_UTILS_H_
#define MXNET_IMPERATIVE_IMPERATIVE_UTILS_H_

namespace mxnet {
namespace imperative {

struct MemoryPlanInfo {
  int storage_id;
  uint32_t root;
  size_t size;
  bool inplace;
};

struct EngineOprDeleter {
  void operator()(engine::Opr* handle) {
    Engine::Get()->DeleteOperator(handle);
  }
};

struct EngineOprSeg {
  bool skip;
  size_t next_nid;
  std::unique_ptr<engine::Opr, EngineOprDeleter> opr;
};

using MemoryPlanVector = std::vector<MemoryPlanInfo>;

inline Context GetContext(const nnvm::NodeAttrs& attrs,
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
  // Non-default context (pinned, shared) does not propagate
  if (ctx.dev_mask() != ctx.dev_type && inputs.size() != 0U) {
    ctx = Context::Create(ctx.dev_mask(), ctx.dev_id);
  }
#if !MXNET_USE_CUDA
  if (ctx.dev_mask() == gpu::kDevMask) {
    LOG(INFO) << "GPU support is disabled. Compile MXNet with "
              << "USE_CUDA=1 to enable GPU support.";
  }
#endif  // MXNET_USE_CUDA
  return ctx;
}

// Set the shape, dtype, storage type and dispatch mode via the attribute inference functions
inline void SetShapeType(const Context& ctx,
                         const nnvm::NodeAttrs& attrs,
                         const std::vector<NDArray*>& inputs,
                         const std::vector<NDArray*>& outputs,
                         DispatchMode* dispatch_mode) {
  static auto& infershape = nnvm::Op::GetAttr<mxnet::FInferShape>("FInferShape");
  static auto& infertype = nnvm::Op::GetAttr<nnvm::FInferType>("FInferType");
  static auto& inferstorage = nnvm::Op::GetAttr<FInferStorageType>("FInferStorageType");
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  // infer shape
  mxnet::ShapeVector& in_shapes  = ret->arg_shapes;
  in_shapes.clear();
  in_shapes.reserve(inputs.size());
  for (auto& i : inputs) {
    in_shapes.push_back(i->shape());
  }
  mxnet::ShapeVector& out_shapes = ret->out_shapes;
  out_shapes.clear();
  out_shapes.reserve(outputs.size());
  for (auto& i : outputs) {
    out_shapes.push_back(i->shape());
  }
  bool is_dynamic_shape_existing = false;
  if (!infershape.count(attrs.op)) {
    is_dynamic_shape_existing = true;
  } else {
    if (!Imperative::Get()->is_np_shape()) {
      common::ConvertToNumpyShape(&in_shapes);
      common::ConvertToNumpyShape(&out_shapes);
    }
    const bool success = infershape[attrs.op](attrs, &in_shapes, &out_shapes);
    if (!success) {
      std::stringstream os;
      os << "Operator " << attrs.op->name << " inferring shapes failed.\n";
      os << "input shapes:\n";
      for (const auto& s : in_shapes) {
        os << s << '\n';
      }
      os << "output shapes:\n";
      for (const auto& s : out_shapes) {
        os << s << '\n';
      }
      os << "operator attributes:\n";
      for (const auto& kv : attrs.dict) {
        os << kv.first << " : " << kv.second << '\n';
      }
      LOG(FATAL) << os.str();
    }
    CHECK_EQ(out_shapes.size(), outputs.size());
  }
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
  bool infer_type_success = false;
  if (infertype.count(attrs.op)) {
    infer_type_success = infertype[attrs.op](attrs, &in_types, &out_types);
  } else {
    infer_type_success = common::SameType(attrs, &in_types, &out_types);
  }
  CHECK(infer_type_success) << "Operator " << attrs.op->name << " is missing FInferType attribute";
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
  bool infer_stype_success = false;
  if (inferstorage.count(attrs.op)) {
    infer_stype_success = inferstorage[attrs.op](attrs, ctx.dev_mask(), dispatch_mode,
                                                 &in_storage_types, &out_storage_types);
  } else {
    // if infer storage attr is not present, apply the default infer storage function
    infer_stype_success = common::DefaultStorageType(attrs, ctx.dev_mask(), dispatch_mode,
                                                   &in_storage_types, &out_storage_types);
  }
  CHECK(infer_stype_success) << "Operator not implemented: "
     << common::operator_stype_string(attrs, ctx.dev_mask(), in_storage_types, out_storage_types);
  if (*dispatch_mode == DispatchMode::kFComputeFallback) {
    common::LogStorageFallback(attrs, ctx.dev_mask(), &in_storage_types, &out_storage_types);
  }

  CHECK_EQ(out_storage_types.size(), outputs.size());
  CHECK(*dispatch_mode != DispatchMode::kUndefined);

  for (size_t i = 0; i < outputs.size(); ++i) {
    NDArrayStorageType storage_type = static_cast<NDArrayStorageType>(out_storage_types[i]);
    if (outputs[i]->is_none() || mxnet::op::shape_is_none(outputs[i]->shape())) {
      if (is_dynamic_shape_existing) {
        // once there is dynamic shape somewhere, we could not pre-determine the shape.
        *outputs[i] = NDArray(ctx, out_types[i]);
      } else if (storage_type == kDefaultStorage) {
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

inline void SetDependency(const nnvm::NodeAttrs& attrs,
                   const Context& ctx,
                   const std::vector<NDArray*>& inputs,
                   const std::vector<NDArray*>& outputs,
                   std::vector<engine::VarHandle> *p_read_vars,
                   std::vector<engine::VarHandle> *p_write_vars,
                   std::vector<Resource> *p_requested,
                   std::vector<uint32_t> *p_mutate_idx,
                   const DispatchMode dispatch_mode) {
  static auto& fmutate = nnvm::Op::GetAttr<nnvm::FMutateInputs>("FMutateInputs");
  static auto& ftmp_resource = nnvm::Op::GetAttr<FResourceRequest>("FResourceRequest");
  static auto& ftmp_resource_ex = nnvm::Op::GetAttr<FResourceRequestEx>("FResourceRequestEx");

  std::vector<engine::VarHandle>& read_vars  = *p_read_vars;
  std::vector<engine::VarHandle>& write_vars = *p_write_vars;
  std::vector<Resource>& requested = *p_requested;
  std::vector<uint32_t>& mutate_idx = *p_mutate_idx;

  if (fmutate.count(attrs.op)) {
    mutate_idx = fmutate[attrs.op](attrs);
  }
  const bool rsc_req = (ftmp_resource.count(attrs.op) != 0);
  const bool rsc_ex_req = (ftmp_resource_ex.count(attrs.op) != 0);
  if (rsc_req || rsc_ex_req) {
    int ntmp = 0;
    auto resource_reqs = rsc_ex_req ? ftmp_resource_ex[attrs.op](attrs,
                                          static_cast<int>(ctx.dev_mask()), dispatch_mode)
                                    : ftmp_resource[attrs.op](attrs);
    for (const auto& req : resource_reqs) {
      switch (req.type) {
       case ResourceRequest::kTempSpace:
        ++ntmp;
       case ResourceRequest::kRandom:
        requested.push_back(ResourceManager::Get()->Request(ctx, req));
        write_vars.push_back(requested.back().var);
        break;
       case ResourceRequest::kParallelRandom:
        requested.push_back(ResourceManager::Get()->Request(ctx, req));
        write_vars.push_back(requested.back().var);
        break;
#if MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 7
       case ResourceRequest::kCuDNNDropoutDesc:
        requested.push_back(ResourceManager::Get()->Request(ctx, req));
        write_vars.push_back(requested.back().var);
        break;
#endif  // MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 7
       default:
        LOG(FATAL) << "resource type not yet supported";
      }
    }
    CHECK_LE(ntmp, 1) << "Only support 1 temp space request";
  }

  // append extra resource requests for storage fallback
  if (dispatch_mode == DispatchMode::kFComputeFallback) {
    requested.push_back(ResourceManager::Get()->Request(ctx, ResourceRequest::kTempSpace));
    write_vars.push_back(requested.back().var);
  }

  read_vars.reserve(inputs.size());
  for (auto& i : inputs) {
    read_vars.push_back(i->var());
  }
  write_vars.reserve(outputs.size() + mutate_idx.size());
  for (auto& i : outputs) {
    write_vars.push_back(i->var());
  }
  for (auto & i : mutate_idx) {
    write_vars.push_back(inputs[i]->var());
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

/*!
 * \brief Parse parameter attributes into a nnvm::NodeAttrs structure
 * \param op Pointer to the nnvm Operator object
 * \param num_inputs Number of operator inputs
 * \param num_params Number of parameters
 * \param param_keys Array of string pointers representing the parameter keys
 * \param param_vals Array of string pointers representing the associated values
 * \return nnvm::NodeAttrs structure representing the parsed attributes
 */
inline nnvm::NodeAttrs ParseAttrs(const nnvm::Op *op,
                                  const int num_inputs,
                                  const int num_params,
                                  const char **param_keys,
                                  const char **param_vals) {
  static auto& num_args = nnvm::Op::GetAttr<std::string>("key_var_num_args");

  nnvm::NodeAttrs attrs;
  attrs.op = op;
  attrs.dict.reserve(num_params+1);
  for (int i = 0; i < num_params; ++i) {
    attrs.dict.emplace(param_keys[i], param_vals[i]);
  }
  if (num_args.count(op)) {
    attrs.dict.emplace(num_args[op], std::to_string(num_inputs));
  }
  if (op->attr_parser != nullptr) {
    op->attr_parser(&attrs);
  }

  return attrs;
}

/*!
 * \brief Determine number of outputs for the given operator
 * \param op Pointer to the nnvm Operator object
 * \param attrs  nnvm::NodeAttrs structure representing the operator's attributes
 * \param num_inputs Number of inputs tot he operator
 * \param infered_num_outputs The inferred number of outputs
 * \param num_visible_outputs The actual number of visible outputs
 */
inline void SetNumOutputs(const nnvm::Op *op,
                          const nnvm::NodeAttrs& attrs,
                          const int& num_inputs,
                          int* infered_num_outputs,
                          int* num_visible_outputs) {
  static auto& visible_out = nnvm::Op::GetAttr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs");
  int infered_num_inputs;
  if (op->get_num_inputs != nullptr) {
    infered_num_inputs = op->get_num_inputs(attrs);
  } else {
    infered_num_inputs = op->num_inputs;
  }
  CHECK_EQ(num_inputs, infered_num_inputs)
    << "Operator " << op->name << " expects " << infered_num_inputs
    << " inputs, but got " << num_inputs << " instead.";
  if (op->get_num_outputs != nullptr) {
    *infered_num_outputs = op->get_num_outputs(attrs);
  } else {
    *infered_num_outputs = op->num_outputs;
  }
  *num_visible_outputs = *infered_num_outputs;
  if (visible_out.count(op)) {
    *num_visible_outputs = visible_out[op](attrs);
    CHECK_LE(*num_visible_outputs, *infered_num_outputs);
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

inline void PushFCompute(const FCompute& fn,
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

  bool is_train = Imperative::Get()->is_training();
  bool need_grad = Imperative::Get()->is_recording();
  ExecType exec_type = fexec_type.count(op) ? fexec_type[op](attrs) : ExecType::kSync;
  CHECK(exec_type == ExecType::kSync);
  std::vector<NDArray> inputs, outputs;
  DerefInputOutput(p_inputs, p_outputs, &inputs, &outputs);
  Engine::Get()->PushSync(
    [=](RunContext rctx) {
      std::vector<TBlob> input_blobs, output_blobs;
      // pre-fcompute and post-fcompute storage fallback src NDArrays and dst NDArrays
      std::vector<NDArray> pre_temp_src, pre_temp_dst, post_temp_dst, post_temp_src;
      // mapping from index in input_blobs to index in pre_temp_dst
      std::unordered_map<uint32_t, uint32_t> in_temp_idx_map;
#if MXNET_USE_MKLDNN == 1
      InvalidateOutputs(outputs, req);
#endif
      std::vector<OpReqType> tmp_req = req;
      // setup blobs
      SetupDefaultBlobsInOut(inputs, outputs, nullptr, nullptr, &tmp_req,
                             &input_blobs, &output_blobs, &pre_temp_src, &pre_temp_dst,
                             &post_temp_src, &post_temp_dst, &in_temp_idx_map, mutate_idx);
      // setup context
      OpContext opctx{need_grad, is_train, rctx, engine::CallbackOnComplete(), requested};
      bool is_gpu = ctx.dev_mask() == gpu::kDevMask;
      // pre-fcompute fallback, cast to default storage type
      CastNonDefaultStorage(pre_temp_src, pre_temp_dst, opctx, is_gpu);
      fn(attrs, opctx, input_blobs, tmp_req, output_blobs);
      // post-fcompute fallback, cast to original storage type
      CastNonDefaultStorage(post_temp_src, post_temp_dst, opctx, is_gpu);
      if (is_gpu && !rctx.is_bulk) {
        rctx.get_stream<gpu>()->Wait();
      }
    }, ctx, read_vars, write_vars, FnProperty::kNormal,
    0, op->name.c_str());
}

inline void PushFComputeEx(const FComputeEx& fn,
                    const nnvm::Op* op,
                    const nnvm::NodeAttrs& attrs,
                    const Context& ctx,
                    const std::vector<engine::VarHandle>& read_vars,
                    const std::vector<engine::VarHandle>& write_vars,
                    const std::vector<Resource>& requested,
                    const std::vector<NDArray*>& p_inputs,
                    const std::vector<NDArray*>& p_outputs,
                    const std::vector<OpReqType>& req) {
  static auto& fexec_type = nnvm::Op::GetAttr<FExecType>("FExecType");

  bool is_train = Imperative::Get()->is_training();
  bool need_grad = Imperative::Get()->is_recording();
  ExecType exec_type = fexec_type.count(op) ? fexec_type[op](attrs) : ExecType::kSync;
  std::vector<NDArray> inputs, outputs;
  DerefInputOutput(p_inputs, p_outputs, &inputs, &outputs);
  const auto& run = [=](RunContext rctx) {
      OpContext opctx{need_grad, is_train, rctx, engine::CallbackOnComplete(), requested};
#if MXNET_USE_MKLDNN == 1
      InvalidateOutputs(outputs, req);
#endif
      fn(attrs, opctx, inputs, req, outputs);
      if (ctx.dev_mask() == gpu::kDevMask && exec_type == ExecType::kSync && !rctx.is_bulk) {
        rctx.get_stream<gpu>()->Wait();
      }
    };

  if (exec_type == ExecType::kCrossDeviceCopy) {
    run(RunContext{ctx, nullptr, nullptr, false});
  } else {
    CHECK(exec_type == ExecType::kSync);
    Engine::Get()->PushSync(run, ctx, read_vars, write_vars, FnProperty::kNormal,
                            0, op->name.c_str());
  }
}

inline void PushOperator(const OpStatePtr& state,
                  const nnvm::Op* op,
                  const nnvm::NodeAttrs& attrs,
                  const Context& ctx,
                  const std::vector<engine::VarHandle>& read_vars,
                  const std::vector<engine::VarHandle>& write_vars,
                  const std::vector<Resource>& requested,
                  const std::vector<NDArray*>& p_inputs,
                  const std::vector<NDArray*>& p_outputs,
                  const std::vector<uint32_t>& mutate_idx,
                  const std::vector<OpReqType>& req,
                  const DispatchMode dispatch_mode) {
  using namespace common;
  static auto& fexec_type = nnvm::Op::GetAttr<FExecType>("FExecType");

  bool is_train = Imperative::Get()->is_training();
  bool need_grad = Imperative::Get()->is_recording();
  ExecType exec_type = fexec_type.count(op) ? fexec_type[op](attrs) : ExecType::kSync;
  std::vector<NDArray> inputs, outputs;
  DerefInputOutput(p_inputs, p_outputs, &inputs, &outputs);

  auto fcompute =
      common::GetFCompute<FStatefulCompute>(op, "FStatefulCompute", ctx);
  auto fcompute_ex =
      common::GetFCompute<FStatefulComputeEx>(op, "FStatefulComputeEx", ctx);
  if (fcompute_ex != nullptr && dispatch_mode == DispatchMode::kFComputeEx) {
    const auto& run = [=](RunContext rctx,
                          engine::CallbackOnComplete on_complete) {
      OpContext opctx{need_grad, is_train, rctx, on_complete, requested};
#if MXNET_USE_MKLDNN == 1
      InvalidateOutputs(outputs, req);
#endif
      fcompute_ex(state, opctx, inputs, req, outputs);
      if (ctx.dev_mask() == gpu::kDevMask && exec_type == ExecType::kSync
          && rctx.get_stream<gpu>() && !rctx.is_bulk) {
        rctx.get_stream<gpu>()->Wait();
      }
    };

    // For operators with subgraphs, we need to invoke them in the main thread
    // instead of the threaded engine.
    if (exec_type == ExecType::kSubgraphExec) {
      RunContext rctx{ctx, nullptr, nullptr, false};
      run(rctx, engine::CallbackOnComplete());
    } else if (exec_type == ExecType::kSync) {
      Engine::Get()->PushSync(
          [=](RunContext rctx) { run(rctx, engine::CallbackOnComplete()); },
          ctx, read_vars, write_vars, FnProperty::kNormal, 0,
          op->name.c_str());
    } else {
      CHECK(exec_type == ExecType::kAsync);
      Engine::Get()->PushAsync(run, ctx, read_vars, write_vars,
                               FnProperty::kAsync, 0,
                               op->name.c_str());
    }
  } else {
    CHECK(fcompute != nullptr)
        << "One of FStatefulCompute and FStatefulComputeEx must be registered "
        << "for stateful operator " << op->name;

    const auto& run = [=](RunContext rctx, engine::CallbackOnComplete on_complete) {
        OpContext opctx{need_grad, is_train, rctx, on_complete, requested};

        std::vector<TBlob> input_blobs, output_blobs;
        // pre-fcompute and post-fcompute storage fallback src NDArrays and dst NDArrays
        std::vector<NDArray> pre_temp_src, pre_temp_dst, post_temp_dst, post_temp_src;
        // mapping from index in input_blobs to index in pre_temp_dst
        std::unordered_map<uint32_t, uint32_t> in_temp_idx_map;
#if MXNET_USE_MKLDNN == 1
        InvalidateOutputs(outputs, req);
#endif
        std::vector<OpReqType> tmp_req = req;
        // populate input blobs and output blobs
        SetupDefaultBlobsInOut(inputs, outputs, nullptr, nullptr, &tmp_req,
                               &input_blobs, &output_blobs, &pre_temp_src, &pre_temp_dst,
                               &post_temp_src, &post_temp_dst, &in_temp_idx_map, mutate_idx);
        // setup contexts
        bool is_gpu = rctx.get_ctx().dev_mask() == gpu::kDevMask;
        // pre-fcompute fallback
        CastNonDefaultStorage(pre_temp_src, pre_temp_dst, opctx, is_gpu);
        fcompute(state, opctx, input_blobs, tmp_req, output_blobs);
        // post-fcompute fallback, cast to original storage type, if necessary
        CastNonDefaultStorage(post_temp_src, post_temp_dst, opctx, is_gpu);
        if (is_gpu && exec_type == ExecType::kSync
            && rctx.get_stream<gpu>() && !rctx.is_bulk) {
          rctx.get_stream<gpu>()->Wait();
        }
      };

    if (exec_type == ExecType::kSubgraphExec) {
      RunContext rctx{ctx, nullptr, nullptr, false};
      run(rctx, engine::CallbackOnComplete());
    } else if (exec_type == ExecType::kSync) {
      Engine::Get()->PushSync(
          [=](RunContext rctx) {
            run(rctx, engine::CallbackOnComplete());
          }, ctx, read_vars, write_vars, FnProperty::kNormal,
          0, op->name.c_str());
    } else {
      CHECK(exec_type == ExecType::kAsync);
      Engine::Get()->PushAsync(
          run, ctx, read_vars, write_vars, FnProperty::kAsync,
          0, op->name.c_str());
    }
  }
}

inline bool CheckAndInferShape(nnvm::Graph* p_g, mxnet::ShapeVector&& shapes,
                               bool use_inputs,
                               std::pair<uint32_t, uint32_t> node_range = {0, 0},
                               std::pair<uint32_t, uint32_t> entry_range = {0, 0},
                               bool *contain_unknown = nullptr) {
  using namespace nnvm;
  if (contain_unknown != nullptr) {
    *contain_unknown = false;
  }
  nnvm::Graph& g = *p_g;
  if (use_inputs) {
    if (g.attrs.count("shape_inputs") && g.GetAttr<mxnet::ShapeVector>("shape_inputs") == shapes)
      return true;
  } else if (g.attrs.count("shape")) {
    const auto& prev_shapes = g.GetAttr<mxnet::ShapeVector>("shape");
    if (prev_shapes.size() == shapes.size()) {
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
      if (match) return true;
    }
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
  if (contain_unknown == nullptr) {
    CHECK_EQ(g.GetAttr<size_t>("shape_num_unknown_nodes"), 0U);
  } else {
    *contain_unknown = g.GetAttr<size_t>("shape_num_unknown_nodes") != 0U;
  }
  return false;
}


inline bool CheckAndInferType(nnvm::Graph* p_g, nnvm::DTypeVector&& dtypes,
                              bool use_inputs,
                              std::pair<uint32_t, uint32_t> node_range = {0, 0},
                              std::pair<uint32_t, uint32_t> entry_range = {0, 0}) {
  using namespace nnvm;
  nnvm::Graph& g = *p_g;
  if (use_inputs) {
    if (g.attrs.count("dtype_inputs") &&
        g.GetAttr<DTypeVector>("dtype_inputs") == dtypes) return true;
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
    if (match) return true;
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

  return false;
}

inline bool CheckAndInferStorageType(nnvm::Graph* p_g, exec::DevMaskVector&& dev_mask,
                                     StorageTypeVector&& storage_types, bool use_inputs,
                                     std::pair<uint32_t, uint32_t> node_range = {0, 0},
                                     std::pair<uint32_t, uint32_t> entry_range = {0, 0}) {
  using namespace nnvm;
  nnvm::Graph& g = *p_g;
  bool dev_match = g.attrs.count("dev_mask") &&
                   g.GetAttr<exec::DevMaskVector>("dev_mask") == dev_mask;
  if (!dev_match) {
    g.attrs["dev_mask"] = std::make_shared<dmlc::any>(std::move(dev_mask));
  }

  if (dev_match && use_inputs) {
    if (g.attrs.count("storage_type_inputs") &&
        g.GetAttr<StorageTypeVector>("storage_type_inputs") == storage_types) return true;
  } else if (dev_match && g.attrs.count("storage_type")) {
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
    if (match) return true;
  }
  g.attrs.erase("dispatch_mode");
  g.attrs.erase("storage_type");
  g.attrs.erase("storage_type_inputs");
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
  return false;
}


inline std::vector<Context> PlaceDevice(const nnvm::IndexedGraph& idx) {
  static const auto& _copyto = Op::Get("_copyto");

  std::vector<Context> vctx(
      idx.num_nodes(), Context::Create(static_cast<Context::DeviceType>(-1), 0));
  // forward pass
  for (size_t i = 0; i < idx.num_nodes(); ++i) {
    if (!idx[i].source->info.empty()) {
      vctx[i] = dmlc::get<Imperative::AGInfo>(idx[i].source->info).ctx;
    } else if (idx[i].source->op() == _copyto) {
      CHECK_GT(idx[i].source->control_deps.size(), 0);
      auto fwd_nid = idx.node_id(idx[i].source->control_deps[0].get());
      CHECK_EQ(idx[fwd_nid].source->op(), _copyto);
      vctx[i] = vctx[idx[fwd_nid].inputs[0].node_id];
    } else if (idx[i].control_deps.size() &&
               vctx[idx[i].control_deps[0]].dev_type != static_cast<Context::DeviceType>(-1)) {
      vctx[i] = vctx[idx[i].control_deps[0]];
    } else {
      for (const auto& in : idx[i].inputs) {
        if (vctx[in.node_id].dev_type == static_cast<Context::DeviceType>(-1)) continue;
        vctx[i] = vctx[in.node_id];
        break;
      }
    }
  }
  // backward pass
  for (int i = idx.num_nodes() - 1; i >= 0; --i) {
    if (vctx[i].dev_type == static_cast<Context::DeviceType>(-1)) continue;
    if (idx[i].source->op() == _copyto) {
      auto in_nid = idx[i].inputs[0].node_id;
      if (vctx[in_nid].dev_type != static_cast<Context::DeviceType>(-1)) continue;
      CHECK_GT(idx[i].source->control_deps.size(), 0);
      auto fwd_nid = idx.node_id(idx[i].source->control_deps[0].get());
      CHECK_EQ(idx[fwd_nid].source->op(), _copyto);
      vctx[in_nid] = vctx[fwd_nid];
      continue;
    }
    for (const auto& j : idx[i].inputs) {
      if (vctx[j.node_id].dev_type != static_cast<Context::DeviceType>(-1)) continue;
      vctx[j.node_id] = vctx[i];
    }
  }
  // check all context initialized
  for (size_t i = 0; i < idx.num_nodes(); ++i) {
    CHECK_NE(vctx[i].dev_type, -1)
        << "Cannot decide context for node " << idx[i].source->attrs.name;
    // Non-default context do not propagate.
    vctx[i].dev_type = vctx[i].dev_mask();
  }

  return vctx;
}


inline MemoryPlanVector PlanMemory(
    nnvm::Graph* p_g,
    nnvm::StorageVector&& storage,
    const std::vector<uint32_t>& ref_count,
    const std::pair<uint32_t, uint32_t>& node_range = {0, 0},
    const std::pair<uint32_t, uint32_t>& entry_range = {0, 0},
    bool detect_inplace_addto = false) {
  using namespace nnvm;
  nnvm::Graph& g = *p_g;
  const auto& idx = g.indexed_graph();
  if (node_range.second > node_range.first) {
    g.attrs["node_range"] = std::make_shared<dmlc::any>(node_range);
  }
  g.attrs["ref_count"] = std::make_shared<dmlc::any>(ref_count);
  g.attrs["storage"] = std::make_shared<dmlc::any>(std::move(storage));
  g = nnvm::ApplyPass(g, "MXPlanMemory");
  if (detect_inplace_addto) g = exec::DetectInplaceAddTo(g);

  const auto& dtypes = g.GetAttr<DTypeVector>("dtype");
  const auto& shapes = g.GetAttr<mxnet::ShapeVector>("shape");
  const auto& storage_inplace = g.GetAttr<std::vector<int> >("storage_inplace_index");
  const auto& storage_ids = g.GetAttr<StorageVector>("storage_id");
  uint32_t entry_start = entry_range.first;
  uint32_t entry_end =
      entry_range.second > entry_start ? entry_range.second : idx.num_node_entries();
  MemoryPlanVector mem_plan(idx.num_node_entries());
  std::unordered_map<int, uint32_t> sid_to_root;

  for (uint32_t i = entry_start; i < entry_end; ++i) {
    if (storage_ids[i] < 0) {
      mem_plan[i] = {storage_ids[i], i, 0, false};
    } else if (!sid_to_root.count(storage_ids[i])) {
      CHECK_LT(storage_inplace[i], 0);
      sid_to_root[storage_ids[i]] = i;
      mem_plan[i] = {storage_ids[i], i,
                     mshadow::mshadow_sizeof(dtypes[i]) * shapes[i].Size(),
                     false};
    } else {
      uint32_t root = sid_to_root[storage_ids[i]];
      mem_plan[i] = {storage_ids[i], root, 0, storage_inplace[i] >= 0};
      mem_plan[root].size = std::max(mem_plan[root].size,
          mshadow::mshadow_sizeof(dtypes[i]) * shapes[i].Size());
    }
  }

  return mem_plan;
}


inline std::multimap<size_t, NDArray> AllocateMemory(
    const nnvm::Graph& g,
    const nnvm::IndexedGraph& idx,
    const Context& default_ctx,
    const uint32_t entry_start, const uint32_t entry_end,
    const MemoryPlanVector& mem_plan,
    const std::vector<NDArray*>& arrays,
    std::vector<OpReqType> *array_reqs,
    std::multimap<size_t, NDArray>&& pool = std::multimap<size_t, NDArray>()) {
  using namespace nnvm;
  const auto& dtypes = g.GetAttr<DTypeVector>("dtype");
  const auto& shapes = g.GetAttr<mxnet::ShapeVector>("shape");
  const auto& stypes = g.GetAttr<StorageTypeVector>("storage_type");

  std::multimap<size_t, NDArray> new_pool;

  for (uint32_t i = entry_start; i < entry_end; ++i) {
    if (mem_plan[i].storage_id == exec::kExternalStorageID) continue;
    CHECK(arrays[i]->is_none());
    if (mem_plan[i].storage_id == exec::kDynamicStorageID) {
      *arrays[i] = NDArray(static_cast<NDArrayStorageType>(stypes[i]),
                           shapes[i], default_ctx, true, dtypes[i]);
      continue;
    }
    CHECK_EQ(stypes[i], kDefaultStorage);
    if (mem_plan[i].root == i) {
      auto iter = pool.lower_bound(mem_plan[i].size);
      if (iter != pool.end()) {
        *arrays[i] = iter->second.AsArray(shapes[i], dtypes[i]);
        new_pool.insert(*iter);
        pool.erase(iter);
      } else {
        NDArray buff(mxnet::TShape({static_cast<nnvm::dim_t>(mem_plan[i].size)}),
                     default_ctx, true, mshadow::kUint8);
        *arrays[i] = buff.AsArray(shapes[i], dtypes[i]);
        new_pool.insert({mem_plan[i].size, buff});
      }
    } else {
      CHECK_GE(mem_plan[mem_plan[i].root].storage_id, 0);
      *arrays[i] = arrays[mem_plan[i].root]->AsArray(shapes[i], dtypes[i]);
      if (mem_plan[i].inplace && array_reqs->at(i) == kWriteTo) {
        array_reqs->at(i) = kWriteInplace;
      }
    }
  }

  return new_pool;
}

inline void SetupOpExec(
    const nnvm::Graph& g,
    size_t nid,
    const std::shared_ptr<exec::OpExecutor>& exec,
    const std::vector<NDArray*> arrays,
    const std::vector<OpReqType> array_reqs) {
  const auto& idx = g.indexed_graph();
  const auto& inode = idx[nid];
  CHECK_EQ(exec->in_array.size(), 0U);
  CHECK_EQ(exec->out_array.size(), 0U);
  for (const auto& e : inode.inputs) {
    CHECK(!arrays[idx.entry_id(e)]->is_none()) << inode.source->attrs.name;
    exec->in_array.push_back(*arrays[idx.entry_id(e)]);
  }
  for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) {
    uint32_t eid = idx.entry_id(nid, index);
    CHECK(!arrays[eid]->is_none()) << inode.source->attrs.name;
    exec->out_array.push_back(*arrays[eid]);
    exec->req.push_back(array_reqs[eid]);
  }

  exec->Setup();
}

inline Engine::OprHandle CreateEngineOp(
    const Context& default_ctx,
    const std::vector<std::shared_ptr<exec::OpExecutor> >& execs) {
  CHECK_GT(execs.size(), 0);
  std::vector<Engine::VarHandle> use_vars, mutate_vars;

  for (const auto& exec : execs) {
    CHECK_GT(exec->out_array.size(), 0);
    CHECK(execs.size() == 1 || exec->exec_type() == ExecType::kSync);

    // the variables
    for (const auto& nd : exec->in_array) {
      use_vars.push_back(nd.var());
    }
    for (auto& r : exec->op_ctx.requested) {
      mutate_vars.push_back(r.var);
    }
    for (auto& nd : exec->out_array) {
      mutate_vars.push_back(nd.var());
    }
    if (exec->var() != nullptr) {
      mutate_vars.push_back(exec->var());
    }
  }

  // dedup vars
  Engine::Get()->DeduplicateVarHandle(&use_vars, &mutate_vars);
  bool is_gpu = default_ctx.dev_mask() == gpu::kDevMask;
  bool is_async = execs.size() > 1 ? false : execs[0]->exec_type() == ExecType::kAsync;

  auto exec_fun = [execs, is_async, is_gpu] (
      RunContext ctx, Engine::CallbackOnComplete on_complete) {
    if (is_async) {
      execs[0]->op_ctx.async_on_complete = on_complete;
    }
    for (const auto& exec : execs) exec->Run(ctx, is_gpu);
    // call on complete only if it is async op
    if (!is_async) {
      if (is_gpu) {
      #if MXNET_USE_CUDA
        // Wait GPU kernel to finish.
        ctx.get_stream<gpu>()->Wait();
      #else
        LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
      #endif
      }
      on_complete();
    }
  };

  return Engine::Get()->NewOperator(
      exec_fun, use_vars, mutate_vars, FnProperty::kNormal);
}

inline void CreateEngineOpSeg(
    const nnvm::IndexedGraph& idx,
    const Context default_ctx,
    const size_t start_nid,
    const size_t end_nid,
    const size_t bulk_size,
    const std::vector<std::shared_ptr<exec::OpExecutor> >& execs,
    const std::vector<int> skip_plus_node,
    std::vector<EngineOprSeg> *opr_segs) {
  size_t seg_start = start_nid;
  std::vector<std::shared_ptr<exec::OpExecutor> > seg_execs;
  for (size_t nid = start_nid; nid < end_nid; ++nid) {
    const auto& node = idx[nid];
    if (node.source->is_variable()) continue;
    if (skip_plus_node.size() && skip_plus_node[nid]) continue;
    auto& exec = execs[nid];
    bool is_async = exec->exec_type() != ExecType::kSync;
    bool valid = exec->out_array.size() > 0;

    // Stop at async nodes and invalid node (due to input/output is not allocated)
    bool stop = is_async || !valid || seg_execs.size() >= bulk_size;

    // Create opr segment for previous nodes.
    if (stop && nid > seg_start) {
      auto& seg = (*opr_segs)[seg_start];
      if (seg_execs.size()) {
        seg = EngineOprSeg{false, nid};
        seg.opr.reset(CreateEngineOp(default_ctx, seg_execs));
      } else {
        seg = EngineOprSeg{true, nid, nullptr};
      }
      seg_start = nid;
      seg_execs.clear();
    }

    seg_execs.push_back(exec);

    auto& seg = (*opr_segs)[nid];
    if (!valid) {
      seg = EngineOprSeg{false, nid + 1, nullptr};
      seg_execs.clear();
      seg_start = nid + 1;
    } else if (is_async) {
      seg = EngineOprSeg{false, nid + 1};
      seg.opr.reset(CreateEngineOp(default_ctx, seg_execs));
      seg_execs.clear();
      seg_start = nid + 1;
    }
  }
  // The last segment
  if (end_nid > seg_start) {
    auto& seg = (*opr_segs)[seg_start];
    if (seg_execs.size()) {
      seg = EngineOprSeg{false, end_nid};
      seg.opr.reset(CreateEngineOp(default_ctx, seg_execs));
    } else {
      seg = EngineOprSeg{true, end_nid, nullptr};
    }
  }
}


void RunGraph(const bool retain_graph,
              const nnvm::IndexedGraph& idx,
              const std::vector<NDArray*>& arrays,
              size_t node_start, size_t node_end,
              std::vector<OpReqType>&& array_reqs,
              std::vector<uint32_t>&& ref_count,
              std::vector<OpStatePtr> *p_states,
              const DispatchModeVector &dispatch_modes,
              bool recording,
              mxnet::ShapeVector *shapes = nullptr);

void NaiveRunGraph(const bool retain_graph,
                   const Context& default_ctx,
                   const nnvm::IndexedGraph& idx,
                   const std::vector<NDArray*>& arrays,
                   size_t node_start, size_t node_end,
                   std::vector<OpReqType>&& array_reqs,
                   std::vector<uint32_t>&& ref_count,
                   std::vector<OpStatePtr> *p_states,
                   const DispatchModeVector &dispatch_modes,
                   bool recording,
                   mxnet::ShapeVector *shapes);

}  // namespace imperative
}  // namespace mxnet

#endif  // MXNET_IMPERATIVE_IMPERATIVE_UTILS_H_
