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
 *  Copyright (c) 2016 by Contributors
 * \file c_api_ndarray.cc
 * \brief C API of mxnet
 */

#include <mxnet/base.h>
#include <mxnet/c_api.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <string>
#include "./c_api_common.h"
#include "../common/utils.h"
#include "../ndarray/autograd.h"

using namespace mxnet;
using mxnet::autograd::AutogradRuntime;

void SetOpAttrs(const nnvm::Op *op,
                nnvm::NodeAttrs *p_attrs,
                const int& num_inputs,
                const int& num_params,
                const char **param_keys,
                const char **param_vals) {
  static auto& num_args = nnvm::Op::GetAttr<std::string>("key_var_num_args");
  nnvm::NodeAttrs& attrs = *p_attrs;
  attrs.op = op;
  for (int i = 0; i < num_params; ++i) {
    attrs.dict.emplace(param_keys[i], param_vals[i]);
  }

  if (num_args.count(op)) {
    attrs.dict.emplace(num_args[op], std::to_string(num_inputs));
  }
  if (op->attr_parser != nullptr) {
    op->attr_parser(&attrs);
  }
}

void SetNumOutputs(const nnvm::Op *op,
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
    << "Expecting " << infered_num_inputs << " inputs, got "
    << num_inputs << " in operator " << op->name;
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

void SetNDInputsOutputs(const nnvm::Op* op,
                        std::vector<NDArray>* p_ndinputs,
                        std::vector<NDArray>* p_ndoutputs,
                        const int& num_inputs,
                        const NDArrayHandle *inputs,
                        int *num_outputs,
                        const int& infered_num_outputs,
                        const int& num_visible_outputs,
                        NDArray** outarray) {
  std::vector<NDArray>& ndinputs  = *p_ndinputs;
  std::vector<NDArray>& ndoutputs = *p_ndoutputs;
  ndinputs.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    ndinputs.emplace_back(*reinterpret_cast<NDArray*>(inputs[i]));
  }
  if (outarray == nullptr) {
    *num_outputs = num_visible_outputs;
    ndoutputs.resize(infered_num_outputs);
  } else {
    CHECK(*num_outputs == infered_num_outputs || *num_outputs == num_visible_outputs)
      << "Expecting " << infered_num_outputs << " (all) or "
      << num_visible_outputs << " (visible only) outputs, got "
      << *num_outputs << " in operator " << op->name;
    ndoutputs.reserve(infered_num_outputs);
    for (int i = 0; i < num_visible_outputs; ++i) {
      ndoutputs.emplace_back(std::move(*outarray[i]));
    }
    ndoutputs.resize(infered_num_outputs);
  }
}

void SetContext(Context* p_ctx,
                const nnvm::NodeAttrs& attrs,
                const std::vector<NDArray>& ndinputs,
                const std::vector<NDArray>& ndoutputs,
                const Context& default_ctx) {
  Context& ctx = *p_ctx;
  if (ndinputs.size()) {
    ctx = ndinputs[0].ctx();
    for (size_t i = 1; i < ndinputs.size(); ++i) {
      CHECK_EQ(ndinputs[i].ctx().dev_mask(), ctx.dev_mask())
          << "All inputs must live on the same context. "
          << "But the first argument is on "
          << ctx << " while the " << i+1 << "-th argument is on "
          << ndinputs[i].ctx();
    }
  } else if (ndoutputs.size() && !ndoutputs[0].is_none()) {
    ctx = ndoutputs[0].ctx();
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
}

// Set the shape, dtype and storage type
void SetShapeType(const nnvm::Op* op,
                  const nnvm::NodeAttrs& attrs,
                  const Context& ctx,
                  const std::vector<NDArray>& ndinputs,
                  std::vector<NDArray>* p_ndoutputs,
                  int* dispatch_stype) {
  std::vector<NDArray>& ndoutputs = *p_ndoutputs;
  static auto& infershape = nnvm::Op::GetAttr<nnvm::FInferShape>("FInferShape");
  static auto& infertype = nnvm::Op::GetAttr<nnvm::FInferType>("FInferType");
  static auto& inferstorage = nnvm::Op::GetAttr<FInferStorageType>("FInferStorageType");
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  // infer shape
  std::vector<TShape>& in_shapes  = ret->arg_shapes;
  std::vector<TShape>& out_shapes = ret->out_shapes;
  in_shapes.clear();
  out_shapes.clear();

  for (auto& i : ndinputs) {
    in_shapes.emplace_back(i.shape());
  }
  for (auto& i : ndoutputs) {
    out_shapes.emplace_back(i.shape());
  }
  CHECK(infershape.count(op))
    << "Operator " << op->name << " is missing FInferShape attribute";
  CHECK(infershape[op](attrs, &in_shapes, &out_shapes));
  CHECK_EQ(out_shapes.size(), ndoutputs.size());

  // infer type
  std::vector<int>& in_types = ret->arg_types;
  std::vector<int>& out_types = ret->out_types;
  in_types.clear();
  out_types.clear();

  for (auto& i : ndinputs) {
    in_types.push_back(i.dtype());
  }
  for (auto& i : ndoutputs) {
    out_types.push_back(i.dtype());
  }
  CHECK(infertype.count(op))
    << "Operator " << op->name << " is missing FInferType attribute";
  CHECK(infertype[op](attrs, &in_types, &out_types));
  CHECK_EQ(out_types.size(), ndoutputs.size());

  // infer storage type
  auto& in_storage_types = ret->arg_storage_types;
  auto& out_storage_types = ret->out_storage_types;
  in_storage_types.clear();
  out_storage_types.clear();
  for (auto& i : ndinputs) {
    in_storage_types.push_back(i.storage_type());
  }
  for (auto& i : ndoutputs) {
    out_storage_types.push_back(i.storage_type());
  }
  if (inferstorage.count(op)) {
    CHECK(inferstorage[op](attrs, ctx, &in_storage_types, &out_storage_types));
    CHECK_EQ(out_storage_types.size(), ndoutputs.size());
  }

  bool contains_non_default = common::ContainsNonDefaultStorage(in_storage_types);
  contains_non_default |= common::ContainsNonDefaultStorage(out_storage_types);
  int kNonDefaultStorage = -2;
  *dispatch_stype = contains_non_default ? kNonDefaultStorage : kDefaultStorage;
  for (size_t i = 0; i < ndoutputs.size(); ++i) {
    NDArrayStorageType storage_type = static_cast<NDArrayStorageType>(out_storage_types[i]);
    if (ndoutputs[i].is_none()) {
      // if failed to infer the storage type, assume the output storage is dense
      if (storage_type == kDefaultStorage || out_storage_types[i] == kUndefinedStorage) {
        ndoutputs[i] = NDArray(out_shapes[i], ctx, true, out_types[i]);
      } else {
        ndoutputs[i] = NDArray(storage_type, out_shapes[i], ctx, true, out_types[i]);
      }
    } else {
      CHECK_EQ(ndoutputs[i].shape(), out_shapes[i])
        << i << "th output has invalid shape. "
        << "Expecting " << out_shapes[i] << " got "
        << ndoutputs[i].shape() << " in operator " << op->name;
      CHECK_EQ(ndoutputs[i].dtype(), out_types[i])
        << i << "th output has invalid shape. "
        << "Expecting " << out_types[i] << " got "
        << ndoutputs[i].dtype()  << " in operator " << op->name;
    }
  }
}

void SetDependency(std::vector<engine::VarHandle> *p_read_vars,
                   std::vector<engine::VarHandle> *p_write_vars,
                   std::vector<Resource> *p_requested,
                   std::vector<uint32_t> *p_mutate_idx,
                   const nnvm::Op* op,
                   const nnvm::NodeAttrs& attrs,
                   const Context& ctx,
                   const std::vector<NDArray>& ndinputs,
                   const std::vector<NDArray>& ndoutputs) {
  static auto& mutate = nnvm::Op::GetAttr<nnvm::FMutateInputs>("FMutateInputs");
  static auto& tmp_resource = nnvm::Op::GetAttr<FResourceRequest>("FResourceRequest");

  std::vector<engine::VarHandle>& read_vars  = *p_read_vars;
  std::vector<engine::VarHandle>& write_vars = *p_write_vars;
  std::vector<Resource>& requested = *p_requested;
  std::vector<uint32_t>& mutate_idx = *p_mutate_idx;

  if (tmp_resource.count(op)) {
    int ntmp = 0;
    for (const auto& req : tmp_resource[op](attrs)) {
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

  for (auto& i : ndinputs) {
    read_vars.push_back(i.var());
  }
  for (auto& i : ndoutputs) {
    write_vars.push_back(i.var());
  }
  if (mutate.count(op)) {
    mutate_idx = mutate[op](attrs);
    std::sort(mutate_idx.begin(), mutate_idx.end());
    for (auto & i : mutate_idx) {
      write_vars.push_back(ndinputs[i].var());
    }
  }
  Engine::Get()->DeduplicateVarHandle(&read_vars, &write_vars);
}

inline void SetWriteInplaceReq(const std::vector<NDArray> &ndinputs,
                               const std::vector<NDArray> &ndoutputs,
                               std::vector<OpReqType> *req) {
  std::unordered_set<engine::VarHandle> in_vars;
  for (auto &nd : ndinputs) {
    in_vars.insert(nd.var());
  }
  for (size_t i = 0; i < ndoutputs.size(); i++) {
    // output NDArray shares the memory with the input NDArray
    if (in_vars.find(ndoutputs[i].var()) != in_vars.end()) {
      req->at(i) = kWriteInplace;
    }
  }
}

void PushFCompute(const FCompute& fn,
                  const nnvm::Op* op,
                  const nnvm::NodeAttrs& attrs,
                  const Context& ctx,
                  const std::vector<engine::VarHandle>& read_vars,
                  const std::vector<engine::VarHandle>& write_vars,
                  const std::vector<Resource>& requested,
                  const std::vector<NDArray>& ndinputs,
                  const std::vector<NDArray>& ndoutputs,
                  const std::vector<uint32_t>& mutate_idx) {
  using namespace common;
  bool is_train = AutogradRuntime::Get()->IsTraining();
  Engine::Get()->PushAsync(
    [ctx, attrs, fn, ndinputs, ndoutputs, requested, is_train, mutate_idx](
        RunContext rctx,
        engine::CallbackOnComplete on_complete) {
      std::vector<TBlob> input_blobs, output_blobs;
      // pre-fcompute and post-fcompute storage fallback src NDArrays and dst NDArrays
      std::vector<NDArray> pre_temp_src, pre_temp_dst, post_temp_dst, post_temp_src;
      // mapping from index in input_blobs to index in pre_temp_dst
      std::unordered_map<uint32_t, uint32_t> in_temp_idx_map;
      // populate input blobs and output blobs
      SetupDefaultBlobs(ndinputs, &input_blobs, &pre_temp_src, &pre_temp_dst, &in_temp_idx_map);
      SetupDefaultBlobs(ndoutputs, &output_blobs, &post_temp_dst, &post_temp_src);
      // add mutable inputs to post temp list
      for (const auto idx : mutate_idx) {
        auto map_iter = in_temp_idx_map.find(idx);
        if (map_iter != in_temp_idx_map.end()) {
          post_temp_src.push_back(pre_temp_dst[map_iter->second]);
          post_temp_dst.push_back(ndinputs[idx]);
        }
      }
      OpContext opctx{is_train, rctx,
                      engine::CallbackOnComplete(),
                      requested};
      std::vector<OpReqType> req(output_blobs.size(), kWriteTo);
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
                    const std::vector<NDArray>& ndinputs,
                    const std::vector<NDArray>& ndoutputs) {
  Engine::Get()->PushAsync(
    [ctx, attrs, fn, ndinputs, ndoutputs, requested](
        RunContext rctx,
        engine::CallbackOnComplete on_complete) {
      std::vector<TBlob> input_blobs, output_blobs;
      OpContext opctx{false, rctx,
                      engine::CallbackOnComplete(),
                      requested};
      std::vector<OpReqType> req(ndoutputs.size(), kWriteTo);
      SetWriteInplaceReq(ndinputs, ndoutputs, &req);
      fn(attrs, opctx, ndinputs, req, ndoutputs);
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
                  const std::vector<NDArray>& ndinputs,
                  const std::vector<NDArray>& ndoutputs,
                  const std::vector<uint32_t>& mutate_idx) {
  using namespace common;
  static auto& fexec_type = nnvm::Op::GetAttr<FExecType>("FExecType");

  bool is_train = AutogradRuntime::Get()->IsTraining();
  ExecType exec_type = ExecType::kSync;
  if (fexec_type.count(op)) {
    exec_type = fexec_type[op](attrs);
  }

  auto fcompute = common::GetFCompute<FStatefulCompute>(op, "FStatefulCompute", ctx);
  if (fcompute != nullptr) {
    CHECK(exec_type == ExecType::kSync || exec_type == ExecType::kAsync);
    Engine::Get()->PushAsync(
      [state, fcompute, ndinputs, ndoutputs, requested, is_train, exec_type, mutate_idx](
          RunContext rctx,
          engine::CallbackOnComplete on_complete) {
        OpContext opctx{is_train, rctx, on_complete, requested};

        std::vector<TBlob> input_blobs, output_blobs;
        // pre-fcompute and post-fcompute storage fallback src NDArrays and dst NDArrays
        std::vector<NDArray> pre_temp_src, pre_temp_dst, post_temp_dst, post_temp_src;
        // mapping from index in input_blobs to index in pre_temp_dst
        std::unordered_map<uint32_t, uint32_t> in_temp_idx_map;
        // populate input blobs and output blobs
        SetupDefaultBlobs(ndinputs, &input_blobs, &pre_temp_src, &pre_temp_dst, &in_temp_idx_map);
        SetupDefaultBlobs(ndoutputs, &output_blobs, &post_temp_dst, &post_temp_src);
        // add mutable inputs to post temp list
        for (const auto idx : mutate_idx) {
          if (in_temp_idx_map.find(idx) != in_temp_idx_map.end()) {
            post_temp_src.push_back(pre_temp_dst[in_temp_idx_map[idx]]);
            post_temp_dst.push_back(ndinputs[idx]);
          }
        }
        std::vector<OpReqType> req(output_blobs.size(), kWriteTo);
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
    const auto& run = [state, fcompute_ex, ndinputs, ndoutputs, requested, is_train, exec_type](
          RunContext rctx,
          engine::CallbackOnComplete on_complete) {
        OpContext opctx{is_train, rctx, on_complete, requested};
        std::vector<OpReqType> req(ndoutputs.size(), kWriteTo);
        SetWriteInplaceReq(ndinputs, ndoutputs, &req);
        fcompute_ex(state, opctx, ndinputs, req, ndoutputs);
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

void ImperativeInvokeImpl(const Context& default_ctx,
                          nnvm::NodeAttrs&& attrs,
                          std::vector<NDArray>* p_ndinputs,
                          std::vector<NDArray>* p_ndoutputs,
                          std::vector<bool>* p_save_inputs = nullptr,
                          std::vector<bool>* p_save_outputs = nullptr) {
  static auto& ndfunc = nnvm::Op::GetAttr<FNDArrayFunction>("FNDArrayFunction");
  static auto& createop = nnvm::Op::GetAttr<FCreateOpState>("FCreateOpState");
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();

  const nnvm::Op *op = attrs.op;
  std::vector<NDArray>& ndinputs  = *p_ndinputs;
  std::vector<NDArray>& ndoutputs = *p_ndoutputs;


  if (ndfunc.count(op)) {
    ndfunc[op](attrs, ndinputs, &ndoutputs);
  } else {
    // TODO(piiswrong): infer ctx
    Context ctx;
    int stype;
    SetContext(&ctx, attrs, ndinputs, ndoutputs, default_ctx);
    SetShapeType(op, attrs, ctx, ndinputs, &ndoutputs, &stype);

    std::vector<engine::VarHandle> read_vars, write_vars;
    std::vector<Resource> requested;
    std::vector<uint32_t> mutate_idx;
    SetDependency(&read_vars, &write_vars, &requested, &mutate_idx,
        op, attrs, ctx, ndinputs, ndoutputs);

    FCompute fn = common::GetFCompute<FCompute>(op, "FCompute", ctx);
    FComputeEx fn_ex = common::GetFCompute<FComputeEx>(op, "FComputeEx", ctx);
    if (fn_ex && stype != kDefaultStorage) {
      PushFComputeEx(fn_ex, op, attrs, ctx, read_vars, write_vars,
          requested, ndinputs, ndoutputs);
      if (AutogradRuntime::Get()->IsRecording()) {
        AutogradRuntime::Get()->RecordOp(
            std::move(attrs), &ndinputs, &ndoutputs, OpStatePtr(),
            p_save_inputs, p_save_outputs);
      }
    } else if (fn) {
      PushFCompute(fn, op, attrs, ctx, read_vars, write_vars,
          requested, ndinputs, ndoutputs, mutate_idx);
      if (AutogradRuntime::Get()->IsRecording()) {
        AutogradRuntime::Get()->RecordOp(
            std::move(attrs), &ndinputs, &ndoutputs, OpStatePtr(),
            p_save_inputs, p_save_outputs);
      }
    } else if (createop.count(op)) {
      auto state =
          createop[op](attrs, ctx, ret->arg_shapes, ret->arg_types);
      write_vars.push_back(state.get_var());
      PushOperator(state, op, attrs, ctx, read_vars, write_vars,
          requested, ndinputs, ndoutputs, mutate_idx);
      if (AutogradRuntime::Get()->IsRecording()) {
        AutogradRuntime::Get()->RecordOp(
            std::move(attrs), &ndinputs, &ndoutputs, state,
            p_save_inputs, p_save_outputs);
      }
    } else {
      LOG(FATAL)
        << "Operator " << op->name << " is not implemented for "
        << (ctx.dev_mask() == gpu::kDevMask ? "GPU." : "CPU.");
    }
  }
}

inline void MXImperativeInvokeImpl(AtomicSymbolCreator creator,
                                   int num_inputs,
                                   NDArrayHandle *inputs,
                                   int *num_outputs,
                                   NDArrayHandle **outputs,
                                   int num_params,
                                   const char **param_keys,
                                   const char **param_vals) {
  const nnvm::Op* op = static_cast<nnvm::Op*>(creator);
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  NDArray** outarray = *reinterpret_cast<NDArray***>(outputs);

  nnvm::NodeAttrs attrs;
  SetOpAttrs(op, &attrs, num_inputs, num_params, param_keys, param_vals);

  int infered_num_outputs;
  int num_visible_outputs;
  SetNumOutputs(op, attrs, num_inputs, &infered_num_outputs, &num_visible_outputs);

  std::vector<NDArray> ndinputs, ndoutputs;
  SetNDInputsOutputs(op, &ndinputs, &ndoutputs, num_inputs, inputs,
      num_outputs, infered_num_outputs, num_visible_outputs, outarray);

  ImperativeInvokeImpl(Context::CPU(), std::move(attrs), &ndinputs, &ndoutputs);

  if (outarray == nullptr) {
    ret->ret_handles.clear();
    for (int i = 0; i < num_visible_outputs; ++i) {
      ret->ret_handles.push_back(
        reinterpret_cast<NDArrayHandle>(new NDArray(std::move(ndoutputs[i]))));
    }
    *outputs = dmlc::BeginPtr(ret->ret_handles);
  } else {
    for (int i = 0; i < *num_outputs; ++i) {
      *outarray[i] = std::move(ndoutputs[i]);
    }
  }
}

int MXImperativeInvoke(AtomicSymbolCreator creator,
                       int num_inputs,
                       NDArrayHandle *inputs,
                       int *num_outputs,
                       NDArrayHandle **outputs,
                       int num_params,
                       const char **param_keys,
                       const char **param_vals) {
  API_BEGIN();
  MXImperativeInvokeImpl(creator, num_inputs, inputs, num_outputs,
                         outputs, num_params, param_keys, param_vals);
  API_END();
}

int MXImperativeInvokeEx(AtomicSymbolCreator creator,
                         int num_inputs,
                         NDArrayHandle *inputs,
                         int *num_outputs,
                         NDArrayHandle **outputs,
                         int num_params,
                         const char **param_keys,
                         const char **param_vals,
                         const int **out_stypes) {  // outputs storage types
  API_BEGIN();
  MXImperativeInvokeImpl(creator, num_inputs, inputs, num_outputs, outputs,
                         num_params, param_keys, param_vals);
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  NDArray** output_nds = reinterpret_cast<NDArray**>(*outputs);
  ret->out_types.resize(*num_outputs);
  for (int i = 0; i < *num_outputs; ++i) {
    ret->out_types[i] = output_nds[i]->storage_type();
  }
  *out_stypes = dmlc::BeginPtr(ret->out_types);
  API_END();
}

int MXCreateCachedOp(SymbolHandle handle,
                     CachedOpHandle *out) {
  nnvm::Symbol* sym = static_cast<nnvm::Symbol*>(handle);

  API_BEGIN();
  nnvm::Graph *g = new nnvm::Graph;
  g->outputs = sym->outputs;
  auto vars = sym->ListInputs(nnvm::Symbol::kAll);
  CHECK_GE(vars.size(), 1) << "CachedOp must have at least 1 input.";
  g->attrs["vars"] = std::make_shared<dmlc::any>(std::move(vars));

  const nnvm::IndexedGraph& idx = g->indexed_graph();
  std::vector<std::vector<bool> > save_inputs(idx.num_nodes());
  std::vector<std::vector<bool> > save_outputs(idx.num_nodes());
  for (size_t i = 0; i < idx.num_nodes(); ++i) {
    nnvm::NodePtr node = nnvm::Node::Create();
    node->attrs = idx[i].source->attrs;
    AutogradRuntime::Get()->GetBackwardDependency(
        node, idx[i].source->num_inputs(), idx[i].source->num_outputs(),
        &save_inputs[i], &save_outputs[i]);
  }
  g->attrs["save_inputs"] = std::make_shared<dmlc::any>(std::move(save_inputs));
  g->attrs["save_outputs"] = std::make_shared<dmlc::any>(std::move(save_outputs));

  *out = g;
  API_END();
}

int MXFreeCachedOp(CachedOpHandle handle) {
  nnvm::Graph *g = static_cast<nnvm::Graph*>(handle);
  API_BEGIN();
  delete g;
  API_END();
}

int MXInvokeCachedOp(CachedOpHandle handle,
                     int num_inputs,
                     NDArrayHandle *inputs,
                     int *num_outputs,
                     NDArrayHandle **outputs) {
  nnvm::Graph *g = static_cast<nnvm::Graph*>(handle);
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  NDArray** outarray = *reinterpret_cast<NDArray***>(outputs);

  API_BEGIN();
  const std::vector<nnvm::NodePtr>& vars =
      g->GetAttr<std::vector<nnvm::NodePtr> >("vars");
  std::vector<std::vector<bool> > save_inputs =
      g->GetAttr<std::vector<std::vector<bool> > >("save_inputs");
  std::vector<std::vector<bool> > save_outputs =
      g->GetAttr<std::vector<std::vector<bool> > >("save_outputs");
  const nnvm::IndexedGraph& idx = g->indexed_graph();
  CHECK_EQ(static_cast<size_t>(num_inputs), vars.size())
      << "Actually number of inputs differs from expected number of inputs";
  Context default_ctx = static_cast<NDArray*>(inputs[0])->ctx();

  std::vector<NDArray> buff(idx.num_node_entries());
  for (size_t i = 0; i < vars.size(); ++i) {
    buff[idx.entry_id(idx.node_id(vars[i].get()), 0)] =
        *static_cast<NDArray*>(inputs[i]);
  }

  for (size_t i = 0; i < idx.num_nodes(); ++i) {
    const nnvm::IndexedGraph::Node& node = idx[i];
    if (node.source->attrs.op == nullptr) continue;
    std::vector<NDArray> in;
    in.reserve(node.inputs.size());
    for (const auto& j : node.inputs) {
      in.emplace_back(buff[idx.entry_id(j)]);
    }
    std::vector<NDArray> out(node.source->num_outputs());
    ImperativeInvokeImpl(default_ctx, nnvm::NodeAttrs(node.source->attrs), &in, &out,
                         &save_inputs[i], &save_outputs[i]);

    for (size_t j = 0; j < node.source->num_outputs(); ++j) {
      buff[idx.entry_id(i, j)] = std::move(out[j]);
    }
  }

  if (outarray == nullptr) {
    ret->ret_handles.clear();
    for (const auto& i : idx.outputs()) {
      ret->ret_handles.push_back(
        reinterpret_cast<NDArrayHandle>(
          new NDArray(buff[idx.entry_id(i)])));
    }
    *num_outputs = idx.outputs().size();
    *outputs = dmlc::BeginPtr(ret->ret_handles);
  } else {
    CHECK_EQ(static_cast<size_t>(*num_outputs), idx.outputs().size())
        << "Specifed number of output differs from expected number of outputs";
    for (size_t i = 0; i < idx.outputs().size(); ++i) {
      *outarray[i] = buff[idx.entry_id(idx.outputs()[i])];
    }
  }
  API_END();
}

int MXInvokeCachedOpEx(CachedOpHandle handle,
                       int num_inputs,
                       NDArrayHandle *inputs,
                       int *num_outputs,
                       NDArrayHandle **outputs,
                       const int **out_stypes) {  // outputs storage types
  API_BEGIN();
  MXInvokeCachedOp(handle, num_inputs, inputs, num_outputs, outputs);
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  NDArray** output_nds = reinterpret_cast<NDArray**>(*outputs);
  ret->out_types.resize(*num_outputs);
  for (int i = 0; i < *num_outputs; ++i) {
    ret->out_types[i] = output_nds[i]->storage_type();
  }
  *out_stypes = dmlc::BeginPtr(ret->out_types);
  API_END();
}

int MXAutogradIsTraining(bool* curr) {
  API_BEGIN();
  *curr = AutogradRuntime::Get()->IsTraining();
  API_END();
}

int MXAutogradSetIsTraining(int is_training, int* prev) {
  API_BEGIN();
  *prev = AutogradRuntime::Get()->SetIsTraining(static_cast<bool>(is_training));
  API_END();
}

int MXAutogradIsRecording(bool* curr) {
  API_BEGIN();
  *curr = AutogradRuntime::Get()->IsRecording();
  API_END();
}

int MXAutogradSetIsRecording(int is_recording, int* prev) {
  API_BEGIN();
  *prev = AutogradRuntime::Get()->SetIsRecording(static_cast<bool>(is_recording));
  API_END();
}

int MXAutogradMarkVariables(mx_uint num_var,
                            NDArrayHandle *var_handles,
                            mx_uint *reqs_array,
                            NDArrayHandle *grad_handles) {
  API_BEGIN();
  std::vector<NDArray*> variables, gradients;
  std::vector<mx_uint> grad_reqs;
  variables.reserve(num_var);
  gradients.reserve(num_var);
  grad_reqs.reserve(num_var);
  for (mx_uint i = 0; i < num_var; ++i) {
    variables.emplace_back(static_cast<NDArray*>(var_handles[i]));
    gradients.emplace_back(static_cast<NDArray*>(grad_handles[i]));
    grad_reqs.emplace_back(reqs_array[i]);
  }
  AutogradRuntime::Get()->MarkVariables(variables, grad_reqs, gradients);
  API_END();
}

int MXAutogradComputeGradient(mx_uint num_output,
                              NDArrayHandle *output_handles) {
  return MXAutogradBackward(num_output, output_handles, nullptr, 0);
}

int MXAutogradBackward(mx_uint num_output,
                       NDArrayHandle *output_handles,
                       NDArrayHandle *ograd_handles,
                       int retain_graph) {
  return MXAutogradBackwardEx(num_output, output_handles, ograd_handles, retain_graph, true);
}

int MXAutogradBackwardEx(mx_uint num_output,
                         NDArrayHandle *output_handles,
                         NDArrayHandle *ograd_handles,
                         int retain_graph,
                         int is_train) {
  API_BEGIN();
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();

  std::vector<NDArray> outputs, ograds;
  outputs.reserve(num_output);
  for (mx_uint i = 0; i < num_output; ++i) {
    outputs.emplace_back(*static_cast<NDArray*>(output_handles[i]));
  }

  ograds.reserve(num_output);
  for (mx_uint i = 0; i < num_output; ++i) {
    if (ograd_handles != nullptr && ograd_handles[i] != nullptr) {
      ograds.emplace_back(*static_cast<NDArray*>(ograd_handles[i]));
    } else {
      ograds.emplace_back();
    }
  }

  AutogradRuntime::Get()->ComputeGradient(outputs, ograds, retain_graph, is_train);
  API_END();
}

int MXAutogradGetSymbol(NDArrayHandle handle, SymbolHandle *out) {
  API_BEGIN();
  NDArray *head = reinterpret_cast<NDArray*>(handle);
  auto sym = new nnvm::Symbol(head->get_autograd_symbol());
  *out = reinterpret_cast<SymbolHandle>(sym);
  API_END();
}
