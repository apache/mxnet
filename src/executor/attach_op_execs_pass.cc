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
 * Copyright (c) 2016 by Contributors
 * \file attach_op_execs_pass.cc
 * \brief Operator executor to execute each operator.
 */
#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <mxnet/op_attr_types.h>
#include <mxnet/graph_attr_types.h>
#include <nnvm/graph_attr_types.h>
#include "../common/utils.h"
#include "../common/exec_utils.h"
#include "./exec_pass.h"
#if MXNET_USE_MKL2017 == 1
#include <mkl_memory.h>
#include "../operator/mkl/mkl_memory-inl.h"
#include "../operator/mkl/mkl_util-inl.h"
#endif
namespace mxnet {

namespace op {
const OperatorProperty* OpPropGetOpProperty(const NodeAttrs& attrs);
}  // namespace op

namespace exec {

// abstract OpExecutor which provides storage fallback procedure on
// non-default inputs and outputs
// FComputeExecutor and FStatefulComputeExecutor inherit from this class
class StorageFallbackOpExecutor : public OpExecutor {
 public:
  explicit StorageFallbackOpExecutor(const std::vector<uint32_t> &mutate_idx)
      : mutate_idx_(mutate_idx) {}

  void Setup() override {
    init_ = false;
  }

 protected:
  // initialize the data blobs
  void InitBlobs() {
    using namespace common;
    if (!init_) {
      in_data_.clear(); out_data_.clear();
      pre_temp_src_.clear(); pre_temp_dst_.clear();
      post_temp_src_.clear(); post_temp_dst_.clear();
      in_temp_idx_map_.clear();
      SetupDefaultBlobsInOut(in_array, out_array, &in_data_, &out_data_,
                             &pre_temp_src_, &pre_temp_dst_,
                             &post_temp_src_, &post_temp_dst_,
                             &in_temp_idx_map_, mutate_idx_);
      init_ = true;
    }
  }

  // storage fallback before fcompute is launched
  void PreFCompute(bool is_gpu) {
    InitBlobs();
    common::CastNonDefaultStorage(pre_temp_src_, pre_temp_dst_, op_ctx, is_gpu);
  }

  // storage fallback after fcompute is completed
  void PostFCompute(bool is_gpu) {
    common::CastNonDefaultStorage(post_temp_src_, post_temp_dst_, op_ctx, is_gpu);
  }

  // default storage tensor blobs for fcompute
  std::vector<TBlob> in_data_, out_data_;
  // source NDArray for cast storage
  std::vector<NDArray> pre_temp_src_, post_temp_src_;
  // destination NDArray for cast storage
  std::vector<NDArray> pre_temp_dst_, post_temp_dst_;
  // mapping from index in input_blobs to index in pre_temp_dst
  std::unordered_map<uint32_t, uint32_t> in_temp_idx_map_;
  // indices of mutatable inputs
  std::vector<uint32_t> mutate_idx_;
  // whether blobs are initialized
  bool init_;
};


// stateful compute executor
class StatefulComputeExecutor : public StorageFallbackOpExecutor {
 public:
  void Run(RunContext rctx, bool is_gpu) override {
    op_ctx.run_ctx = rctx;
    PreFCompute(is_gpu);
    fcompute_(state_, op_ctx, in_data_, req, out_data_);
    PostFCompute(is_gpu);
#if MKL_EXPERIMENTAL == 1
    mkl_tblobs_prv_to_cpu(in_data_);
    mkl_tblobs_prv_to_cpu(out_data_);
#endif
  }

  ExecType exec_type() const override {
    return exec_type_;
  }

  engine::VarHandle var() const override {
    return state_.get_var();
  }

  explicit StatefulComputeExecutor(const OpStatePtr& state,
                                   const FStatefulCompute& fcompute,
                                   ExecType exec_type,
                                   const std::vector<uint32_t> &mutate_idx)
      : StorageFallbackOpExecutor(mutate_idx),
        state_(state), fcompute_(fcompute), exec_type_(exec_type) {}

 private:
  friend Graph AttachOpExecs(Graph g);
  OpStatePtr state_;
  FStatefulCompute fcompute_;
  ExecType exec_type_;
};


// stateful compute_ex executor
class StatefulComputeExExecutor : public OpExecutor {
 public:
  void Run(RunContext rctx, bool is_gpu) override {
    op_ctx.run_ctx = rctx;
    fcompute_(state_, op_ctx, in_array, req, out_array);
  }

  void Setup() override {}

  ExecType exec_type() const override {
    return exec_type_;
  }

  engine::VarHandle var() const override {
    return state_.get_var();
  }

  explicit StatefulComputeExExecutor(const OpStatePtr& state,
                                     const FStatefulComputeEx& fcompute,
                                     ExecType exec_type)
      : state_(state), fcompute_(fcompute), exec_type_(exec_type) {}

 private:
  friend Graph AttachOpExecs(Graph g);
  OpStatePtr state_;
  FStatefulComputeEx fcompute_;
  ExecType exec_type_;
};


// fcompute executor
class FComputeExecutor : public StorageFallbackOpExecutor {
 public:
  void Run(RunContext rctx, bool is_gpu) override {
    using namespace common;
    op_ctx.run_ctx = rctx;
    PreFCompute(is_gpu);
    fcompute_(attrs_, op_ctx, in_data_, req, out_data_);
    PostFCompute(is_gpu);
#if MKL_EXPERIMENTAL == 1
    mkl_tblobs_prv_to_cpu(in_data_);
    mkl_tblobs_prv_to_cpu(out_data_);
#endif
  }

  ExecType exec_type() const override {
    return exec_type_;
  }

  explicit FComputeExecutor(const NodeAttrs& attrs, FCompute fcompute,
                            ExecType exec_type, const std::vector<uint32_t> &mutate_idx)
      : StorageFallbackOpExecutor(mutate_idx),
        attrs_(attrs), fcompute_(fcompute), exec_type_(exec_type) {
  }

 private:
  NodeAttrs attrs_;
  FCompute fcompute_;
  ExecType exec_type_;
};

// fcompute_ex executor
class FComputeExExecutor : public OpExecutor {
 public:
  void Run(RunContext rctx, bool is_gpu) override {
    op_ctx.run_ctx = rctx;
    fcompute_(attrs_, op_ctx, in_array, req, out_array);
  }

  void Setup() override {}

  ExecType exec_type() const override {
    return exec_type_;
  }

  explicit FComputeExExecutor(const NodeAttrs& attrs, FComputeEx fcompute,
                              ExecType exec_type)
      : attrs_(attrs), fcompute_(fcompute), exec_type_(exec_type) {
  }

 private:
  NodeAttrs attrs_;
  FComputeEx fcompute_;
  ExecType exec_type_;
};

// pass to attach operator executors
Graph AttachOpExecs(Graph g) {
  using nnvm::DTypeVector;
  using nnvm::ShapeVector;
  using nnvm::FMutateInputs;

  auto& fcreate_op_state = nnvm::Op::GetAttr<FCreateOpState>("FCreateOpState");
  auto& fmutate_inputs = nnvm::Op::GetAttr<FMutateInputs>("FMutateInputs");
  auto& fexec_type = nnvm::Op::GetAttr<FExecType>("FExecType");
  auto& is_layer_backward = nnvm::Op::GetAttr<bool>("TIsLayerOpBackward");

  const auto& vdtype = g.GetAttr<DTypeVector>("dtype");
  const auto& vshape = g.GetAttr<ShapeVector>("shape");
  const auto& vctx = g.GetAttr<ContextVector>("context");
  const auto& saved_states = g.GetAttr<
    std::unordered_map<const nnvm::Node*, OpStatePtr> >("saved_states");
  const auto& dispatch_modes = g.GetAttr<DispatchModeVector>("dispatch_mode");

  // get the graph
  const auto& idx = g.indexed_graph();
  std::vector<std::shared_ptr<OpExecutor> > ret(idx.num_nodes());

  // initialize the nodes
  for (size_t i = 0; i < idx.num_nodes(); ++i) {
    const auto& inode = idx[i];
    if (inode.source->is_variable()) continue;
    const nnvm::Op *op = inode.source->op();
    ExecType exec_type = ExecType::kSync;
    std::vector<uint32_t> mutate_index;
    if (fmutate_inputs.count(op)) {
      mutate_index = fmutate_inputs[op](inode.source->attrs);
    }
    if (fexec_type.count(op)) {
      exec_type = fexec_type[op](inode.source->attrs);
    }
    CHECK(dispatch_modes[i] != DispatchMode::kUndefined);
    if (fcreate_op_state.count(op)) {
      std::vector<TShape> ishape;
      std::vector<int> itype;
      for (const auto& e : inode.inputs) {
        ishape.emplace_back(vshape[idx.entry_id(e)]);
        itype.emplace_back(vdtype[idx.entry_id(e)]);
      }

      OpStatePtr state;
      if (saved_states.count(inode.source)) {
        state = saved_states.at(inode.source);
      } else {
        state = fcreate_op_state[op](
            inode.source->attrs, vctx[i], ishape, itype);
      }
      FStatefulComputeEx fcompute_ex = common::GetFCompute<FStatefulComputeEx>(
          op, "FStatefulComputeEx", vctx[i]);
      // FStatefulComputeEx is dispatched only when dispatch_mode is DispatchMode::kFComputeEx
      if (fcompute_ex != nullptr && dispatch_modes[i] == DispatchMode::kFComputeEx) {
        ret[i] = std::make_shared<StatefulComputeExExecutor>(state, fcompute_ex, exec_type);
      } else {
        FStatefulCompute fcompute = common::GetFCompute<FStatefulCompute>(
            op, "FStatefulCompute", vctx[i]);
        CHECK(fcompute != nullptr)
            << "One of FStatefulCompute and FStatefulComputeEx must be registered "
            << "for stateful operator " << op->name;
        ret[i] = std::make_shared<StatefulComputeExecutor>(state, fcompute,
                                                           exec_type, mutate_index);
      }
    } else if (is_layer_backward.get(op, false)) {
      CHECK_GE(inode.control_deps.size(), 1);
      uint32_t fwd_id = inode.control_deps[0];
      CHECK(vctx[fwd_id] == vctx[i]);
      CHECK(ret[fwd_id] != nullptr);
      FStatefulComputeEx fcompute_ex = common::GetFCompute<FStatefulComputeEx>(
          op, "FStatefulComputeEx", vctx[i]);
      // FStatefulComputeEx is dispatched only when dispatch_mode is DispatchMode::kFComputeEx
      if (fcompute_ex != nullptr && dispatch_modes[i] == DispatchMode::kFComputeEx) {
        ret[i] = std::make_shared<StatefulComputeExExecutor>(
            dynamic_cast<StatefulComputeExExecutor*>(ret[fwd_id].get())->state_,
            fcompute_ex, exec_type);
      } else {
        FStatefulCompute fcompute = common::GetFCompute<FStatefulCompute>(
            op, "FStatefulCompute", vctx[i]);
        CHECK(fcompute != nullptr)
            << "One of FStatefulCompute and FStatefulComputeEx must be registered "
            << "for stateful operator " << op->name;
        ret[i] = std::make_shared<StatefulComputeExecutor>(
            dynamic_cast<StatefulComputeExecutor*>(ret[fwd_id].get())->state_,
            fcompute, exec_type, mutate_index);
      }
    } else {
      FCompute fcompute = common::GetFCompute<FCompute>(op, "FCompute", vctx[i]);
      FComputeEx fcomp_ex = common::GetFCompute<FComputeEx>(op, "FComputeEx", vctx[i]);
      if (fcomp_ex != nullptr && dispatch_modes[i] == DispatchMode::kFComputeEx) {
        ret[i] = std::make_shared<FComputeExExecutor>(
            inode.source->attrs, fcomp_ex, exec_type);
      } else if (fcompute != nullptr) {
        ret[i] = std::make_shared<FComputeExecutor>(
            inode.source->attrs, fcompute, exec_type, mutate_index);
      } else {
        LOG(INFO) << "Neither FCompute nor FComputeEx registered " << op->name;
      }
    }
  }
  g.attrs["op_execs"] = std::make_shared<nnvm::any>(ret);
  return g;
}

}  // namespace exec
}  // namespace mxnet
