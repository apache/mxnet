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

// stateful compute executor
class StatefulComputeExecutor : public OpExecutor {
 public:
  void Run(RunContext rctx, bool is_gpu) override {
    using namespace common;
    op_ctx.run_ctx = rctx;
    if (is_gpu) {
#if MXNET_USE_CUDA
      CastNonDefaultStorage<gpu>(temp_in_src_, temp_in_dst_, op_ctx);
      CastNonDefaultStorage<gpu>(temp_out_src_, temp_out_dst_, op_ctx);
      fcompute_(state_, op_ctx, in_data_, req, out_data_);
      CastNonDefaultStorage<gpu>(temp_out_dst_, temp_out_src_, op_ctx);
#else
      LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
#endif
    } else {
      CastNonDefaultStorage<cpu>(temp_in_src_, temp_in_dst_, op_ctx);
      CastNonDefaultStorage<cpu>(temp_out_src_, temp_out_dst_, op_ctx);
      fcompute_(state_, op_ctx, in_data_, req, out_data_);
      CastNonDefaultStorage<cpu>(temp_out_dst_, temp_out_src_, op_ctx);
    }
#if MKL_EXPERIMENTAL == 1
    mkl_tblobs_prv_to_cpu(in_data_);
    mkl_tblobs_prv_to_cpu(out_data_);
#endif
  }

  void Setup() override {
    using namespace common;
    in_data_.clear(); out_data_.clear();
    temp_in_src_.clear(); temp_in_dst_.clear();
    temp_out_src_.clear(); temp_out_dst_.clear();
    GetDefaultBlobs(in_array, &in_data_, &temp_in_src_, &temp_in_dst_);
    GetDefaultBlobs(out_array, &out_data_, &temp_out_src_, &temp_out_dst_);
  }

  ExecType exec_type() const override {
    return exec_type_;
  }

  engine::VarHandle var() const override {
    return state_.get_var();
  }

  explicit StatefulComputeExecutor(const OpStatePtr& state,
                                   const FStatefulCompute& fcompute,
                                   ExecType exec_type)
      : state_(state), fcompute_(fcompute), exec_type_(exec_type) {}

 private:
  friend Graph AttachOpExecs(Graph g);
  OpStatePtr state_;
  FStatefulCompute fcompute_;
  ExecType exec_type_;
  std::vector<TBlob> in_data_, out_data_;
  std::vector<NDArray> temp_in_src_, temp_in_dst_, temp_out_src_, temp_out_dst_;
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
class FComputeExecutor : public OpExecutor {
 public:
  void Run(RunContext rctx, bool is_gpu) override {
    using namespace common;
    // TODO(haibin) avoid repeating this if all inputs are already in default-storage
    op_ctx.run_ctx = rctx;
    if (is_gpu) {
#if MXNET_USE_CUDA
      CastNonDefaultStorage<gpu>(temp_in_src_, temp_in_dst_, op_ctx);
      CastNonDefaultStorage<gpu>(temp_out_src_, temp_out_dst_, op_ctx);
      fcompute_(attrs_, op_ctx, in_data_, req, out_data_);
      CastNonDefaultStorage<gpu>(temp_out_dst_, temp_out_src_, op_ctx);
#else
      LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
#endif
    } else {
      CastNonDefaultStorage<cpu>(temp_in_src_, temp_in_dst_, op_ctx);
      CastNonDefaultStorage<cpu>(temp_out_src_, temp_out_dst_, op_ctx);
      fcompute_(attrs_, op_ctx, in_data_, req, out_data_);
      CastNonDefaultStorage<cpu>(temp_out_dst_, temp_out_src_, op_ctx);
    }
#if MKL_EXPERIMENTAL == 1
    mkl_tblobs_prv_to_cpu(in_data_);
    mkl_tblobs_prv_to_cpu(out_data_);
#endif
  }

  void Setup() override {
    using namespace common;
    in_data_.clear(); out_data_.clear();
    temp_in_src_.clear(); temp_in_dst_.clear();
    temp_out_src_.clear(); temp_out_dst_.clear();
    GetDefaultBlobs(in_array, &in_data_, &temp_in_src_, &temp_in_dst_);
    GetDefaultBlobs(out_array, &out_data_, &temp_out_src_, &temp_out_dst_);
  }

  ExecType exec_type() const override {
    return exec_type_;
  }

  explicit FComputeExecutor(const NodeAttrs& attrs, FCompute fcompute,
                            ExecType exec_type)
      : attrs_(attrs), fcompute_(fcompute), exec_type_(exec_type) {
  }

 private:
  NodeAttrs attrs_;
  FCompute fcompute_;
  ExecType exec_type_;
  std::vector<TBlob> in_data_, out_data_;
  std::vector<NDArray> temp_in_src_, temp_in_dst_, temp_out_src_, temp_out_dst_;
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
  const auto& dispatch_stypes = g.GetAttr<StorageTypeVector>("dispatch_stypes");


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
      FStatefulCompute fcompute = common::GetFCompute<FStatefulCompute>(
          op, "FStatefulCompute", vctx[i]);
      if (fcompute != nullptr) {
        ret[i] = std::make_shared<StatefulComputeExecutor>(state, fcompute, exec_type);
      } else {
        FStatefulComputeEx fcompute_ex = common::GetFCompute<FStatefulComputeEx>(
            op, "FStatefulComputeEx", vctx[i]);
        CHECK(fcompute_ex != nullptr)
            << "One of FStatefulCompute and FStatefulComputeEx must be registered "
            << "for stateful operator " << op->name;
        ret[i] = std::make_shared<StatefulComputeExExecutor>(state, fcompute_ex, exec_type);
      }
    } else if (is_layer_backward.get(op, false)) {
      CHECK_GE(inode.control_deps.size(), 1);
      uint32_t fwd_id = inode.control_deps[0];
      CHECK(vctx[fwd_id] == vctx[i]);
      CHECK(ret[fwd_id] != nullptr);
      FStatefulCompute fcompute = common::GetFCompute<FStatefulCompute>(
          op, "FStatefulCompute", vctx[i]);
      if (fcompute != nullptr) {
        ret[i] = std::make_shared<StatefulComputeExecutor>(
            dynamic_cast<StatefulComputeExecutor*>(ret[fwd_id].get())->state_,
            fcompute, exec_type);
      } else {
        FStatefulComputeEx fcompute_ex = common::GetFCompute<FStatefulComputeEx>(
            op, "FStatefulComputeEx", vctx[i]);
        CHECK(fcompute_ex != nullptr)
            << "One of FStatefulCompute and FStatefulComputeEx must be registered "
            << "for stateful operator " << op->name;
        ret[i] = std::make_shared<StatefulComputeExExecutor>(
            dynamic_cast<StatefulComputeExExecutor*>(ret[fwd_id].get())->state_,
            fcompute_ex, exec_type);
      }
    } else {
      FCompute fcompute = common::GetFCompute<FCompute>(op, "FCompute", vctx[i]);
      FComputeEx fcomp_ex = common::GetFCompute<FComputeEx>(op, "FComputeEx", vctx[i]);
      if (fcomp_ex != nullptr && dispatch_stypes[i] != kDefaultStorage) {
        ret[i] = std::make_shared<FComputeExExecutor>(
            inode.source->attrs, fcomp_ex, exec_type);
      } else if (fcompute != nullptr) {
        ret[i] = std::make_shared<FComputeExecutor>(
            inode.source->attrs, fcompute, exec_type);
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
