/*!
 * Copyright (c) 2016 by Contributors
 * \file attach_op_execs_pass.cc
 * \brief Operator executor to execute each operator.
 */
#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <mxnet/op_attr_types.h>
#include <nnvm/graph_attr_types.h>
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

// forward executor
class StatefulComputeExecutor : public OpExecutor {
 public:
  void Run(RunContext rctx) override {
    static auto& fcompute_cpu =
        nnvm::Op::GetAttr<FStatefulCompute>("FStatefulCompute<cpu>");
    static auto& fcompute_gpu =
        nnvm::Op::GetAttr<FStatefulCompute>("FStatefulCompute<gpu>");

    op_ctx.run_ctx = rctx;
    if (rctx.get_ctx().dev_mask() == cpu::kDevMask) {
      fcompute_cpu[op_](state_, op_ctx, in_data_, req, out_data_);
    } else if (rctx.get_ctx().dev_mask() == gpu::kDevMask) {
      fcompute_gpu[op_](state_, op_ctx, in_data_, req, out_data_);
    } else {
      LOG(FATAL) << "Unknown device mask";
    }
#if MKL_EXPERIMENTAL == 1
    mkl_tblobs_prv_to_cpu(in_data_);
    mkl_tblobs_prv_to_cpu(out_data_);
#endif
  }

  void Setup() override {
    in_data_.clear();
    for (size_t i = 0; i < in_array.size(); ++i) {
      in_data_.push_back(in_array[i].data());
    }
    out_data_.clear();
    for (size_t i = 0; i < out_array.size(); ++i) {
      out_data_.push_back(out_array[i].data());
    }
  }
  Operator::ExecType exec_type() const override {
    return Operator::kSync;// op_->exec_type();
  }
  explicit StatefulComputeExecutor(const nnvm::Op* op,
                             std::shared_ptr<dmlc::any> state)
      : op_(op), state_(state) {}

 private:
  friend Graph AttachOpExecs(Graph g);
  const nnvm::Op *op_;
  std::shared_ptr<dmlc::any> state_;
  std::vector<TBlob> in_data_, out_data_;
};


// fcompute executor executor
class FComputeExecutor : public OpExecutor {
 public:
  void Run(RunContext rctx) override {
    op_ctx.run_ctx = rctx;
    fcompute_(attrs_, op_ctx, in_data_, req, out_data_);
#if MKL_EXPERIMENTAL == 1
    mkl_tblobs_prv_to_cpu(in_data_);
    mkl_tblobs_prv_to_cpu(out_data_);
#endif
  }
  void Setup() override {
    in_data_.resize(in_array.size());
    out_data_.resize(out_array.size());
    auto get_blob =  [](const NDArray& nd) {
      return nd.data();
    };
    std::transform(in_array.begin(), in_array.end(), in_data_.begin(), get_blob);
    std::transform(out_array.begin(), out_array.end(), out_data_.begin(), get_blob);
  }
  Operator::ExecType exec_type() const override {
    return Operator::kSync;
  }
  explicit FComputeExecutor(FCompute fcompute, const NodeAttrs& attrs)
      : fcompute_(fcompute), attrs_(attrs) {
  }

  static FCompute GetFCompute(const Op* op, Context ctx) {
    static auto& fcompute_cpu = nnvm::Op::GetAttr<FCompute>("FCompute<cpu>");
    static auto& fcompute_gpu = nnvm::Op::GetAttr<FCompute>("FCompute<gpu>");
    if (ctx.dev_mask() == cpu::kDevMask) {
      return fcompute_cpu.get(op, nullptr);
    } else if (ctx.dev_mask() == gpu::kDevMask) {
      return fcompute_gpu.get(op, nullptr);
    } else {
      LOG(FATAL) << "Unknown device mask";
      return nullptr;
    }
  }

 private:
  FCompute fcompute_;
  NodeAttrs attrs_;
  std::vector<TBlob> in_data_, out_data_;
};

// pass to attach operator executors
Graph AttachOpExecs(Graph g) {
  using nnvm::DTypeVector;
  using nnvm::ShapeVector;
  using nnvm::FMutateInputs;

  auto& fcreate_op_state = nnvm::Op::GetAttr<FCreateOpState>("FCreateOpState");
  auto& fmutate_inputs = nnvm::Op::GetAttr<FMutateInputs>("FMutateInputs");
  auto& is_layer_backward = nnvm::Op::GetAttr<bool>("TIsLayerOpBackward");

  const auto& vdtype = g.GetAttr<DTypeVector>("dtype");
  const auto& vshape = g.GetAttr<ShapeVector>("shape");
  const auto& vctx = g.GetAttr<ContextVector>("context");
  const auto& saved_states = g.GetAttr<
    std::unordered_map<const nnvm::Node*, std::shared_ptr<dmlc::any>>>("saved_states");

  // get the graph
  const auto& idx = g.indexed_graph();
  std::vector<std::shared_ptr<OpExecutor> > ret(idx.num_nodes());

  // initialize the nodes
  for (size_t i = 0; i < idx.num_nodes(); ++i) {
    const auto& inode = idx[i];
    if (inode.source->is_variable()) continue;
    std::vector<uint32_t> mutate_index;
    if (fmutate_inputs.count(inode.source->op())) {
      mutate_index = fmutate_inputs[inode.source->op()](inode.source->attrs);
    }
    if (fcreate_op_state.count(inode.source->op())) {
      std::vector<TShape> ishape;
      std::vector<int> itype;
      for (const auto& e : inode.inputs) {
        ishape.emplace_back(vshape[idx.entry_id(e)]);
        itype.emplace_back(vdtype[idx.entry_id(e)]);
      }
      std::shared_ptr<dmlc::any> state;
      if (saved_states.count(inode.source)) {
        state = saved_states.at(inode.source);
      } else {
        state = fcreate_op_state[inode.source->op()](
            inode.source->attrs, vctx[i], ishape, itype);
      }
      ret[i] = std::make_shared<StatefulComputeExecutor>(inode.source->op(), state);
    } else if (is_layer_backward.get(inode.source->op(), false)) {
      CHECK_GE(inode.control_deps.size(), 1);
      uint32_t fwd_id = inode.control_deps[0];
      CHECK(vctx[fwd_id] == vctx[i]);
      CHECK(ret[fwd_id] != nullptr);
      ret[i] = std::make_shared<StatefulComputeExecutor>(
          inode.source->op(),
          dynamic_cast<StatefulComputeExecutor*>(ret[fwd_id].get())->state_);
    } else {
      FCompute fcompute = FComputeExecutor::GetFCompute(inode.source->op(), vctx[i]);
      if (fcompute != nullptr) {
        ret[i] = std::make_shared<FComputeExecutor>(fcompute, inode.source->attrs);
      } else {
        LOG(INFO) << "FCompute not registered " << inode.source->op()->name;
      }
    }
  }
  g.attrs["op_execs"] = std::make_shared<nnvm::any>(ret);
  return g;
}

}  // namespace exec
}  // namespace mxnet
