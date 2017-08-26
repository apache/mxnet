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
#include "../operator/mkl/mkldnn_memory-inl.h"
#include "../operator/mkl/mkl_util-inl.h"
#endif
#if MKL_EXPERIMENTAL == 1
#include <mkl_memory.h>
#include "../operator/mkl/mkldnn_memory-inl.h"
#include "../operator/mkl/mkl_util-inl.h"
#endif
#include <iostream>
#include <string>
#include <map>

#if defined(_MSC_VER)
#include <windows.h>
#endif

//#define ENABLE_TIME_PROFILE_NEW
#ifdef ENABLE_TIME_PROFILE_NEW
#include "../operator/mkl/ptimerutil.hpp"
#endif
// #define ENABLE_TIME_PROFILE
#ifdef ENABLE_TIME_PROFILE
#include <time.h>
class Timer {
public:
  Timer();
  virtual void Start();
  virtual void Stop();
  virtual float MilliSeconds();
private:
  float elapsed_milliseconds_;
#if !defined(_MSC_VER)
  timeval start, end;
#else
  LARGE_INTEGER Frequency, StartingTime, EndingTime;
#endif
};
#endif
#ifdef ENABLE_TIME_PROFILE
#if !defined(_MSC_VER)
#include <sys/time.h>
#endif
Timer::Timer()
{
  elapsed_milliseconds_ = 0;
#if defined(_MSC_VER)
  QueryPerformanceFrequency(&Frequency);
#endif
}

void Timer::Start()
{
#if !defined(_MSC_VER)
  gettimeofday(&start, NULL);
#else
  QueryPerformanceCounter(&StartingTime);
#endif
}
void Timer::Stop()
{
#if !defined(_MSC_VER)
  gettimeofday(&end, NULL);
#else
  QueryPerformanceCounter(&EndingTime);
#endif
}
float Timer::MilliSeconds()
{
#if !defined(_MSC_VER)
  long seconds, useconds;
  seconds = end.tv_sec - start.tv_sec;
  useconds = end.tv_usec - start.tv_usec;

  elapsed_milliseconds_ = ((seconds) * 1000 + useconds / 1000.0) + 0.5;
#else
  LARGE_INTEGER ElapsedMicroseconds;
  ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
  //
  // We now have the elapsed number of ticks, along with the
  // number of ticks-per-second. We use these values
  // to convert to the number of elapsed microseconds.
  // To guard against loss-of-precision, we convert
  // to microseconds *before* dividing by ticks-per-second.
  //

  ElapsedMicroseconds.QuadPart *= 1000000;
  ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
  elapsed_milliseconds_ = ElapsedMicroseconds.QuadPart * 1000;
#endif
  return elapsed_milliseconds_;
}


static std::map<std::string, float> map_perf_db;
typedef std::map<std::string, float>::iterator map_perf_iter;

static void mkl_update_perf(std::string k, float v) {
  map_perf_iter findIter = map_perf_db.find(k);
  if (map_perf_db.end() == findIter) {
    map_perf_db.insert(std::pair<std::string, float>(k, v));
    std::cout << "insert " << k << " with " << v << std::endl;
  } else {
    std::cout << "update " << k << " with " << findIter->second << " + " << v << std::endl;
    findIter->second += v;
  }
}
void dump_perf_statistic() {
  for (map_perf_iter it = map_perf_db.begin(); it != map_perf_db.end(); ++it)
    std::cout << it->first << " => " << it->second << std::endl;
}
#endif

namespace mxnet {

namespace op {
const OperatorProperty* OpPropGetOpProperty(const NodeAttrs& attrs);
}  // namespace op

namespace exec {

// forward executor
class ForwardOpExecutor : public OpExecutor {
 public:
   std::string getName() {
     std::string display = "Forward_";
     if (name_.length() == 0)
       display += typeid((*op_)).name();
     else
       display += name_;
     return display;
   }
  void Run(RunContext rctx) override {
    op_ctx.run_ctx = rctx;
#ifdef ENABLE_TIME_PROFILE
    Timer timer;
    timer.Start();
#endif
#ifdef ENABLE_TIME_PROFILE_NEW
      PTimerUtil::tstart(getName(),1000);
#endif
    op_->Forward(op_ctx, in_data_, req, out_data_, aux_data_);
#if MKL_EXPERIMENTAL == 1
    mkl_tblobs_prv_to_cpu(in_data_);
    mkl_tblobs_prv_to_cpu(out_data_);
    mkl_tblobs_prv_to_cpu(aux_data_);
#endif
#ifdef ENABLE_TIME_PROFILE_NEW
      PTimerUtil::tprint(getName());
#endif
#ifdef ENABLE_TIME_PROFILE
   timer.Stop();
   mkl_update_perf(getName(), timer.MilliSeconds());
#endif
  }

  void Setup() override {
    in_data_.clear(); aux_data_.clear();
    for (size_t i = 0; i < in_array.size(); ++i) {
      if (!std::binary_search(aux_index_.begin(), aux_index_.end(), i)) {
        in_data_.push_back(in_array[i].data());
      } else {
        aux_data_.push_back(in_array[i].data());
      }
    }
    out_data_.resize(out_array.size());
    std::transform(out_array.begin(), out_array.end(), out_data_.begin(), [](const NDArray& nd) {
        return nd.data();
      });
  }
  Operator::ExecType exec_type() const override {
    return op_->exec_type();
  }
  explicit ForwardOpExecutor(std::shared_ptr<Operator> op,
      std::vector<uint32_t> aux_index, std::string name)
      : op_(op), aux_index_(aux_index) {
    name_ = name;
    std::sort(aux_index_.begin(), aux_index_.end());
  }

 private:
  friend Graph AttachOpExecs(Graph g);
  std::shared_ptr<Operator> op_;
  std::vector<uint32_t> aux_index_;
  std::vector<TBlob> in_data_, out_data_, aux_data_;
  std::string name_;
};

// backward executor
class BackwardOpExecutor : public OpExecutor {
 public:
  std::string getName() {
    std::string display = "Backward_";
    if (name_.length() == 0)
      display += typeid((*op_)).name();
    else
      display += name_;
    return display;
  }
  void Run(RunContext rctx) override {
    op_ctx.run_ctx = rctx;
#ifdef ENABLE_TIME_PROFILE
     Timer timer;
     timer.Start();
#endif
    op_->Backward(op_ctx, out_grad_, in_data_, out_data_,
                  req, in_grad_, aux_data_);
#if MKL_EXPERIMENTAL == 1
    mkl_tblobs_prv_to_cpu(out_grad_);
    mkl_tblobs_prv_to_cpu(in_data_);
    mkl_tblobs_prv_to_cpu(out_data_);
    mkl_tblobs_prv_to_cpu(in_grad_);
    mkl_tblobs_prv_to_cpu(aux_data_);
#endif
#ifdef ENABLE_TIME_PROFILE
     timer.Stop();
     mkl_update_perf(getName(), timer.MilliSeconds());
#endif
  }
  void Setup() override {
    size_t arg_top = 0, aux_top = 0;
    aux_data_.resize(aux_index_.size());
    for (size_t i = 0; i < in_array.size(); ++i) {
      if (!std::binary_search(aux_index_.begin(), aux_index_.end(), i)) {
        CHECK_GT(arg_data_ptr_.size(), arg_top);
        *arg_data_ptr_[arg_top++] = in_array[i].data();
      } else {
        aux_data_.at(aux_top++) = in_array[i].data();
      }
    }
    CHECK_EQ(out_array.size(), in_grad_.size());
    std::transform(out_array.begin(), out_array.end(),
                   in_grad_.begin(), [](const NDArray& nd) {
        return nd.data();
      });
  }
  Operator::ExecType exec_type() const override {
    return op_->exec_type();
  }
  explicit BackwardOpExecutor(std::shared_ptr<Operator> op,
                              const OperatorProperty* prop,
                              std::vector<uint32_t> aux_index, std::string name)
      : op_(op), aux_index_(aux_index) {
    name_ = name;
    std::sort(aux_index_.begin(), aux_index_.end());
    out_grad_.resize(prop->NumVisibleOutputs());
    in_data_.resize(prop->ListArguments().size());
    in_grad_.resize(in_data_.size());
    out_data_.resize(prop->NumOutputs());

    std::vector<TBlob*> out_grad_ptr(out_grad_.size());
    for (size_t i = 0; i < out_grad_.size(); ++i) {
      out_grad_ptr[i] = &out_grad_[i];
    }
    std::vector<TBlob*> in_data_ptr(in_data_.size());
    for (size_t i = 0; i < in_data_.size(); ++i) {
      in_data_ptr[i] = &in_data_[i];
    }
    std::vector<TBlob*> out_data_ptr(out_data_.size());
    for (size_t i = 0; i < out_data_.size(); ++i) {
      out_data_ptr[i] = &out_data_[i];
    }
    arg_data_ptr_ = prop->BackwardInputs(
        out_grad_ptr, in_data_ptr, out_data_ptr);
  }

 private:
  std::shared_ptr<Operator> op_;
  std::vector<uint32_t> aux_index_;
  std::vector<TBlob> out_grad_, in_grad_, in_data_, out_data_, aux_data_;
  std::vector<TBlob*> arg_data_ptr_;
  std::string name_;
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

  auto& fcreate_layer_op = nnvm::Op::GetAttr<FCreateLayerOp>("FCreateLayerOp");
  auto& fmutate_inputs = nnvm::Op::GetAttr<FMutateInputs>("FMutateInputs");
  auto& is_layer_backward = nnvm::Op::GetAttr<bool>("TIsLayerOpBackward");

  const auto& vdtype = g.GetAttr<DTypeVector>("dtype");
  const auto& vshape = g.GetAttr<ShapeVector>("shape");
  const auto& vctx = g.GetAttr<ContextVector>("context");
  const auto& saved_opr = g.GetAttr<
    std::unordered_map<const nnvm::Node*, std::shared_ptr<Operator>>>("saved_opr");

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
    FCompute fcompute = FComputeExecutor::GetFCompute(inode.source->op(), vctx[i]);
    if (fcreate_layer_op.count(inode.source->op())) {
      std::vector<TShape> ishape;
      std::vector<int> itype;
      for (const auto& e : inode.inputs) {
        ishape.emplace_back(vshape[idx.entry_id(e)]);
        itype.emplace_back(vdtype[idx.entry_id(e)]);
      }
      std::shared_ptr<Operator> opr;
      if (saved_opr.count(inode.source)) {
        opr = saved_opr.at(inode.source);
      } else {
        opr.reset(fcreate_layer_op[inode.source->op()](
              inode.source->attrs, vctx[i], ishape, itype));
      }
      ret[i] = std::make_shared<ForwardOpExecutor>(opr, mutate_index, inode.source->op()->name);
    } else if (is_layer_backward.get(inode.source->op(), false)) {
      CHECK_GE(inode.control_deps.size(), 1);
      uint32_t fwd_id = inode.control_deps[0];
      CHECK(vctx[fwd_id] == vctx[i]);
      CHECK(ret[fwd_id] != nullptr);
      ret[i] = std::make_shared<BackwardOpExecutor>(
          dynamic_cast<ForwardOpExecutor*>(ret[fwd_id].get())->op_,
          mxnet::op::OpPropGetOpProperty(inode.source->attrs),
          mutate_index, inode.source->op()->name);
    } else if (fcompute != nullptr) {
      ret[i] = std::make_shared<FComputeExecutor>(fcompute, inode.source->attrs);
    } else {
      LOG(INFO) << "FCompute not registered " << inode.source->op()->name;
    }
  }
  g.attrs["op_execs"] = std::make_shared<nnvm::any>(ret);
  return g;
}

}  // namespace exec
}  // namespace mxnet
