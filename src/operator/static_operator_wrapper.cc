/*!
 *  Copyright (c) 2015 by Contributors
 * \file static_operator.cc
 * \brief the implementation of static operator
 * \author Naiyan Wang
 */
#include <dmlc/base.h>
#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <mxnet/narray.h>
#include <mxnet/dag_engine.h>
#include <vector>

namespace mxnet {
namespace op {
/*!
 * \brief StaticOperatorWrapper that wraps a static_operator
 *  This class do not need to be seen by others, so it sit in cc file.
 * \sa Operator, StaticOperator
 */
class StaticOperatorWrapper: public Operator {
 public:
  StaticOperatorWrapper(StaticOperator* op, Context ctx)
      : op_(op), ctx_(ctx) {}

  virtual ~StaticOperatorWrapper() {
    delete op_;
  }

  virtual int DescribeProperty() const {
    return op_->DescribeProperty();
  }

  virtual void Forward(Option opt,
                       RunContext ctx,
                       const std::vector<NArray> &in_data,
                       const std::vector<NArray> &out_data) {
    std::vector<DAGEngine::Variable> used_var;
    std::vector<DAGEngine::Variable> mutate_var;
    std::vector<TBlob> in;
    std::vector<TBlob> out;
    for (size_t i = 0; i < in_data.size(); ++i) {
      used_var.push_back(in_data[i].var());
      in.push_back(in_data[i].data());
    }
    for (size_t i = 0; i < out_data.size(); ++i) {
      mutate_var.push_back(out_data[i].var());
      out.push_back(out_data[i].data());
    }
    DAGEngine::Get()->Push([this, opt, ctx, in, out](RunContext ctx) {
        op_->Forward(opt, ctx, in, out);
      }, ctx_, used_var, mutate_var);
  }

  virtual void Backward(RunContext ctx,
                        const std::vector<NArray> &grad_next,
                        const std::vector<NArray> &in_data,
                        const std::vector<NArray> &out_grad,
                        const std::vector<GradReqType> &req) {
    std::vector<DAGEngine::Variable> used_var;
    std::vector<DAGEngine::Variable> mutate_var;
    std::vector<TBlob> grad_in;
    std::vector<TBlob> grad_out;
    std::vector<TBlob> data;
    for (size_t i = 0; i < grad_next.size(); ++i) {
      used_var.push_back(grad_next[i].var());
      grad_in.push_back(grad_next[i].data());
    }
    for (size_t i = 0; i < in_data.size(); ++i) {
      used_var.push_back(in_data[i].var());
      data.push_back(in_data[i].data());
    }
    for (size_t i = 0; i < out_grad.size(); ++i) {
      mutate_var.push_back(out_grad[i].var());
      grad_out.push_back(out_grad[i].data());
    }
    DAGEngine::Get()->Push([this, ctx, grad_in, grad_out, data, req](RunContext ctx) {
        op_->Backward(ctx, grad_in, data, grad_out, req);
      }, ctx_, used_var, mutate_var);
  }

 private:
  /* \brief the static operator */
  StaticOperator* op_;
  /** \brief the global context denots the device info. */
  Context ctx_;
};
}  // namespace op

// implements CreateWrapper
Operator *Operator::CreateWrapper(StaticOperator *op, Context ctx) {
  return new op::StaticOperatorWrapper(op, ctx);
}

}  // namespace mxnet
