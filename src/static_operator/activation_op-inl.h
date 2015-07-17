/*!
 *  Copyright (c) 2015 by Contributors
 * \file activation_op-inl.h
 * \brief activation operator of mxnet
 */

#ifndef MXNET_STATIC_OPERATOR_ACTIVATION_OP_INL_H_
#define MXNET_STATIC_OPERATOR_ACTIVATION_OP_INL_H_

#include <dmlc/logging.h>
#include <mxnet/static_operator.h>
#include <vector>
#include "./static_operator_common.h"

namespace mxnet {
namespace op {
template<typename xpu, typename ForwardOp, typename BackOp>
class ActivationOp : public StaticOperator {
 public:
  virtual void InferShape(std::vector<TShape> *in_shape,
                          std::vector<TShape> *out_shape) {
    CHECK_EQ(in_shape->size(), 1) << "Only 1 input is allowed";
    CHECK_NE((*in_shape)[0].ndim(), 0) << "Require data shape to be known";
    out_shape->clear();
    out_shape->push_back((*in_shape)[0]);
  }
  virtual void Forward(Option opt,
                       RunContext ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<TBlob> &out_data) {
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(in_data.size(), 1);
    mshadow::Stream<xpu> *stream = \
      static_cast<mshadow::Stream<xpu> *>(ctx.stream);
    mshadow::Tensor<xpu, 2> in = in_data[0].FlatTo2D<xpu, real_t>(stream);
    mshadow::Tensor<xpu, 2> out = out_data[0].FlatTo2D<xpu, real_t>(stream);
    out = mshadow::expr::F<ForwardOp>(in);
  }
  virtual void Backward(RunContext ctx,
                        const std::vector<TBlob> &grad_next,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<GradReqType> &req) {
    CHECK_EQ(grad_next.size(), 1);
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_grad.size(), 1);
    CHECK_EQ(req.size(), 1);
    mshadow::Stream<xpu> *stream = \
      static_cast<mshadow::Stream<xpu> *>(ctx.stream);
    mshadow::Tensor<xpu, 2> grad = grad_next[0].FlatTo2D<xpu, real_t>(stream);
    mshadow::Tensor<xpu, 2> data = in_data[0].FlatTo2D<xpu, real_t>(stream);
    mshadow::Tensor<xpu, 2> out = out_grad[0].FlatTo2D<xpu, real_t>(stream);
    Assign(out, req[0], mshadow::expr::F<BackOp>(
        mshadow::expr::F<ForwardOp>(data)) * grad);
  }
};  // class ActivationOp
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_STATIC_OPERATOR_ACTIVATION_OP_INL_H_
