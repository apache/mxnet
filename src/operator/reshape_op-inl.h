/*!
 * Copyright (c) 2015 by Contributors
 * \file reshape_op-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_RESHAPE_OP_INL_H_
#define MXNET_OPERATOR_RESHAPE_OP_INL_H_

#include <mxnet/operator.h>
#include <vector>

namespace mxnet {
namespace op {
template<typename xpu, bool flatten>
class ReshapeOp : public Operator {
 public:
  virtual void SetParam(const char *name, const char *val) {
    if (!strcmp(name, "out_ch")) oshape_[1] = atoi(val);
    if (!strcmp(name, "out_y")) oshape_[2] = atoi(val);
    if (!strcmp(name, "out_x")) oshape_[3] = atoi(val);
  }
  virtual void InferShape(std::vector<TShape> *in_shape,
                          std::vector<TShape> *out_shape) {
    CHECK_EQ(in_shape->size(), 1);
    ishape_ = (*in_shape)[0].get<4>();
    oshape_[0] = ishape_[0];
    if (flatten) {
      oshape_[1] = 1;
      oshape_[2] = 1;
      oshape_[3] = ishape_[1] * ishape_[2] * ishape_[3];
    }
    CHECK_EQ(oshape_.Size(), ishape_.Size()) << "Incorrect new shape";
    TShape ts;
    ts = oshape_;
    out_shape->clear();
    out_shape->push_back(ts);
  }
  virtual void Forward(Option opt,
                       RunContext ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<TBlob> &out_data) {
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<xpu> *s = static_cast<Stream<xpu> *>(ctx.stream);
    Tensor<xpu, 4> data = in_data[0].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> out = out_data[0].get<xpu, 4, real_t>(s);
    out = reshape(data, oshape_);
  }
  virtual void Backward(RunContext ctx,
                        const std::vector<TBlob> &grad_next,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<GradReqType> &req) {
    CHECK_EQ(grad_next.size(), 1);
    CHECK_EQ(out_grad.size(), 1);
    CHECK_EQ(req.size(), 1);
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<xpu> *s = static_cast<Stream<xpu> *>(ctx.stream);
    Tensor<xpu, 4> grad = grad_next[0].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> out = out_grad[0].get<xpu, 4, real_t>(s);
    Assign(out, req[0], reshape(grad, ishape_));
  }

 private:
  mshadow::Shape<4> oshape_;
  mshadow::Shape<4> ishape_;
};  // class Operator

}  //  namespace op
}  //  namespace mxnet
#endif  // MXNET_OPERATOR_RESHAPE_OP_INL_H_
