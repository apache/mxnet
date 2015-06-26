/*!
 * Copyright (c) 2015 by Contributors
 * \file dropout_op-inl.h
 * \brief dropout operator
 * \author Bing Xu
*/
#ifndef MXNET_DROPOUT_OP_INL_H_
#define MXNET_DROPOUT_OP_INL_H_

#include <vector>
#include <mxnet/operator.h>
#include "./mshadow_op.h"

namespace mxnet {
namespace op {
template<typename xpu>
class DropoutOp : public Operator {
 public:
  DropoutOp(mshadow::Random<xpu> *prnd)
      : prnd_(prnd), mask_used_(false) {}
  virtual int DescribeProperty() const {
    return kForwardRequireRnd | kContainInteralState;
  }
  virtual void SetParam(const char *name, const char* val) {
    if (!strcmp("threshold", name)) pkeep_ = \
      static_cast<real_t>(1.0f - atof(val));
    CHECK(pkeep_ > 0) << "invalid dropout threshold";
  }
  virtual void InferShape(std::vector<TShape> *in_shape,
                          std::vector<TShape> *out_shape) {
    CHECK(in_shape->size() == 1) << "Input: [data]";
    out_shape->clear();
    out_shape->push_back((*in_shape)[0]);
  }
  virtual void Forward(Option opt,
                       RunContext ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<TBlob> &out_data) {
    CHECK(in_data.size() == 1);
    CHECK(out_data.size() == 1);
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<xpu> *s = static_cast<Stream<xpu> *>(ctx.stream);
    Tensor<xpu, 4> data = in_data[0].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> out = out_data[0].get<xpu, 4, real_t>(s);
    if (mask_.shape_!= out.shape_) {
      mask_.Resize(out.shape_);
    }
    if (opt.is_train && pkeep_ != 1.0f) {
      mask_ = F<op::threshold>(prnd_->uniform(mask_.shape_), pkeep_)  * \
             (1.0f / pkeep_);
      out = data * mask_;
      mask_used_ = true;
    } else {
      out = data;
      mask_used_ = false;
    }
  }
  virtual void Backward(RunContext ctx,
                        const std::vector<TBlob> &grad_next,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<GradReqType> &req) {
    CHECK(grad_next.size() == 1);
    CHECK(out_grad.size() == 1);
    CHECK(req.size() == 1);
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<xpu> *s = static_cast<Stream<xpu> *>(ctx.stream);
    Tensor<xpu, 4> grad = grad_next[0].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> out = out_grad[0].get<xpu, 4, real_t>(s);
    // mask won't be initialized in when
    if (mask_used_) {
      Assign(out, req[0], grad * mask_);
    } else {
      // avoid directly assign tensor to tensor
      Assign(out, req[0], F<op::identity>(grad));
    }
  }

 private:
  /*! \brief random number generator */
  mshadow::Random<xpu> *prnd_;
  /*! \brief random mask */
  mshadow::TensorContainer<xpu, 4> mask_;
  /*! \brief probability to keep */
  real_t pkeep_;
  /*! \brief record whether mask is used in last forward */
  bool mask_used_;
};  // class DropoutOp
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_DROPOUT_OP_INL_H_
