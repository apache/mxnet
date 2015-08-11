/*!
 * Copyright (c) 2015 by Contributors
 * \file pooling_op-inl.h
 * \brief pooling operator
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_STATIC_OPERATOR_POOLING_OP_INL_H_
#define MXNET_OPERATOR_STATIC_OPERATOR_POOLING_OP_INL_H_

#include <mxnet/operator.h>
#include <algorithm>
#include <vector>
#include "./param.h"
#include "./static_operator_common.h"


namespace mxnet {
namespace op {
template<typename xpu, typename Reducer, OpType mode>
class PoolingOp : public StaticOperator {
 public:
  virtual void SetParam(const char *name, const char *val) {
    param_.SetParam(name, val);
  }
  virtual void InferShape(std::vector<TShape> *in_shape,
                          std::vector<TShape> *out_shape) {
    CHECK_EQ(in_shape->size(), 1) << "Input: [data]";
    CHECK_GT(param_.kernel_y, 0);
    CHECK_GT(param_.kernel_x, 0);
    const int ksize_y = static_cast<index_t>(param_.kernel_y);
    const int ksize_x = static_cast<index_t>(param_.kernel_x);
    const int pad_y = static_cast<index_t>(param_.pad_y);
    const int pad_x = static_cast<index_t>(param_.pad_x);
    // TODO(bing): dual stride
    const int kstride = static_cast<index_t>(param_.stride_y);
    mshadow::Shape<4> ishape = (*in_shape)[0].get<4>();
    oshape_ = ishape;
    fea_shape_ = mshadow::Shape2(ishape[2], ishape[3]);
    oshape_[2] = std::min(ishape[2] + 2 * pad_y - ksize_y + kstride - 1,
                             ishape[2] + 2 * pad_y - 1) / kstride + 1;
    oshape_[3] = std::min(ishape[3] + 2 * pad_x - ksize_x + kstride - 1,
                             ishape[3] + 2 * pad_x - 1) / kstride + 1;
    CHECK(oshape_[2] > 0 && oshape_[3] > 0) << "kernel size exceed input";
    out_shape->clear();
    out_shape->push_back((*in_shape)[0]);
    (*out_shape)[0][2] = oshape_[2];
    (*out_shape)[0][3] = oshape_[3];
  }
  virtual void Forward(Option opt,
                       RunContext ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<TBlob> &out_data) {
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 0);
    if (!(temp_.shape_ == oshape_)) {
      temp_.Resize(oshape_);
    }
    const int ksize_y = param_.kernel_y;
    const int ksize_x = param_.kernel_x;
    const int pad_y = param_.pad_y;
    const int pad_x = param_.pad_x;
    // TODO(bing): dual stride
    const int kstride = param_.stride_y;
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<xpu> *s = static_cast<Stream<xpu> *>(ctx.stream);
    Tensor<xpu, 4> data = in_data[0].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> out = out_data[0].get<xpu, 4, real_t>(s);
    mshadow::Shape<2> pshape = Shape2(out.shape_[2], out.shape_[3]);
    if (mode == kMaxPooling || mode == kSumPooling) {
      temp_ = pool<Reducer>(pad(data, pad_y, pad_x),
                          pshape,
                          ksize_y,
                          ksize_x,
                          kstride);
    } else if (mode == kAvgPooling) {
      temp_ = (1.0f / (ksize_y * ksize_x)) * \
            pool<Reducer>(pad(data, pad_y, pad_x),
                          pshape,
                          ksize_y,
                          ksize_x,
                          kstride);
    } else {
      LOG(FATAL) << "Unknown pooling mode";
    }
    Copy(out, temp_, s);
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
    const int ksize_y = param_.kernel_y;
    const int ksize_x = param_.kernel_x;
    const int pad_y = param_.pad_y;
    const int pad_x = param_.pad_x;
    // TODO(bing): dual stride
    const int kstride = param_.stride_y;
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<xpu> *s = static_cast<Stream<xpu> *>(ctx.stream);
    Tensor<xpu, 4> grad = grad_next[0].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> data = in_data[0].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> out = out_grad[0].get<xpu, 4, real_t>(s);
    if (mode == kMaxPooling || mode == kSumPooling) {
      Assign(out,
             req[0],
             crop(unpool<Reducer>(pad(data, pad_y, pad_x),
                                  pad(temp_, 0, 0),
                                  pad(grad, 0, 0),
                                  ksize_y,
                                  ksize_x,
                                  kstride),
                  fea_shape_,
                  pad_y,
                  pad_x));
    } else if (mode == kAvgPooling) {
      Assign(out,
             req[0],
             (1.0f / (ksize_y * ksize_x)) * \
             crop(unpool<Reducer>(pad(data, pad_y, pad_x),
                                  pad(temp_, 0, 0),
                                  pad(grad, 0, 0),
                                  ksize_y,
                                  ksize_x,
                                  kstride),
                  fea_shape_,
                  pad_y,
                  pad_x));
    } else {
      LOG(FATAL) << "Unknown pooling mode";
    }
  }

 private:
  /*! \brief parameters that potentially be useful */
  Param param_;
  /*! \brief temp space to save pooled result */
  mshadow::TensorContainer<xpu, 4> temp_;
  /*! \brief pooled output shape */
  mshadow::Shape<4> oshape_;
  /*! \brief input feature map shape */
  mshadow::Shape<2> fea_shape_;
};  // class PoolingOp

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_STATIC_OPERATOR_POOLING_OP_INL_H_
