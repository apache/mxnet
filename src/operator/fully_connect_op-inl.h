/*!
 * Copyright (c) 2015 by Contributors
 * \file fully_connect_op-inl.hpp
 * \brief fully connect operator
 * \author Bing Xu
*/

#ifndef MXNET_FULLY_CONNECT_OP_INL_HPP_
#define MXNET_FULLY_CONNECT_OP_INL_HPP_

#include <mxnet/operator.h>
#include <vector>
#include "./assign_helper.h"
#include "./param.h"

namespace mxnet {
namespace op {
template<typename xpu>
class FullyConnectOp : public Operator {
 public:
  virtual void DescribeArgs(std::vector<ArgReqType> *args) {
    args->clear();
    args->push_back(kDataArg);
    args->push_back(kWeightArg);
    args->push_back(kBiasArg);
  }
  virtual void SetParam(const char *name, const char *val) {
    param_.SetParam(name, val);
  }
  virtual void InferShape(std::vector<TShape> &in_shape,
                          std::vector<TShape> *out_shape) {
    CHECK(in_shape.size() == 3) << "Input:[data, weight, bias]";
    CHECK(param_.num_input_node > 0);
    CHECK(param_.num_hidden > 0);
    TShape &dshape = in_shape[0];
    TShape &wshape = in_shape[1];
    TShape &bshape = in_shape[2];
    if (wshape.Size() == 0) {
      mshadow::Shape<2> ws = mshadow::Shape2(param_.num_hidden,
                                             param_.num_input_node);
      wshape = ws;
    } else {
      CHECK(wshape[0] == param_.num_hidden);
      CHECK(wshape[1] == param_.num_input_node);
    }
    if (bshape.Size() == 0) {
      mshadow::Shape<1> bs = mshadow::Shape1(param_.num_hidden);
      bshape = bs;
    } else {
      CHECK(bshape[0] == param_.num_hidden);
    }
    CHECK(dshape.ndim() == 4 && dshape[3] == param_.num_input_node) << \
                         "Input data should be 4D in batch-1-1-hidden";
    out_shape->clear();
    out_shape->push_back(dshape);
    out_shape->at(0)[3] = param_.num_hidden;
  }
  virtual void Forward(Option opt,
                       RunContext ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<TBlob> &out_data) {
    CHECK(in_data.size() == 3) << "Input:[data, weight, bias]";
    CHECK(out_data.size() == 1);
    mshadow::Stream<xpu> *stream = \
      static_cast<mshadow::Stream<xpu> *>(ctx.stream);
    mshadow::Tensor<xpu, 2> wmat = in_data[0].get<xpu, 2, real_t>(stream);
    mshadow::Tensor<xpu, 1> bias = in_data[1].get<xpu, 1, real_t>(stream);
    mshadow::Tensor<xpu, 2> data = in_data[2].FlatTo2D<xpu, real_t>(stream);
    mshadow::Tensor<xpu, 2> out = out_data[0].FlatTo2D<xpu, real_t>(stream);
    out = mshadow::expr::dot(data, wmat.T());
    if (!param_.no_bias) {
      out += mshadow::expr::repmat(bias, data.size(0));
    }
  }
  virtual void Backward(RunContext ctx,
                        const std::vector<TBlob> &grad_next,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<GradReqType> &req) {
    CHECK(grad_next.size() == 1);
    CHECK(in_data.size() == 3) << "Input: [data, weight, bias]";
    CHECK(out_grad.size() == 3) << "Output: [gdata, gweight, gbias]";
    CHECK(req.size() == 3);
    mshadow::Stream<xpu> *stream = \
      static_cast<mshadow::Stream<xpu> *>(ctx.stream);
    mshadow::Tensor<xpu, 2> data = in_data[0].FlatTo2D<xpu, real_t>(stream);
    mshadow::Tensor<xpu, 2> wmat = in_data[1].get<xpu, 2, real_t>(stream);
    mshadow::Tensor<xpu, 2> grad = grad_next[0].FlatTo2D<xpu, real_t>(stream);
    mshadow::Tensor<xpu, 2> gdata = out_grad[0].FlatTo2D<xpu, real_t>(stream);
    mshadow::Tensor<xpu, 2> gwmat = out_grad[1].get<xpu, 2, real_t>(stream);
    mshadow::Tensor<xpu, 1> gbias = out_grad[2].get<xpu, 1, real_t>(stream);
    //  backprop
    CHECK(req[0] != kWriteInplace);
    Assign(gwmat, mshadow::expr::dot(grad.T(), data), req[0]);
    if (!param_.no_bias) {
      Assign(gbias, mshadow::expr::sum_rows(grad), req[1]);
    }
    if (req[0] != kNullOp) {
      CHECK(req[0] != kWriteInplace);
      Assign(gdata, mshadow::expr::dot(grad, wmat), req[2]);
    }
  }
 private:
  Param param_;
};  // class FullyConnectOp
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_FULLY_CONNECT_OP_INL_HPP


