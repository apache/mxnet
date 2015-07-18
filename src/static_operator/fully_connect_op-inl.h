/*!
 * Copyright (c) 2015 by Contributors
 * \file fully_connect_op-inl.h
 * \brief fully connect operator
 * \author Bing Xu
*/
#ifndef MXNET_STATIC_OPERATOR_FULLY_CONNECT_OP_INL_H_
#define MXNET_STATIC_OPERATOR_FULLY_CONNECT_OP_INL_H_

#include <dmlc/logging.h>
#include <mxnet/static_operator.h>
#include <vector>
#include "./static_operator_common.h"
#include "./param.h"

namespace mxnet {
namespace op {
template<typename xpu>
class FullyConnectOp : public StaticOperator {
 public:
  FullyConnectOp () {
    // Do nothing.
  }

  FullyConnectOp (param p) {
    this->param = p;
  }

  virtual std::vector<ArgType> DescribeArgs() const {
    ArgType ret[] = {kDataArg, kWeightArg, kBiasArg};
    if (param_.no_bias == 0) {
      return std::vector<ArgType>(ret, ret + 3);
    } else {
      return std::vector<ArgType>(ret, ret + 2);
    }
  }
  virtual void Forward(Option opt,
                       RunContext ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    using namespace mshadow::expr;
    size_t expected = param_.no_bias == 0 ? 3 : 2;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1);
    // TODO(bing): check the BLAS Handle, be careful
    // maybe need blas handle from context
    Stream<xpu> *s = static_cast<Stream<xpu> *>(ctx.stream);
    Tensor<xpu, 2> data = in_data[0].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> wmat = in_data[1].get<xpu, 2, real_t>(s);
    Tensor<xpu, 2> out = out_data[0].FlatTo2D<xpu, real_t>(s);
    out = dot(data, wmat.T());
    if (param_.no_bias == 0) {
      Tensor<xpu, 1> bias = in_data[2].get<xpu, 1, real_t>(s);
      out += repmat(bias, data.size(0));
    }
  }
  virtual void Backward(RunContext ctx,
                        const std::vector<TBlob> &grad_next,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<GradReqType> &req) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(grad_next.size(), 1);
    size_t expected = param_.no_bias == 0 ? 3 : 2;
    CHECK(in_data.size() == expected && out_grad.size() == expected);
    CHECK_EQ(req.size(), expected);
    // TODO(bing): check the BLAS Handle, be careful
    //  maybe need blas handle from context
    Stream<xpu> *s = static_cast<Stream<xpu> *>(ctx.stream);
    Tensor<xpu, 2> data = in_data[0].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> wmat = in_data[1].get<xpu, 2, real_t>(s);
    Tensor<xpu, 2> grad = grad_next[0].FlatTo2D<xpu, real_t>(s);
    //  backprop
    CHECK_NE(req[1], kWriteInplace) << "cannot write weight inplace";
    // gradient of weight
    Tensor<xpu, 2> gwmat = out_grad[1].get<xpu, 2, real_t>(s);
    Assign(gwmat, req[1], dot(grad.T(), data));
    // gradient of bias
    if (param_.no_bias == 0) {
      Tensor<xpu, 1> gbias = out_grad[2].get<xpu, 1, real_t>(s);
      Assign(gbias, req[2], sum_rows(grad));
    }
    // gradient of data
    Tensor<xpu, 2> gdata = out_grad[0].FlatTo2D<xpu, real_t>(s);
    Assign(gdata, req[0], dot(grad, wmat));
  }
 private:
  Param param_;
};  // class FullyConnectOp
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_STATIC_OPERATOR_FULLY_CONNECT_OP_INL_H_

