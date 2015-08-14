/*!
 * Copyright (c) 2015 by Contributors
 * \file fully_connect_op-inl.h
 * \brief fully connect operator and symbol
*/
#ifndef MXNET_OPERATOR_STATIC_OPERATOR_FULLY_CONNECT_OP_INL_H_
#define MXNET_OPERATOR_STATIC_OPERATOR_FULLY_CONNECT_OP_INL_H_

#include <dmlc/logging.h>
#include <mxnet/operator.h>
#include <mxnet/symbolic.h>
#include <vector>
#include <string>
#include "./static_operator_common.h"
#include "./param.h"

namespace mxnet {
namespace op {
/**
 * \brief This is the implementation of fully connected layer.
 *
 * \tparam xpu The device that the op will be executed on.
 */
template<typename xpu>
class FullyConnectOp : public StaticOperator {
 public:
  /*!
   * \brief constructor with parameters. Used in Bind() in corresponding symbol.
   */
  explicit FullyConnectOp(Param p) {
    this->param_ = p;
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
                        const std::vector<TBlob> &out_data,
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
  /** The param of the fully connected layer.*/
  Param param_;
};  // class FullyConnectOp

/**
 * @brief The symbol part of the fully connected layer.
 */
class FullyConnectSymbol : public AtomicSymbol {
 public:
  virtual std::vector<std::string> ListArguments() const {
    std::string ret[] = {"data", "weight", "bias"};
    if (param_.no_bias == 0) {
      return std::vector<std::string>(ret, ret + 3);
    } else {
      return std::vector<std::string>(ret, ret + 2);
    }
  }

  virtual void SetParam(const char *name, const char *val) {
    param_.SetParam(name, val);
  }

  virtual bool InferShape(std::vector<TShape> *in_shape,
                          std::vector<TShape> *out_shape) const {
    using namespace mshadow;
    if (param_.no_bias == 0) {
      CHECK_EQ(in_shape->size(), 3) << "Input:[data, weight, bias]";
    } else {
      CHECK_EQ(in_shape->size(), 2) << "Input:[data, weight]";
    }
    CHECK_GT(param_.num_hidden, 0);
    const TShape &dshape = (*in_shape)[0];
    CHECK_EQ(dshape.ndim(), 4) << \
        "Input data should be 4D in batch-1-1-hidden";
    CHECK_NE(dshape.ndim(), 0) << "Require data shape to be known";
    ShapeAssignCheck((*in_shape)[1], Shape2(param_.num_hidden, dshape[3]));
    if (param_.no_bias == 0) {
      ShapeAssignCheck((*in_shape)[2], Shape1(param_.num_hidden));
    }
    out_shape->clear();
    out_shape->push_back(dshape);
    (*out_shape)[0][3] = param_.num_hidden;
    return true;
  }

  virtual AtomicSymbol* Copy() const {
    FullyConnectSymbol* fc_sym = new FullyConnectSymbol();
    fc_sym->param_ = this->param_;
    return fc_sym;
  }

  virtual std::string TypeString() const {
    return "FullyConnected";
  }

  /**
   * @brief This is the template function of bind() implementation.
   *
   * @param ctx The device context
   * @return A device dependent static operator can be used for execution.
   */
  template<typename xpu>
  StaticOperator* Bind_(Context ctx) const;
  // the real bind
  StaticOperator* Bind(Context ctx) const;

 private:
  /** The param of the fully connected layer.*/
  Param param_;
};  // class FullyConnectSymbol

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_STATIC_OPERATOR_FULLY_CONNECT_OP_INL_H_
