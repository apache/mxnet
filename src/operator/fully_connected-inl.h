/*!
 * Copyright (c) 2015 by Contributors
 * \file fully_connect_op-inl.h
 * \brief fully connect operator and symbol
*/
#ifndef MXNET_OPERATOR_FULLY_CONNECTED_INL_H_
#define MXNET_OPERATOR_FULLY_CONNECTED_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"


namespace mxnet {
namespace op {

// Declare enumeration of input order to make code more intuitive.
// These enums are only visible within this header
enum FullyConnectedOpInputs {kData, kWeight, kBias};
enum FullyConnectedOpOutputs {kOut};

struct FullyConnectedParam : public dmlc::Parameter<FullyConnectedParam> {
  int num_hidden;
  bool no_bias;
  DMLC_DECLARE_PARAMETER(FullyConnectedParam) {
    // TODO(bing) change to only set lower bound
    // add support for boolean
    DMLC_DECLARE_FIELD(num_hidden).set_range(1, 100000)
    .describe("Number of hidden nodes of the output.");
    DMLC_DECLARE_FIELD(no_bias).set_default(false)
    .describe("Whether to disable bias parameter.");
  }
};

/**
 * \brief This is the implementation of fully connected operator.
 * \tparam xpu The device that the op will be executed on.
 */
template<typename xpu>
class FullyConnectedOp : public Operator {
 public:
  explicit FullyConnectedOp(FullyConnectedParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(req[kOut], kWriteTo);
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1);
    // TODO(bing): check the BLAS Handle, be careful
    // maybe need blas handle from context
    // TODO(bing): judge shape to remove flatten op
    Stream<xpu> *s = ctx.get_stream<xpu>();
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif  // __CUDACC__
    Tensor<xpu, 2> data = in_data[kData].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> wmat = in_data[kWeight].get<xpu, 2, real_t>(s);
    Tensor<xpu, 2> out = out_data[kOut].FlatTo2D<xpu, real_t>(s);
    out = dot(data, wmat.T());
    if (!param_.no_bias) {
      Tensor<xpu, 1> bias = in_data[kBias].get<xpu, 1, real_t>(s);
      out += repmat(bias, data.size(0));
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1);
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK(in_data.size() == expected && in_grad.size() == expected);
    CHECK_EQ(req.size(), expected);
    // TODO(bing): check the BLAS Handle, be careful
    //  maybe need blas handle from context
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> data = in_data[kData].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> wmat = in_data[kWeight].get<xpu, 2, real_t>(s);
    Tensor<xpu, 2> grad = out_grad[kOut].FlatTo2D<xpu, real_t>(s);
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif
    //  backprop
    CHECK_NE(req[kWeight], kWriteInplace) << "cannot write weight inplace";
    // gradient of weight
    Tensor<xpu, 2> gwmat = in_grad[kWeight].get<xpu, 2, real_t>(s);
    Assign(gwmat, req[kWeight], dot(grad.T(), data));
    // gradient of bias
    if (!param_.no_bias) {
      Tensor<xpu, 1> gbias = in_grad[kBias].get<xpu, 1, real_t>(s);
      Assign(gbias, req[kBias], sum_rows(grad));
    }
    // gradient of data
    Tensor<xpu, 2> gdata = in_grad[kData].FlatTo2D<xpu, real_t>(s);
    Assign(gdata, req[kData], dot(grad, wmat));
  }

 private:
  FullyConnectedParam param_;
};  // class FullyConnectedOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(FullyConnectedParam param);

#if DMLC_USE_CXX11
class FullyConnectedProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    if (!param_.no_bias) {
      return {"data", "weight", "bias"};
    } else {
      return {"data", "weight"};
    }
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    if (!param_.no_bias) {
      CHECK_EQ(in_shape->size(), 3) << "Input:[data, weight, bias]";
    } else {
      CHECK_EQ(in_shape->size(), 2) << "Input:[data, weight]";
    }
    const TShape &dshape = (*in_shape)[kData];
    // require data to be known
    if (dshape.ndim() ==  0) return false;

    index_t num_input = 0;
    mshadow::Shape<2> ishape = dshape.FlatTo2D();
    num_input = ishape[1];
    SHAPE_ASSIGN_CHECK(*in_shape, kWeight, Shape2(param_.num_hidden, num_input));
    if (!param_.no_bias) {
      SHAPE_ASSIGN_CHECK(*in_shape, kBias, Shape1(param_.num_hidden));
    }
    out_shape->clear();
    out_shape->push_back(Shape2(dshape[0], param_.num_hidden));
    return true;
  }

  OperatorProperty* Copy() const override {
    FullyConnectedProp* fc_sym = new FullyConnectedProp();
    fc_sym->param_ = this->param_;
    return fc_sym;
  }

  std::string TypeString() const override {
    return "FullyConnected";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[kOut], in_data[kData], in_data[kWeight]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{in_data[kData], in_grad[kData]}};
  }

  Operator* CreateOperator(Context ctx) const;

 private:
  FullyConnectedParam param_;
};  // class FullyConnectedSymbol
#endif
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_FULLY_CONNECTED_INL_H_
