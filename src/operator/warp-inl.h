/*!
 * Copyright (c) 2016 by Contributors
 * \file warp-inl.h
 * \brief warp operator and symbol
 * \author Xu Dong 
*/
#ifndef MXNET_OPERATOR_WARP_INL_H_
#define MXNET_OPERATOR_WARP_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./mshadow_op.h"
#include "./operator_common.h"
namespace mxnet {
namespace op {
//  Declare enumeration of input order to make code more intuitive.
//  These enums are only visible within this header
namespace Warp {
enum  WarpOpInputs{kData, kGrid};
enum  WarpOpOutputs{kOut};
}  // namespace Warp

struct WarpParam : public dmlc::Parameter<WarpParam> {
  bool only_grid;
  DMLC_DECLARE_PARAMETER(WarpParam) {
  DMLC_DECLARE_FIELD(only_grid).set_default(false)
  .describe("indicate whether WarpOp only calculate the gradient of the grid ");
  }
};
template<typename xpu>
class WarpOp : public Operator {
 public:
  explicit WarpOp(WarpParam param) {
    this->param_ = param;
  }
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    CHECK_EQ(in_data.size(), 2);
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> data = in_data[Warp::kData].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> grid = in_data[Warp::kGrid].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> out  = out_data[Warp::kOut].get<xpu, 4, real_t>(s);
    out = 0.0f;
    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(grid.CheckContiguous(), true);
    CHECK_EQ(out.CheckContiguous(), true);
    WarpForward(data, grid, out);
  }
  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> data = in_data[Warp::kData].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> grid = in_data[Warp::kGrid].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> grad_data = in_grad[Warp::kData].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> grad_grid = in_grad[Warp::kGrid].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> out_g = out_grad[Warp::kOut].get<xpu, 4, real_t>(s);

    CHECK_EQ(grad_data.CheckContiguous(), true);
    CHECK_EQ(grad_grid.CheckContiguous(), true);
    CHECK_EQ(out_g.CheckContiguous(), true);

    grad_data = 0.0f;
    grad_grid = 0.0f;
    WarpBackward(grad_data, grad_grid, out_g, data, grid, param_.only_grid);
  }

 private:
    WarpParam param_;
};   //  class WarpOp
//  Decalre Factory function
template<typename xpu>
Operator* CreateOp(WarpParam param);
#if DMLC_USE_CXX11
class WarpProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "grid"};
  }
  std::vector<std::string> ListOutputs() const override {
    return {"output"};
  }
  int NumOutputs() const override {
    return 1;
  }
  int NumVisibleOutputs() const override {
    return 1;
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
    CHECK_EQ(in_shape->size(), 2) << "Input:[data, grid]";
    TShape dshape1 = in_shape->at(Warp::kData);
    TShape dshape2 = in_shape->at(Warp::kGrid);
    CHECK_EQ(dshape1.ndim(), 4) << "data should be a 4D tensor";
    CHECK_EQ(dshape2.ndim(), 4) << "data should be a 4D tensor";
    CHECK_EQ(dshape1[0], dshape2[0]) << "The shapes should be identical except for the 4th axis";
    CHECK_EQ(dshape1[1], dshape2[1]);
    CHECK_EQ(dshape1[2], dshape2[2]);
    CHECK_EQ(dshape2[3], 2) <<"We assume [batch, y, x, channel] format on grid."
     "WarpOp only support 2D grid, channel : (x_displacement, y_displacement) ";

    out_shape->clear();
    out_shape->push_back(dshape1);
    return true;
  }
  OperatorProperty* Copy() const override {
    WarpProp* Warp_sym = new WarpProp();
    Warp_sym->param_ = this->param_;
    return Warp_sym;
  }
  std::string TypeString() const override {
    return "Warp";
  }
  //  decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
     return {out_grad[Warp::kOut], in_data[Warp::kData], in_data[Warp::kGrid]};
}
  Operator* CreateOperator(Context ctx) const override;

 private:
  WarpParam param_;
};  //  class WarpProp
#endif
}  //  namespace op
}  //  namespace mxnet
#endif  // MXNET_OPERATOR_WARP_INL_H_
