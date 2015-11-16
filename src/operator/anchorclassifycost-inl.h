/*!
 * Copyright (c) 2015 by Contributors
 * \file anchorclassifycost-inl.h
 * \brief
 * \author Ming Zhang
*/
#ifndef MXNET_OPERATOR_ANCHORCLASSIFYCOST_INL_H_
#define MXNET_OPERATOR_ANCHORCLASSIFYCOST_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"

#define MIN_NUM 1e-37f

namespace mxnet {
namespace op {

namespace anchorclscost_enum {
enum AnchorClsCostOpInputs {kData, kLabel, kMarkLabel};
enum AnchorClsCostOpOutputs {kOut};
};


struct AnchorClsCostParam : public dmlc::Parameter<AnchorClsCostParam> {
  // use int for enumeration
  uint32_t anchornum;
  DMLC_DECLARE_PARAMETER(AnchorClsCostParam) {
    DMLC_DECLARE_FIELD(anchornum)
    .set_default(0)
    .describe("The Anchor Number.");
  }
};


template<typename xpu>
class AnchorClsCostOp : public Operator {
 public:
  explicit AnchorClsCostOp(AnchorClsCostParam p) {
    CHECK_NE(p.anchornum, 0) << "anchornum can not be equal 0.";
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    
    TBlob data_in = in_data[anchorclscost_enum::kData];
    TBlob label = in_data[anchorclscost_enum::kLabel];
    TBlob marklabel = in_data[anchorclscost_enum::kMarkLabel];
    TBlob data_out = out_data[anchorclscost_enum::kOut];

    Tensor<xpu, 2> tdata_in = data_in.FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> tlabel = label.FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> tmarklabel = marklabel.FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> tdata_out = data_out.FlatTo2D<xpu, real_t>(s);

//    std::cout << "class:" << tmarklabel[0][0] << "\n";

    tdata_out = F<mshadow_op::log>(tdata_in + MIN_NUM) * tlabel +
                (1.0f - tlabel) * F<mshadow_op::log>(1.0f - tdata_in + MIN_NUM);
    tdata_out = F<mshadow_op::negation>(tdata_out * tmarklabel);

//    tdata_out = 10.f;
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
    Stream<xpu> *s = ctx.get_stream<xpu>();

    TBlob data_in = in_data[anchorclscost_enum::kData];
    TBlob label = in_data[anchorclscost_enum::kLabel];
    TBlob marklabel = in_data[anchorclscost_enum::kMarkLabel];
    TBlob grad_in = in_grad[anchorclscost_enum::kOut];

    Tensor<xpu, 2> tdata_in = data_in.FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> tlabel = label.FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> tmarklabel = marklabel.FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> tgrad_in = grad_in.FlatTo2D<xpu, real_t>(s);
    
    tgrad_in = tlabel / (tdata_in + MIN_NUM) - \
              (1.0f - tlabel) / (1.0f - tdata_in + MIN_NUM);
    tgrad_in = F<mshadow_op::negation>(tgrad_in * tmarklabel);
  }

  AnchorClsCostParam param_;
};


template<typename xpu>
Operator* CreateOp(AnchorClsCostParam param);


#if DMLC_USE_CXX11
class AnchorClsCostProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "label", "marklabel"};
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
    CHECK_EQ(in_shape->size(), 3);

    TShape &datashape = (*in_shape)[anchorclscost_enum::kData];
    TShape &labelshape = (*in_shape)[anchorclscost_enum::kLabel];
    TShape &markshape = (*in_shape)[anchorclscost_enum::kMarkLabel];
    
    labelshape = datashape;
    markshape = datashape;
    
    out_shape->clear();
    out_shape->push_back(datashape);

    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new AnchorClsCostProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "AnchorClsCost";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_data[anchorclscost_enum::kOut],
            in_data[anchorclscost_enum::kData],
            in_data[anchorclscost_enum::kLabel],
            in_data[anchorclscost_enum::kMarkLabel],
            };
  };


  Operator* CreateOperator(Context ctx) const override;

 private:
  AnchorClsCostParam param_;
};  // class AnchorClsCostProp
#endif  // DMLC_USE_CXX11


}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_ANCHORCLASSIFYCOST_INL_H_


