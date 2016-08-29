/*!
 * Copyright (c) 2016 by Contributors
 * \file postprocess_rpn-inl.h
 * \brief
 * \author Ming Zhang
*/
#ifndef MXNET_POSTPROCESS_RPN_INL_H_
#define MXNET_POSTPROCESS_RPN_INL_H_

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

namespace postprocess_rpn_enum {
enum PostProcessRPNOpInputs {kDataCls, kDataReg, kAnchorInfo, kOtherInfo};
enum PostProcessRPNOpOutputs {kOut};
};


struct PostProcessRPNParam : public dmlc::Parameter<PostProcessRPNParam> {
  int maxoutbbnum;
  DMLC_DECLARE_PARAMETER(PostProcessRPNParam) {
    DMLC_DECLARE_FIELD(maxoutbbnum)
    .set_default(0)
    .describe("max out bounding box number[0].");
  }
};


template<typename xpu>
class PostProcessRPNOp : public Operator {
 public:
  explicit PostProcessRPNOp(PostProcessRPNParam p) {
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
    
    TBlob datacls_in = in_data[postprocess_rpn_enum::kDataCls];
    TBlob datareg_in = in_data[postprocess_rpn_enum::kDataReg];
    TBlob anchorinfo_in = in_data[postprocess_rpn_enum::kAnchorInfo];
    TBlob otherinfo_in = in_data[postprocess_rpn_enum::kOtherInfo];
    TBlob bb_out = out_data[postprocess_rpn_enum::kOut];

    Tensor<xpu, 4> tdatacls_in = datacls_in.get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> tdatareg_in = datareg_in.get<xpu, 4, real_t>(s);
    Tensor<xpu, 2> tanchorinfo_in = anchorinfo_in.get<xpu, 2, real_t>(s);
    Tensor<xpu, 1> totherinfo_in = otherinfo_in.get<xpu, 1, real_t>(s);
    Tensor<xpu, 3> tbb_out = bb_out.get<xpu, 3, real_t>(s);

    CHECK_EQ(tdatacls_in.CheckContiguous(), true);
    CHECK_EQ(tdatareg_in.CheckContiguous(), true);
    CHECK_EQ(tanchorinfo_in.CheckContiguous(), true);
    CHECK_EQ(totherinfo_in.CheckContiguous(), true);
    CHECK_EQ(tbb_out.CheckContiguous(), true);

    PostProcessRPNForward(tdatacls_in, tdatareg_in, tanchorinfo_in, totherinfo_in, tbb_out);
  }

  virtual void Backward(const OpContext &ctx,
                       const std::vector<TBlob> &out_grad,
                       const std::vector<TBlob> &in_data,
                       const std::vector<TBlob> &out_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &in_grad,
                       const std::vector<TBlob> &aux_args) {

  }

  PostProcessRPNParam param_;
};


template<typename xpu>
Operator* CreateOp(PostProcessRPNParam param);


#if DMLC_USE_CXX11
class PostProcessRPNProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"datacls", "datareg", "anchorinfo", "otherinfo"};
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
    CHECK_EQ(in_shape->size(), 4);

    TShape &clsshape = (*in_shape)[postprocess_rpn_enum::kDataCls];
    TShape &regshape = (*in_shape)[postprocess_rpn_enum::kDataReg];
    TShape &anchorinfoshape = (*in_shape)[postprocess_rpn_enum::kAnchorInfo];
    TShape &otherinfoshape = (*in_shape)[postprocess_rpn_enum::kOtherInfo];
    
    CHECK_EQ(clsshape[1]*4, regshape[1]);
    
    anchorinfoshape = Shape2(clsshape[1], 2);
    otherinfoshape = Shape1(3);

    TShape outbbshape = Shape3(clsshape[0], param_.maxoutbbnum, 5); //batch, bbnum, [score, cx, cy, w, h]

    out_shape->clear();
    out_shape->push_back(outbbshape);

    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new PostProcessRPNProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "PostProcessRPN";
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  PostProcessRPNParam param_;
};  // class PostProcessRPNProp
#endif  // DMLC_USE_CXX11


}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_ANCHOR_REGRESSIONCOST_INL_H_


