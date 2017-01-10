/*!
 * Copyright (c) 2016 by Contributors
 * \file multibox_detection-inl.h
 * \brief post-process multibox detection predictions
 * \author Joshua Zhang
*/
#ifndef MXNET_OPERATOR_MULTIBOX_DETECTION_INL_H_
#define MXNET_OPERATOR_MULTIBOX_DETECTION_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/base.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <valarray>
#include "./operator_common.h"

namespace mxnet {
namespace op {
namespace mboxdet_enum {
enum MultiBoxDetectionOpInputs {kClsProb, kLocPred, kAnchor};
enum MultiBoxDetectionOpOutputs {kOut};
enum MultiBoxDetectionOpResource {kTempSpace};
}  // namespace mboxdet_enum

struct VarInfo {
  VarInfo() {}
  explicit VarInfo(std::vector<float> in) : info(in) {}

  std::vector<float> info;
};  // struct VarInfo

inline std::istream &operator>>(std::istream &is, VarInfo &size) {
  while (true) {
    char ch = is.get();
    if (ch == '(') break;
    if (!isspace(ch)) {
      is.setstate(std::ios::failbit);
      return is;
    }
  }
  float f;
  std::vector<float> tmp;
  // deal with empty case
  // safe to remove after stop using target_size
  size_t pos = is.tellg();
  char ch = is.get();
  if (ch == ')') {
    size.info = tmp;
    return is;
  }
  is.seekg(pos);
  // finish deal
  while (is >> f) {
    tmp.push_back(f);
    char ch;
    do {
      ch = is.get();
    } while (isspace(ch));
    if (ch == ',') {
      while (true) {
        ch = is.peek();
        if (isspace(ch)) {
          is.get(); continue;
        }
        if (ch == ')') {
          is.get(); break;
        }
        break;
      }
      if (ch == ')') break;
    } else if (ch == ')') {
      break;
    } else {
      is.setstate(std::ios::failbit);
      return is;
    }
  }
  size.info = tmp;
  return is;
}

inline std::ostream &operator<<(std::ostream &os, const VarInfo &size) {
  os << '(';
  for (index_t i = 0; i < size.info.size(); ++i) {
    if (i != 0) os << ',';
    os << size.info[i];
  }
  // python style tuple
  if (size.info.size() == 1) os << ',';
  os << ')';
  return os;
}

struct MultiBoxDetectionParam : public dmlc::Parameter<MultiBoxDetectionParam> {
  bool clip;
  float threshold;
  int background_id;
  float nms_threshold;
  bool force_suppress;
  VarInfo variances;
  DMLC_DECLARE_PARAMETER(MultiBoxDetectionParam) {
    DMLC_DECLARE_FIELD(clip).set_default(true)
    .describe("Clip out-of-boundary boxes.");
    DMLC_DECLARE_FIELD(threshold).set_default(0.01f)
    .describe("Threshold to be a positive prediction.");
    DMLC_DECLARE_FIELD(background_id).set_default(0)
    .describe("Background id.");
    DMLC_DECLARE_FIELD(nms_threshold).set_default(0.5f)
    .describe("Non-maximum suppression threshold.");
    DMLC_DECLARE_FIELD(force_suppress).set_default(false)
    .describe("Suppress all detections regardless of class_id.");
    DMLC_DECLARE_FIELD(variances).set_default(VarInfo({0.1f, 0.1f, 0.2f, 0.2f}))
    .describe("Variances to be decoded from box regression output.");
  }
};  // struct MultiBoxDetectionParam

template<typename xpu, typename DType>
class MultiBoxDetectionOp : public Operator {
 public:
  explicit MultiBoxDetectionOp(MultiBoxDetectionParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
     using namespace mshadow;
     using namespace mshadow::expr;
     CHECK_EQ(in_data.size(), 3) << "Input: [cls_prob, loc_pred, anchor]";
     TShape ashape = in_data[mboxdet_enum::kAnchor].shape_;
     CHECK_EQ(out_data.size(), 1);

     Stream<xpu> *s = ctx.get_stream<xpu>();
     Tensor<xpu, 3, DType> cls_prob = in_data[mboxdet_enum::kClsProb]
       .get<xpu, 3, DType>(s);
     Tensor<xpu, 2, DType> loc_pred = in_data[mboxdet_enum::kLocPred]
       .get<xpu, 2, DType>(s);
     Tensor<xpu, 2, DType> anchors = in_data[mboxdet_enum::kAnchor]
       .get_with_shape<xpu, 2, DType>(Shape2(ashape[1], 4), s);
     Tensor<xpu, 3, DType> out = out_data[mboxdet_enum::kOut]
       .get<xpu, 3, DType>(s);
     Tensor<xpu, 3, DType> temp_space = ctx.requested[mboxdet_enum::kTempSpace]
       .get_space_typed<xpu, 3, DType>(out.shape_, s);
     out = -1.f;
     MultiBoxDetectionForward(out, cls_prob, loc_pred, anchors, temp_space,
       param_.threshold, param_.clip, param_.variances.info, param_.nms_threshold,
       param_.force_suppress);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
}

 private:
  MultiBoxDetectionParam param_;
};  // class MultiBoxDetectionOp

template<typename xpu>
Operator *CreateOp(MultiBoxDetectionParam, int dtype);

#if DMLC_USE_CXX11
class MultiBoxDetectionProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  std::vector<std::string> ListArguments() const override {
    return {"cls_prob", "loc_pred", "anchor"};
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 3) << "Inputs: [cls_prob, loc_pred, anchor]";
    TShape cshape = in_shape->at(mboxdet_enum::kClsProb);
    TShape lshape = in_shape->at(mboxdet_enum::kLocPred);
    TShape ashape = in_shape->at(mboxdet_enum::kAnchor);
    CHECK_EQ(cshape.ndim(), 3) << "Provided: " << cshape;
    CHECK_EQ(lshape.ndim(), 2) << "Provided: " << lshape;
    CHECK_EQ(ashape.ndim(), 3) << "Provided: " << ashape;
    CHECK_EQ(cshape[2], ashape[1]) << "Number of anchors mismatch";
    CHECK_EQ(cshape[2] * 4, lshape[1]) << "# anchors mismatch with # loc";
    CHECK_GT(ashape[1], 0) << "Number of anchors must > 0";
    CHECK_EQ(ashape[2], 4);
    TShape oshape = TShape(3);
    oshape[0] = cshape[0];
    oshape[1] = ashape[1];
    oshape[2] = 6;  // [id, prob, xmin, ymin, xmax, ymax]
    out_shape->clear();
    out_shape->push_back(oshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new MultiBoxDetectionProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "MultiBoxDetection";
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not implemented";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  MultiBoxDetectionParam param_;
};  // class MultiBoxDetectionProp
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_MULTIBOX_DETECTION_INL_H_
