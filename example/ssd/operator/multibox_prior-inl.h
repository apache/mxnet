/*!
 * Copyright (c) 2016 by Contributors
 * \file multibox_prior-inl.h
 * \brief generate multibox prior boxes
 * \author Joshua Zhang
*/
#ifndef MXNET_OPERATOR_MULTIBOX_PRIOR_INL_H_
#define MXNET_OPERATOR_MULTIBOX_PRIOR_INL_H_
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

namespace mshadow_op {
struct clip_zero_one {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    if (a < 0.f) return DType(0.f);
    if (a > 1.f) return DType(1.f);
    return DType(a);
  }
};  // struct clip_zero_one
}  // namespace mshadow_op

namespace mboxprior_enum {
enum MultiBoxPriorOpInputs {kData};
enum MultiBoxPriorOpOutputs {kOut};
}  // namespace mboxprior_enum

struct SizeInfo {
  SizeInfo() {}
  explicit SizeInfo(std::vector<float> in) : info(in) {}

  std::vector<float> info;
};  // struct SizeInfo

inline std::istream &operator>>(std::istream &is, SizeInfo &size) {
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

inline std::ostream &operator<<(std::ostream &os, const SizeInfo &size) {
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

struct MultiBoxPriorParam : public dmlc::Parameter<MultiBoxPriorParam> {
  SizeInfo sizes;
  SizeInfo ratios;
  bool clip;
  DMLC_DECLARE_PARAMETER(MultiBoxPriorParam) {
    DMLC_DECLARE_FIELD(sizes).set_default(SizeInfo({1.0f}))
    .describe("List of sizes of generated MultiBoxPriores.");
    DMLC_DECLARE_FIELD(ratios).set_default(SizeInfo({1.0f}))
    .describe("List of aspect ratios of generated MultiBoxPriores.");
    DMLC_DECLARE_FIELD(clip).set_default(false)
    .describe("Whether to clip out-of-boundary boxes.");
  }
};  // struct MultiBoxPriorParam

template<typename xpu, typename DType>
class MultiBoxPriorOp : public Operator {
 public:
  explicit MultiBoxPriorOp(MultiBoxPriorParam param)
    : clip_(param.clip), sizes_(param.sizes.info), ratios_(param.ratios.info) {
      CHECK_GT(sizes_.size(), 0);
      CHECK_GT(ratios_.size(), 0);
    }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(static_cast<int>(in_data.size()), 1);
    CHECK_GE(in_data[mboxprior_enum::kData].ndim(), 4);  // require spatial information
    int in_height = in_data[mboxprior_enum::kData].size(2);
    CHECK_GT(in_height, 0);
    int in_width = in_data[mboxprior_enum::kData].size(3);
    CHECK_GT(in_width, 0);
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> out;
    // TODO(Joshua Zhang): this implementation is to be compliant to original ssd in caffe
    // The prior boxes could be implemented in more versatile ways
    // since input sizes are same in each batch, we could share MultiBoxPrior
    const int num_sizes = static_cast<int>(sizes_.size());
    const int num_ratios = static_cast<int>(ratios_.size());
    const int num_anchors = num_sizes - 1 + num_ratios;  // anchors per location
    Shape<2> oshape = Shape2(num_anchors * in_width * in_height, 4);
    out = out_data[mboxprior_enum::kOut].get_with_shape<xpu, 2, DType>(oshape, s);
    MultiBoxPriorForward(out, sizes_, ratios_, in_width, in_height);

    if (clip_) {
      Assign(out, req[mboxprior_enum::kOut], F<mshadow_op::clip_zero_one>(out));
    }
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
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> grad = in_grad[mboxprior_enum::kData].FlatTo2D<xpu, DType>(s);
    grad = 0.f;
  }

 private:
  bool clip_;
  std::vector<float> sizes_;
  std::vector<float> ratios_;
};  // class MultiBoxPriorOp

template<typename xpu>
Operator *CreateOp(MultiBoxPriorParam, int dtype);

#if DMLC_USE_CXX11
class MultiBoxPriorProp: public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  std::vector<std::string> ListArguments() const override {
    return {"data"};
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 1) << "Inputs: [data]" << in_shape->size();
    TShape dshape = in_shape->at(mboxprior_enum::kData);
    CHECK_GE(dshape.ndim(), 4) << "Input data should be 4D: batch-channel-y-x";
    int in_height = dshape[2];
    CHECK_GT(in_height, 0) << "Input height should > 0";
    int in_width = dshape[3];
    CHECK_GT(in_width, 0) << "Input width should > 0";
    // since input sizes are same in each batch, we could share MultiBoxPrior
    TShape oshape = TShape(3);
    int num_sizes = param_.sizes.info.size();
    int num_ratios = param_.ratios.info.size();
    oshape[0] = 1;
    oshape[1] = in_height * in_width * (num_sizes + num_ratios - 1);
    oshape[2] = 4;
    out_shape->clear();
    out_shape->push_back(oshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new MultiBoxPriorProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "MultiBoxPrior";
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not implemented";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  MultiBoxPriorParam param_;
};  // class MultiBoxPriorProp
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_MULTIBOX_PRIOR_INL_H_
