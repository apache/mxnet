/*!
 * Copyright (c) 2015 by Contributors
 * \file reshape-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_RESHAPE_INL_H_
#define MXNET_OPERATOR_RESHAPE_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"

namespace mxnet {
namespace op {

namespace reshape_enum {
enum ReshapeOpInputs {kData};
enum ReshapeOpOutputs {kOut};
}  // namespace reshape_enum


struct ShapeInfo {
  inline size_t ndim() const {
    return info.size();
  }

  inline size_t Size() const {
    size_t sz = 1;
    for (size_t i = 0; i < info.size(); ++i) {
      sz *= info[i];
    }
    return sz;
  }

  std::vector<int> info;
};


inline std::istream &operator>>(std::istream &is, ShapeInfo &shape) {
  while (true) {
    char ch = is.get();
    if (ch == '(') break;
    if (!isspace(ch)) {
      is.setstate(std::ios::failbit);
      return is;
    }
  }
  int idx;
  std::vector<int> tmp;
  // deal with empty case
  // safe to remove after stop using target_shape
  size_t pos = is.tellg();
  char ch = is.get();
  if (ch == ')') {
    shape.info = tmp;
    return is;
  }
  is.seekg(pos);
  // finish deal
  while (is >> idx) {
    tmp.push_back(idx);
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
  shape.info = tmp;
  return is;
}


inline std::ostream &operator<<(std::ostream &os, const ShapeInfo &shape) {
  os << '(';
  for (index_t i = 0; i < shape.info.size(); ++i) {
    if (i != 0) os << ',';
    os << shape.info[i];
  }
  // python style tuple
  if (shape.info.size() == 1) os << ',';
  os << ')';
  return os;
}

struct ReshapeParam : public dmlc::Parameter<ReshapeParam> {
  TShape target_shape;
  bool keep_highest;
  ShapeInfo shape;
  DMLC_DECLARE_PARAMETER(ReshapeParam) {
    int tmp[] = {0, 0};
    DMLC_DECLARE_FIELD(target_shape)
    .set_default(TShape(tmp, tmp + 2))
    .describe("(Deprecated! Use shape instead.) Target new shape. One and only one dim can be 0, "
              "in which case it will be inferred from the rest of dims");
    DMLC_DECLARE_FIELD(keep_highest).set_default(false)
    .describe("(Deprecated! Use shape instead.) Whether keep the highest dim unchanged."
              "If set to true, then the first dim in target_shape is ignored,"
              "and always fixed as input");
    DMLC_DECLARE_FIELD(shape)
    .set_default(ShapeInfo())
    .describe("Target new shape. If the dim is same, set it to 0. If the dim is set "
              "to be -1, it will be inferred from the rest of dims. One and only one dim "
              "can be -1");
  }
};

template<typename xpu, typename DType>
class ReshapeOp : public Operator {
 public:
  explicit ReshapeOp(ReshapeParam param) {}  // Do nothing

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(req.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    if (req[reshape_enum::kOut] == kNullOp) return;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> data = in_data[reshape_enum::kData].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out = out_data[reshape_enum::kOut].FlatTo2D<xpu, DType>(s);
    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(out.CheckContiguous(), true);
    if (data.dptr_ == out.dptr_) return;
    CHECK_EQ(data.shape_.Size(), out.shape_.Size());
    Assign(out, req[reshape_enum::kOut], reshape(data, out.shape_));
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
    CHECK_EQ(req.size(), 1);
    if (req[reshape_enum::kData] == kNullOp) return;
    CHECK_EQ(out_grad.size(), 1);
    CHECK_EQ(in_grad.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> grad_in = in_grad[reshape_enum::kOut].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> grad_out = out_grad[reshape_enum::kData].FlatTo2D<xpu, DType>(s);
    CHECK_EQ(grad_out.CheckContiguous(), true);
    CHECK_EQ(grad_in.CheckContiguous(), true);
    if (grad_out.dptr_ == grad_in.dptr_) return;
    CHECK_EQ(grad_out.shape_.Size(), grad_in.shape_.Size());
    Assign(grad_in, req[reshape_enum::kData], reshape(grad_out, grad_in.shape_));
  }
};  // class ReshapeOp

template<typename xpu>
Operator* CreateOp(ReshapeParam, int dtype);

#if DMLC_USE_CXX11
class ReshapeProp : public OperatorProperty {
 public:
  ReshapeProp() {}

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    CHECK_EQ(in_shape->size(), 1) << "Input: [data]";
    CHECK_EQ(param_.target_shape.ndim() > 0 ||
             param_.shape.info.size() > 0, true) << "targe_shape or shape must be present.";
    const TShape &dshape = in_shape->at(reshape_enum::kData);
    if (dshape.ndim() == 0) return false;
    if (param_.shape.ndim() != 0) {
      std::vector<int> tmp;
      int src_idx = 0;
      int neg_idx = -1;
      size_t new_size = dshape.Size();
      bool keep = true;
      for (index_t i = 0; i < param_.shape.info.size(); ++i) {
        int proposed_dim = param_.shape.info[i];
        if (proposed_dim == 0) {
          // keep same
          CHECK_EQ(keep, true) << "After set manual dim, can't keep original dim";
          tmp.push_back(dshape[src_idx++]);
          new_size /= tmp.back();
        } else if (proposed_dim < 0) {
          // infer
          CHECK_LT(neg_idx, 0) << "One and only one dim can be inferred";
          neg_idx = i;
          tmp.push_back(0);
          src_idx++;
        } else {
          // greater than 0, new shape
          CHECK_EQ(new_size % proposed_dim, 0) << "Illegal dim setting, can't be divided.";
          tmp.push_back(proposed_dim);
          new_size /= proposed_dim;
          // after set manual shape, can't keep same
          if (param_.shape.info.size() != dshape.ndim()) {
            keep = false;
          } else {
            src_idx++;
          }
        }
      }

      if (neg_idx >= 0) {
        tmp[neg_idx] = new_size;
      }
      TShape oshape(tmp.begin(), tmp.end());
      CHECK_EQ(oshape.Size(), dshape.Size())
        << "Target shape size is different to source. "
        << "Target: " << param_.target_shape.Size()
        << "\nSource: " << dshape.Size();
      out_shape->clear();
      out_shape->push_back(oshape);
    } else {
      LOG(INFO) << "Using target_shape will be deprecated.";
      TShape oshape = param_.target_shape;
      int neg_count = 0;
      index_t neg_idx = 0;
      index_t start_idx = param_.keep_highest ? 1 : 0;
      if (param_.keep_highest) {
        oshape[0] = dshape[0];
      }
      for (index_t i = start_idx; i < oshape.ndim(); ++i) {
        if (oshape[i] == 0) {
          neg_count++;
          neg_idx = i;
        }
      }
      if (neg_count == 1) {
        oshape[neg_idx] = 1;
        oshape[neg_idx] = dshape.Size() / oshape.Size();
      }

      CHECK(oshape.Size() == dshape.Size())
          << "Target shape size is different to source. "
          << "Target: " << param_.target_shape.Size()
          << "\nSource: " << dshape.Size();
      out_shape->clear();
      out_shape->push_back(oshape);
    }
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), 1);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";

    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new ReshapeProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "Reshape";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[reshape_enum::kOut]};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[reshape_enum::kData], out_data[reshape_enum::kOut]}};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[reshape_enum::kOut], in_grad[reshape_enum::kData]}};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not implemented";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 protected:
  ReshapeParam param_;
};  // class ReshapeProp

class FlattenProp : public ReshapeProp {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {}

  std::map<std::string, std::string> GetParams() const override {
    // need to use this on osx
    return std::map<std::string, std::string>();
  }

  std::string TypeString() const override {
    return "Flatten";
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    CHECK_EQ(in_shape->size(), 1) << "Input: [data]";
    const TShape &dshape = in_shape->at(reshape_enum::kData);
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    uint32_t target_dim = 1;
    for (uint32_t i = 1; i < dshape.ndim(); ++i) {
      target_dim *= dshape[i];
    }
    out_shape->push_back(mshadow::Shape2(dshape[0], target_dim));
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new FlattenProp();
    return ptr;
  }
};  // class FlattenProp
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_RESHAPE_INL_H_
