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
  bool reverse;
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
    .describe("Target shape, a tuple, t=(t_1,t_2,..,t_m).\n"
              "Let the input dims be s=(s_1,s_2,..,s_n).\n"
              "The output dims u=(u_1,u_2,..,u_p) are computed from s and t.\n"
              "The target shape tuple elements t_i are read in order, and used to "
              " generate successive output dims u_p:\n"
              "t_i:       meaning:      behavior:\n"
              "+ve        explicit      u_p = t_i\n"
              "0          copy          u_p = s_i\n"
              "-1         infer         u_p = (Prod s_i) / (Prod u_j | j != p)\n"
              "-2         copy all      u_p = s_i, u_p+1 = s_i+1, ...\n"
              "-3         merge two     u_p = s_i * s_i+1\n"
              "-4,a,b     split two     u_p = a, u_p+1 = b | a * b = s_i\n"
              "The split directive (-4) in the target shape tuple is followed by "
              "two dimensions, one of which can be -1, which means it will be "
              "inferred from the other one and the original dimension.\n"
              "The can only be one globally inferred dimension (-1), aside from "
              "any -1 occuring in a split directive.");
    DMLC_DECLARE_FIELD(reverse)
      .set_default(false)
      .describe("Whether to match the shapes from the backward. If reverse is true, "
      "0 values in the `shape` argument will be searched from the backward. E.g the "
      "original shape is (10, 5, 4) and the shape argument is (-1, 0). If reverse is true, "
      "the new shape should be (50, 4). Otherwise it will be (40, 5).");
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
      std::vector<int> dshape_vec;
      std::vector<int> param_shape_vec(param_.shape.info);
      for (index_t i = 0; i < dshape.ndim(); ++i) {
        dshape_vec.push_back(dshape[i]);
      }
      std::vector<int> tmp;
      int src_idx = 0;
      int inf_idx = -1;
      size_t new_size = dshape.Size();
      if (param_.reverse) {
        std::reverse(dshape_vec.begin(), dshape_vec.end());
        std::reverse(param_shape_vec.begin(), param_shape_vec.end());
      }
      auto dshape_len = dshape_vec.size();
      auto params_len = param_shape_vec.size();
      for (index_t i = 0; i < params_len; ++i) {
        int proposed_dim = param_shape_vec[i];
        if (proposed_dim == 0) {
          // keep same
          CHECK_LT(src_idx, dshape_len);
          tmp.push_back(dshape_vec[src_idx++]);
          new_size /= tmp.back();
        } else if (proposed_dim == -1) {
          // infer
          CHECK_LT(inf_idx, 0) << "One and only one dim can be inferred";
          inf_idx = i;
          tmp.push_back(0);
          src_idx++;
        } else if (proposed_dim == -2) {
          // copy all remaining dims from source
          while (src_idx < dshape_len) {
            size_t dn = dshape_vec[src_idx++];
            new_size /= dn;
            tmp.push_back(dn);
          }
        } else if (proposed_dim == -3) {
          // merge two dims from source
          CHECK_LT(src_idx, dshape_len-1);
          size_t d1 = dshape_vec[src_idx++];
          size_t d2 = dshape_vec[src_idx++];
          size_t dn = d1 * d2;
          new_size /= dn;
          tmp.push_back(dn);
        } else if (proposed_dim == -4) {
          // split the source dim s into two dims
          // read the left dim and then the right dim (either can be -1)
          CHECK_LT(i + 2, params_len);
          CHECK_LT(src_idx, dshape_len);
          size_t d0 = dshape_vec[src_idx++];
          int d1 = param_shape_vec[++i];
          int d2 = param_shape_vec[++i];
          CHECK(d1 != -1 || d2 != -1) << "Split dims cannot both be -1.";
          if (d1 == -1) d1 = d0 / d2;
          if (d2 == -1) d2 = d0 / d1;
          CHECK_EQ(d1 * d2, d0) <<
            "Split dims " << d1 << ", " << d2 << " do not divide original dim " << d0;
          new_size /= d0;
          tmp.push_back(d1);
          tmp.push_back(d2);
        } else {
          // greater than 0, new shape
          CHECK_EQ(new_size % proposed_dim, 0) << "Illegal dim setting, can't be divided.";
          tmp.push_back(proposed_dim);
          new_size /= proposed_dim;
          src_idx++;
        }
      }

      if (inf_idx >= 0) {
        tmp[inf_idx] = new_size;
      }
      if (param_.reverse) {
        std::reverse(param_shape_vec.begin(), param_shape_vec.end());
        std::reverse(dshape_vec.begin(), dshape_vec.end());
        std::reverse(tmp.begin(), tmp.end());
      }
      TShape oshape(tmp.begin(), tmp.end());
      CHECK_EQ(oshape.Size(), dshape.Size())
        << "Target shape size is different to source. "
        << "Target: " << oshape
        << "\nSource: " << dshape;
      out_shape->clear();
      out_shape->push_back(oshape);
    } else {
      LOG(INFO) << "Using target_shape will be deprecated.";
      TShape oshape = param_.target_shape;
      int neg_count = 0;
      index_t inf_idx = 0;
      index_t start_idx = param_.keep_highest ? 1 : 0;
      if (param_.keep_highest) {
        oshape[0] = dshape[0];
      }
      for (index_t i = start_idx; i < oshape.ndim(); ++i) {
        if (oshape[i] == 0) {
          neg_count++;
          inf_idx = i;
        }
      }
      if (neg_count == 1) {
        oshape[inf_idx] = 1;
        oshape[inf_idx] = dshape.Size() / oshape.Size();
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
