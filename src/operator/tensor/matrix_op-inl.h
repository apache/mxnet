/*!
 *  Copyright (c) 2015 by Contributors
 * \file broadcast_reduce_op-inl.h
 * \brief Function defintion of matrix related operators
 */
#ifndef MXNET_OPERATOR_TENSOR_MATRIX_OP_INL_H_
#define MXNET_OPERATOR_TENSOR_MATRIX_OP_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

struct ReshapeParam : public dmlc::Parameter<ReshapeParam> {
  TShape target_shape;
  bool keep_highest;
  nnvm::Tuple<int> shape;
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
    .set_default(nnvm::Tuple<int>())
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

inline bool ReshapeShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape> *in_attrs,
                             std::vector<TShape> *out_attrs) {
  const ReshapeParam& param_ = nnvm::get<ReshapeParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1) << "Input: [data]";
  CHECK_EQ(out_attrs->size(), 1);
  CHECK_EQ(param_.target_shape.ndim() > 0 ||
           param_.shape.ndim() > 0, true) << "targe_shape or shape must be present.";
  const TShape &dshape = (*in_attrs)[0];
  if (dshape.ndim() == 0) return false;
  if (param_.shape.ndim() != 0) {
    std::vector<int> dshape_vec;
    std::vector<int> param_shape_vec(param_.shape.begin(), param_.shape.end());
    for (index_t i = 0; i < dshape.ndim(); ++i) {
      dshape_vec.push_back(dshape[i]);
    }
    std::vector<int> tmp;
    size_t src_idx = 0;
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
    out_attrs->clear();
    out_attrs->push_back(oshape);
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
    out_attrs->clear();
    out_attrs->push_back(oshape);
  }
  return true;
}

inline bool FlattenShape(const nnvm::NodeAttrs& attrs,
                         std::vector<TShape> *in_attrs,
                         std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1) << "Input: [data]";
  CHECK_EQ(out_attrs->size(), 1);
  const TShape &dshape = (*in_attrs)[0];
  if (dshape.ndim() == 0) return false;
  out_attrs->clear();
  uint32_t target_dim = 1;
  for (uint32_t i = 1; i < dshape.ndim(); ++i) {
    target_dim *= dshape[i];
  }
  out_attrs->push_back(mshadow::Shape2(dshape[0], target_dim));
  return true;
}

struct TransposeParam : public dmlc::Parameter<TransposeParam> {
  TShape axes;
  DMLC_DECLARE_PARAMETER(TransposeParam) {
    DMLC_DECLARE_FIELD(axes).set_default(TShape())
    .describe("Target axis order. By default the axes will be inverted.");
  }
};

template<typename xpu>
void TransposeImpl(RunContext ctx,
                   const TBlob& src,
                   const TBlob& ret,
                   const TShape& axes) {
  using namespace mshadow;
  using namespace mshadow::expr;
  CHECK_EQ(src.type_flag_, ret.type_flag_);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(ret.type_flag_, DType, {
    switch (axes.ndim()) {
     case 0:
      break;
     case 1: {
      Tensor<xpu, 1, DType> in = src.get<xpu, 1, DType>(s);
      Tensor<xpu, 1, DType> out = ret.get<xpu, 1, DType>(s);
      Copy(out, in, s);
      break;
     }
     case 2: {
      mshadow::Tensor<xpu, 2, DType> in = src.FlatTo2D<xpu, DType>(s);
      mshadow::Tensor<xpu, 2, DType> out = ret.FlatTo2D<xpu, DType>(s);
      if (axes[0] == 1 && axes[1] == 0) {
        out = in.T();
      } else {
        Copy(out, in, s);
      }
      break;
     }
     case 3: {
      Tensor<xpu, 3, DType> in = src.get<xpu, 3, DType>(s);
      Tensor<xpu, 3, DType> out = ret.get<xpu, 3, DType>(s);
      out = transpose(in, axes.get<3>());
      break;
     }
     case 4: {
      Tensor<xpu, 4, DType> in = src.get<xpu, 4, DType>(s);
      Tensor<xpu, 4, DType> out = ret.get<xpu, 4, DType>(s);
      out = transpose(in, axes.get<4>());
      break;
     }
     case 5: {
      Tensor<xpu, 5, DType> in = src.get<xpu, 5, DType>(s);
      Tensor<xpu, 5, DType> out = ret.get<xpu, 5, DType>(s);
      out = transpose(in, axes.get<5>());
      break;
     }
     default:
      LOG(FATAL) << "Transpose support at most 5 dimensions";
      break;
    }
  });
}

// matrix transpose
template<typename xpu>
void Transpose(const nnvm::NodeAttrs& attrs,
               const OpContext& ctx,
               const std::vector<TBlob>& inputs,
               const std::vector<OpReqType>& req,
               const std::vector<TBlob>& outputs) {
  const TransposeParam& param = nnvm::get<TransposeParam>(attrs.parsed);
  CHECK_EQ(req[0], kWriteTo) << "Transpose does not support inplace";
  if (param.axes.ndim() == 0) {
    TShape axes = TShape(inputs[0].ndim());
    for (index_t i = 0; i < axes.ndim(); ++i) {
      axes[i] = axes.ndim() - 1 - i;
    }
    TransposeImpl<xpu>(ctx.run_ctx, inputs[0], outputs[0], axes);
  } else {
    TransposeImpl<xpu>(ctx.run_ctx, inputs[0], outputs[0], param.axes);
  }
}

inline bool TransposeShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape> *in_attrs,
                             std::vector<TShape> *out_attrs) {
  const TransposeParam& param = nnvm::get<TransposeParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 1);
  TShape& shp = (*in_attrs)[0];
  CHECK_LE(shp.ndim(), 5) << "Transpose support at most 5 dimensions";
  TShape ret(shp.ndim());
  if (param.axes.ndim() == 0) {
    for (index_t i = 0; i < shp.ndim(); ++i) {
      ret[i] = shp[shp.ndim()-1-i];
    }
  } else {
    CHECK_EQ(shp.ndim(), param.axes.ndim());
    for (index_t i = 0; i < shp.ndim(); ++i) {
      CHECK(param.axes[i] < shp.ndim());
      ret[i] = shp[param.axes[i]];
    }
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, ret);
  return true;
}


struct ExpandDimParam : public dmlc::Parameter<ExpandDimParam> {
  index_t axis;
  DMLC_DECLARE_PARAMETER(ExpandDimParam) {
    DMLC_DECLARE_FIELD(axis)
    .describe("Position (amongst axes) where new axis is to be inserted.");
  }
};


inline bool ExpandDimShape(const nnvm::NodeAttrs& attrs,
                           std::vector<TShape> *in_attrs,
                           std::vector<TShape> *out_attrs) {
  const ExpandDimParam& param = nnvm::get<ExpandDimParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 1);
  TShape& shp = (*in_attrs)[0];
  CHECK_LE(param.axis, shp.ndim())
      << "axis exceeds the dimension of the array";
  TShape ret(shp.ndim() + 1);
  for (index_t i = 0; i < param.axis; ++i) ret[i] = shp[i];
  ret[param.axis] = 1;
  for (index_t i = param.axis+1; i < ret.ndim(); ++i) ret[i] = shp[i-1];
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, ret);
  return true;
}

template<typename xpu>
void DotForward_(const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx,
                 const std::vector<TBlob>& inputs,
                 const std::vector<OpReqType>& req,
                 const std::vector<TBlob>& outputs) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(outputs[0].type_flag_, inputs[0].type_flag_)
      << "Binary function only support input/output with the same type";
  CHECK_EQ(outputs[0].type_flag_, inputs[1].type_flag_)
      << "Binary function only support input/output with the same type";
  CHECK_EQ(outputs[0].type_flag_, mshadow::kFloat32)
      << "dot only support 32 bit float so far";

  if (inputs[0].ndim() == 2 && inputs[1].ndim() == 2) {
    mshadow::Tensor<xpu, 2, real_t> out = outputs[0].FlatTo2D<xpu, real_t>(s);
    ASSIGN_DISPATCH(out, req[0],
                    dot(inputs[0].get<xpu, 2, real_t>(s),
                        inputs[1].get<xpu, 2, real_t>(s)));
  } else {
    CHECK_NE(req[0], kAddTo) << "AddTo not yet suported";
    mshadow::Tensor<xpu, 1, real_t> out = outputs[0].get<xpu, 1, real_t>(s);
    mshadow::VectorDot(out,
                       inputs[0].get<xpu, 1, real_t>(s),
                       inputs[1].get<xpu, 1, real_t>(s));
  }
}

template<typename xpu>
void DotBackward_(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_NE(req[0], kWriteInplace);
  CHECK_NE(req[1], kWriteInplace);

  if (inputs[1].ndim() == 2 && inputs[2].ndim() == 2) {
    mshadow::Tensor<xpu, 2, real_t> mout_grad = inputs[0].get<xpu, 2, real_t>(s);
    mshadow::Tensor<xpu, 2, real_t> mlhs_data = inputs[1].get<xpu, 2, real_t>(s);
    mshadow::Tensor<xpu, 2, real_t> mrhs_data = inputs[2].get<xpu, 2, real_t>(s);
    mshadow::Tensor<xpu, 2, real_t> mlhs_grad = outputs[0].get<xpu, 2, real_t>(s);
    mshadow::Tensor<xpu, 2, real_t> mrhs_grad = outputs[1].get<xpu, 2, real_t>(s);
    ASSIGN_DISPATCH(mrhs_grad, req[1], dot(mlhs_data.T(), mout_grad));
    ASSIGN_DISPATCH(mlhs_grad, req[0], dot(mout_grad, mrhs_data.T()));
  } else {
    mshadow::Tensor<xpu, 1, real_t> mout_grad = inputs[0].get<xpu, 1, real_t>(s);
    mshadow::Tensor<xpu, 1, real_t> mlhs_data = inputs[1].get<xpu, 1, real_t>(s);
    mshadow::Tensor<xpu, 1, real_t> mrhs_data = inputs[2].get<xpu, 1, real_t>(s);
    mshadow::Tensor<xpu, 1, real_t> mlhs_grad = outputs[0].get<xpu, 1, real_t>(s);
    mshadow::Tensor<xpu, 1, real_t> mrhs_grad = outputs[1].get<xpu, 1, real_t>(s);
    ASSIGN_DISPATCH(mrhs_grad, req[1],
                    broadcast_scalar(mout_grad, mlhs_data.shape_) * mlhs_data);
    ASSIGN_DISPATCH(mlhs_grad, req[0],
                    broadcast_scalar(mout_grad, mlhs_data.shape_) * mrhs_data);
  }
}

inline bool DotShape(const nnvm::NodeAttrs& attrs,
                     std::vector<TShape> *in_attrs,
                     std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2);
  CHECK_EQ(out_attrs->size(), 1);
  TShape& lshape = (*in_attrs)[0];
  TShape& rshape = (*in_attrs)[1];
  if (lshape.ndim() == 2 && rshape.ndim() == 2) {
    CHECK_EQ(lshape[1], rshape[0]) << "dot shape error: " << lshape << " X " << rshape;
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::Shape2(lshape[0], rshape[1]));
  } else if (lshape.ndim() == 1 && rshape.ndim() == 1) {
    CHECK_EQ(lshape[0], rshape[0]) << "dot shape error: " << lshape << " X " << rshape;
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::Shape1(1));
  } else {
    LOG(FATAL) << "dot currently only support 2D*2D array or 1D*1D array"
               << lshape << " v.s. " << rshape;
    return false;
  }
  return true;
}

template<typename xpu>
void BatchDotForward_(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(outputs[0].type_flag_, inputs[0].type_flag_)
      << "Binary function only support input/output with the same type";
  CHECK_EQ(outputs[0].type_flag_, inputs[1].type_flag_)
      << "Binary function only support input/output with the same type";
  CHECK_EQ(outputs[0].type_flag_, mshadow::kFloat32)
      << "dot only support 32 bit float so far";

  mshadow::Tensor<xpu, 3, real_t> out = outputs[0].get<xpu, 3, real_t>(s);
  mshadow::Tensor<xpu, 3, real_t> mlhs = inputs[0].get<xpu, 3, real_t>(s);
  mshadow::Tensor<xpu, 3, real_t> mrhs = inputs[1].get<xpu, 3, real_t>(s);
  mshadow::Tensor<xpu, 1, real_t*> workspace =
    ctx.requested[0].get_space_typed<xpu, 1, real_t*>(mshadow::Shape1(3 * out.size(0)), s);
  if (kNullOp != req[0]) {
    mshadow::BatchGEMM<false, false>(out, mlhs, mrhs, 1.0f,
                                     (kAddTo == req[0]) ? 1.0f : 0.0f,
                                     workspace);
  }
}

template<typename xpu>
void BatchDotBackward_(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_NE(req[1], kWriteInplace);
  CHECK_NE(req[0], kWriteInplace);

  mshadow::Tensor<xpu, 3, real_t> mout_grad = inputs[0].get<xpu, 3, real_t>(s);
  mshadow::Tensor<xpu, 3, real_t> mlhs_data = inputs[1].get<xpu, 3, real_t>(s);
  mshadow::Tensor<xpu, 3, real_t> mrhs_data = inputs[2].get<xpu, 3, real_t>(s);
  mshadow::Tensor<xpu, 3, real_t> mlhs_grad = outputs[0].get<xpu, 3, real_t>(s);
  mshadow::Tensor<xpu, 3, real_t> mrhs_grad = outputs[1].get<xpu, 3, real_t>(s);
  mshadow::Tensor<xpu, 2, real_t*> workspace =
    ctx.requested[0].get_space_typed<xpu, 2, real_t*>(
      mshadow::Shape2(2, 3 * mout_grad.size(0)), s);
  mshadow::Tensor<xpu, 1, real_t*> rhs_workspace = workspace[0];
  mshadow::Tensor<xpu, 1, real_t*> lhs_workspace = workspace[1];
  if (kNullOp != req[1]) {
    mshadow::BatchGEMM<true, false>(mrhs_grad, mlhs_data, mout_grad, 1.0f,
                                    (kAddTo == req[1]) ? 1.0f : 0.0f,
                                    rhs_workspace);
  }
  if (kNullOp != req[0]) {
    mshadow::BatchGEMM<false, true>(mlhs_grad, mout_grad, mrhs_data, 1.0f,
                                    (kAddTo == req[0]) ? 1.0f : 0.0f,
                                    lhs_workspace);
  }
}

inline bool BatchDotShape(const nnvm::NodeAttrs& attrs,
                          std::vector<TShape> *in_attrs,
                          std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2);
  CHECK_EQ(out_attrs->size(), 1);
  TShape& lshape = (*in_attrs)[0];
  TShape& rshape = (*in_attrs)[1];
  if (lshape.ndim() == 3 && rshape.ndim() == 3) {
    CHECK(lshape[0] == rshape[0] && lshape[2] == rshape[1])
      << "batch_dot shape error: " << lshape << " X " << rshape;
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::Shape3(lshape[0], lshape[1], rshape[2]));
  } else {
    LOG(FATAL) << "batch_dot currently only support 3D*3D array"
               << lshape << " v.s. " << rshape;
  }
  return true;
}


struct SimpleCropParam : public dmlc::Parameter<SimpleCropParam> {
  TShape begin, end;
  DMLC_DECLARE_PARAMETER(SimpleCropParam) {
    DMLC_DECLARE_FIELD(begin)
    .describe("starting coordinates");
    DMLC_DECLARE_FIELD(end)
    .describe("ending coordinates");
  }
};

// matrix crop for multi dimensional cropping: see also slice
template<typename xpu>
void Crop(const nnvm::NodeAttrs& attrs,
          const OpContext& ctx,
          const std::vector<TBlob>& inputs,
          const std::vector<OpReqType>& req,
          const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  const SimpleCropParam& param = nnvm::get<SimpleCropParam>(attrs.parsed);
  CHECK_EQ(inputs[0].type_flag_, outputs[0].type_flag_);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    switch (inputs[0].ndim()) {
     case 0:
      break;
     case 1: {
      Tensor<xpu, 1, DType> in = inputs[0].get<xpu, 1, DType>(s);
      Tensor<xpu, 1, DType> out = outputs[0].get<xpu, 1, DType>(s);
      out = slice(in, param.begin.get<1>(), param.end.get<1>());
      break;
     }
     case 2: {
      Tensor<xpu, 2, DType> in = inputs[0].get<xpu, 2, DType>(s);
      Tensor<xpu, 2, DType> out = outputs[0].get<xpu, 2, DType>(s);
      out = slice(in, param.begin.get<2>(), param.end.get<2>());
      break;
     }
     case 3: {
      Tensor<xpu, 3, DType> in = inputs[0].get<xpu, 3, DType>(s);
      Tensor<xpu, 3, DType> out = outputs[0].get<xpu, 3, DType>(s);
      out = slice(in, param.begin.get<3>(), param.end.get<3>());
      break;
     }
     case 4: {
      Tensor<xpu, 4, DType> in = inputs[0].get<xpu, 4, DType>(s);
      Tensor<xpu, 4, DType> out = outputs[0].get<xpu, 4, DType>(s);
      out = slice(in, param.begin.get<4>(), param.end.get<4>());
      break;
     }
     case 5: {
      Tensor<xpu, 5, DType> in = inputs[0].get<xpu, 5, DType>(s);
      Tensor<xpu, 5, DType> out = outputs[0].get<xpu, 5, DType>(s);
      out = slice(in, param.begin.get<5>(), param.end.get<5>());
      break;
     }
     default:
      LOG(FATAL) << "crop supports at most 5 dimensions";
      break;
    }
  });
}

inline bool CropShape(const nnvm::NodeAttrs& attrs,
                      std::vector<TShape> *in_attrs,
                      std::vector<TShape> *out_attrs) {
  const SimpleCropParam& param = nnvm::get<SimpleCropParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 1);
  TShape& shp = (*in_attrs)[0];
  CHECK_EQ(shp.ndim(), param.begin.ndim());
  CHECK_EQ(shp.ndim(), param.end.ndim());
  TShape ret(shp.ndim());
  for (index_t i = 0; i < shp.ndim(); ++i) {
    CHECK(param.begin[i] < shp[i]
          && param.end[i] <= shp[i]
          && param.begin[i] < param.end[i]);
    ret[i] = param.end[i] - param.begin[i];
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, ret);
  return true;
}


template<typename xpu>
void CropAssign(const nnvm::NodeAttrs& attrs,
                const OpContext& ctx,
                const std::vector<TBlob>& inputs,
                const std::vector<OpReqType>& req,
                const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;

  const SimpleCropParam& param = nnvm::get<SimpleCropParam>(attrs.parsed);
  Stream<xpu> *s = ctx.get_stream<xpu>();

  if (req[0] == kWriteTo) {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      Tensor<xpu, 1, DType> in = inputs[0].FlatTo1D<xpu, DType>(s);
      Tensor<xpu, 1, DType> out = outputs[0].FlatTo1D<xpu, DType>(s);
      Copy(out, in, s);
    });
  } else if (req[0] != kWriteInplace) {
    LOG(FATAL) << "CropAssignScalar only supports kWriteTo and kWriteInplace";
  }

  MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    switch (outputs[0].shape_.ndim()) {
      case 0:
        break;
      case 1: {
        Tensor<xpu, 1, DType> out = outputs[0].get<xpu, 1, DType>(s);
        Tensor<xpu, 1, DType> in = inputs[1].get<xpu, 1, DType>(s);
        slice(out, param.begin.get<1>(), param.end.get<1>()) = in;
        break;
      }
      case 2: {
        Tensor<xpu, 2, DType> out = outputs[0].get<xpu, 2, DType>(s);
        Tensor<xpu, 2, DType> in = inputs[1].get<xpu, 2, DType>(s);
        slice(out, param.begin.get<2>(), param.end.get<2>()) = in;
        break;
      }
      case 3: {
        Tensor<xpu, 3, DType> out = outputs[0].get<xpu, 3, DType>(s);
        Tensor<xpu, 3, DType> in = inputs[1].get<xpu, 3, DType>(s);
        slice(out, param.begin.get<3>(), param.end.get<3>()) = in;
        break;
      }
      case 4: {
        Tensor<xpu, 4, DType> out = outputs[0].get<xpu, 4, DType>(s);
        Tensor<xpu, 4, DType> in = inputs[1].get<xpu, 4, DType>(s);
        slice(out, param.begin.get<4>(), param.end.get<4>()) = in;
        break;
      }
      case 5: {
        Tensor<xpu, 5, DType> out = outputs[0].get<xpu, 5, DType>(s);
        Tensor<xpu, 5, DType> in = inputs[1].get<xpu, 5, DType>(s);
        slice(out, param.begin.get<5>(), param.end.get<5>()) = in;
        break;
      }
      default:
        LOG(FATAL) << "CropAssign supports at most 5 dimensions";
        break;
    }
  });
}

inline bool CropAssignShape(const nnvm::NodeAttrs& attrs,
                            std::vector<TShape> *in_attrs,
                            std::vector<TShape> *out_attrs) {
  const SimpleCropParam& param = nnvm::get<SimpleCropParam>(attrs.parsed);
  TShape& lshape = (*in_attrs)[0];
  TShape& rshape = (*in_attrs)[1];
  CHECK_EQ(lshape.ndim(), rshape.ndim());
  CHECK_EQ(lshape.ndim(), param.begin.ndim());
  CHECK_EQ(lshape.ndim(), param.end.ndim());
  for (index_t i = 0; i < rshape.ndim(); ++i) {
    CHECK_LT(param.begin[i], param.end[i]);
    CHECK_LE(param.end[i], lshape[i]);
    CHECK_EQ(param.end[i] - param.begin[i], rshape[i]);
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, lshape);
  return true;
}


struct SimpleCropAssignScalarParam : public dmlc::Parameter<SimpleCropAssignScalarParam> {
  real_t scalar;
  TShape begin, end;
  DMLC_DECLARE_PARAMETER(SimpleCropAssignScalarParam) {
    DMLC_DECLARE_FIELD(scalar)
    .set_default(0)
    .describe("The scalar value for assignment.");
    DMLC_DECLARE_FIELD(begin)
    .describe("starting coordinates");
    DMLC_DECLARE_FIELD(end)
    .describe("ending coordinates");
  }
};

template<typename xpu>
void CropAssignScalar(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  const SimpleCropAssignScalarParam& param = nnvm::get<SimpleCropAssignScalarParam>(attrs.parsed);
  Stream<xpu> *s = ctx.get_stream<xpu>();

  if (req[0] == kWriteTo) {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      Tensor<xpu, 1, DType> in = inputs[0].FlatTo1D<xpu, DType>(s);
      Tensor<xpu, 1, DType> out = outputs[0].FlatTo1D<xpu, DType>(s);
      Copy(out, in, s);
    });
  } else if (req[0] != kWriteInplace) {
    LOG(FATAL) << "CropAssignScalar only supports kWriteTo and kWriteInplace";
  }

  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    switch (outputs[0].shape_.ndim()) {
      case 0:
        break;
      case 1: {
        Tensor<xpu, 1, DType> out = outputs[0].get<xpu, 1, DType>(s);
        slice(out, param.begin.get<1>(), param.end.get<1>()) = \
            static_cast<DType>(param.scalar);
        break;
      }
      case 2: {
        Tensor<xpu, 2, DType> out = outputs[0].get<xpu, 2, DType>(s);
        slice(out, param.begin.get<2>(), param.end.get<2>()) = \
            static_cast<DType>(param.scalar);
        break;
      }
      case 3: {
        Tensor<xpu, 3, DType> out = outputs[0].get<xpu, 3, DType>(s);
        slice(out, param.begin.get<3>(), param.end.get<3>()) = \
            static_cast<DType>(param.scalar);
        break;
      }
      case 4: {
        Tensor<xpu, 4, DType> out = outputs[0].get<xpu, 4, DType>(s);
        slice(out, param.begin.get<4>(), param.end.get<4>()) = \
            static_cast<DType>(param.scalar);
        break;
      }
      case 5: {
        Tensor<xpu, 5, DType> out = outputs[0].get<xpu, 5, DType>(s);
        slice(out, param.begin.get<5>(), param.end.get<5>()) = \
            static_cast<DType>(param.scalar);
        break;
      }
      default:
        LOG(FATAL) << "CropAssign supports at most 5 dimensions";
        break;
    }
  });
}

inline bool CropAssignScalarShape(const nnvm::NodeAttrs& attrs,
                                  std::vector<TShape> *in_attrs,
                                  std::vector<TShape> *out_attrs) {
  const SimpleCropAssignScalarParam& param = nnvm::get<SimpleCropAssignScalarParam>(attrs.parsed);
  TShape& lshape = (*in_attrs)[0];
  CHECK_EQ(lshape.ndim(), param.begin.ndim());
  CHECK_EQ(lshape.ndim(), param.end.ndim());
  for (index_t i = 0; i < lshape.ndim(); ++i) {
    CHECK_LT(param.begin[i], param.end[i]);
    CHECK_LE(param.end[i], lshape[i]);
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, lshape);
  return true;
}

struct SliceParam : public dmlc::Parameter<SliceParam> {
  int axis;
  int begin;
  int end;
  DMLC_DECLARE_PARAMETER(SliceParam) {
    DMLC_DECLARE_FIELD(axis).set_lower_bound(0)
      .describe("The axis to be sliced");
    DMLC_DECLARE_FIELD(begin).set_lower_bound(0)
      .describe("The beginning index to be sliced");
    DMLC_DECLARE_FIELD(end).set_lower_bound(0)
      .describe("The end index to be sliced");
  }
};

inline bool SliceShape(const nnvm::NodeAttrs& attrs,
                       std::vector<TShape> *in_attrs,
                       std::vector<TShape> *out_attrs) {
  const SliceParam& param = nnvm::get<SliceParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 1);
  TShape& ishape = (*in_attrs)[0];
  CHECK(param.axis < static_cast<int>(ishape.ndim())) <<
    "axis must be smaller than the source ndim! Recieved axis=" <<
      param.axis << ", src_ndim=" << ishape.ndim();
  int axis_size = static_cast<int>(ishape[param.axis]);
  CHECK_LE(param.end, axis_size);
  CHECK_LT(param.begin, param.end);

  TShape shape(ishape.ndim());
  for (index_t i = 0; i < ishape.ndim(); ++i) {
    if (static_cast<int>(i) == param.axis) {
      shape[i] = static_cast<index_t>(param.end - param.begin);
    } else {
      shape[i] = ishape[i];
    }
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, shape);
  return true;
}


template<typename xpu>
void Slice(const nnvm::NodeAttrs& attrs,
           const OpContext& ctx,
           const std::vector<TBlob>& inputs,
           const std::vector<OpReqType>& req,
           const std::vector<TBlob>& outputs) {
  using namespace mshadow::expr;
  const SliceParam& param = nnvm::get<SliceParam>(attrs.parsed);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  int ndim = static_cast<int>(outputs[0].ndim());

  if (param.axis + 1 == ndim) {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
        mshadow::Tensor<xpu, 2, DType> in =
            inputs[0].FlatTo2D<xpu, DType>(s);
        mshadow::Tensor<xpu, 2, DType> out =
            outputs[0].FlatTo2D<xpu, DType>(s);
        ASSIGN_DISPATCH(out, req[0], slice<1>(in, param.begin, param.end));
      });
  } else {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
        mshadow::Tensor<xpu, 3, DType> in =
            inputs[0].FlatTo3D<xpu, DType>(param.axis, s);
        mshadow::Tensor<xpu, 3, DType> out =
            outputs[0].FlatTo3D<xpu, DType>(param.axis, s);
        ASSIGN_DISPATCH(out, req[0], slice<1>(in, param.begin, param.end));
      });
  }
}

// Backward pass of broadcast over the given axis
template<typename xpu>
void SliceGrad_(const nnvm::NodeAttrs& attrs,
                const OpContext& ctx,
                const std::vector<TBlob>& inputs,
                const std::vector<OpReqType>& req,
                const std::vector<TBlob>& outputs) {
  const SliceParam& param = nnvm::get<SliceParam>(attrs.parsed);
  using namespace mshadow::op;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  int ndim = static_cast<int>(outputs[0].shape_.ndim());

  if (param.axis + 1 == ndim) {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
        mshadow::Tensor<xpu, 2, DType> ograd =
            inputs[0].FlatTo2D<xpu, DType>(s);
        mshadow::Tensor<xpu, 2, DType> igrad =
            outputs[0].FlatTo2D<xpu, DType>(s);
        if (req[0] == kAddTo) {
          slice<1>(igrad, param.begin, param.end) += F<identity>(ograd);
        } else if (req[0] == kWriteTo) {
          igrad = 0.0f;
          slice<1>(igrad, param.begin, param.end) = F<identity>(ograd);
        } else {
          CHECK_EQ(req[0], kNullOp);
        }
      });
  } else {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
        mshadow::Tensor<xpu, 3, DType> ograd =
            inputs[0].FlatTo3D<xpu, DType>(param.axis, s);
        mshadow::Tensor<xpu, 3, DType> igrad =
            outputs[0].FlatTo3D<xpu, DType>(param.axis, s);
        if (req[0] == kAddTo) {
          slice<1>(igrad, param.begin, param.end) += F<identity>(ograd);
        } else if (req[0] == kWriteTo) {
          igrad = 0.0f;
          slice<1>(igrad, param.begin, param.end) = F<identity>(ograd);
        } else {
          CHECK_EQ(req[0], kNullOp);
        }
      });
  }
}

struct FlipParam : public dmlc::Parameter<FlipParam> {
  int axis;
  DMLC_DECLARE_PARAMETER(FlipParam) {
    DMLC_DECLARE_FIELD(axis)
    .describe("The dimension to flip");
  }
};

// matrix crop
template<typename xpu>
void Flip(const nnvm::NodeAttrs& attrs,
          const OpContext& ctx,
          const std::vector<TBlob>& inputs,
          const std::vector<OpReqType>& req,
          const std::vector<TBlob>& outputs) {
  const FlipParam& param = nnvm::get<FlipParam>(attrs.parsed);
  using namespace mshadow;
  using namespace mshadow::expr;
  CHECK_EQ(inputs[0].type_flag_, outputs[0].type_flag_);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    switch (inputs[0].shape_.ndim()) {
     case 0:
      break;
     case 1: {
      Tensor<xpu, 1, DType> in = inputs[0].get<xpu, 1, DType>(s);
      Tensor<xpu, 1, DType> out = outputs[0].get<xpu, 1, DType>(s);
      out = flip(in, param.axis);
      break;
     }
     case 2: {
      Tensor<xpu, 2, DType> in = inputs[0].get<xpu, 2, DType>(s);
      Tensor<xpu, 2, DType> out = outputs[0].get<xpu, 2, DType>(s);
      out = flip(in, param.axis);
      break;
     }
     case 3: {
      Tensor<xpu, 3, DType> in = inputs[0].get<xpu, 3, DType>(s);
      Tensor<xpu, 3, DType> out = outputs[0].get<xpu, 3, DType>(s);
      out = flip(in, param.axis);
      break;
     }
     case 4: {
      Tensor<xpu, 4, DType> in = inputs[0].get<xpu, 4, DType>(s);
      Tensor<xpu, 4, DType> out = outputs[0].get<xpu, 4, DType>(s);
      out = flip(in, param.axis);
      break;
     }
     case 5: {
      Tensor<xpu, 5, DType> in = inputs[0].get<xpu, 5, DType>(s);
      Tensor<xpu, 5, DType> out = outputs[0].get<xpu, 5, DType>(s);
      out = flip(in, param.axis);
      break;
     }
     default:
      LOG(FATAL) << "flip supports at most 5 dimensions";
      break;
    }
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_MATRIX_OP_INL_H_
