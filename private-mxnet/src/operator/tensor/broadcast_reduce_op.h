/*!
 *  Copyright (c) 2015 by Contributors
 * \file elementwise_unary_op-inl.h
 * \brief Function definition of elementwise unary operators
 */
#ifndef MXNET_OPERATOR_TENSOR_BROADCAST_REDUCE_OP_H_
#define MXNET_OPERATOR_TENSOR_BROADCAST_REDUCE_OP_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <utility>
#include <algorithm>
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "./elemwise_binary_broadcast_op.h"
#include "../mxnet_op.h"

namespace mxnet {
namespace op {
struct ReduceAxesParam : public dmlc::Parameter<ReduceAxesParam> {
  TShape axis;
  bool keepdims;
  bool exclude;
  DMLC_DECLARE_PARAMETER(ReduceAxesParam) {
    DMLC_DECLARE_FIELD(axis).set_default(TShape())
      .describe(R"code(The axis or axes along which to perform the reduction.

      The default, `axis=()`, will compute over all elements into a
      scalar array with shape `(1,)`.

      If `axis` is int, a reduction is performed on a particular axis.

      If `axis` is a tuple of ints, a reduction is performed on all the axes
      specified in the tuple.

      If `exclude` is true, reduction will be performed on the axes that are
      NOT in axis instead.)code");
    DMLC_DECLARE_FIELD(keepdims).set_default(false)
      .describe("If this is set to `True`, the reduced axes are left "
                "in the result as dimension with size one.");
    DMLC_DECLARE_FIELD(exclude).set_default(false)
      .describe("Whether to perform reduction on axis that are NOT in axis instead.");
  }
};

struct ReduceAxisParam : public dmlc::Parameter<ReduceAxisParam> {
  dmlc::optional<int> axis;
  bool keepdims;
  DMLC_DECLARE_PARAMETER(ReduceAxisParam) {
    DMLC_DECLARE_FIELD(axis).set_default(dmlc::optional<int>())
      .describe("The axis along which to perform the reduction. "
                "Negative values means indexing from right to left. "
                "``Requires axis to be set as int, because global reduction "
                "is not supported yet.``");
    DMLC_DECLARE_FIELD(keepdims).set_default(false)
      .describe("If this is set to `True`, the reduced axis is left "
                "in the result as dimension with size one.");
  }
};

struct PickParam : public dmlc::Parameter<PickParam> {
  dmlc::optional<int> axis;
  int mode;
  bool keepdims;
  DMLC_DECLARE_PARAMETER(PickParam) {
    DMLC_DECLARE_FIELD(axis).set_default(dmlc::optional<int>(-1))
      .describe("int or None. The axis to picking the elements. "
                "Negative values means indexing from right to left. "
                "If is `None`, the elements in the index w.r.t the "
                "flattened input will be picked.");
    DMLC_DECLARE_FIELD(keepdims).set_default(false)
      .describe("If true, the axis where we pick the elements is left "
                "in the result as dimension with size one.");
  }
};

struct BroadcastAxesParam : public dmlc::Parameter<BroadcastAxesParam> {
  TShape axis;
  TShape size;
  DMLC_DECLARE_PARAMETER(BroadcastAxesParam) {
    DMLC_DECLARE_FIELD(axis).set_default(TShape())
      .describe("The axes to perform the broadcasting.");
    DMLC_DECLARE_FIELD(size).set_default(TShape())
      .describe("Target sizes of the broadcasting axes.");
  }
};

struct BroadcastToParam : public dmlc::Parameter<BroadcastToParam> {
  TShape shape;
  DMLC_DECLARE_PARAMETER(BroadcastToParam) {
    DMLC_DECLARE_FIELD(shape).set_default(TShape())
      .describe("The shape of the desired array."
                " We can set the dim to zero if it's same as the original."
                " E.g `A = broadcast_to(B, shape=(10, 0, 0))` "
                "has the same meaning as `A = broadcast_axis(B, axis=0, size=10)`.");
  }
};

inline int CheckAxis(int axis, int ndim) {
  CHECK(axis < ndim && axis >= -ndim)
    << "axis " << axis << " exceeds the input dimension of " << ndim;
  return (axis + ndim)%ndim;
}

inline TShape AxisShapeCompact(TShape shape, int *axis, bool allow_2d) {
  int ndim = static_cast<int>(shape.ndim());
  index_t leading = 1, trailing = 1, M = shape[*axis];
  for (int i = 0; i < *axis; ++i) leading *= shape[i];
  for (int i = *axis + 1; i < ndim; ++i) trailing *= shape[i];
  if (allow_2d && trailing == 1) {
    *axis = 1;
    return mshadow::Shape2(leading, M);
  }
  if (allow_2d && leading == 1) {
    *axis = 0;
    return mshadow::Shape2(M, trailing);
  }
  *axis = 1;
  return mshadow::Shape3(leading, M, trailing);
}

inline TShape ReduceAxisShapeImpl(const ReduceAxisParam& param, const TShape& ishape) {
  if (!param.axis || ishape.ndim() == 1) {
    if (param.keepdims) {
      return TShape(ishape.ndim());
    } else {
      return mshadow::Shape1(1);
    }
  } else {
    int axis = CheckAxis(param.axis.value(), ishape.ndim());
    if (param.keepdims) {
      TShape oshape = ishape;
      oshape[axis] = 1;
      return oshape;
    } else {
      TShape oshape(ishape.ndim() - 1);
      for (int i = 0; i < axis; ++i) oshape[i] = ishape[i];
      for (int i = axis+1; i < static_cast<int>(ishape.ndim()); ++i) {
        oshape[i-1] = ishape[i];
      }
      return oshape;
    }
  }
}

inline bool ReduceAxisShape(const nnvm::NodeAttrs& attrs,
                            std::vector<TShape> *in_attrs,
                            std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  TShape& ishape = (*in_attrs)[0];
  if (ishape.ndim() == 0) return false;

  const ReduceAxisParam& param = nnvm::get<ReduceAxisParam>(attrs.parsed);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, ReduceAxisShapeImpl(param, ishape));
  return true;
}

inline TShape ReduceAxesShapeImpl(const TShape& ishape, const TShape& axis,
                                  bool keepdims, bool exclude) {
  if (axis.ndim() == 0) {
    if (keepdims) {
      return TShape(ishape.ndim());
    } else {
      return TShape(1);
    }
  }

  CHECK_LT(axis[axis.ndim()-1], ishape.ndim())
    << "Reduction axis " << axis[axis.ndim()-1]
    << " Exceeds input dimensions " << ishape;

  if (keepdims) {
    TShape oshape(ishape);
    if (exclude) {
      for (index_t i = 0, j = 0; i < ishape.ndim(); ++i) {
        if (j < axis.ndim() && i == axis[j]) {
          ++j;
          continue;
        }
        oshape[i] = 1;
      }
      return oshape;
    }

    for (index_t i = 0; i < axis.ndim(); ++i) {
      oshape[axis[i]] = 1;
    }
    return oshape;
  }

  if (exclude) {
    TShape oshape = TShape(axis.ndim());
    for (index_t i = 0; i < axis.ndim(); ++i) {
      oshape[i] = ishape[axis[i]];
    }
    return oshape;
  }

  TShape oshape = TShape(std::max<index_t>(1, ishape.ndim() - axis.ndim()));
  for (index_t i = 0, j = 0, k = 0; i < ishape.ndim(); ++i) {
    if (j < axis.ndim() && i == axis[j]) {
      ++j;
      continue;
    }
    oshape[k++] = ishape[i];
  }
  return oshape;
}

inline bool ReduceAxesShape(const nnvm::NodeAttrs& attrs,
                            std::vector<TShape> *in_attrs,
                            std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  if ((*in_attrs)[0].ndim() == 0) return false;
  const ReduceAxesParam& param = nnvm::get<ReduceAxesParam>(attrs.parsed);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0,
                     ReduceAxesShapeImpl((*in_attrs)[0], param.axis,
                                         param.keepdims, param.exclude));
  return true;
}

inline bool BroadcastAxesShape(const nnvm::NodeAttrs& attrs,
                               std::vector<TShape> *in_attrs,
                               std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  if ((*in_attrs)[0].ndim() == 0) return false;
  const BroadcastAxesParam& param = nnvm::get<BroadcastAxesParam>(attrs.parsed);
  CHECK_EQ(param.axis.ndim() , param.size.ndim());
  TShape &ishape = (*in_attrs)[0];
  TShape oshape = ishape;
  for (index_t i = 0; i < param.axis.ndim(); ++i) {
    CHECK_EQ(oshape[param.axis[i]], 1U) << "Broadcasting axis must have size 1";
    oshape[param.axis[i]] = param.size[i];
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  return true;
}

inline bool BroadcastToShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape> *in_attrs,
                            std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  TShape& ishape = (*in_attrs)[0];
  if (ishape.ndim() == 0) return false;
  const BroadcastToParam& param = nnvm::get<BroadcastToParam>(attrs.parsed);
  CHECK_EQ(ishape.ndim(), param.shape.ndim())
    << "Operand of shape " << ishape << " cannot be broadcasted to " << param.shape;
  TShape oshape = param.shape;
  for (index_t i = 0; i < ishape.ndim(); ++i) {
    if (oshape[i] != 0) {
      CHECK(ishape[i] == oshape[i] || ishape[i] == 1)
        << "Array cannot be broadcasted from " << ishape << " to " << param.shape;
    } else {
      oshape[i] = ishape[i];
    }
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  return true;
}

inline void BroadcastReduceShapeCompact(const TShape& big, const TShape& small,
                                        TShape *new_big, TShape *new_small) {
  index_t idim = std::max<index_t>(big.ndim(), MXNET_SPECIAL_MAX_NDIM);
  *new_big = TShape(idim);
  *new_small = TShape(idim);
  index_t j = 0;
  if (small.Size() == 1) {
    (*new_big)[j++] = big.Size();
  } else {
    index_t bprod = 1, sprod = 1;
    for (index_t i = 0, k = 0; i < big.ndim(); ++i) {
      bool red_axis = big[i] != small[i];
      if ((red_axis && sprod > 1) || (!red_axis && bprod != sprod)) {
        (*new_big)[j] = bprod;
        (*new_small)[j] = sprod;
        bprod = sprod = 1; ++j;
      }
      bprod *= big[i];
      if (red_axis) {
        ++k;
      } else {
        sprod *= big[i];
      }
    }
    if (bprod > 1 || sprod > 1) {
      (*new_big)[j] = bprod;
      (*new_small)[j] = sprod;
      ++j;
    }
  }
  if (j <= 2) {
    new_small->assign(&(*new_small)[0], &(*new_small)[2]);
    new_big->assign(&(*new_big)[0], &(*new_big)[2]);
  } else if (j <= MXNET_SPECIAL_MAX_NDIM) {
    new_small->assign(&(*new_small)[0], &(*new_small)[MXNET_SPECIAL_MAX_NDIM]);
    new_big->assign(&(*new_big)[0], &(*new_big)[MXNET_SPECIAL_MAX_NDIM]);
  } else {
    LOG(FATAL) << "Too many reduction axes from " << big << " to " << small;
  }
}

template<typename xpu, typename reducer>
void SearchAxisCompute(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  const ReduceAxisParam& param = nnvm::get<ReduceAxisParam>(attrs.parsed);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  if (!param.axis) LOG(FATAL) << "Global reduction not supported yet";

  int axis = CheckAxis(param.axis.value(), inputs[0].shape_.ndim());
  TShape shape = AxisShapeCompact(inputs[0].shape_, &axis, false);
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<xpu, 2, DType> out = outputs[0].get_with_shape<xpu, 2, DType>(
      Shape2(shape[0], shape[2]), s);
    Tensor<xpu, 3, DType> in = inputs[0].get_with_shape<xpu, 3, DType>(
      shape.get<3>(), s);
    CHECK(req[0] != kAddTo) << "AddTo is not supported";
    ASSIGN_DISPATCH(out, req[0], (reduce_with_axis<reducer, true>(in, 1)));
  });
}

template<typename xpu, typename reducer, bool normalize = false>
void ReduceAxesComputeImpl(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs,
                           const TShape& small) {
  using namespace mshadow;
  using namespace mshadow::expr;

  TShape src_shape, dst_shape;
  BroadcastReduceShapeCompact(inputs[0].shape_, small, &src_shape, &dst_shape);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    const TBlob in_data = inputs[0].reshape(src_shape);
    const TBlob out_data = outputs[0].reshape(dst_shape);
    BROADCAST_NDIM_SWITCH(dst_shape.ndim(), NDim, {
      size_t workspace_size = broadcast::ReduceWorkspaceSize<NDim, DType>(
          s, out_data, req[0], in_data);
      Tensor<xpu, 1, char> workspace =
          ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);
      broadcast::Reduce<reducer, NDim, DType, mshadow::op::identity>(
          s, out_data, req[0], workspace, in_data);
      if (normalize) {
        auto out = out_data.FlatTo2D<xpu, DType>(s);
        out /= scalar<DType>(src_shape.Size()/dst_shape.Size());
      }
    });
  });
}

template<typename xpu, typename reducer, bool normalize = false>
void ReduceAxesCompute(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  const ReduceAxesParam& param = nnvm::get<ReduceAxesParam>(attrs.parsed);
  TShape small;
  if (param.keepdims) {
    small = outputs[0].shape_;
  } else {
    small = ReduceAxesShapeImpl(inputs[0].shape_, param.axis, true, param.exclude);
  }

  ReduceAxesComputeImpl<xpu, reducer, normalize>(attrs, ctx, inputs, req, outputs, small);
}

// works when shape inference of output is given
template<typename xpu, typename OP, bool normalize = false>
void ReduceAxesBackwardUseInOut(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<TBlob>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  const ReduceAxesParam& param = nnvm::get<ReduceAxesParam>(attrs.parsed);
  TShape small;
  if (param.keepdims) {
    small = inputs[0].shape_;
  } else {
    small = ReduceAxesShapeImpl(outputs[0].shape_, param.axis, true, param.exclude);
  }

  TShape src_shape, dst_shape;
  BroadcastReduceShapeCompact(outputs[0].shape_, small, &src_shape, &dst_shape);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    if (dst_shape.ndim() == 2) {
      Tensor<xpu, 2, DType> igrad =
        outputs[0].get_with_shape<xpu, 2, DType>(src_shape.get<2>(), s);
      Tensor<xpu, 2, DType> ograd =
        inputs[0].get_with_shape<xpu, 2, DType>(dst_shape.get<2>(), s);
      Tensor<xpu, 2, DType> data =
        inputs[1].get_with_shape<xpu, 2, DType>(src_shape.get<2>(), s);
      Tensor<xpu, 2, DType> out =
        inputs[2].get_with_shape<xpu, 2, DType>(dst_shape.get<2>(), s);
      ASSIGN_DISPATCH(igrad, req[0],
          broadcast_to(ograd, src_shape)*F<OP>(data, broadcast_to(out, src_shape)));
      if (normalize) igrad /= scalar<DType>(src_shape.Size()/dst_shape.Size());
    } else {
      const int ndim = MXNET_SPECIAL_MAX_NDIM;
      Tensor<xpu, ndim, DType> igrad =
        outputs[0].get_with_shape<xpu, ndim, DType>(src_shape.get<ndim>(), s);
      Tensor<xpu, ndim, DType> ograd =
        inputs[0].get_with_shape<xpu, ndim, DType>(dst_shape.get<ndim>(), s);
      Tensor<xpu, ndim, DType> data =
        inputs[1].get_with_shape<xpu, ndim, DType>(src_shape.get<ndim>(), s);
      Tensor<xpu, ndim, DType> out =
        inputs[2].get_with_shape<xpu, ndim, DType>(dst_shape.get<ndim>(), s);
      ASSIGN_DISPATCH(igrad, req[0],
          broadcast_to(ograd, src_shape)*F<OP>(data, broadcast_to(out, src_shape)));
      if (normalize) igrad /= scalar<DType>(src_shape.Size()/dst_shape.Size());
    }
  });
}

template<typename xpu>
inline void BroadcastComputeImpl(const nnvm::NodeAttrs& attrs,
                                 const OpContext& ctx,
                                 const std::vector<TBlob>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<TBlob>& outputs,
                                 const TShape& small) {
  using namespace mshadow;
  using namespace mshadow::expr;
  TShape src_shape, dst_shape;
  BroadcastReduceShapeCompact(outputs[0].shape_, small, &dst_shape, &src_shape);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    if (dst_shape.ndim() == 2) {
      Tensor<xpu, 2, DType> out =
        outputs[0].get_with_shape<xpu, 2, DType>(dst_shape.get<2>(), s);
      Tensor<xpu, 2, DType> data =
        inputs[0].get_with_shape<xpu, 2, DType>(src_shape.get<2>(), s);
      ASSIGN_DISPATCH(out, req[0], broadcast_to(data, dst_shape));
    } else {
      const int ndim = MXNET_SPECIAL_MAX_NDIM;
      Tensor<xpu, ndim, DType> out =
        outputs[0].get_with_shape<xpu, ndim, DType>(dst_shape.get<ndim>(), s);
      Tensor<xpu, ndim, DType> data =
        inputs[0].get_with_shape<xpu, ndim, DType>(src_shape.get<ndim>(), s);
      ASSIGN_DISPATCH(out, req[0], broadcast_to(data, dst_shape));
    }
  });
}

template<typename xpu>
inline void BroadcastCompute(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  BroadcastComputeImpl<xpu>(attrs, ctx, inputs, req, outputs, inputs[0].shape_);
}

template<typename xpu, bool normalize = false>
inline void ReduceAxesBackwardUseNone(const nnvm::NodeAttrs& attrs,
                                      const OpContext& ctx,
                                      const std::vector<TBlob>& inputs,
                                      const std::vector<OpReqType>& req,
                                      const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  const ReduceAxesParam& param = nnvm::get<ReduceAxesParam>(attrs.parsed);
  TShape small;
  if (param.keepdims) {
    small = inputs[0].shape_;
  } else {
    small = ReduceAxesShapeImpl(outputs[0].shape_, param.axis, true, param.exclude);
  }

  BroadcastComputeImpl<xpu>(attrs, ctx, inputs, req, outputs, small);
  if (normalize)  {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      Tensor<xpu, 1, DType> igrad = outputs[0].FlatTo1D<xpu, DType>(s);
      igrad /= scalar<DType>(outputs[0].Size()/inputs[0].Size());
    });
  }
}

template<typename PType>
inline void AxesParamParser(nnvm::NodeAttrs* attrs) {
  PType param;
  param.Init(attrs->dict);
  std::sort(&param.axis[0], &param.axis[param.axis.ndim()]);
  attrs->parsed = std::move(param);
}

struct ReduceGrad {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) {
    return MakeNonlossGradNode(
        op_name, n,
        ograds, {n->inputs[0], nnvm::NodeEntry{n, 0, 0}},
        n->attrs.dict);
  }
};

template<typename xpu>
void L2NormCompute(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    mshadow::Tensor<xpu, 1, DType> out = outputs[0].get<xpu, 1, DType>(s);
    mshadow::Tensor<xpu, 1, DType> in = inputs[0].get_with_shape<xpu, 1, DType>(
      mshadow::Shape1(inputs[0].shape_.Size()), s);
    mshadow::VectorDot(out, in, in);
    ASSIGN_DISPATCH(out, req[0], mshadow::expr::F<mxnet::op::mshadow_op::square_root>(out));
  });
}

/*! \brief index element from array along axes */
template<int ndim>
struct pick {
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i, DType* out, const DType* a,
                                  const IType *idx, int M, int stride,
                                  mshadow::Shape<ndim> bshape,
                                  mshadow::Shape<ndim> sshape) {
    using namespace broadcast;
    int j = static_cast<int>(idx[i]);
    if (j < 0) j = 0;
    else if (j >= M) j = M-1;
    j = ravel(unravel(i, sshape), bshape) + j*stride;
    out[i] = a[j];
  }
};

/*! \brief index element from array along axes */
template<int ndim>
struct pick_grad {
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i, DType* igrad, const DType* ograd,
                                  const IType *idx, int M, int stride,
                                  mshadow::Shape<ndim> bshape,
                                  mshadow::Shape<ndim> sshape) {
    using namespace broadcast;
    int j = static_cast<int>(idx[i]);
    if (j < 0) j = 0;
    else if (j >= M) j = M-1;
    j = ravel(unravel(i, sshape), bshape) + j*stride;
    igrad[j] += ograd[i];
  }
};

inline bool PickOpShape(const nnvm::NodeAttrs& attrs,
                        std::vector<TShape> *in_attrs,
                        std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2);
  CHECK_EQ(out_attrs->size(), 1);
  const TShape& ishape = (*in_attrs)[0];
  if (ishape.ndim() == 0) return false;
  const PickParam& param = nnvm::get<PickParam>(attrs.parsed);
  if (!param.axis) LOG(FATAL)
    << "axis=None is not supported by pick yet. Must specify an axis.";
  ReduceAxisParam tmp_param;
  tmp_param.axis = param.axis;
  tmp_param.keepdims = param.keepdims;
  TShape oshape = ReduceAxisShapeImpl(tmp_param, ishape);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  SHAPE_ASSIGN_CHECK(*in_attrs, 1, oshape);
  return true;
}

inline bool PickOpType(const nnvm::NodeAttrs& attrs,
                       std::vector<int> *in_attrs,
                       std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  CHECK_NE((*in_attrs)[1], -1) << "Index type must be set for pick operator";
  TYPE_ASSIGN_CHECK(*out_attrs, 0, (*in_attrs)[0]);
  TYPE_ASSIGN_CHECK(*in_attrs, 0, (*out_attrs)[0]);
  return (*out_attrs)[0] != -1;
}

template<typename xpu>
void PickOpForward(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  CHECK_EQ(req[0], kWriteTo);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const PickParam& param = nnvm::get<PickParam>(attrs.parsed);

  const TShape& ishape = inputs[0].shape_;
  int axis = CheckAxis(param.axis.value(), ishape.ndim());
  int leading = 1, trailing = 1, M = ishape[axis];
  for (index_t i = 0; i < axis; ++i) leading *= ishape[i];
  for (index_t i = axis+1; i < ishape.ndim(); ++i) trailing *= ishape[i];

  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {  // output type
    MSHADOW_TYPE_SWITCH(inputs[1].type_flag_, IType, {  // index type
      if (trailing == 1) {
        Kernel<pick<2>, xpu>::Launch(s, outputs[0].Size(), outputs[0].dptr<DType>(),
                                     inputs[0].dptr<DType>(), inputs[1].dptr<IType>(),
                                     M, 1, Shape2(leading, M), Shape2(leading, 1));
      } else {
        Kernel<pick<3>, xpu>::Launch(s, outputs[0].Size(), outputs[0].dptr<DType>(),
                                     inputs[0].dptr<DType>(), inputs[1].dptr<IType>(),
                                     M, trailing, Shape3(leading, M, trailing),
                                     Shape3(leading, 1, trailing));
      }
    });
  });
}

template<typename xpu>
void PickOpBackward(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  if (req[0] == kNullOp) return;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const PickParam& param = nnvm::get<PickParam>(attrs.parsed);

  const TShape& ishape = outputs[0].shape_;
  const index_t axis = CheckAxis(param.axis.value(), ishape.ndim());
  int leading = 1, trailing = 1, M = ishape[axis];
  for (index_t i = 0; i < axis; ++i) leading *= ishape[i];
  for (index_t i = axis+1; i < ishape.ndim(); ++i) trailing *= ishape[i];

  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {  // output type
    MSHADOW_TYPE_SWITCH(inputs[1].type_flag_, IType, {  // index type
      if (req[0] != kAddTo) outputs[0].FlatTo1D<xpu, DType>(s) = 0;
      if (trailing == 1) {
        Kernel<pick_grad<2>, xpu>::Launch(s, inputs[0].Size(), outputs[0].dptr<DType>(),
                                     inputs[0].dptr<DType>(), inputs[1].dptr<IType>(),
                                     M, 1, Shape2(leading, M), Shape2(leading, 1));
      } else {
        Kernel<pick_grad<3>, xpu>::Launch(s, inputs[0].Size(), outputs[0].dptr<DType>(),
                                     inputs[0].dptr<DType>(), inputs[1].dptr<IType>(),
                                     M, trailing, Shape3(leading, M, trailing),
                                     Shape3(leading, 1, trailing));
      }
    });
  });
}

#define MXNET_OPERATOR_REGISTER_REDUCE_AXIS(name)               \
  NNVM_REGISTER_OP(name)                                        \
  .set_num_inputs(1)                                            \
  .set_num_outputs(1)                                           \
  .set_attr_parser(ParamParser<ReduceAxisParam>)                \
  .set_attr<nnvm::FInferShape>("FInferShape", ReduceAxisShape)  \
  .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>) \
  .add_argument("data", "NDArray-or-Symbol", "The input")       \
  .add_arguments(ReduceAxisParam::__FIELDS__())

#define MXNET_OPERATOR_REGISTER_REDUCE(name)                    \
  NNVM_REGISTER_OP(name)                                        \
  .set_num_inputs(1)                                            \
  .set_num_outputs(1)                                           \
  .set_attr_parser(AxesParamParser<ReduceAxesParam>)            \
  .set_attr<nnvm::FInferShape>("FInferShape", ReduceAxesShape)  \
  .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>) \
  .add_argument("data", "NDArray-or-Symbol", "The input")       \
  .add_arguments(ReduceAxesParam::__FIELDS__())

#define MXNET_OPERATOR_REGISTER_REDUCE_BACKWARD(name)               \
  NNVM_REGISTER_OP(name)                                            \
  .set_num_outputs(1)                                               \
  .set_attr_parser(AxesParamParser<ReduceAxesParam>)                \
  .set_attr<nnvm::TIsBackward>("TIsBackward", true)

#define MXNET_OPERATOR_REGISTER_BROADCAST(name)                 \
  NNVM_REGISTER_OP(name)                                        \
  .set_num_inputs(1)                                            \
  .set_num_outputs(1)                                           \
  .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>) \
  .set_attr<nnvm::FGradient>("FGradient",                       \
    [](const nnvm::NodePtr& n,                                  \
       const std::vector<nnvm::NodeEntry>& ograds) {            \
      return MakeNonlossGradNode("_broadcast_backward", n, ograds, {},    \
                                 {{"keepdims", "true"}});              \
    })                                                          \
  .add_argument("data", "NDArray-or-Symbol", "The input")

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_TENSOR_BROADCAST_REDUCE_OP_H_
