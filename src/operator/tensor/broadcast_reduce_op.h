/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2015 by Contributors
 * \file broadcast_reduce_op.h
 * \brief Function definition of broadcast and reduce operators
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
  dmlc::optional<TShape> axis;
  bool keepdims;
  bool exclude;
  DMLC_DECLARE_PARAMETER(ReduceAxesParam) {
    DMLC_DECLARE_FIELD(axis).set_default(dmlc::optional<TShape>())
      .describe(R"code(The axis or axes along which to perform the reduction.

      The default, `axis=()`, will compute over all elements into a
      scalar array with shape `(1,)`.

      If `axis` is int, a reduction is performed on a particular axis.

      If `axis` is a tuple of ints, a reduction is performed on all the axes
      specified in the tuple.

      If `exclude` is true, reduction will be performed on the axes that are
      NOT in axis instead.

      Negative values means indexing from right to left.)code");
    DMLC_DECLARE_FIELD(keepdims).set_default(false)
      .describe("If this is set to `True`, the reduced axes are left "
                "in the result as dimension with size one.");
    DMLC_DECLARE_FIELD(exclude).set_default(false)
      .describe("Whether to perform reduction on axis that are NOT in axis instead.");
  }
};

struct NormParam : public dmlc::Parameter<NormParam> {
  int ord;
  dmlc::optional<TShape> axis;
  bool keepdims;
  DMLC_DECLARE_PARAMETER(NormParam) {
    DMLC_DECLARE_FIELD(ord).set_default(2)
      .describe("Order of the norm. Currently ord=1 and ord=2 is supported.");
    DMLC_DECLARE_FIELD(axis).set_default(dmlc::optional<TShape>())
      .describe(R"code(The axis or axes along which to perform the reduction.
      The default, `axis=()`, will compute over all elements into a
      scalar array with shape `(1,)`.
      If `axis` is int, a reduction is performed on a particular axis.
      If `axis` is a 2-tuple, it specifies the axes that hold 2-D matrices,
      and the matrix norms of these matrices are computed.)code");
    DMLC_DECLARE_FIELD(keepdims).set_default(false)
      .describe("If this is set to `True`, the reduced axis is left "
                "in the result as dimension with size one.");
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

enum PickOpMode {kWrap, kClip};

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
    DMLC_DECLARE_FIELD(mode)
    .add_enum("wrap", kWrap)
    .add_enum("clip", kClip)
    .set_default(kClip)
    .describe("Specify how out-of-bound indices behave. Default is \"clip\"."
              " \"clip\" means clip to the range. So, if all indices mentioned are too large,"
              " they are replaced by the index that addresses the last element along an axis. "
              " \"wrap\" means to wrap around.");
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

struct BroadcastLikeParam : public dmlc::Parameter<BroadcastLikeParam> {
  dmlc::optional<TShape> lhs_axes;
  dmlc::optional<TShape> rhs_axes;
  DMLC_DECLARE_PARAMETER(BroadcastLikeParam) {
    DMLC_DECLARE_FIELD(lhs_axes).set_default(dmlc::optional<TShape>())
      .describe("Axes to perform broadcast on in the first input array");
    DMLC_DECLARE_FIELD(rhs_axes).set_default(dmlc::optional<TShape>())
      .describe("Axes to copy from the second input array");
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

inline TShape ReduceAxisShapeImpl(const TShape& ishape, const dmlc::optional<int>& axis,
                                  bool keepdims) {
  if (!axis || ishape.ndim() == 1) {
    if (keepdims) {
      return TShape(ishape.ndim());
    }
    return mshadow::Shape1(1);
  }

  int new_axis = CheckAxis(axis.value(), ishape.ndim());
  if (keepdims) {
    TShape oshape = ishape;
    oshape[new_axis] = 1;
    return oshape;
  }

  TShape oshape(ishape.ndim() - 1);
  for (int i = 0; i < new_axis; ++i) oshape[i] = ishape[i];
  for (int i = new_axis+1; i < static_cast<int>(ishape.ndim()); ++i) {
    oshape[i-1] = ishape[i];
  }
  return oshape;
}

inline bool ReduceAxisShape(const nnvm::NodeAttrs& attrs,
                            std::vector<TShape> *in_attrs,
                            std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  TShape& ishape = (*in_attrs)[0];
  if (ishape.ndim() == 0) return false;

  const ReduceAxisParam& param = nnvm::get<ReduceAxisParam>(attrs.parsed);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0,
                     ReduceAxisShapeImpl(ishape, param.axis, param.keepdims));
  return true;
}

inline TShape ReduceAxesShapeImpl(const TShape& ishape, const dmlc::optional<TShape>& axis,
                                  bool keepdims, bool exclude) {
  // if axis doesn't have value, treat it same TShape().
  if (!axis.has_value() || axis.value().ndim() == 0) {
    if (keepdims) {
      return TShape(ishape.ndim());
    } else {
      return TShape(1);
    }
  }
  // axis has value
  TShape axes(axis.value());
  for (index_t i = 0; i < axes.ndim(); i++) {
    if (axes[i] < 0) {
      axes[i] += ishape.ndim();
    }
  }
  std::sort(axes.begin(), axes.end());

  for (index_t i = 1; i < axes.ndim(); i++) {
    CHECK_LT(axes[i-1], axes[i])
      << "Reduction axes have duplicates "
      << axes;
  }
  CHECK_LT(axes[axes.ndim()-1], ishape.ndim())
    << "Reduction axis " << axes[axes.ndim()-1]
    << " Exceeds input dimensions " << ishape;
  CHECK_GE(axes[0], 0)
    << "Reduction axis " << axis.value()
    << " Exceeds input dimensions " << ishape;

  TShape oshape;
  if (keepdims) {
    oshape = TShape(ishape);
  } else if (exclude) {
    oshape = TShape(axes.ndim());
  } else {
    oshape = TShape(std::max<index_t>(1, ishape.ndim() - axes.ndim()));
  }

  if (keepdims && exclude) {
    for (index_t i = 0, j = 0; i < ishape.ndim(); ++i) {
      if (j < axes.ndim() && i == axes[j]) {
        ++j;
        continue;
      }
      oshape[i] = 1;
    }
  } else if (keepdims) {
    for (index_t i = 0; i < axes.ndim(); ++i) {
      oshape[axes[i]] = 1;
    }
  } else if (exclude) {
    for (index_t i = 0; i < axes.ndim(); ++i) {
      oshape[i] = ishape[axes[i]];
    }
  } else {
    for (index_t i = 0, j = 0, k = 0; i < ishape.ndim(); ++i) {
      if (j < axes.ndim() && i == axes[j]) {
        ++j;
        continue;
      }
      oshape[k++] = ishape[i];
    }
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

inline bool NormShape(const nnvm::NodeAttrs& attrs,
                      std::vector<TShape> *in_attrs,
                      std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  if ((*in_attrs)[0].ndim() == 0) return false;
  const NormParam& param = nnvm::get<NormParam>(attrs.parsed);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0,
                     ReduceAxesShapeImpl((*in_attrs)[0], param.axis,
                                         param.keepdims, false));
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

inline bool BroadcastLikeShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape> *in_attrs,
                            std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  TShape& lhs_shape = (*in_attrs)[0];
  TShape& rhs_shape = (*in_attrs)[1];

  if ((lhs_shape.ndim() == 0) || (lhs_shape.ndim() == 0)) {
    return false;
  }

  const BroadcastLikeParam& param = nnvm::get<BroadcastLikeParam>(attrs.parsed);
  TShape oshape;

  // lhs or rhs or both params were not specified
  if (!param.lhs_axes.has_value() || !param.rhs_axes.has_value()) {
    CHECK_EQ(lhs_shape.ndim(), rhs_shape.ndim())
      << "Operand of shape " << lhs_shape << " cannot be broadcasted to " << rhs_shape;

    oshape = TShape(rhs_shape);
    for (index_t i = 0; i < lhs_shape.ndim(); ++i) {
      if (rhs_shape[i] != 0) {
        CHECK(lhs_shape[i] == rhs_shape[i] || lhs_shape[i] == 1)
          << "Array cannot be broadcasted from " << lhs_shape << " to " << rhs_shape;
      } else {
        oshape[i] = lhs_shape[i];
      }
    }
  } else {
    auto lhs_axes = param.lhs_axes.value();
    auto rhs_axes = param.rhs_axes.value();

    CHECK(rhs_axes.ndim() == lhs_axes.ndim())
      << "Input_axis and other_axis size does not match";

    CHECK(lhs_axes.ndim() > 0)
      << "Empty axes tuple is not allowed";

    oshape = TShape(lhs_shape);
    for (index_t i = 0; i < lhs_axes.ndim(); ++i) {
      auto copyfrom = lhs_axes[i];
      if (copyfrom < 0) {
        copyfrom =  lhs_shape.ndim() + copyfrom;
      }
      CHECK(copyfrom >= 0 && copyfrom < oshape.ndim())
        << "Invalid dimension specified in lhs_axes: " << lhs_axes[i];

      auto copyto = rhs_axes[i];
      if (copyto < 0) {
        copyto =  rhs_shape.ndim() + copyto;
      }
      CHECK(copyto >= 0 && copyto < rhs_shape.ndim())
        << "Invalid dimension specified in rhs_axes: " << rhs_axes[i];

      CHECK(lhs_shape[copyfrom] == 1) << "Input axis " << lhs_axes[i]
        << " at dimension " << i << " cannot be broadcasted to " << rhs_shape[copyto];
      oshape[copyfrom] = rhs_shape[copyto];
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

// infer storage function for sum(csr) and mean(csr)
inline bool ReduceAxesOpForwardStorage(const nnvm::NodeAttrs& attrs,
                                       const int dev_mask,
                                       DispatchMode* dispatch_mode,
                                       std::vector<int>* in_attrs,
                                       std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  const ReduceAxesParam& param = nnvm::get<ReduceAxesParam>(attrs.parsed);
  const int in_stype = in_attrs->at(0);
  int& out_stype = out_attrs->at(0);
  // sum and reduce only supports for CPU for now.
  const bool invalid_ctx = dev_mask != mshadow::cpu::kDevMask;
  const auto dispatch_ex =
      invalid_ctx ? DispatchMode::kFComputeFallback : DispatchMode::kFComputeEx;
  bool dispatched = false;
  if (!dispatched && in_stype == kDefaultStorage) {
    // When input is dense output storage is set as dense and dispatched to
    // dense operator
    dispatched = storage_type_assign(&out_stype, kDefaultStorage, dispatch_mode,
                                     DispatchMode::kFCompute);
  }
  TShape axis = param.axis.has_value() ? param.axis.value() : TShape();
  if (!dispatched && in_stype == kCSRStorage && axis.ndim() == 1 &&
      (axis[0] == 0 || axis[0] == 1) && !param.keepdims && !param.exclude) {
    // If input is csr and axis is 0 or 1, and neither of keepdims or exclude
    // are set, dipsatch to sparse operator and output storage is set as dense
    dispatched = storage_type_assign(&out_stype, kDefaultStorage, dispatch_mode,
                                     dispatch_ex);
  }

  if (!dispatched) {
    // If input is csr, but keepdims or exclude is set or summing along a axis
    // different from 0 or 1
    dispatched = dispatch_fallback(out_attrs, dispatch_mode);
  }
  return dispatched;
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

template<typename xpu, typename reducer, bool normalize = false,
         typename OP = op::mshadow_op::identity>
void ReduceAxesComputeImpl(const OpContext& ctx,
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
          s, out_data.shape_, req[0], in_data.shape_);
      Tensor<xpu, 1, char> workspace =
          ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);
      broadcast::Reduce<reducer, NDim, DType, OP>(
          s, out_data, req[0], workspace, in_data);
      if (normalize) {
        auto out = out_data.FlatTo2D<xpu, DType>(s);
        out /= scalar<DType>(src_shape.Size()/dst_shape.Size());
      }
    });
  });
}

template<typename xpu, typename reducer, bool normalize = false,
         typename OP = op::mshadow_op::identity>
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

  ReduceAxesComputeImpl<xpu, reducer, normalize, OP>(ctx, inputs, req, outputs, small);
}

template <typename red_op, int req, int axis>
struct ReduceCsrKernel;

template <typename red_op, int req>
/* \brief The number of columns are divided equally among the number of threads
 * available.
 * Each thread gets a subset of columns. It iterates through all rows for the
 * subset of columns.
 * In each iteration, it tries to do a binary search for the first column
 * index between in_idx[in_indptr[row]] in_idx[in_indptr[row+1]]. After we find
 * an index that is equal to the first column or close to the first column,
 * it does a linear search for the rest of the indices and adds their data
 * to the intermediate sum. At the end of iteration through all
 * rows we have the sum along the axis for the subset of columns.
 */
struct ReduceCsrKernel<red_op, req, 0> {
  template <typename RType, typename IType, typename DType>
  MSHADOW_XINLINE static void Map(int j, DType* out_data,
                                  const RType* in_indptr, const IType* in_idx,
                                  const DType* in_data,
                                  DType* sum,
                                  DType* residual,
                                  RType num_rows,
                                  IType num_cols,
                                  const nnvm::dim_t seg_len) {
    const IType seg_start = j * seg_len;
    if (seg_start >= num_cols) return;
    const IType seg_end = std::min(seg_start + seg_len, num_cols);

    for (RType row = 0; row < num_rows; ++row) {
      // row specific seg starts
      IType row_seg_start = seg_start;
      IType row_seg_end = seg_end;

      // Cache starting and ending indptr values for the row
      IType row_indptr_start = in_indptr[row];
      IType row_indptr_end = in_indptr[row + 1] - 1;
      if (row_indptr_start == (row_indptr_end + 1)) continue;

      // If row_seg_start is less than the first index for the row, move the
      // row_seg_start forward
      while (row_seg_start < in_idx[row_indptr_start] &&
             row_seg_start < row_seg_end) {
        row_seg_start++;
      }

      // If row_seg_start is greater than last index for the row, move on to
      // the next row
      if (row_seg_start > in_idx[row_indptr_end]) continue;

      // Do binary search for row_seg_start between in_idx[in_indptr[i]] and
      // in_idx[in_indptr[i + 1]]
      IType start = row_indptr_start;
      IType end = row_indptr_end;

      // Initialize mid with the first indice of the row
      IType mid = start;
      while (start <= end) {
        mid = start + (end - start) / 2;
        if (in_idx[mid] == row_seg_start) {
          break;
        } else if (in_idx[mid] < row_seg_start) {
          start = mid + 1;
        } else {
          end = mid - 1;
        }
      }

      // At this point we have a in_idx[mid] which is close to row_seg_start
      // Safety check to make sure mid is a valid indptr value
      if (mid < row_indptr_start || mid > row_indptr_end)
          mid = row_indptr_start;


      // Linear search for nnzs for column subset between row_seg_start
      // and row_seg_end
      for (IType col = row_seg_start;
           col < row_seg_end && mid <= row_indptr_end;) {
        if (col == in_idx[mid]) {
          red_op::Reduce(sum[col], in_data[mid], residual[col]);
          mid++;
          col++;
        } else if (in_idx[mid] < col) {
          mid++;
        } else {
          col++;
        }
      }
    }

    for (IType col = seg_start; col < seg_end; col++) {
        KERNEL_ASSIGN(out_data[col], req, sum[col]);
    }
  }
};

template <typename red_op, int req>
struct ReduceCsrKernel<red_op, req, 1> {
  template <typename RType, typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data,
                                  const RType* in_indptr,
                                  const DType* in_data) {
    DType sum, residual;
    red_op::SetInitValue(sum, residual);
    for (RType k = in_indptr[i]; k < in_indptr[i + 1]; k++) {
      red_op::Reduce(sum, in_data[k], residual);
    }
    KERNEL_ASSIGN(out_data[i], req, sum);
  }
};

template <typename xpu, typename red_op, bool normalize = false>
void ReduceCsrImpl(mshadow::Stream<xpu>* s, const OpContext& ctx,
                   const NDArray& input, const OpReqType req,
                   NDArray* output, const TShape reduce_axis) {
  if (req == kNullOp) return;
  int64_t out_data_size = 0;
  if (reduce_axis[0] == 0) {
    out_data_size = input.shape()[1];
  } else {
    out_data_size = input.shape()[0];
  }
  // only dense output storage type is supported
  CHECK_EQ(output->storage_type(), kDefaultStorage);

  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;
  using namespace csr;
  using nnvm::dim_t;

  if (req == kWriteTo || req == kWriteInplace) {
    MSHADOW_TYPE_SWITCH(output->data().type_flag_, DType, {
      Kernel<set_zero, xpu>::Launch(s, out_data_size,
                                    output->data().dptr<DType>());
    })
  }

  if (!input.storage_initialized()) {
    return;
  }

  if (0 == reduce_axis[0]) {
    MSHADOW_IDX_TYPE_SWITCH(input.aux_type(kIndPtr), RType, {
      MSHADOW_IDX_TYPE_SWITCH(input.aux_type(kIdx), IType, {
        MSHADOW_TYPE_SWITCH(input.dtype(), DType, {
          MXNET_ASSIGN_REQ_SWITCH(req, req_type, {
            const RType* in_indptr = input.aux_data(kIndPtr).dptr<RType>();
            const IType* in_idx = input.aux_data(kIdx).dptr<IType>();
            const DType* in_data = input.data().dptr<DType>();
            const RType num_rows = input.shape()[0];
            const IType num_cols = input.shape()[1];
            dim_t num_threads = mxnet_op::get_num_threads<xpu>(16);
            dim_t seg_len = (out_data_size + num_threads - 1) / num_threads;
            mshadow::Tensor<xpu, 1, DType> workspace =
                ctx.requested[0].get_space_typed<xpu, 1, DType>(
                    Shape1(2 * out_data_size), s);
            mshadow::Tensor<xpu, 1, DType> sum(
                reinterpret_cast<DType*>(workspace.dptr_),
                Shape1(out_data_size));
            mshadow::Tensor<xpu, 1, DType> residual(
                reinterpret_cast<DType*>(workspace.dptr_ +
                                         out_data_size),
                Shape1(out_data_size));

            Kernel<set_zero, xpu>::Launch(s, out_data_size, sum.dptr_);
            Kernel<set_zero, xpu>::Launch(s, out_data_size, residual.dptr_);
            Kernel<ReduceCsrKernel<red_op, req_type, 0>, xpu>::Launch(
                s, num_threads, output->data().dptr<DType>(), in_indptr, in_idx,
                in_data, sum.dptr_, residual.dptr_, num_rows, num_cols,
                seg_len);
            if (normalize) {
              mxnet_op::Kernel<
                  mxnet_op::op_with_req<op::mshadow_op::div, req_type>,
                  xpu>::Launch(s, out_data_size, output->data().dptr<DType>(),
                               output->data().dptr<DType>(), DType(num_rows));
            }
          });
        });
      });
    });
  } else if (1 == reduce_axis[0]) {
    MSHADOW_IDX_TYPE_SWITCH(input.aux_type(kIndPtr), RType, {
      MSHADOW_IDX_TYPE_SWITCH(input.aux_type(kIdx), IType, {
        MSHADOW_TYPE_SWITCH(input.dtype(), DType, {
          MXNET_ASSIGN_REQ_SWITCH(req, req_type, {
            const RType* in_indptr = input.aux_data(kIndPtr).dptr<RType>();
            const DType* in_data = input.data().dptr<DType>();
            const IType num_cols = input.shape()[1];
            Kernel<ReduceCsrKernel<red_op, req_type, 1>, xpu>::Launch(
                s, out_data_size, output->data().dptr<DType>(), in_indptr,
                in_data);
            if (normalize) {
              mxnet_op::Kernel<
                  mxnet_op::op_with_req<op::mshadow_op::div, req_type>,
                  xpu>::Launch(s, out_data_size, output->data().dptr<DType>(),
                               output->data().dptr<DType>(), DType(num_cols));
            }
          });
        });
      });
    });
  }
}

/*! \brief If normalize is true, the mean should be computed instead of sum */
template <typename xpu, typename red_op, bool normalize = false>
void ReduceCsr(const nnvm::NodeAttrs& attrs, mshadow::Stream<xpu>* s, const OpContext& ctx,
               const NDArray& input, const OpReqType req, NDArray* output) {
  const ReduceAxesParam& param = nnvm::get<ReduceAxesParam>(attrs.parsed);
  CHECK(param.axis.has_value());
  const TShape axis = param.axis.value();
  CHECK_EQ(axis.ndim(), 1U) << "sum(csr)/mean(csr) only supports axis 0 or 1";
  CHECK(axis[0] == 0 || axis[0] == 1)
     << "sum(csr)/mean(csr) only support axis 0 or 1";
  CHECK(!param.keepdims) << "keepdims not supported for sparse";
  CHECK(!param.exclude) << "exclude not supported for sparse";
  ReduceCsrImpl<xpu, red_op, normalize>(s, ctx, input, req, output, axis);
}

template <typename xpu, typename reducer, bool normalize = false>
void ReduceAxesOpForwardEx(const nnvm::NodeAttrs& attrs, const OpContext& ctx,
                           const std::vector<NDArray>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  mshadow::Stream<xpu>* s = ctx.get_stream<xpu>();
  const NDArrayStorageType istype = inputs[0].storage_type();
  if (istype == kCSRStorage) {
    NDArray output = outputs[0];
    ReduceCsr<xpu, mshadow::red::sum, normalize>(attrs, s, ctx, inputs[0],
                                                 req[0], &output);
  } else {
    LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
  }
}

template<typename xpu, typename OP, bool normalize = false>
void ReduceAxesBackwardUseInOutImpl(const OpContext& ctx,
                                    const TShape &small,
                                    const std::vector<TBlob>& inputs,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;

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
  ReduceAxesBackwardUseInOutImpl<xpu, OP, normalize>(ctx, small, inputs, req, outputs);
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

inline bool LpNormStorageType(const nnvm::NodeAttrs& attrs,
                              const int dev_mask,
                              DispatchMode* dispatch_mode,
                              std::vector<int>* in_attrs,
                              std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  const int in_stype = in_attrs->at(0);
  int& out_stype = out_attrs->at(0);
  const NormParam& param = nnvm::get<NormParam>(attrs.parsed);
  bool dispatched = false;
  // l2 norm on a particular axis only supports cpu
  const bool invalid_ctx = dev_mask != mshadow::cpu::kDevMask;
  const auto dispatch_ex =
      invalid_ctx ? DispatchMode::kFComputeFallback : DispatchMode::kFComputeEx;
  if (!dispatched && in_stype == kDefaultStorage) {
    // dns -> dns
    dispatched = storage_type_assign(&out_stype, kDefaultStorage, dispatch_mode,
                                     DispatchMode::kFCompute);
  }
  if (param.ord == 2) {
    const TShape axis = param.axis.has_value() ? param.axis.value() : TShape();
    if (!dispatched && (in_stype == kRowSparseStorage || in_stype == kCSRStorage) &&
        axis.ndim() == 0 && param.ord == 2) {
      // l2 norm: rsp/csr, axis = () -> dns
      dispatched = storage_type_assign(&out_stype, kDefaultStorage, dispatch_mode,
                                       DispatchMode::kFComputeEx);
    }
    if (!dispatched && in_stype == kCSRStorage && axis.ndim() == 1 && !param.keepdims &&
        (axis[0] == 0 || axis[0] == 1) && param.ord == 2) {
      // l2 norm: csr, axis = 0/1 -> dns
      dispatched = storage_type_assign(&out_stype, kDefaultStorage, dispatch_mode,
                                       dispatch_ex);
    }
  }
  if (!dispatched) {
    dispatched = dispatch_fallback(out_attrs, dispatch_mode);
  }
  return dispatched;
}

/*! \brief compute square on each element and sum reducer */
struct sq_sum {
  /*! \brief do reduction into dst */
  template<typename DType>
  MSHADOW_XINLINE static void Reduce(volatile DType& dst,  volatile DType src) { // NOLINT(*)
    dst += src * src;
  }
  /*! \brief do stable reduction into dst */
  template<typename DType>
  MSHADOW_XINLINE static void Reduce(volatile DType& dst,  volatile DType src, volatile DType& residual) { // NOLINT(*)
    DType y = src * src - residual;
    DType t = dst + y;
    residual = (t - dst) - y;
    dst = t;
  }
  /*!
   *\brief calculate gradient of redres with respect to redsrc,
   * redres: reduced result, redsrc: one of reduction element
   */
  template<typename DType>
  MSHADOW_XINLINE static DType PartialGrad(DType redres, DType redsrc) {
    // This won't be called in backward.
    return 1;
  }
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType &initv) { // NOLINT(*)
    initv = 0;
  }
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType &initv, DType &residual) { // NOLINT(*)
    SetInitValue(initv);
    residual = 0;
  }
};

template<typename xpu>
void L2NormComputeImpl(mshadow::Stream<xpu> *s,
                       const TBlob& input,
                       const OpReqType req,
                       const TBlob& output) {
  MSHADOW_REAL_TYPE_SWITCH(output.type_flag_, DType, {
    // assign_req switch exits immediately for null req
    MXNET_ASSIGN_REQ_SWITCH(req, Req, {
      mshadow::Tensor<xpu, 1, DType> out = output.get_with_shape<xpu, 1, DType>(
        mshadow::Shape1(output.shape_.Size()), s);
      mshadow::Tensor<xpu, 1, DType> in = input.get_with_shape<xpu, 1, DType>(
        mshadow::Shape1(input.shape_.Size()), s);
      mshadow::VectorDot(out, in, in);
      DType* out_data = output.dptr<DType>();
      using namespace mxnet_op;
      Kernel<op_with_req<mshadow_op::square_root, Req>, xpu>::Launch(
        s, output.Size(), out_data, out_data);
    });
  });
}

template<typename xpu>
void SqRootForL2(const OpContext& ctx, OpReqType req, const TBlob &output) {
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH(output.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req, Req, {
      DType* out_data = output.dptr<DType>();
      using namespace mxnet_op;
      Kernel<op_with_req<mshadow_op::square_root, Req>, xpu>::Launch(
        s, output.Size(), out_data, out_data);
    });
  });
}

template<typename xpu>
void LpNormCompute(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
  const NormParam& param = nnvm::get<NormParam>(attrs.parsed);
  CHECK(param.ord == 1 || param.ord == 2) << "norm only supports ord=1 and ord=2";
  if (req[0] == kNullOp) return;

  TShape small;
  if (param.keepdims) {
    small = outputs[0].shape_;
  } else {
    small = ReduceAxesShapeImpl(inputs[0].shape_, param.axis, true, false);
  }
  if (param.ord == 1) {
    ReduceAxesComputeImpl<xpu, mshadow::red::sum, false, mshadow_op::abs>(
          ctx, inputs, req, outputs, small);
  } else if (param.ord == 2) {
    ReduceAxesComputeImpl<xpu, mshadow_op::nrm2, false, mshadow_op::identity>(
        ctx, inputs, req, outputs, small);
  }
}

template<typename xpu>
void LpNormGradCompute(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  if (req[0] == kNullOp) return;

  const NormParam& param = nnvm::get<NormParam>(attrs.parsed);
  TShape small;
  if (param.keepdims) {
    small = inputs[0].shape_;
  } else {
    small = ReduceAxesShapeImpl(outputs[0].shape_, param.axis, true, false);
  }
  if (param.ord == 1) {
    TShape src_shape, dst_shape;
    BroadcastReduceShapeCompact(outputs[0].shape_, small, &src_shape, &dst_shape);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      if (dst_shape.ndim() == 2) {
        Tensor<xpu, 2, DType> ograd =
          inputs[0].get_with_shape<xpu, 2, DType>(dst_shape.get<2>(), s);
        Tensor<xpu, 2, DType> igrad =
          outputs[0].get_with_shape<xpu, 2, DType>(src_shape.get<2>(), s);
        Tensor<xpu, 2, DType> data =
          inputs[1].get_with_shape<xpu, 2, DType>(src_shape.get<2>(), s);
        ASSIGN_DISPATCH(igrad, req[0],
          broadcast_to(ograd, src_shape)*F<mshadow_op::sign>(data));
      } else {
        const int ndim = MXNET_SPECIAL_MAX_NDIM;
        Tensor<xpu, ndim, DType> igrad =
          outputs[0].get_with_shape<xpu, ndim, DType>(src_shape.get<ndim>(), s);
        Tensor<xpu, ndim, DType> ograd =
          inputs[0].get_with_shape<xpu, ndim, DType>(dst_shape.get<ndim>(), s);
        Tensor<xpu, ndim, DType> data =
          inputs[1].get_with_shape<xpu, ndim, DType>(src_shape.get<ndim>(), s);
        ASSIGN_DISPATCH(igrad, req[0],
          broadcast_to(ograd, src_shape)*F<mshadow_op::sign>(data));
      }
    });
  } else if (param.ord == 2) {
    ReduceAxesBackwardUseInOutImpl<xpu, mshadow_op::div, false>(ctx, small, inputs,
                                                                req, outputs);
  }
}

template<typename xpu>
void L2NormComputeSparseImpl(mshadow::Stream<xpu> *s,
                             const NDArray& input,
                             const OpReqType req,
                             const TBlob& output) {
  if (req == kNullOp) return;
  // input is zeros
  if (!input.storage_initialized()) {
    // Add zeros. No op.
    if (req == kAddTo) return;
    Fill<false>(s, output, req, 0);
  } else {
    L2NormComputeImpl(s, input.data(), req, output);
  }
}

template<typename xpu>
void L2NormComputeEx(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<NDArray>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<NDArray>& outputs);

/*! \brief index element from array along axes */
template<int ndim, bool clip = true>
struct pick {
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i, DType* out, const DType* a,
                                  const IType *idx, int M, int stride,
                                  mshadow::Shape<ndim> bshape,
                                  mshadow::Shape<ndim> sshape) {
    using namespace broadcast;
    int j = static_cast<int>(idx[i]);
    if (clip) {
      if (j <= 0) j = 0;
      else if (j >= M) j = M - 1;
    } else {
      j = j % M;
      j += (j < 0) ? M : 0;
    }
    j = ravel(unravel(i, sshape), bshape) + j*stride;
    out[i] = a[j];
  }
};

/*! \brief index element from array along axes */
template<int ndim, bool clip = true>
struct pick_grad {
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i, DType* igrad, const DType* ograd,
                                  const IType *idx, int M, int stride,
                                  mshadow::Shape<ndim> bshape,
                                  mshadow::Shape<ndim> sshape) {
    using namespace broadcast;
    int j = static_cast<int>(idx[i]);
    if (clip) {
      if (j <= 0) j = 0;
      else if (j >= M) j = M - 1;
    } else {
      j = j % M;
      j += (j < 0) ? M : 0;
    }
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
  TShape oshape = ReduceAxisShapeImpl(ishape, param.axis, param.keepdims);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  if (!(*in_attrs)[1].ndim()) return false;
  if ((*in_attrs)[1].ndim() == ishape.ndim()) {
    SHAPE_ASSIGN_CHECK(*in_attrs, 1,
                       ReduceAxisShapeImpl(ishape, param.axis, true));
  } else {
    SHAPE_ASSIGN_CHECK(*in_attrs, 1,
                       ReduceAxisShapeImpl(ishape, param.axis, false));
  }
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
  index_t axis = CheckAxis(param.axis.value(), ishape.ndim());
  int leading = 1, trailing = 1, M = ishape[axis];
  for (index_t i = 0; i < axis; ++i) leading *= ishape[i];
  for (index_t i = axis+1; i < ishape.ndim(); ++i) trailing *= ishape[i];

  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {  // output type
    MSHADOW_TYPE_SWITCH(inputs[1].type_flag_, IType, {  // index type
      if (param.mode == kWrap) {
        if (trailing == 1) {
            Kernel<pick<2, false>, xpu>::Launch(s, outputs[0].Size(), outputs[0].dptr<DType>(),
                                    inputs[0].dptr<DType>(), inputs[1].dptr<IType>(),
                                    M, 1, Shape2(leading, M), Shape2(leading, 1));
        } else {
            Kernel<pick<3, false>, xpu>::Launch(s, outputs[0].Size(), outputs[0].dptr<DType>(),
                                    inputs[0].dptr<DType>(), inputs[1].dptr<IType>(),
                                    M, trailing, Shape3(leading, M, trailing),
                                    Shape3(leading, 1, trailing));
        }
      } else {
        if (trailing == 1) {
            Kernel<pick<2, true>, xpu>::Launch(s, outputs[0].Size(), outputs[0].dptr<DType>(),
                                   inputs[0].dptr<DType>(), inputs[1].dptr<IType>(),
                                   M, 1, Shape2(leading, M), Shape2(leading, 1));
        } else {
            Kernel<pick<3, true>, xpu>::Launch(s, outputs[0].Size(), outputs[0].dptr<DType>(),
                                   inputs[0].dptr<DType>(), inputs[1].dptr<IType>(),
                                   M, trailing, Shape3(leading, M, trailing),
                                   Shape3(leading, 1, trailing));
        }
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
      if (param.mode == kWrap) {
        if (trailing == 1) {
          Kernel<pick_grad<2, false>, xpu>::Launch(s, inputs[0].Size(), outputs[0].dptr<DType>(),
                                      inputs[0].dptr<DType>(), inputs[1].dptr<IType>(),
                                      M, 1, Shape2(leading, M), Shape2(leading, 1));
        } else {
          Kernel<pick_grad<3, false>, xpu>::Launch(s, inputs[0].Size(), outputs[0].dptr<DType>(),
                                      inputs[0].dptr<DType>(), inputs[1].dptr<IType>(),
                                      M, trailing, Shape3(leading, M, trailing),
                                      Shape3(leading, 1, trailing));
        }
      } else {
          if (trailing == 1) {
          Kernel<pick_grad<2, true>, xpu>::Launch(s, inputs[0].Size(), outputs[0].dptr<DType>(),
                                      inputs[0].dptr<DType>(), inputs[1].dptr<IType>(),
                                      M, 1, Shape2(leading, M), Shape2(leading, 1));
        } else {
          Kernel<pick_grad<3, true>, xpu>::Launch(s, inputs[0].Size(), outputs[0].dptr<DType>(),
                                      inputs[0].dptr<DType>(), inputs[1].dptr<IType>(),
                                      M, trailing, Shape3(leading, M, trailing),
                                      Shape3(leading, 1, trailing));
        }
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
