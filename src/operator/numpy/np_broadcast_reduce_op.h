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
#ifndef MXNET_OPERATOR_NUMPY_NP_BROADCAST_REDUCE_OP_H_
#define MXNET_OPERATOR_NUMPY_NP_BROADCAST_REDUCE_OP_H_

#include <algorithm>
#include <vector>
#include <string>
#include "../nn/moments-inl.h"
#include "../tensor/broadcast_reduce_op.h"
#include "../tensor/ordering_op-inl.h"
#include "../tensor/matrix_op-inl.h"

namespace mxnet {
namespace op {

struct NumpyReduceAxesParam : public dmlc::Parameter<NumpyReduceAxesParam> {
  dmlc::optional<mxnet::Tuple<int>> axis;
  dmlc::optional<int> dtype;
  bool keepdims;
  dmlc::optional<double> initial;
  DMLC_DECLARE_PARAMETER(NumpyReduceAxesParam) {
    DMLC_DECLARE_FIELD(axis)
      .set_default(dmlc::optional<mxnet::Tuple<int>>())
      .describe("Axis or axes along which a sum is performed. The default, axis=None, will sum "
                "all of the elements of the input array. If axis is negative it counts from the "
                "last to the first axis.");
    DMLC_DECLARE_FIELD(dtype)
      .add_enum("float16", mshadow::kFloat16)
      .add_enum("float32", mshadow::kFloat32)
      .add_enum("float64", mshadow::kFloat64)
      .add_enum("int8", mshadow::kInt8)
      .add_enum("int32", mshadow::kInt32)
      .add_enum("int64", mshadow::kInt64)
      .add_enum("bool", mshadow::kBool)
      .set_default(dmlc::optional<int>())
      .describe("The type of the returned array and of the accumulator in which the elements are "
                "summed. The dtype of a is used by default unless a has an integer dtype of less "
                "precision than the default platform integer. In that case, if a is signed then "
                "the platform integer is used while if a is unsigned then an unsigned integer of "
                "the same precision as the platform integer is used.");
    DMLC_DECLARE_FIELD(keepdims).set_default(false)
      .describe("If this is set to `True`, the reduced axes are left "
                "in the result as dimension with size one.");
    DMLC_DECLARE_FIELD(initial).set_default(dmlc::optional<double>())
      .describe("Starting value for the sum.");
  }
};

struct NumpyReduceAxesNoDTypeParam : public dmlc::Parameter<NumpyReduceAxesNoDTypeParam> {
  dmlc::optional<mxnet::Tuple<int>> axis;
  bool keepdims;
  dmlc::optional<double> initial;
  DMLC_DECLARE_PARAMETER(NumpyReduceAxesNoDTypeParam) {
    DMLC_DECLARE_FIELD(axis)
      .set_default(dmlc::optional<mxnet::Tuple<int>>())
      .describe("Axis or axes along which a sum is performed. The default, axis=None, will sum "
                "all of the elements of the input array. If axis is negative it counts from the "
                "last to the first axis.");
    DMLC_DECLARE_FIELD(keepdims).set_default(false)
      .describe("If this is set to `True`, the reduced axes are left "
                "in the result as dimension with size one.");
    DMLC_DECLARE_FIELD(initial).set_default(dmlc::optional<double>())
      .describe("Starting value for the sum.");
  }
};

inline TShape NumpyReduceAxesShapeImpl(const TShape& ishape,
                                       const dmlc::optional<mxnet::Tuple<int>>& axis,
                                       bool keepdims) {
  // If input is a scalar, output should be a scalar too
  if (ishape.ndim() == 0) {
    if (axis.has_value()) {
      const mxnet::Tuple<int>& axes = axis.value();
      if (axes.ndim() > 0) {
        CHECK_EQ(axes.ndim(), 1);
        CHECK(axes[0] == 0 || axes[0] == -1);
      }
    }
    return TShape(0, -1);
  }

  // axis=None, do global reduction
  if (!axis.has_value()) {
    if (keepdims) {
      return TShape(ishape.ndim(), 1);
    } else {
      return TShape(0, -1);
    }
  }

  // axis = (), will return identity(input)
  if (axis.value().ndim() == 0) {
    return ishape;
  }

  // axis has value
  mxnet::Tuple<int> axes(axis.value());
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
  } else {
    oshape = TShape(ishape.ndim() - axes.ndim(), -1);
  }

  if (keepdims) {
    for (index_t i = 0; i < axes.ndim(); ++i) {
      oshape[axes[i]] = 1;
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

inline bool NumpyReduceAxesShape(const nnvm::NodeAttrs& attrs,
                                 std::vector<TShape> *in_attrs,
                                 std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  if (!shape_is_known(in_attrs->at(0))) {
    return false;
  }
  const NumpyReduceAxesParam& param = nnvm::get<NumpyReduceAxesParam>(attrs.parsed);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0,
                     NumpyReduceAxesShapeImpl((*in_attrs)[0], param.axis, param.keepdims));
  return shape_is_known(out_attrs->at(0));
}

inline bool NumpyReduceAxesNoDTypeShape(const nnvm::NodeAttrs& attrs,
                                        std::vector<TShape> *in_attrs,
                                        std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  if (!shape_is_known(in_attrs->at(0))) {
    return false;
  }
  const NumpyReduceAxesNoDTypeParam& param = nnvm::get<NumpyReduceAxesNoDTypeParam>(attrs.parsed);
  // check the case where the reduction axis should not be zero
  bool is_all_reducded_axes_not_zero = true;
  const TShape& ishape = (*in_attrs)[0];
  if (param.axis.has_value()) {
    const mxnet::Tuple<int>& axes = param.axis.value();
    for (int i = 0; i < axes.ndim(); ++i) {
      if (ishape[axes[i]] == 0) {
        is_all_reducded_axes_not_zero = false;
        break;
      }
    }
  } else {
    if (ishape.Size() == 0) {
      // global reduction should excuted only when input have size more than 0
      is_all_reducded_axes_not_zero = false;
    }
  }
  CHECK(is_all_reducded_axes_not_zero)
    << "zero-size array to reduction operation maximum which has no identity";
  SHAPE_ASSIGN_CHECK(*out_attrs, 0,
                     NumpyReduceAxesShapeImpl((*in_attrs)[0], param.axis, param.keepdims));
  return shape_is_known(out_attrs->at(0));
}

template<bool safe_acc_hint = false>
inline bool NeedSafeAcc(int itype, int otype) {
  bool rule = (itype != otype) || (itype != mshadow::kFloat32 && itype != mshadow::kFloat64);
  return safe_acc_hint && rule;
}

void TVMOpReduce(const OpContext& ctx, const TBlob& input,
                 const dmlc::optional<mxnet::Tuple<int>>& axis,
                 const TBlob& output, const OpReqType req, const std::string& reducer_name);

template<typename xpu, typename reducer, bool safe_acc_hint = false, bool normalize = false,
         typename OP = op::mshadow_op::identity>
void NumpyReduceAxesCompute(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  if (req[0] == kNullOp) return;
  const NumpyReduceAxesParam& param = nnvm::get<NumpyReduceAxesParam>(attrs.parsed);
  if (param.initial.has_value()) {
    LOG(FATAL) << "initial is not supported yet";
  }
  Stream<xpu>* s = ctx.get_stream<xpu>();
  if (inputs[0].shape_.Size() == 0 && outputs[0].shape_.Size() != 0) {
    using namespace mxnet_op;
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      Kernel<set_zero, xpu>::Launch(s, outputs[0].shape_.Size(), outputs[0].dptr<DType>());
    });
    return;
  }
  CHECK_NE(req[0], kWriteInplace) << "Reduce does not support write in-place";
#if MXNET_USE_TVM_OP
  // If boolean ndarray, use the kernel generated by TVM
  if (inputs[0].type_flag_ == mshadow::kBool) {
    std::string reducer_name;
    if (std::is_same<reducer, mshadow_op::sum>::value) {
      reducer_name = "sum";
    } else {
      LOG(FATAL) << "Only reduce op: `sum` is supported for boolean ndarrays";
    }
    TVMOpReduce(ctx, inputs[0], param.axis, outputs[0], req[0], reducer_name);
    if (normalize) {
      using namespace mshadow::expr;
      MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, OType, {
        auto out = outputs[0].FlatTo2D<xpu, OType>(s);
        out /= scalar<OType>(inputs[0].Size()/outputs[0].Size());
      });
    }
    return;
  }
#endif
  if (param.axis.has_value() && param.axis.value().ndim() == 0) {
    UnaryOp::IdentityCompute<xpu>(attrs, ctx, inputs, req, outputs);
  }
  TShape small;
  if (param.keepdims) {
    small = outputs[0].shape_;
  } else {
    small = NumpyReduceAxesShapeImpl(inputs[0].shape_, param.axis, true);
  }

  if (NeedSafeAcc<safe_acc_hint>(inputs[0].type_flag_, outputs[0].type_flag_)) {
    ReduceAxesComputeImpl<xpu, reducer, true, normalize, OP>(ctx, inputs, req, outputs, small);
  } else {
    ReduceAxesComputeImpl<xpu, reducer, false, normalize, OP>(ctx, inputs, req, outputs, small);
  }
}

template<typename xpu, typename reducer, typename OP = op::mshadow_op::identity>
void NumpyReduceAxesNoDTypeCompute(const nnvm::NodeAttrs& attrs,
                                   const OpContext& ctx,
                                   const std::vector<TBlob>& inputs,
                                   const std::vector<OpReqType>& req,
                                   const std::vector<TBlob>& outputs) {
  const NumpyReduceAxesNoDTypeParam& param = nnvm::get<NumpyReduceAxesNoDTypeParam>(attrs.parsed);
  if (param.initial.has_value()) {
    LOG(FATAL) << "initial is not supported yet";
  }
  if (inputs[0].shape_.Size() == 0U || outputs[0].shape_.Size() == 0U) return;  // zero-size tensor
  if (param.axis.has_value() && param.axis.value().ndim() == 0) {
    UnaryOp::IdentityCompute<xpu>(attrs, ctx, inputs, req, outputs);
  }
  TShape small;
  if (param.keepdims) {
    small = outputs[0].shape_;
  } else {
    small = NumpyReduceAxesShapeImpl(inputs[0].shape_, param.axis, true);
  }
  ReduceAxesComputeImpl<xpu, reducer, false, false, OP>(ctx, inputs, req, outputs, small);
}


template<typename xpu, bool normalize = false>
inline void NumpyReduceAxesBackwardUseNone(const nnvm::NodeAttrs& attrs,
                                           const OpContext& ctx,
                                           const std::vector<TBlob>& inputs,
                                           const std::vector<OpReqType>& req,
                                           const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  CHECK_NE(outputs[0].type_flag_, kBool) << "reduce operators do not support gradient calculation "
                                            "for input tensors of boolean type.";
  const NumpyReduceAxesParam& param = nnvm::get<NumpyReduceAxesParam>(attrs.parsed);
  TShape small;
  if (param.keepdims) {
    small = inputs[0].shape_;
  } else {
    small = NumpyReduceAxesShapeImpl(outputs[0].shape_, param.axis, true);
  }

  BroadcastComputeImpl<xpu>(attrs, ctx, inputs, req, outputs, small);
  if (normalize) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, IType, {
      Tensor<xpu, 1, IType> igrad = outputs[0].FlatTo1D<xpu, IType>(s);
      igrad /= scalar<IType>(outputs[0].Size()/inputs[0].Size());
    });
  }
}

template<typename xpu, typename OP, bool normalize = false>
void NumpyReduceAxesBackwardUseInOut(const nnvm::NodeAttrs& attrs,
                                     const OpContext& ctx,
                                     const std::vector<TBlob>& inputs,
                                     const std::vector<OpReqType>& req,
                                     const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  const NumpyReduceAxesParam& param = nnvm::get<NumpyReduceAxesParam>(attrs.parsed);
  TShape small;
  if (param.keepdims) {
    small = inputs[0].shape_;
  } else {
    small = NumpyReduceAxesShapeImpl(outputs[0].shape_, param.axis, true);
  }
  ReduceAxesBackwardUseInOutImpl<xpu, OP, normalize>(ctx, small, inputs, req, outputs);
}

struct NumpyMomentsParam : public dmlc::Parameter<NumpyMomentsParam> {
  dmlc::optional<mxnet::Tuple<int>> axis;
  dmlc::optional<int> dtype;
  bool keepdims;
  int ddof;
  DMLC_DECLARE_PARAMETER(NumpyMomentsParam) {
    DMLC_DECLARE_FIELD(axis)
      .set_default(dmlc::optional<mxnet::Tuple<int>>())
      .describe("Axis or axes along which a sum is performed. The default, axis=None, will sum "
                "all of the elements of the input array. If axis is negative it counts from the "
                "last to the first axis.");
    DMLC_DECLARE_FIELD(dtype)
      .add_enum("float16", mshadow::kFloat16)
      .add_enum("float32", mshadow::kFloat32)
      .add_enum("float64", mshadow::kFloat64)
      .add_enum("int8", mshadow::kInt8)
      .add_enum("int32", mshadow::kInt32)
      .add_enum("int64", mshadow::kInt64)
      .set_default(dmlc::optional<int>())
      .describe("The type of the returned array and of the accumulator in which the elements are "
                "summed. The dtype of a is used by default unless a has an integer dtype of less "
                "precision than the default platform integer. In that case, if a is signed then "
                "the platform integer is used while if a is unsigned then an unsigned integer of "
                "the same precision as the platform integer is used.");
    DMLC_DECLARE_FIELD(ddof).set_default(0)
      .describe("Starting value for the sum.");
    DMLC_DECLARE_FIELD(keepdims).set_default(false)
      .describe("If this is set to `True`, the reduced axes are left "
                "in the result as dimension with size one.");
  }
};

template<typename xpu, typename reducer, bool safe_acc, bool normalize = false,
         typename OP = op::mshadow_op::identity>
void ReduceAxesComputeWithWorkspaceImpl(const OpContext& ctx,
                                        const std::vector<TBlob>& inputs,
                                        const std::vector<OpReqType>& req,
                                        const std::vector<TBlob>& outputs,
                                        const mshadow::Tensor<xpu, 1, char>& workspace,
                                        const mxnet::TShape& src_shape,
                                        const mxnet::TShape& dst_shape,
                                        const int ddof = 0) {
  using namespace mshadow;
  using namespace mshadow::expr;

  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, OType, {
      const TBlob in_data = inputs[0].reshape(src_shape);
      const TBlob out_data = outputs[0].reshape(dst_shape);
      BROADCAST_NDIM_SWITCH(dst_shape.ndim(), NDim, {
        broadcast::Reduce<reducer, NDim, DType, OP, safe_acc>(
            s, out_data, req[0], workspace, in_data);
        if (normalize) {
          auto out = out_data.FlatTo2D<xpu, OType>(s);
          out /= scalar<OType>(src_shape.Size()/dst_shape.Size() - ddof);
        }
      });
    });
  });
}

template<typename xpu, bool sqrt>
void NumpyMomentsForward(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mshadow_op;
  using namespace mxnet_op;

  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(req.size(), 2U);
  CHECK_EQ(outputs.size(), 2U);

  const NumpyMomentsParam& param = nnvm::get<NumpyMomentsParam>(attrs.parsed);

  Stream<xpu> *s = ctx.get_stream<xpu>();

  const TBlob& data = inputs[0];
  const TBlob& moment = outputs[0];
  const TBlob& mean = outputs[1];

  mxnet::TShape small;
  if (param.keepdims) {
    small = moment.shape_;
  } else {
    small = NumpyReduceAxesShapeImpl(data.shape_, param.axis, true);
  }

  mxnet::TShape src_shape, dst_shape;
  BroadcastReduceShapeCompact(data.shape_, small, &src_shape, &dst_shape);

  MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, OType, {
      // Get workspace and temp space for data - mean
      size_t workspace_size = 0;
      BROADCAST_NDIM_SWITCH(dst_shape.ndim(), NDim, {
        workspace_size = broadcast::ReduceWorkspaceSize<NDim, DType>(
          s, dst_shape, req[0], src_shape);;
      });
      size_t temp_data_size = data.shape_.Size() * sizeof(DType);
      size_t temp_mem_size = temp_data_size + workspace_size;
      Tensor<xpu, 1, char> temp_mem =
        ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(temp_mem_size), s);
      DType *temp_data_ptr = reinterpret_cast<DType*>(temp_mem.dptr_);
      char *workspace_ptr = temp_mem.dptr_ + temp_data_size;
      Tensor<xpu, 1, char> workspace(workspace_ptr, Shape1(workspace_size), s);
      // Compute mean
      ReduceAxesComputeWithWorkspaceImpl<xpu, mshadow_op::sum, true, true>(
        ctx, inputs, {kWriteTo}, {mean}, workspace, src_shape, dst_shape);
      // Compute data - mean
      Shape<6> data_shape, mean_shape;
      for (int i = 0; i < 6; ++i) {
        data_shape[i] = (i < data.shape_.ndim()) ? data.shape_[i] : 1;
        mean_shape[i] = (i < small.ndim()) ? small[i] : 1;
      }
      Kernel<VarBroadcastKernel, xpu>::Launch(s, data_shape.Size(), temp_data_ptr,
        data.dptr<DType>(), mean.dptr<DType>(), data_shape, mean_shape);
      Tensor<xpu, 1, DType> temp_data_tensor(temp_data_ptr, Shape1(data.shape_.Size()), s);
      TBlob temp_data_blob = TBlob(temp_data_tensor).reshape(data.shape_);
      ReduceAxesComputeWithWorkspaceImpl<xpu, mshadow_op::sum, true, true>(
        ctx, {temp_data_blob}, {req[0]}, {moment}, workspace, src_shape, dst_shape, param.ddof);
      if (sqrt) {
        Tensor<xpu, 1, OType> moment_tensor = moment.FlatTo1D<xpu, OType>(s);
        moment_tensor = F<mshadow_op::square_root>(moment_tensor);
      }
    });
  });
}

namespace quantile_enum {
enum InterpolationType {kLinear, kLower, kHigher, kMidpoint, kNearest};
}

struct NumpyQuantileParam : public dmlc::Parameter<NumpyQuantileParam> {
  dmlc::optional<mxnet::Tuple<int>> axis;
  int interpolation;
  bool keepdims;
  DMLC_DECLARE_PARAMETER(NumpyQuantileParam) {
    DMLC_DECLARE_FIELD(axis)
      .set_default(dmlc::optional<mxnet::Tuple<int>>())
      .describe("Axis or axes along which the quantiles are computed. "
                "The default is to compute the "
                "quantile(s) along a flattened version of the array.");
    DMLC_DECLARE_FIELD(interpolation)
      .set_default(quantile_enum::kLinear)
      .add_enum("linear", quantile_enum::kLinear)
      .add_enum("lower", quantile_enum::kLower)
      .add_enum("higher", quantile_enum::kHigher)
      .add_enum("midpoint", quantile_enum::kMidpoint)
      .add_enum("nearest", quantile_enum::kNearest)
      .describe("This optional parameter specifies the interpolation method to use when the desired"
                " quantile lies between two data points i < j");
    DMLC_DECLARE_FIELD(keepdims)
      .set_default(false)
      .describe("If this is set to `True`, the reduced axes are left "
                "in the result as dimension with size one.");
  }
};

inline bool NumpyQuantileShape(const nnvm::NodeAttrs& attrs,
                               std::vector<TShape> *in_attrs,
                               std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  const mxnet::TShape& a_shape = in_attrs->at(0);
  const mxnet::TShape& q_shape = in_attrs->at(1);
  auto &param = nnvm::get<NumpyQuantileParam>(attrs.parsed);
  CHECK_LE(q_shape.ndim(), 1);
  auto small = NumpyReduceAxesShapeImpl(a_shape, param.axis, param.keepdims);
  if (q_shape.ndim() == 0) {
    // q is scalar
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, small);
  } else {
    CHECK_EQ(q_shape.ndim(), 1U);
    mxnet::TShape shape(small.ndim()+1, 0);
    shape[0] = q_shape[0];
    for (int i = 1; i < shape.ndim(); ++i) {
      shape[i] = small[i-1];
    }
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, shape);
  }
  return shape_is_known(*in_attrs) && shape_is_known(*out_attrs);
}

template<int NDim>
struct quantile_take {
  template<typename DType, typename QType>
  MSHADOW_XINLINE static void Map(int i,
                                  DType* out,
                                  const QType* q,
                                  const DType* a_sort,
                                  const int interpolation,
                                  mshadow::Shape<NDim> t_shape,
                                  mshadow::Shape<NDim> r_shape) {
    using namespace mshadow;
    using namespace mxnet_op;

    auto r_coord = unravel(i, r_shape);
    size_t q_idx = r_coord[0];

    Shape<NDim> t_coord(t_shape);

    for (int j = 0; j < NDim-1; ++j) {
      t_coord[j] = r_coord[j+1];
    }

    float idx = q[q_idx] * (t_shape[NDim-1]-1);
    int integral_idx = -1;
    if (interpolation == quantile_enum::kLower) {
      integral_idx = floor(idx);
    } else if (interpolation == quantile_enum::kHigher) {
      integral_idx = ceil(idx);
    } else if (interpolation == quantile_enum::kMidpoint) {
      idx = (floor(idx) + ceil(idx)) / 2;
    } else if (interpolation == quantile_enum::kNearest) {
      integral_idx = round(idx);
    }

    if (integral_idx >= 0) {
      t_coord[NDim-1] = integral_idx;
      size_t t_idx = ravel(t_coord, t_shape);
      out[i] = a_sort[t_idx];
    } else {
      int idx_below = floor(idx);
      int idx_above = idx_below + 1;
      idx_above = idx_above > t_shape[NDim-1] - 1 ? t_shape[NDim-1] - 1 : idx_above;
      float weight_above = idx - idx_below;
      float weight_below = 1 - weight_above;
      t_coord[NDim-1] = idx_below;
      size_t t_idx1 = ravel(t_coord, t_shape);
      size_t t_idx2 = t_idx1 + (idx_above - idx_below);
      DType x1 = a_sort[t_idx1] * weight_below;
      DType x2 = a_sort[t_idx2] * weight_above;
      out[i] = x1 + x2;
    }
  }
};

template<typename xpu>
void NumpyQuantileImpl(const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs,
                       const dmlc::optional<mxnet::Tuple<int>>& axis,
                       const int interpolation) {
  using namespace mxnet;
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);

  Stream<xpu> *s = ctx.get_stream<xpu>();

  const TBlob& a = inputs[0];
  const TBlob& q = inputs[1];
  const TBlob& r = outputs[0];

  auto small = NumpyReduceAxesShapeImpl(a.shape_, axis, false);

  TShape r_shape;
  r_shape = TShape(small.ndim()+1, 1);
  for (int i = 1; i < r_shape.ndim(); ++i) {
    r_shape[i] = small[i-1];
  }
  if (q.shape_.ndim() != 0) {
    r_shape[0] = q.shape_[0];
  }

  TShape axes;
  if (!axis.has_value()) {
    axes = TShape(a.shape_.ndim(), 1);
    for (int i = 0; i < a.shape_.ndim(); ++i) {
      axes[i] = i;
    }
  } else {
    auto axis_tuple = axis.value();
    axes = TShape(axis_tuple.ndim(), 1);
    for (int i = 0; i < axis_tuple.ndim(); ++i) {
      axes[i] = axis_tuple[i];
    }
  }

  TShape t_axes(a.shape_.ndim(), 1);
  int j = 0;
  for (int i = 0; i < t_axes.ndim(); ++i) {
    bool red = false;
    for (int k = 0; k < axes.ndim(); ++k) {
      if (axes[k] == i) {
        red = true;
      }
    }
    if (!red) {
      t_axes[j] = i;
      j++;
    }
  }
  for (int jj = j; jj < t_axes.ndim(); ++jj) {
    t_axes[jj] = axes[jj-j];
  }

  TShape t_shape(small.ndim()+1, 1);
  for (int i = 0; i < small.ndim(); ++i) {
    t_shape[i] = small[i];
  }
  size_t red_size = 1;
  for (int i = 0; i < axes.ndim(); ++i) {
    red_size *= a.shape_[axes[i]];
  }
  t_shape[t_shape.ndim()-1] = red_size;
  TShape t_shape_ex(a.shape_.ndim(), 1);
  for (int i = 0; i < small.ndim(); ++i) {
    t_shape_ex[i] = small[i];
  }
  for (int i = small.ndim(); i < a.shape_.ndim(); ++i) {
    t_shape_ex[i] = a.shape_[axes[i-small.ndim()]];
  }

  MSHADOW_TYPE_SWITCH(a.type_flag_, DType, {
    size_t temp_data_size = a.shape_.Size() * sizeof(DType);
    size_t idx_size = a.shape_.Size() * sizeof(index_t);
    size_t temp_mem_size = 2 * temp_data_size + idx_size;
    Tensor<xpu, 1, char> temp_mem =
        ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(temp_mem_size), s);
    DType* trans_ptr, *sort_ptr;
    index_t* idx_ptr;
    if (sizeof(DType) >= sizeof(index_t)) {
      trans_ptr = reinterpret_cast<DType*>(temp_mem.dptr_);
      sort_ptr = reinterpret_cast<DType*>(temp_mem.dptr_ + temp_data_size);
      idx_ptr = reinterpret_cast<index_t*>(temp_mem.dptr_ + 2 * temp_data_size);
    } else {
      idx_ptr = reinterpret_cast<index_t*>(temp_mem.dptr_);
      trans_ptr = reinterpret_cast<DType*>(temp_mem.dptr_ + idx_size);
      sort_ptr = reinterpret_cast<DType*>(temp_mem.dptr_ + temp_data_size + idx_size);
    }

    TBlob a_trans = TBlob(trans_ptr, t_shape_ex, xpu::kDevMask);

    TransposeImpl<xpu>(ctx.run_ctx, a, a_trans, t_axes);

    TopKParam topk_param;
    topk_param.axis = dmlc::optional<int>(-1);
    topk_param.is_ascend = true;
    topk_param.k = 0;
    topk_param.ret_typ = topk_enum::kReturnValue;

    TBlob a_sort = TBlob(sort_ptr, t_shape, xpu::kDevMask);
    TBlob a_idx = TBlob(idx_ptr, t_shape, xpu::kDevMask);
    TopKImpl<xpu, DType, index_t>(ctx.run_ctx,
                                  ctx.requested[1], {kWriteTo, kNullOp}, a_trans.reshape(t_shape),
                                  {a_sort, a_idx},
                                  topk_param);

    MSHADOW_TYPE_SWITCH(q.type_flag_, QType, {
      MXNET_NDIM_SWITCH(small.ndim()+1, NDim, {
        Kernel<quantile_take<NDim>, xpu>::Launch(
            s, r_shape.Size(), r.dptr<DType>(), q.dptr<QType>(), a_sort.dptr<DType>(),
            interpolation, t_shape.get<NDim>(), r_shape.get<NDim>());
      })
    })
  })
}

template<typename xpu>
void NumpyQuantileForward(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
  if (req[0] == kNullOp)
    return;
  auto &param = nnvm::get<NumpyQuantileParam>(attrs.parsed);
  NumpyQuantileImpl<xpu>(
      ctx, inputs, req, outputs,
      param.axis, param.interpolation);
}

template<typename xpu>
void NumpyBroadcastToForward(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  if (outputs[0].shape_.Size() == 0U) return;  // zero-size tensor
  TShape expanded_ishape(outputs[0].shape_.ndim(), 1);
  const TShape& ishape = inputs[0].shape_;
  CHECK_LE(ishape.ndim(), expanded_ishape.ndim()) << "output ndim cannot be less than input ndim";
  const int ndim_delta = expanded_ishape.ndim() - ishape.ndim();
  for (int i = 0; i < ishape.ndim(); ++i) {
    expanded_ishape[i + ndim_delta] = ishape[i];
  }
  BroadcastComputeImpl<xpu>(attrs, ctx, {inputs[0].reshape(expanded_ishape)},
                            req, outputs, expanded_ishape);
}

template<typename xpu>
void NumpyBroadcastToBackward(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<TBlob>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<TBlob>& outputs) {
  if (inputs[0].shape_.Size() == 0U) return;  // zero-size ograd
  TShape expanded_igrad_shape(inputs[0].shape_.ndim(), 1);
  const TShape& igrad_shape = outputs[0].shape_;
  CHECK_LE(igrad_shape.ndim(), expanded_igrad_shape.ndim())
      << "output ndim cannot be less than input ndim";
  const int ndim_delta = expanded_igrad_shape.ndim() - igrad_shape.ndim();
  for (int i = 0; i < igrad_shape.ndim(); ++i) {
    expanded_igrad_shape[i + ndim_delta] = igrad_shape[i];
  }
  if (NeedSafeAcc<true>(inputs[0].type_flag_, outputs[0].type_flag_)) {
    ReduceAxesComputeImpl<xpu, mshadow_op::sum, true>(
        ctx, inputs, req, {outputs[0].reshape(expanded_igrad_shape)}, expanded_igrad_shape);
  } else {
    ReduceAxesComputeImpl<xpu, mshadow_op::sum, false>(
        ctx, inputs, req, {outputs[0].reshape(expanded_igrad_shape)}, expanded_igrad_shape);
  }
}

template<typename xpu, typename OP>
void NumpyReduceAxesNoDTypeBackward(const nnvm::NodeAttrs& attrs,
                                    const OpContext& ctx,
                                    const std::vector<TBlob>& inputs,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  const NumpyReduceAxesNoDTypeParam& param = nnvm::get<NumpyReduceAxesNoDTypeParam>(attrs.parsed);
  TShape small;
  if (param.keepdims) {
    small = inputs[0].shape_;
  } else {
    small = NumpyReduceAxesShapeImpl(outputs[0].shape_, param.axis, true);
  }
  ReduceAxesBackwardUseInOutImpl<xpu, OP, false>(ctx, small, inputs, req, outputs);
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NUMPY_NP_BROADCAST_REDUCE_OP_H_
