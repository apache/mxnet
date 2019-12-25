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
#include "../tensor/elemwise_binary_broadcast_op.h"

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
      if ((axes[i] >= 0) && (ishape[axes[i]] == 0)) {
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

struct NumpyWeightedAverageParam : public dmlc::Parameter<NumpyWeightedAverageParam> {
  dmlc::optional<mxnet::Tuple<int>> axis;
  bool returned;
  bool weighted;

  DMLC_DECLARE_PARAMETER(NumpyWeightedAverageParam) {
    DMLC_DECLARE_FIELD(axis)
      .set_default(dmlc::optional<mxnet::Tuple<int>>())
      .describe("Axis or axes along which a average is performed. "
                "The default, axis=None, will average "
                "all of the elements of the input array. If axis is negative it counts from the "
                "last to the first axis.");
    DMLC_DECLARE_FIELD(returned)
      .set_default(false)
      .describe("If True, the tuple (average, sum_of_weights) is returned,"
                "otherwise only the average is returned."
                "If weights=None, sum_of_weights is equivalent to"
                "the number of elements over which the average is taken.");
    DMLC_DECLARE_FIELD(weighted)
      .set_default(true)
      .describe("Auxiliary flag to deal with none weights.");
  }
};

inline bool NumpyWeightedAverageShape(const nnvm::NodeAttrs& attrs,
                                      std::vector<TShape> *in_attrs,
                                      std::vector<TShape> *out_attrs) {
  const auto& param = nnvm::get<NumpyWeightedAverageParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), (param.weighted ? 2U : 1U));
  CHECK_EQ(out_attrs->size(), 2U);
  if (!shape_is_known(in_attrs->at(0))) {
    return false;
  }

  const TShape& a_shape = (*in_attrs)[0];
  SHAPE_ASSIGN_CHECK(*out_attrs, 0,
                     NumpyReduceAxesShapeImpl(a_shape, param.axis, false));

  if (param.weighted) {
    const TShape& w_shape = (*in_attrs)[1];
    if (w_shape.ndim() != a_shape.ndim()) {
      CHECK_EQ(w_shape.ndim(), 1U)
        << "1D weights expected when shapes of a and weights differ.";
      CHECK_EQ(param.axis.has_value(), true)
        << "Axis must be specified when shapes of a and weights differ.";
      mxnet::Tuple<int> axes(param.axis.value());
      CHECK_EQ(axes.ndim(), 1U) << "Axis must be int when shapes of a and weights differ.";
      int red_axis = axes[0] < 0 ? axes[0] + a_shape.ndim() : axes[0];
      CHECK_EQ(a_shape[red_axis], w_shape[0])
        << "Length of weights not compatible with specified axis.";
      SHAPE_ASSIGN_CHECK(*out_attrs, 1,
                         NumpyReduceAxesShapeImpl(
                           w_shape, dmlc::optional<mxnet::Tuple<int>>(), false));
    } else {
      for (int i = 0; i < w_shape.ndim(); i++) {
        CHECK_EQ(w_shape[i], a_shape[i]);
      }
      SHAPE_ASSIGN_CHECK(*out_attrs, 1,
                         NumpyReduceAxesShapeImpl(w_shape, param.axis, false));
    }
  } else {
    SHAPE_ASSIGN_CHECK(*out_attrs, 1, TShape(0, -1));
  }

  return shape_is_known(out_attrs->at(0)) && shape_is_known(out_attrs->at(1));
}

template<int req, int NDim, bool onedim = false>
struct avg_grad_a_kernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i,
                                  DType* out,
                                  const DType* w,
                                  const DType* scl,
                                  const DType* ograd,
                                  mshadow::Shape<NDim> small,
                                  mshadow::Shape<NDim> big) {
    // partial a = w / sum(w)
    size_t big_idx = i;
    size_t small_idx = i;
    size_t big_stride = 1;
    size_t small_stride = 1;
    size_t red_axis_idx = 0;
    for (int axis = NDim-1; axis >= 0; --axis) {
      size_t axis_idx = big_idx % big[axis];
      small_idx -= axis_idx * big_stride;
      if (small[axis] != 1) {
        small_idx += axis_idx * small_stride;
      } else if (onedim && small[axis] != big[axis]) {
        red_axis_idx = axis_idx;
      }
      big_idx /= big[axis];
      big_stride *= big[axis];
      small_stride *= small[axis];
    }
    if (onedim) {
      KERNEL_ASSIGN(out[i], req, (ograd[small_idx] * (w[red_axis_idx] / *scl)));
    } else {
      KERNEL_ASSIGN(out[i], req, (ograd[small_idx] * (w[i] / scl[small_idx])));
    }
  }
};

template<int req, int NDim>
struct avg_grad_w_kernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i,
                                  DType* out,
                                  const DType* a,
                                  const DType* scl,
                                  const DType* sum_of_wa,
                                  const DType* ograd,
                                  mshadow::Shape<NDim> small,
                                  mshadow::Shape<NDim> big) {
    // partial w = (a * sum(w) - sum(a*w)) / (sum(w) * sum(w))
    size_t big_idx = i;
    size_t small_idx = i;
    size_t big_stride = 1;
    size_t small_stride = 1;
    for (int axis = NDim-1; axis >= 0; --axis) {
      size_t axis_idx = big_idx % big[axis];
      small_idx -= axis_idx * big_stride;
      if (small[axis] != 1) {
        small_idx += axis_idx * small_stride;
      }
      big_idx /= big[axis];
      big_stride *= big[axis];
      small_stride *= small[axis];
    }
    DType ret = ograd[small_idx] *
      (((a[i] * scl[small_idx] - sum_of_wa[small_idx]) / scl[small_idx]) / scl[small_idx]);
    KERNEL_ASSIGN(out[i], req, ret);
  }
};

template<int req, int NDim>
struct avg_grad_w_1D_kernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i,
                                  DType* out,
                                  const DType* a,
                                  const DType* scl,
                                  const DType* sum_of_wa,
                                  const DType* ograd,
                                  mshadow::Shape<NDim> big,
                                  const int red_axis) {
    DType scl_val = *scl;
    size_t tail = 1;
    size_t head = 1;
    for (int axis = NDim-1; axis > red_axis; --axis) {
      tail *= big[axis];
    }
    for (int axis = 0; axis < red_axis; ++axis) {
      head *= big[axis];
    }
    DType ret = 0;
    for (size_t j = 0; j < head; ++j) {
      for (size_t k = 0; k < tail; ++k) {
        size_t a_idx = j*(tail*big[red_axis]) + i * tail + k;
        size_t small_idx = j*tail + k;
        ret += (ograd[small_idx] *
          (((a[a_idx] * scl_val - sum_of_wa[small_idx]) / scl_val) / scl_val));
      }
    }
    KERNEL_ASSIGN(out[i], req, ret);
  }
};

template<typename xpu, bool back = false>
void NumpyWeightedAverageComputeImpl(const nnvm::NodeAttrs& attrs,
                                     const OpContext& ctx,
                                     const std::vector<TBlob>& inputs,
                                     const std::vector<OpReqType>& req,
                                     const std::vector<TBlob>& outputs,
                                     const dmlc::optional<mxnet::Tuple<int>>& axis) {
  using namespace mshadow;
  using namespace mxnet_op;
  Stream<xpu>* s = ctx.get_stream<xpu>();
  const TBlob& data = inputs[0];
  TShape small1 = NumpyReduceAxesShapeImpl(data.shape_, axis, true);
  // Reshape weights
  TShape small2 = small1;
  TBlob weights = inputs[1];

  bool one_dim = weights.shape_.ndim() != data.shape_.ndim();

  int red_axis = -1;

  if (one_dim) {
    CHECK_EQ(weights.shape_.ndim(), 1U)
      << "1D weights expected when shapes of a and weights differ.";
    CHECK_EQ(axis.has_value(), true)
      << "Axis must be specified when shapes of a and weights differ.";
    Tuple<int> axes(axis.value());
    CHECK_EQ(axes.ndim(), 1U)
      << "Axis must be int when shapes of a and weights differ.";
    red_axis = axes[0] < 0 ? axes[0] + data.shape_.ndim() : axes[0];
    CHECK_EQ(weights.shape_[0], data.shape_[red_axis])
      << "Length of weights not compatible with specified axis.";
    TShape new_w_shape(data.shape_.ndim(), 1);
    new_w_shape[red_axis] = weights.shape_[0];
    weights = weights.reshape(new_w_shape);
    small2 = TShape(new_w_shape.ndim(), 1);
  }
  MSHADOW_TYPE_SWITCH(data.type_flag_, DType, {
    // Get temp space
    size_t temp_data_size = data.shape_.Size() * sizeof(DType);
    size_t temp_sum_size = small1.Size() * sizeof(DType);
    TShape src_shape, dst_shape;
    BroadcastReduceShapeCompact(data.shape_, small1, &src_shape, &dst_shape);
    size_t workspace_size = 0;
    MXNET_NDIM_SWITCH(dst_shape.ndim(), NDim, {
      workspace_size = broadcast::ReduceWorkspaceSize<NDim, DType>(
        s, dst_shape, {kWriteTo}, src_shape);
    });
    size_t temp_mem_size = temp_data_size + temp_sum_size + workspace_size;
    Tensor<xpu, 1, char> temp_mem =
    ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(temp_mem_size), s);
    DType *temp_data_ptr = reinterpret_cast<DType*>(temp_mem.dptr_);
    DType *temp_sum_ptr = reinterpret_cast<DType*>(temp_mem.dptr_ + temp_data_size);
    char *workspace_ptr = temp_mem.dptr_ + temp_data_size + temp_sum_size;
    Tensor<xpu, 1, char> workspace(workspace_ptr, Shape1(workspace_size), s);

    // Compute weighted data
    TBlob wa = TBlob(temp_data_ptr, data.shape_, xpu::kDevMask);
    BinaryBroadcastCompute<xpu, mshadow_op::mul>(
      attrs, ctx, {data, weights}, {kWriteTo}, {wa});

    // Compute sum of weighted data
    TBlob sum_of_wa = TBlob(temp_sum_ptr, small1, xpu::kDevMask);
    ReduceAxesComputeWithWorkspaceImpl<xpu, mshadow_op::sum, true>(
      ctx, {wa}, {kWriteTo}, {sum_of_wa}, workspace, src_shape, dst_shape);
    if (!back) {
      const TBlob& avg = outputs[0];
      const TBlob& sum_of_weights = outputs[1];
      TShape w_src_shape, w_dst_shape;
      BroadcastReduceShapeCompact(weights.shape_, small2, &w_src_shape, &w_dst_shape);
      // Compute sum of weight
      TBlob scl = sum_of_weights.reshape(small2);
      ReduceAxesComputeWithWorkspaceImpl<xpu, mshadow_op::sum, true>(
        ctx, {weights}, {kWriteTo}, {scl}, workspace, w_src_shape, w_dst_shape);

      // Compute avg and assign output
      BinaryBroadcastCompute<xpu, mshadow_op::div>(
        attrs, ctx, {sum_of_wa, scl}, req, {avg.reshape(small1)});
    } else {
      // Compute and assign the derivatives of a and weights
      const TBlob& igrad_a = outputs[0];
      const TBlob& igrad_w = outputs[1];
      const TBlob& scl = inputs[2];
      const TBlob& ograd = inputs[3];
      MXNET_NDIM_SWITCH(igrad_a.shape_.ndim(), NDim, {
        MXNET_ASSIGN_REQ_SWITCH(req[0], req_a, {
          if (one_dim) {
            // 1D weights
            Kernel<avg_grad_a_kernel<req_a, NDim, true>, xpu>::Launch(
                s, igrad_a.shape_.Size(), igrad_a.dptr<DType>(),
                weights.dptr<DType>(), scl.dptr<DType>(), ograd.dptr<DType>(),
                small1.get<NDim>(),
                igrad_a.shape_.get<NDim>());
          } else {
            Kernel<avg_grad_a_kernel<req_a, NDim, false>, xpu>::Launch(
                s, igrad_a.shape_.Size(), igrad_a.dptr<DType>(),
                weights.dptr<DType>(), scl.dptr<DType>(), ograd.dptr<DType>(),
                small1.get<NDim>(),
                igrad_a.shape_.get<NDim>());
          }
        });
        MXNET_ASSIGN_REQ_SWITCH(req[1], req_w, {
          if (one_dim) {
            Kernel<avg_grad_w_1D_kernel<req_w, NDim>, xpu>::Launch(
                s, igrad_w.shape_.Size(), igrad_w.dptr<DType>(),
                data.dptr<DType>(), scl.dptr<DType>(), sum_of_wa.dptr<DType>(), ograd.dptr<DType>(),
                data.shape_.get<NDim>(),
                red_axis);
          } else {
            Kernel<avg_grad_w_kernel<req_w, NDim>, xpu>::Launch(
                s, igrad_w.shape_.Size(), igrad_w.dptr<DType>(),
                data.dptr<DType>(), scl.dptr<DType>(), sum_of_wa.dptr<DType>(), ograd.dptr<DType>(),
                small1.get<NDim>(),
                igrad_w.shape_.get<NDim>());
          }
        });
      })
    }
  });
}

template<typename xpu>
void NumpyWeightedAverageForward(const nnvm::NodeAttrs& attrs,
                                 const OpContext& ctx,
                                 const std::vector<TBlob>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  if (req[0] == kNullOp) return;
  CHECK_NE(req[0], kWriteInplace) << "Average does not support write in-place";
  const auto& param = nnvm::get<NumpyWeightedAverageParam>(attrs.parsed);
  const TBlob& data = inputs[0];
  MSHADOW_TYPE_SWITCH(data.type_flag_, DType, {
    if (!param.weighted) {
      TShape small = NumpyReduceAxesShapeImpl(data.shape_, param.axis, true);
      // Compute sum of weights which equals to the product of sizes of reduced axes
      Stream<xpu>* s = ctx.get_stream<xpu>();
      auto ret = outputs[1].FlatTo1D<xpu, DType>(s);
      ret = scalar<DType>(data.shape_.Size()/small.Size());
      // Compute mean
      ReduceAxesComputeImpl<xpu, mshadow_op::sum, true, true>(
        ctx, inputs, req, {outputs[0]}, small);
    } else {
      NumpyWeightedAverageComputeImpl<xpu>(
        attrs, ctx, inputs, req, outputs, param.axis);
    }
  });
}

template<typename xpu>
void NumpyWeightedAverageBackward(const nnvm::NodeAttrs& attrs,
                                  const OpContext& ctx,
                                  const std::vector<TBlob>& inputs,
                                  const std::vector<OpReqType>& req,
                                  const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  const auto& param = nnvm::get<NumpyWeightedAverageParam>(attrs.parsed);
  if (req[0] == kNullOp && !param.weighted) return;
  CHECK_EQ(inputs.size(), (param.weighted ? 6U : 5U));
  CHECK_EQ(outputs.size(), (param.weighted ? 2U : 1U));
  const TBlob& ograd = inputs[0];
  const TBlob& data = inputs[2];
  MSHADOW_TYPE_SWITCH(data.type_flag_, DType, {
    if (!param.weighted) {
      TShape small = NumpyReduceAxesShapeImpl(outputs[0].shape_, param.axis, true);
      Stream<xpu>* s = ctx.get_stream<xpu>();
      auto ograd_tensor = ograd.FlatTo1D<xpu, DType>(s);
      ograd_tensor /= scalar<DType>(data.shape_.Size()/small.Size());
      BroadcastComputeImpl<xpu>(attrs, ctx, {ograd}, req, {outputs[0]}, small);
    } else {
      const TBlob& weights = inputs[3];
      const TBlob& scl = inputs[5];
      NumpyWeightedAverageComputeImpl<xpu, true>(
        attrs, ctx, {data, weights, scl, ograd}, req, outputs, param.axis);
    }
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
