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
 * \file np_matrix_op-inl.h
 * \brief Function definition of matrix related operators
 */
#ifndef MXNET_OPERATOR_NUMPY_NP_MATRIX_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_MATRIX_OP_INL_H_

#include <vector>
#include <algorithm>
#include <string>
#include <set>
#include <utility>
#include <unordered_map>
#include "../tensor/matrix_op-inl.h"
#include "../nn/concat-inl.h"
#include "../../common/utils.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../tensor/broadcast_reduce_op.h"

namespace mxnet {
namespace op {

struct NumpyTransposeParam : public dmlc::Parameter<NumpyTransposeParam> {
  mxnet::TShape axes;
  DMLC_DECLARE_PARAMETER(NumpyTransposeParam) {
    DMLC_DECLARE_FIELD(axes)
        .set_default(mxnet::TShape(-1, 0))
        .describe(
            "By default, reverse the dimensions, otherwise permute "
            "the axes according to the values given.");
  }

  bool operator==(const NumpyTransposeParam& other) const {
    return this->axes == other.axes;
  }

  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream axes_s;
    axes_s << axes;
    (*dict)["axes"] = axes_s.str();
  }
};

struct NumpyVstackParam : public dmlc::Parameter<NumpyVstackParam> {
  int num_args;
  DMLC_DECLARE_PARAMETER(NumpyVstackParam) {
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(1).describe("Number of inputs to be vstacked.");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream num_args_s;
    num_args_s << num_args;
    (*dict)["num_args"] = num_args_s.str();
  }
};

struct NumpyColumnStackParam : public dmlc::Parameter<NumpyColumnStackParam> {
  int num_args;
  DMLC_DECLARE_PARAMETER(NumpyColumnStackParam) {
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(1).describe(
        "Number of inputs to be column stacked");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream ss;
    ss << num_args;
    (*dict)["num_args"] = ss.str();
  }
};

struct NumpyReshapeParam : public dmlc::Parameter<NumpyReshapeParam> {
  mxnet::TShape newshape;
  std::string order;
  DMLC_DECLARE_PARAMETER(NumpyReshapeParam) {
    DMLC_DECLARE_FIELD(newshape).describe(
        "The new shape should be compatible with the original shape."
        " If an integer, then the result will be a 1-D array of that length."
        " One shape dimension can be -1. In this case, the value is inferred"
        " from the length of the array and remaining dimensions.");
    DMLC_DECLARE_FIELD(order).set_default("C").describe(
        "Read the elements of a using this index order, and place the elements into"
        " the reshaped array using this index order. 'C' means to read/write the elements"
        " using C-like index order, with the last axis index changing fastest,"
        " back to the first axis index changing slowest."
        " Note that currently only C-like order is"
        " supported");
  }
};

struct NumpyXReshapeParam : public dmlc::Parameter<NumpyXReshapeParam> {
  mxnet::TShape newshape;
  bool reverse;
  std::string order;
  DMLC_DECLARE_PARAMETER(NumpyXReshapeParam) {
    DMLC_DECLARE_FIELD(newshape).describe(
        "The new shape should be compatible with the original shape."
        " If an integer, then the result will be a 1-D array of that length."
        " One shape dimension can be -1. In this case, the value is inferred"
        " from the length of the array and remaining dimensions."
        " -2 to -6 are used for data manipulation."
        " -2 copy this dimension from the input to the output shape."
        " -3 will skip current dimension if and only if the current dim size is one."
        " -4 copy all remain of the input dimensions to the output shape."
        " -5 use the product of two consecutive dimensions of the input"
        " shape as the output."
        " -6 split one dimension of the input into two dimensions passed"
        " subsequent to -6 in the new shape.");
    DMLC_DECLARE_FIELD(reverse).set_default(false).describe(
        "If true then the special values are inferred from right to left");
    DMLC_DECLARE_FIELD(order).set_default("C").describe(
        "Read the elements of a using this index order, and place the elements into"
        " the reshaped array using this index order. 'C' means to read/write the elements"
        " using C-like index order, with the last axis index changing fastest,"
        " back to the first axis index changing slowest."
        " Note that currently only C-like order is"
        " supported");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream newshape_s, reverse_s, order_s;
    newshape_s << newshape;
    reverse_s << reverse;
    order_s << order;
    (*dict)["newshape"] = newshape_s.str();
    (*dict)["reverse"]  = reverse_s.str();
    (*dict)["order"]    = order_s.str();
  }
};

bool NumpyXReshapeShape(const nnvm::NodeAttrs& attrs,
                        mxnet::ShapeVector* in_attrs,
                        mxnet::ShapeVector* out_attrs);

template <typename xpu>
void NumpyTranspose(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  if (req[0] == kNullOp)
    return;
  CHECK(req[0] == kWriteTo || req[0] == kAddTo) << "Transpose does not support inplace";
  const NumpyTransposeParam& param = nnvm::get<NumpyTransposeParam>(attrs.parsed);
  mxnet::TShape axes;
  if (ndim_is_known(param.axes)) {
    axes = common::CanonicalizeAxes(param.axes);
  } else {
    axes = mxnet::TShape(inputs[0].ndim(), -1);
    for (int i = 0; i < axes.ndim(); ++i) {
      axes[i] = axes.ndim() - 1 - i;
    }
  }
  mshadow::Tensor<xpu, 1, dim_t> workspace = GetTransposeExWorkspace<xpu>(ctx, axes);
  if (req[0] == kAddTo) {
    TransposeExImpl<xpu, true>(ctx.run_ctx, inputs[0], outputs[0], axes, workspace);
  } else {
    TransposeExImpl<xpu, false>(ctx.run_ctx, inputs[0], outputs[0], axes, workspace);
  }
}

inline bool NumpyTransposeShape(const nnvm::NodeAttrs& attrs,
                                mxnet::ShapeVector* in_attrs,
                                mxnet::ShapeVector* out_attrs) {
  const NumpyTransposeParam& param = nnvm::get<NumpyTransposeParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  mxnet::TShape& shp     = (*in_attrs)[0];
  mxnet::TShape& out_shp = (*out_attrs)[0];

  int ndim = -1;
  if (ndim_is_known(shp)) {
    ndim = shp.ndim();
  } else if (ndim_is_known(out_shp)) {
    ndim = out_shp.ndim();
  }
  if (ndim < 0) {
    return false;
  }
  if (out_shp.ndim() >= 0 && shp.ndim() >= 0) {
    CHECK_EQ(out_shp.ndim(), shp.ndim());
  }

  mxnet::TShape get(ndim, -1);
  mxnet::TShape ret(ndim, -1);

  if (ndim_is_known(param.axes)) {
    CHECK_EQ(ndim, param.axes.ndim())
        << "The number of axes does not match the dimension of the tensor. axes = " << param.axes
        << ", input tensor shape = " << shp;
    mxnet::TShape axes = common::CanonicalizeAxes(param.axes);
    std::set<dim_t> axes_set(axes.begin(), axes.end());
    CHECK_EQ(axes_set.size(), axes.ndim()) << "ValueError: Repeated axis in transpose."
                                           << " param.axes = " << param.axes;
    if (ndim_is_known(shp)) {
      for (int i = 0; i < ndim; ++i) {
        ret[i] = shp[axes[i]];
      }
    }
    if (ndim_is_known(out_shp)) {
      for (int i = 0; i < ndim; ++i) {
        get[axes[i]] = out_shp[i];
      }
    }
  } else {
    if (ndim_is_known(shp)) {
      for (int i = 0; i < ndim; ++i) {
        ret[i] = shp[ndim - 1 - i];
      }
    }
    if (ndim_is_known(out_shp)) {
      for (int i = 0; i < ndim; ++i) {
        get[ndim - 1 - i] = out_shp[i];
      }
    }
  }
  SHAPE_ASSIGN_CHECK(*in_attrs, 0, get);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, ret);
  return shape_is_known(*in_attrs) && shape_is_known(*out_attrs);
}

template <typename xpu>
void NumpyColumnStackForward(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow_op;

  const NumpyColumnStackParam& param = nnvm::get<NumpyColumnStackParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), param.num_args);
  CHECK_EQ(outputs.size(), 1);
  CHECK_EQ(req.size(), 1);

  // reshape if necessary
  std::vector<TBlob> data(param.num_args);
  for (int i = 0; i < param.num_args; i++) {
    if (inputs[i].shape_.ndim() == 0 || inputs[i].shape_.ndim() == 1) {
      TShape shape = Shape2(inputs[i].shape_.Size(), 1);
      data[i]      = inputs[i].reshape(shape);
    } else {
      data[i] = inputs[i];
    }
  }

  // initialize ConcatOp
  ConcatParam cparam;
  cparam.num_args = param.num_args;
  cparam.dim      = 1;
  MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    ConcatOp<xpu, DType> op;
    op.Init(cparam);
    op.Forward(ctx, data, req, outputs);
  });
}

template <typename xpu>
void NumpyColumnStackBackward(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<TBlob>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow_op;

  const NumpyColumnStackParam& param = nnvm::get<NumpyColumnStackParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(outputs.size(), param.num_args);
  CHECK_EQ(req.size(), param.num_args);

  // reshape if necessary
  std::vector<TBlob> data(param.num_args);
  for (int i = 0; i < param.num_args; i++) {
    if (outputs[i].shape_.ndim() == 0 || outputs[i].shape_.ndim() == 1) {
      TShape shape = Shape2(outputs[i].shape_.Size(), 1);
      data[i]      = outputs[i].reshape(shape);
    } else {
      data[i] = outputs[i];
    }
  }

  // initialize ConcatOp
  ConcatParam cparam;
  cparam.num_args = param.num_args;
  cparam.dim      = 1;
  MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    ConcatOp<xpu, DType> op;
    op.Init(cparam);
    op.Backward(ctx, inputs[0], req, data);
  });
}

template <typename xpu>
void NumpyVstackForward(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow_op;

  const NumpyVstackParam& param = nnvm::get<NumpyVstackParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), param.num_args);
  CHECK_EQ(outputs.size(), 1);
  CHECK_EQ(req.size(), 1);

  // reshape if necessary
  std::vector<TBlob> data(param.num_args);
  for (int i = 0; i < param.num_args; i++) {
    if (inputs[i].shape_.ndim() == 0 || inputs[i].shape_.ndim() == 1) {
      TShape shape = Shape2(1, inputs[i].shape_.Size());
      data[i]      = inputs[i].reshape(shape);
    } else {
      data[i] = inputs[i];
    }
  }

  // initialize ConcatOp
  ConcatParam cparam;
  cparam.num_args = param.num_args;
  cparam.dim      = 0;
  MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    ConcatOp<xpu, DType> op;
    op.Init(cparam);
    op.Forward(ctx, data, req, outputs);
  });
}

template <typename xpu>
void NumpyVstackBackward(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow_op;

  const NumpyVstackParam& param = nnvm::get<NumpyVstackParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(outputs.size(), param.num_args);
  CHECK_EQ(req.size(), param.num_args);

  // reshape if necessary
  std::vector<TBlob> data(param.num_args);
  for (int i = 0; i < param.num_args; i++) {
    if (outputs[i].shape_.ndim() == 0 || outputs[i].shape_.ndim() == 1) {
      TShape shape = Shape2(1, outputs[i].shape_.Size());
      data[i]      = outputs[i].reshape(shape);
    } else {
      data[i] = outputs[i];
    }
  }

  // initialize ConcatOp
  ConcatParam cparam;
  cparam.num_args = param.num_args;
  cparam.dim      = 0;
  MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    ConcatOp<xpu, DType> op;
    op.Init(cparam);
    op.Backward(ctx, inputs[0], req, data);
  });
}

struct NumpyTrilindicesParam : public dmlc::Parameter<NumpyTrilindicesParam> {
  index_t n;
  index_t k;
  index_t m;
  DMLC_DECLARE_PARAMETER(NumpyTrilindicesParam) {
    DMLC_DECLARE_FIELD(n).describe(
        "The row dimension of the arrays for which"
        "the returned indices will be valid.");
    DMLC_DECLARE_FIELD(k).set_default(0).describe("Diagonal offset");
    DMLC_DECLARE_FIELD(m).describe(
        "The column dimension of the arrays for "
        "which the returned arrays will be valid."
        "By default m is taken equal to n.");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream n_s, k_s, m_s;
    n_s << n;
    k_s << k;
    m_s << m;
    (*dict)["n"] = n_s.str();
    (*dict)["k"] = k_s.str();
    (*dict)["m"] = m_s.str();
  }
};

template <int req>
struct TrilindicesOpForwardImpl {
  template <typename DType>
  MSHADOW_XINLINE static void Map(index_t i,
                                  DType* out_data0,
                                  DType* out_data1,
                                  index_t* data,
                                  index_t length) {
    KERNEL_ASSIGN(out_data0[i], req, data[i]);
    KERNEL_ASSIGN(out_data1[i], req, data[i + length]);
  }
};

template <typename xpu>
void TrilindicesOpForward(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 0U);
  CHECK_EQ(outputs.size(), 2U);
  Stream<xpu>* s                     = ctx.get_stream<xpu>();
  const NumpyTrilindicesParam& param = nnvm::get<NumpyTrilindicesParam>(attrs.parsed);

  const TBlob& out_data0 = outputs[0];
  const TBlob& out_data1 = outputs[1];

  CHECK_EQ(out_data0.shape_[0], out_data1.shape_[0]);
  index_t length = out_data0.shape_[0];

  std::vector<index_t> indices_cpu(2 * length, 0);
  size_t total_temp_size = 2 * length * sizeof(index_t);
  Tensor<xpu, 1, char> temp_space =
      ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(total_temp_size), s);
  index_t* indices = reinterpret_cast<index_t*>(temp_space.dptr_);

  index_t n = param.n;
  index_t m = param.m;
  index_t k = param.k;

  index_t end = k;
  index_t idx = 0;
  for (index_t i = 0; i < n; i++) {
    for (index_t j = 0; j <= std::min(end, m - 1); j++) {
      indices_cpu[idx]          = i;
      indices_cpu[idx + length] = j;
      idx++;
    }
    end++;
  }

  if (ctx.run_ctx.ctx.dev_mask() == gpu::kDevMask) {
#if MXNET_USE_CUDA
    cudaMemcpyAsync(indices,
                    indices_cpu.data(),
                    indices_cpu.size() * sizeof(index_t),
                    cudaMemcpyHostToDevice,
                    Stream<gpu>::GetStream(ctx.get_stream<gpu>()));
#else
    LOG(FATAL) << "Illegal attempt to use GPU in a CPU-only build";
#endif
  } else {
    std::memcpy(indices, indices_cpu.data(), indices_cpu.size() * sizeof(index_t));
  }

  MSHADOW_IDX_TYPE_SWITCH(out_data0.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<TrilindicesOpForwardImpl<req_type>, xpu>::Launch(
          s, length, out_data0.dptr<DType>(), out_data1.dptr<DType>(), indices, length);
    });
  });
}

struct NumpyRollParam : public dmlc::Parameter<NumpyRollParam> {
  dmlc::optional<mxnet::TShape> shift;
  dmlc::optional<mxnet::TShape> axis;
  DMLC_DECLARE_PARAMETER(NumpyRollParam) {
    DMLC_DECLARE_FIELD(shift)
        .set_default(dmlc::optional<mxnet::TShape>())
        .describe(
            "The number of places by which elements are shifted. If a tuple,"
            "then axis must be a tuple of the same size, and each of the given axes is shifted"
            "by the corresponding number. If an int while axis is a tuple of ints, "
            "then the same value is used for all given axes.");
    DMLC_DECLARE_FIELD(axis)
        .set_default(dmlc::optional<mxnet::TShape>())
        .describe(
            "Axis or axes along which elements are shifted. By default, the array is flattened"
            "before shifting, after which the original shape is restored.");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream shift_s, axis_s;
    shift_s << shift;
    axis_s << axis;
    (*dict)["shift"] = shift_s.str();
    (*dict)["axis"]  = axis_s.str();
  }
};

template <int req>
struct RollAxisNone_forward {
  template <typename DType>
  MSHADOW_XINLINE static void Map(index_t i,
                                  DType* out_data,
                                  const DType* in_data,
                                  const index_t size,
                                  const index_t shift) {
    index_t new_index = i - shift < 0 ? i - shift + size : i - shift;
    KERNEL_ASSIGN(out_data[i], req, in_data[new_index]);
  }
};

template <int req>
struct RollAxis_forward {
  template <typename DType>
  MSHADOW_XINLINE static void Map(index_t i,
                                  DType* out_data,
                                  const DType* in_data,
                                  const size_t* new_index) {
    KERNEL_ASSIGN(out_data[i], req, in_data[new_index[i]]);
  }
};

inline void RollDfs(const std::vector<std::vector<size_t>>& new_axes,
                    const std::vector<size_t>& value,
                    std::vector<size_t>* new_index,
                    index_t index,
                    int ndim,
                    index_t mid) {
  for (int a : new_axes[index]) {
    if (index == ndim - 1) {
      std::vector<size_t>& out = (*new_index);
      out.push_back(mid + a);
    } else {
      mid += a * value[ndim - 1 - index];
      RollDfs(new_axes, value, new_index, index + 1, ndim, mid);
      mid -= a * value[ndim - 1 - index];
    }
  }
}

template <typename xpu>
void NumpyRollCompute(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  if (inputs[0].Size() == 0U)
    return;
  const NumpyRollParam& param = nnvm::get<NumpyRollParam>(attrs.parsed);
  const int ndim(inputs[0].shape_.ndim());
  Stream<xpu>* s = ctx.get_stream<xpu>();
  std::vector<index_t> shifts(ndim, 0);
  index_t input_size = inputs[0].Size();
  if (!param.axis.has_value()) {
    index_t shift = param.shift.value()[0];
    shift         = shift % input_size;
    if (shift < 0) {
      shift += inputs[0].shape_.Size();
    }
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        Kernel<RollAxisNone_forward<req_type>, xpu>::Launch(s,
                                                            outputs[0].Size(),
                                                            outputs[0].dptr<DType>(),
                                                            inputs[0].dptr<DType>(),
                                                            inputs[0].Size(),
                                                            shift);
      });
    });
  } else {
    mxnet::TShape axes(param.axis.value());
    for (int i = 0; i < axes.ndim(); ++i) {
      if (axes[i] < 0) {
        axes[i] += ndim;
      }
    }
    for (int i = 0; i < axes.ndim(); ++i) {
      CHECK_LT(axes[i], ndim) << "axis " << axes[i] << " Exceeds input dimensions "
                              << inputs[0].shape_;
      CHECK_GE(axes[0], 0) << "Reduction axis " << param.axis.value()
                           << " Exceeds input dimensions " << inputs[0].shape_;
    }
    if (param.shift.value().ndim() == 1) {
      for (int i = 0; i < axes.ndim(); ++i) {
        shifts[axes[i]] = param.shift.value()[0];
      }
    } else {
      if (param.shift.value().ndim() != axes.ndim()) {
        LOG(FATAL) << "shift and `axis` must be a tuple of the same size,";
      }
      for (int i = 0; i < axes.ndim(); ++i) {
        shifts[axes[i]] = param.shift.value()[i];
      }
    }
    // keep shift in a legal range
    for (int i = 0; i < ndim; ++i) {
      index_t trans_shift = shifts[i] % inputs[0].shape_[i];
      if (trans_shift < 0) {
        trans_shift = shifts[i] + inputs[0].shape_[i];
      }
      shifts[i] = trans_shift;
    }
    // the result of new axis after shift.
    std::vector<std::vector<size_t>> new_axes;
    std::vector<size_t> new_index;
    std::vector<size_t> temp;
    std::vector<size_t> value(ndim, 0);
    index_t mid_val = 1;
    for (int i = 0; i < ndim; ++i) {
      if (shifts[i] != 0) {
        for (index_t j = 0; j < inputs[0].shape_[i]; ++j) {
          index_t new_axis = (j + inputs[0].shape_[i] - shifts[i]) % inputs[0].shape_[i];
          temp.push_back(new_axis);
        }
      } else {
        for (index_t j = 0; j < inputs[0].shape_[i]; ++j) {
          temp.push_back(j);
        }
      }
      new_axes.push_back(temp);
      temp.clear();
      value[i] = mid_val;
      mid_val *= inputs[0].shape_[ndim - 1 - i];
    }
    RollDfs(new_axes, value, &new_index, 0, ndim, 0);
    size_t workspace_size = new_index.size() * sizeof(size_t);
    Tensor<xpu, 1, char> workspace =
        ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);
    Tensor<cpu, 1, size_t> index_cpu_tensor(new_index.data(), Shape1(new_index.size()));
    Tensor<xpu, 1, size_t> index_xpu_tensor(reinterpret_cast<size_t*>(workspace.dptr_),
                                            Shape1(new_index.size()));
    mshadow::Copy(index_xpu_tensor, index_cpu_tensor, s);
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        Kernel<RollAxis_forward<req_type>, xpu>::Launch(s,
                                                        outputs[0].Size(),
                                                        outputs[0].dptr<DType>(),
                                                        inputs[0].dptr<DType>(),
                                                        index_xpu_tensor.dptr_);
      });
    });
  }
}

struct FlipParam : public dmlc::Parameter<FlipParam> {
  mxnet::Tuple<int> axis;
  DMLC_DECLARE_PARAMETER(FlipParam) {
    DMLC_DECLARE_FIELD(axis).describe("The axis which to flip elements.");
  }

  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream axis_s;
    axis_s << axis;
    (*dict)["axis"] = axis_s.str();
  }
};

#define FLIP_MAX_DIM 10
#define FLIP_MIN_DIM -1

template <typename xpu>
void NumpyFlipForwardImpl(const OpContext& ctx,
                          const std::vector<TBlob>& inputs,
                          const std::vector<TBlob>& outputs,
                          const std::vector<index_t>& stride_,
                          const std::vector<index_t>& trailing_,
                          const index_t& flip_index);

template <typename xpu>
void NumpyFlipForward(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  const FlipParam& param = nnvm::get<FlipParam>(attrs.parsed);
  mxnet::Tuple<int> axistemp;
  CHECK_EQ(inputs[0].type_flag_, outputs[0].type_flag_);
  CHECK_LT(param.axis.ndim(), FLIP_MAX_DIM);
  CHECK_GE(param.axis.ndim(), FLIP_MIN_DIM);
  if (param.axis.ndim() == FLIP_MIN_DIM) {
    if (inputs[0].shape_.ndim() == 0) {
      mshadow::Stream<xpu>* s = ctx.get_stream<xpu>();
      MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
        mshadow::Copy(outputs[0].FlatTo1D<xpu, DType>(s), inputs[0].FlatTo1D<xpu, DType>(s), s);
      });
      return;
    }
    std::vector<int> temp;
    for (int i = 0; i < inputs[0].shape_.ndim(); i++) {
      temp.push_back(i);
    }
    axistemp.assign(temp.begin(), temp.end());
  } else {
    axistemp = param.axis;
  }

  const mxnet::TShape& ishape = inputs[0].shape_;
  if (ishape.ProdShape(0, ishape.ndim()) == 0) {
    return;  // zero shape
  }
  std::vector<index_t> stride_(axistemp.ndim());
  std::vector<index_t> trailing_(axistemp.ndim());
  index_t flip_index = 0;
  for (int axis : axistemp) {
    CHECK_LT(axis, ishape.ndim()) << "axis " << axis << " is out of bounds for array of dimension "
                                  << ishape.ndim() << std::endl;
    CHECK_GE(axis, -ishape.ndim()) << "axis" << axis << " is out of bounds for array of dimension "
                                   << ishape.ndim() << std::endl;
    axis                  = (axis < 0) ? axis + ishape.ndim() : axis;
    stride_[flip_index]   = ishape[axis];
    trailing_[flip_index] = 1;
    for (int i2 = axis + 1; i2 < ishape.ndim(); ++i2) {
      trailing_[flip_index] *= ishape[i2];
    }
    flip_index++;
  }
  NumpyFlipForwardImpl<xpu>(ctx, inputs, outputs, stride_, trailing_, flip_index);
}

struct NumpyRollaxisParam : public dmlc::Parameter<NumpyRollaxisParam> {
  int axis;
  int start;
  DMLC_DECLARE_PARAMETER(NumpyRollaxisParam) {
    DMLC_DECLARE_FIELD(axis).describe(
        "The axis to roll backwards. "
        "The positions of the other axes do not change relative to one another.");
    DMLC_DECLARE_FIELD(start).set_default(0).describe(
        "The axis is rolled until it lies before this position. "
        "The default, 0, results in a “complete” roll.");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream axis_s, start_s;
    axis_s << axis;
    start_s << start;
    (*dict)["axis"]  = axis_s.str();
    (*dict)["start"] = start_s.str();
  }
};

inline mxnet::TShape NumpyRollaxisShapeImpl(int axis, int start, const int& ndim) {
  mxnet::TShape axes(ndim, -1);
  if (axis < 0) {
    axis += ndim;
  }
  if (start < 0) {
    start += ndim;
  }
  if (axis < start) {
    axes[start - 1] = axis;
  } else {
    axes[start] = axis;
  }
  int new_axis = 0;
  for (int i = 0; i < axes.ndim(); i++) {
    if (axes[i] < 0) {
      if (new_axis == axis) {
        new_axis++;
      }
      axes[i] = new_axis++;
    }
  }
  return axes;
}

struct NumpyMoveaxisParam : public dmlc::Parameter<NumpyMoveaxisParam> {
  mxnet::TShape source;
  mxnet::TShape destination;
  DMLC_DECLARE_PARAMETER(NumpyMoveaxisParam) {
    DMLC_DECLARE_FIELD(source).describe(
        "Original positions of the axes to move. These must be unique.");
    DMLC_DECLARE_FIELD(destination)
        .describe(
            "Destination positions for each of the original axes. "
            "These must also be unique.");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream source_s, destination_s;
    source_s << source;
    destination_s << destination;
    (*dict)["source"]      = source_s.str();
    (*dict)["destination"] = destination_s.str();
  }
};

inline mxnet::TShape NumpyMoveaxisShapeImpl(const nnvm::NodeAttrs& attrs, const int& ndim) {
  const NumpyMoveaxisParam& param = nnvm::get<NumpyMoveaxisParam>(attrs.parsed);
  mxnet::TShape axes(ndim, -1);
  std::vector<bool> state_axes(ndim, false);
  mxnet::TShape real_src(param.source.ndim(), -1);
  mxnet::TShape real_des(param.destination.ndim(), -1);
  for (int i = 0; i < param.source.ndim(); ++i) {
    if (param.source[i] >= 0) {
      CHECK_LT(static_cast<size_t>(param.source[i]), ndim);
      real_src[i] = param.source[i];
    } else {
      CHECK_LT(param.source[i] + ndim, ndim);
      real_src[i] = param.source[i] + ndim;
    }
    if (param.destination[i] >= 0) {
      CHECK_LT(static_cast<size_t>(param.destination[i]), ndim);
      real_des[i] = param.destination[i];
    } else {
      CHECK_LT(param.destination[i] + ndim, ndim);
      real_des[i] = param.destination[i] + ndim;
    }
  }
  if (ndim > 1) {
    for (int i = 0; i < param.source.ndim() - 1; ++i) {
      for (int j = i + 1; j < param.source.ndim(); ++j) {
        CHECK_NE(real_src[i], real_src[j]) << "repeated axis in `source` argument";
        CHECK_NE(real_des[i], real_des[j]) << "repeated axis in `destination` argument";
      }
    }
  }
  for (int i = 0; i < param.source.ndim(); ++i) {
    axes[real_des[i]]       = real_src[i];
    state_axes[real_src[i]] = true;
  }
  for (int i = 0; i < axes.ndim(); ++i) {
    if (axes[i] < 0) {
      for (int j = 0; j < axes.ndim(); ++j) {
        if (state_axes[j] == false) {
          axes[i]       = j;
          state_axes[j] = true;
          break;
        }
      }
    }
  }
  return axes;
}

template <typename xpu>
void NumpyMoveaxisCompute(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  const NumpyMoveaxisParam& param = nnvm::get<NumpyMoveaxisParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req[0], kWriteTo) << "Moveaxis does not support inplace";
  CHECK_EQ(param.source.ndim(), param.destination.ndim()) << "source and destination not equal.";
  mxnet::TShape axes;
  axes = NumpyMoveaxisShapeImpl(attrs, inputs[0].ndim());
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, Dtype, {
    TransposeImpl<xpu>(ctx.run_ctx, inputs[0], outputs[0], axes);
  })
}

template <typename xpu>
void NumpyRollaxisCompute(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  if (req[0] == kNullOp)
    return;
  CHECK(req[0] == kWriteTo || req[0] == kAddTo) << "Rollaxis does not support inplace";
  const NumpyRollaxisParam& param = nnvm::get<NumpyRollaxisParam>(attrs.parsed);
  mxnet::TShape axes = NumpyRollaxisShapeImpl(param.axis, param.start, inputs[0].ndim());

  mshadow::Tensor<xpu, 1, dim_t> workspace = GetTransposeExWorkspace<xpu>(ctx, axes);
  if (req[0] == kAddTo) {
    TransposeExImpl<xpu, true>(ctx.run_ctx, inputs[0], outputs[0], axes, workspace);
  } else {
    TransposeExImpl<xpu, false>(ctx.run_ctx, inputs[0], outputs[0], axes, workspace);
  }
}

template <typename xpu>
void NumpyRollaxisBackward(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  if (req[0] == kNullOp)
    return;
  CHECK(req[0] == kWriteTo || req[0] == kAddTo) << "Rollaxis Backward does not support inplace";
  const NumpyRollaxisParam& param = nnvm::get<NumpyRollaxisParam>(attrs.parsed);
  int axis_origin                 = param.axis;
  int start_origin                = param.start;
  int ndim                        = inputs[0].ndim();

  int axis;
  int start;

  if (axis_origin < 0) {
    axis_origin += ndim;
  }

  if (start_origin < 0) {
    start_origin += ndim;
  }

  if (axis_origin < start_origin) {
    axis  = start_origin - 1;
    start = axis_origin;
  } else {
    axis  = start_origin;
    start = axis_origin + 1;
  }
  mxnet::TShape axes = NumpyRollaxisShapeImpl(axis, start, inputs[0].ndim());

  mshadow::Tensor<xpu, 1, dim_t> workspace = GetTransposeExWorkspace<xpu>(ctx, axes);
  if (req[0] == kAddTo) {
    TransposeExImpl<xpu, true>(ctx.run_ctx, inputs[0], outputs[0], axes, workspace);
  } else {
    TransposeExImpl<xpu, false>(ctx.run_ctx, inputs[0], outputs[0], axes, workspace);
  }
}

struct NumpyRot90Param : public dmlc::Parameter<NumpyRot90Param> {
  int k;
  dmlc::optional<mxnet::TShape> axes;
  DMLC_DECLARE_PARAMETER(NumpyRot90Param) {
    DMLC_DECLARE_FIELD(k).set_default(1).describe(
        "Number of times the array is rotated by 90 degrees.");
    DMLC_DECLARE_FIELD(axes)
        .set_default(dmlc::optional<mxnet::TShape>())
        .describe(
            " The array is rotated in the plane defined by the axes. Axes must be different.");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream k_s, axes_s;
    k_s << k;
    axes_s << axes;
    (*dict)["k"]    = k_s.str();
    (*dict)["axes"] = axes_s.str();
  }
};

struct rot90reverse {
  MSHADOW_XINLINE static index_t ReverseIndex(index_t idx,
                                              index_t nreversedim,
                                              const index_t* stride_,
                                              const index_t* trailing_) {
    index_t outputIndex = idx;
    for (index_t i = 0; i < nreversedim; ++i) {
      const index_t low = outputIndex % trailing_[i];
      index_t high      = outputIndex / trailing_[i];
      const index_t x   = high % stride_[i];
      high /= stride_[i];
      outputIndex = (high * stride_[i] + stride_[i] - 1 - x) * trailing_[i] + low;
    }
    return outputIndex;
  }
  template <typename DType>
  MSHADOW_XINLINE static void Map(index_t index,
                                  index_t nreversedim,
                                  const DType* src,
                                  DType* dst,
                                  const index_t* stride_,
                                  const index_t* trailing_) {
    index_t new_idx = ReverseIndex(index, nreversedim, stride_, trailing_);
    dst[new_idx]    = src[index];
  }
};

template <typename xpu>
void NumpyRot90ComputeFlipIml(const OpContext& ctx,
                              const std::vector<TBlob>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<TBlob>& outputs,
                              const index_t axis0,
                              const index_t axis1) {
  using namespace mshadow;
  using namespace mxnet_op;

  const mxnet::TShape& ishape = inputs[0].shape_;
  Stream<xpu>* s              = ctx.get_stream<xpu>();

  std::vector<index_t> stride_(2);
  std::vector<index_t> trailing_(2);
  index_t reverse_index = 0;
  std::vector<index_t> temp{axis0, axis1};
  for (int axis : temp) {
    stride_[reverse_index]   = ishape[axis];
    trailing_[reverse_index] = 1;
    for (int i2 = axis + 1; i2 < ishape.ndim(); ++i2) {
      trailing_[reverse_index] *= ishape[i2];
    }
    reverse_index++;
  }

  index_t workspace_size = 2 * sizeof(index_t);
  Tensor<xpu, 1, char> workspace =
      ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(2 * workspace_size), s);
  Tensor<cpu, 1, index_t> stride_cpu_tensor(stride_.data(), Shape1(stride_.size()));
  Tensor<xpu, 1, index_t> stride_xpu_tensor(reinterpret_cast<index_t*>(workspace.dptr_),
                                            Shape1(stride_.size()));
  Tensor<cpu, 1, index_t> trailing_cpu_tensor(trailing_.data(), Shape1(trailing_.size()));
  Tensor<xpu, 1, index_t> trailing_xpu_tensor(
      reinterpret_cast<index_t*>(workspace.dptr_ + workspace_size), Shape1(trailing_.size()));

  mshadow::Copy(stride_xpu_tensor, stride_cpu_tensor, s);
  mshadow::Copy(trailing_xpu_tensor, trailing_cpu_tensor, s);
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Kernel<rot90reverse, xpu>::Launch(s,
                                      inputs[0].Size(),
                                      reverse_index,
                                      inputs[0].dptr<DType>(),
                                      outputs[0].dptr<DType>(),
                                      stride_xpu_tensor.dptr_,
                                      trailing_xpu_tensor.dptr_);
  });
}

struct rot90Transreverse {
  MSHADOW_XINLINE static index_t ReverseIndex(index_t idx,
                                              const index_t stride_,
                                              const index_t trailing_) {
    index_t outputIndex = idx;
    const index_t low   = outputIndex % trailing_;
    index_t high        = outputIndex / trailing_;
    const index_t x     = high % stride_;
    high /= stride_;
    outputIndex = (high * stride_ + stride_ - 1 - x) * trailing_ + low;

    return outputIndex;
  }
  template <typename DType>
  MSHADOW_XINLINE static void Map(index_t index,
                                  const DType* src,
                                  DType* dst,
                                  const index_t stride_,
                                  const index_t trailing_) {
    index_t new_idx = ReverseIndex(index, stride_, trailing_);
    dst[new_idx]    = src[index];
  }
};

template <typename xpu>
void NumpyRot90ComputeFlipTransposeIml(const OpContext& ctx,
                                       const std::vector<TBlob>& inputs,
                                       const std::vector<OpReqType>& req,
                                       const std::vector<TBlob>& outputs,
                                       const mxnet::TShape axes_list,
                                       const index_t axis) {
  using namespace mshadow;
  using namespace mxnet_op;

  const mxnet::TShape& ishape = inputs[0].shape_;
  Stream<xpu>* s              = ctx.get_stream<xpu>();

  index_t stride_;
  index_t trailing_;

  stride_   = ishape[axis];
  trailing_ = 1;
  for (int i2 = axis + 1; i2 < ishape.ndim(); ++i2) {
    trailing_ *= ishape[i2];
  }

  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    index_t workspace_size = inputs[0].Size() * sizeof(DType);
    Tensor<xpu, 1, char> workspace =
        ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);
    DType* data_ptr = reinterpret_cast<DType*>(workspace.dptr_);
    TBlob mid_data  = TBlob(data_ptr, inputs[0].shape_, xpu::kDevMask);
    Kernel<rot90Transreverse, xpu>::Launch(
        s, inputs[0].Size(), inputs[0].dptr<DType>(), mid_data.dptr<DType>(), stride_, trailing_);
    mxnet::op::TransposeImpl<xpu>(ctx.run_ctx, mid_data, outputs[0], axes_list);
  });
}

template <typename xpu>
void NumpyRot90ComputeTransposeFlipIml(const OpContext& ctx,
                                       const std::vector<TBlob>& inputs,
                                       const std::vector<OpReqType>& req,
                                       const std::vector<TBlob>& outputs,
                                       const mxnet::TShape axes_list,
                                       const index_t axis) {
  using namespace mshadow;
  using namespace mxnet_op;

  Stream<xpu>* s = ctx.get_stream<xpu>();

  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    index_t workspace_size = inputs[0].Size() * sizeof(DType);
    Tensor<xpu, 1, char> workspace =
        ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);
    DType* data_ptr = reinterpret_cast<DType*>(workspace.dptr_);
    mxnet::TShape mid_shape(outputs[0].shape_);
    TBlob mid_data = TBlob(data_ptr, mid_shape, xpu::kDevMask);
    mxnet::op::TransposeImpl<xpu>(ctx.run_ctx, inputs[0], mid_data, axes_list);

    index_t stride_;
    index_t trailing_;
    stride_   = mid_shape[axis];
    trailing_ = 1;
    for (int i2 = axis + 1; i2 < mid_shape.ndim(); ++i2) {
      trailing_ *= mid_shape[i2];
    }
    Kernel<rot90Transreverse, xpu>::Launch(
        s, mid_data.Size(), mid_data.dptr<DType>(), outputs[0].dptr<DType>(), stride_, trailing_);
  });
}

template <int req>
struct rot90 {
  template <typename DType>
  MSHADOW_XINLINE static void Map(index_t i, const DType* in_data, DType* out_data) {
    KERNEL_ASSIGN(out_data[i], req, in_data[i]);
  }
};

template <typename xpu>
void NumpyRot90Compute(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  const NumpyRot90Param& param = nnvm::get<NumpyRot90Param>(attrs.parsed);
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(inputs[0].type_flag_, outputs[0].type_flag_);
  if (outputs[0].Size() == 0)
    return;
  Stream<xpu>* s = ctx.get_stream<xpu>();

  int real_k(param.k);
  real_k = real_k % 4;
  if (real_k < 0) {
    real_k += 4;
  }

  // axis has value
  mxnet::TShape real_axes(param.axes.value());
  for (index_t i = 0; i < real_axes.ndim(); i++) {
    if (real_axes[i] < 0) {
      real_axes[i] += inputs[0].shape_.ndim();
    }
  }
  if (real_k == 0) {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        Kernel<rot90<req_type>, xpu>::Launch(
            s, inputs[0].Size(), inputs[0].dptr<DType>(), outputs[0].dptr<DType>());
      });
    });
  } else if (real_k == 2) {
    NumpyRot90ComputeFlipIml<xpu>(ctx, inputs, req, outputs, real_axes[0], real_axes[1]);
  } else if (real_k == 1) {
    mxnet::TShape axes_list(inputs[0].shape_.ndim(), -1);
    for (int i = 0; i < inputs[0].shape_.ndim(); ++i) {
      axes_list[i] = i;
    }
    axes_list[real_axes[0]] += axes_list[real_axes[1]];
    axes_list[real_axes[1]] = axes_list[real_axes[0]] - axes_list[real_axes[1]];
    axes_list[real_axes[0]] -= axes_list[real_axes[1]];
    NumpyRot90ComputeFlipTransposeIml<xpu>(ctx, inputs, req, outputs, axes_list, real_axes[1]);
  } else if (real_k == 3) {
    mxnet::TShape axes_list(inputs[0].shape_.ndim(), -1);
    for (int i = 0; i < inputs[0].shape_.ndim(); ++i) {
      axes_list[i] = i;
    }
    axes_list[real_axes[0]] += axes_list[real_axes[1]];
    axes_list[real_axes[1]] = axes_list[real_axes[0]] - axes_list[real_axes[1]];
    axes_list[real_axes[0]] -= axes_list[real_axes[1]];
    NumpyRot90ComputeTransposeFlipIml<xpu>(ctx, inputs, req, outputs, axes_list, real_axes[1]);
  }
}

template <typename xpu>
inline void HSplitOpForward(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;
  const SplitParam& param = nnvm::get<SplitParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), param.sections > 0 ? param.sections : param.indices.ndim());
  const TBlob& input_data = inputs[split_enum::kData];
  int real_axis;
  if (input_data.ndim() > 1) {
    real_axis = 1;
  } else {
    real_axis = 0;
  }
  if (param.sections > 0) {
    CHECK_EQ(inputs[0].shape_[real_axis] % param.sections, 0U)
        << "ValueError: array split does not result in an equal division";
  }
  SplitOpForwardImpl<xpu>(attrs, ctx, inputs, req, outputs, real_axis);
}

template <typename xpu>
inline void HSplitOpBackward(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;
  const SplitParam& param = nnvm::get<SplitParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), (param.sections > 0) ? param.sections : param.indices.ndim())
      << "out grad vector size mush match the output size";
  CHECK_EQ(outputs.size(), 1U);
  int real_axis;
  if (outputs[split_enum::kData].ndim() > 1) {
    real_axis = 1;
  } else {
    real_axis = 0;
  }
  SplitOpBackwardImpl<xpu>(attrs, ctx, inputs, req, outputs, real_axis);
}

struct NumpyConcatenateParam : public dmlc::Parameter<NumpyConcatenateParam> {
  int num_args;
  dmlc::optional<int> axis;
  DMLC_DECLARE_PARAMETER(NumpyConcatenateParam) {
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(1).describe("Number of inputs to be concated.");
    DMLC_DECLARE_FIELD(axis)
        .set_default(dmlc::optional<int>(0))
        .describe(
            "The axis along which `values` are appended.  If `axis` is not"
            "given, both `arr` and `values` are flattened before use.");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream num_args_s, axis_s;
    num_args_s << num_args;
    axis_s << axis;
    (*dict)["num_args"] = num_args_s.str();
    (*dict)["axis"]     = axis_s.str();
  }
};

template <typename xpu>
void NumpyConcatenateForward(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow_op;

  const NumpyConcatenateParam& param = nnvm::get<NumpyConcatenateParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), param.num_args);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);

  std::vector<TBlob> data(param.num_args);
  for (int i = 0; i < param.num_args; i++) {
    if (!param.axis.has_value()) {
      data[i] = inputs[i].reshape(Shape1(inputs[i].shape_.Size()));
    } else {
      data[i] = inputs[i];
    }
  }

  ConcatParam cparam;
  cparam.num_args = param.num_args;
  cparam.dim      = param.axis.has_value() ? param.axis.value() : 0;
  MSHADOW_TYPE_SWITCH_WITH_BOOL(inputs[0].type_flag_, DType, {
    ConcatOp<xpu, DType> op;
    op.Init(cparam);
    op.Forward(ctx, data, req, outputs);
  });
}

template <typename xpu>
void NumpyConcatenateBackward(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<TBlob>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow_op;

  const NumpyConcatenateParam& param = nnvm::get<NumpyConcatenateParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), param.num_args);
  CHECK_EQ(req.size(), param.num_args);

  std::vector<TBlob> data(param.num_args);
  for (int i = 0; i < param.num_args; i++) {
    if (!param.axis.has_value()) {
      data[i] = outputs[i].reshape(Shape1(outputs[i].shape_.Size()));
    } else {
      data[i] = outputs[i];
    }
  }

  ConcatParam cparam;
  cparam.num_args = param.num_args;
  cparam.dim      = param.axis.has_value() ? param.axis.value() : 0;
  MSHADOW_TYPE_SWITCH_WITH_BOOL(inputs[0].type_flag_, DType, {
    ConcatOp<xpu, DType> op;
    op.Init(cparam);
    op.Backward(ctx, inputs[0], req, data);
  });
}

struct NumpyDiagParam : public dmlc::Parameter<NumpyDiagParam> {
  index_t k;
  DMLC_DECLARE_PARAMETER(NumpyDiagParam) {
    DMLC_DECLARE_FIELD(k).set_default(0).describe(
        "Diagonal in question. The default is 0. "
        "Use k>0 for diagonals above the main diagonal, "
        "and k<0 for diagonals below the main diagonal. ");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream k_s;
    k_s << k;
    (*dict)["k"] = k_s.str();
  }
};

inline mxnet::TShape NumpyDiagShapeImpl(const mxnet::TShape& ishape, const index_t k) {
  CHECK_LE(ishape.ndim(), 2) << "Input must be 1- or 2-d";

  if (ishape.ndim() == 1) {
    auto s = ishape[0] + std::abs(k);
    return mxnet::TShape({s, s});
  }

  auto h = ishape[0];
  auto w = ishape[1];

  if (k > 0) {
    w -= k;
  } else if (k < 0) {
    h += k;
  }
  dim_t a = 0;
  auto s  = std::max(std::min(h, w), a);
  // s is the length of diagonal with k as the offset

  int n_dim = ishape.ndim() - 1;
  mxnet::TShape oshape(n_dim, -1);
  oshape[n_dim - 1] = s;
  return oshape;
}

inline bool NumpyDiagOpShape(const nnvm::NodeAttrs& attrs,
                             mxnet::ShapeVector* in_attrs,
                             mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  const mxnet::TShape& ishape = (*in_attrs)[0];
  if (!mxnet::ndim_is_known(ishape)) {
    return false;
  }

  const NumpyDiagParam& param = nnvm::get<NumpyDiagParam>(attrs.parsed);
  mxnet::TShape oshape        = NumpyDiagShapeImpl(ishape, param.k);

  if (shape_is_none(oshape)) {
    LOG(FATAL) << "Diagonal does not exist.";
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);

  return shape_is_known(out_attrs->at(0));
}

inline bool NumpyDiagOpType(const nnvm::NodeAttrs& attrs,
                            std::vector<int>* in_attrs,
                            std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, (*in_attrs)[0]);
  TYPE_ASSIGN_CHECK(*in_attrs, 0, (*out_attrs)[0]);
  return (*out_attrs)[0] != -1;
}

template <int ndim, int req, bool back>
struct diag {
  template <typename DType>
  MSHADOW_XINLINE static void Map(index_t i,
                                  DType* out,
                                  const DType* a,
                                  index_t stride,
                                  index_t offset) {
    using namespace mxnet_op;
    index_t j = offset + stride * i;

    if (back) {
      KERNEL_ASSIGN(out[j], req, a[i]);
    } else {
      KERNEL_ASSIGN(out[i], req, a[j]);
    }
  }
};

template <int req, bool back>
struct diag_gen {
  template <typename DType>
  MSHADOW_XINLINE static void Map(index_t i,
                                  DType* out,
                                  const DType* a,
                                  mshadow::Shape<2> oshape,
                                  index_t k) {
    using namespace mxnet_op;

    auto j = unravel(i, oshape);
    if (j[1] == (j[0] + k)) {
      auto l = j[0] < j[1] ? j[0] : j[1];
      if (back) {
        KERNEL_ASSIGN(out[l], req, a[i]);
      } else {
        KERNEL_ASSIGN(out[i], req, a[l]);
      }
    } else if (!back) {
      KERNEL_ASSIGN(out[i], req, static_cast<DType>(0));
    }
  }
};

template <typename xpu, bool back>
void NumpyDiagOpImpl(const TBlob& in_data,
                     const TBlob& out_data,
                     const mxnet::TShape& ishape,
                     const mxnet::TShape& oshape,
                     index_t dsize,
                     const index_t& k,
                     mxnet_op::Stream<xpu>* s,
                     const OpReqType& req) {
  using namespace mxnet_op;
  using namespace mshadow;
  if (ishape.ndim() > 1) {
    index_t stride1 = ishape[1], stride2 = 1;
    // stride1 + stride2 is the stride for
    // iterating over the diagonal in question

    // the extra index offset introduced by k
    index_t offset;
    if (k > 0) {
      offset = stride2 * k;
    } else if (k < 0) {
      offset = stride1 * -k;
    } else {
      offset = 0;
    }

    MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req, req_type, {
        if (back && req != kAddTo && req != kNullOp) {
          out_data.FlatTo1D<xpu, DType>(s) = 0;
        }

        Kernel<diag<2, req_type, back>, xpu>::Launch(
            s, dsize, out_data.dptr<DType>(), in_data.dptr<DType>(), stride1 + stride2, offset);
      });
    });
  } else {
    MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req, req_type, {
        Kernel<diag_gen<req_type, back>, xpu>::Launch(s,
                                                      dsize,
                                                      out_data.dptr<DType>(),
                                                      in_data.dptr<DType>(),
                                                      Shape2(oshape[0], oshape[1]),
                                                      k);
      });
    });
  }
}

template <typename xpu>
void NumpyDiagOpForward(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  CHECK_EQ(req[0], kWriteTo);
  Stream<xpu>* s              = ctx.get_stream<xpu>();
  const TBlob& in_data        = inputs[0];
  const TBlob& out_data       = outputs[0];
  const mxnet::TShape& ishape = inputs[0].shape_;
  const mxnet::TShape& oshape = outputs[0].shape_;
  const NumpyDiagParam& param = nnvm::get<NumpyDiagParam>(attrs.parsed);

  NumpyDiagOpImpl<xpu, false>(
      in_data, out_data, ishape, oshape, out_data.Size(), param.k, s, req[0]);
}

template <typename xpu>
void NumpyDiagOpBackward(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  Stream<xpu>* s = ctx.get_stream<xpu>();

  const TBlob& in_data        = inputs[0];
  const TBlob& out_data       = outputs[0];
  const mxnet::TShape& ishape = inputs[0].shape_;
  const mxnet::TShape& oshape = outputs[0].shape_;
  const NumpyDiagParam& param = nnvm::get<NumpyDiagParam>(attrs.parsed);

  NumpyDiagOpImpl<xpu, true>(in_data, out_data, oshape, ishape, in_data.Size(), param.k, s, req[0]);
}

struct NumpyDiagonalParam : public dmlc::Parameter<NumpyDiagonalParam> {
  index_t offset;
  int axis1;
  int axis2;
  DMLC_DECLARE_PARAMETER(NumpyDiagonalParam) {
    DMLC_DECLARE_FIELD(offset).set_default(0).describe(
        "Diagonal in question. The default is 0. "
        "Use k>0 for diagonals above the main diagonal, "
        "and k<0 for diagonals below the main diagonal. "
        "If input has shape (S0 S1) k must be between -S0 and S1");
    DMLC_DECLARE_FIELD(axis1).set_default(0).describe(
        "The first axis of the sub-arrays of interest. "
        "Ignored when the input is a 1-D array.");
    DMLC_DECLARE_FIELD(axis2).set_default(1).describe(
        "The second axis of the sub-arrays of interest. "
        "Ignored when the input is a 1-D array.");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream offset_s, axis1_s, axis2_s;
    offset_s << offset;
    axis1_s << axis1;
    axis2_s << axis2;
    (*dict)["offset"] = offset_s.str();
    (*dict)["axis1"]  = axis1_s.str();
    (*dict)["axis2"]  = axis2_s.str();
  }
};

inline mxnet::TShape NumpyDiagonalShapeImpl(const mxnet::TShape& ishape,
                                            const index_t k,
                                            const int axis1,
                                            const int axis2) {
  int x1 = CheckAxis(axis1, ishape.ndim());
  int x2 = CheckAxis(axis2, ishape.ndim());

  CHECK_NE(x1, x2) << "axis1 and axis2 cannot refer to the same axis " << x1;

  auto h = ishape[x1];
  auto w = ishape[x2];
  if (k > 0) {
    w -= k;
  } else if (k < 0) {
    h += k;
  }
  auto s = std::min(h, w);
  if (s < 0)
    s = 0;
  if (x1 > x2)
    std::swap(x1, x2);

  int n_dim = ishape.ndim() - 1;
  mxnet::TShape oshape(n_dim, -1);

  // remove axis1 and axis2 and append the new axis to the end
  int idx = 0;
  for (int i = 0; i <= n_dim; ++i) {
    if (i != x1 && i != x2) {
      oshape[idx++] = ishape[i];
    }
  }
  oshape[n_dim - 1] = s;
  return oshape;
}

inline bool NumpyDiagonalOpShape(const nnvm::NodeAttrs& attrs,
                                 mxnet::ShapeVector* in_attrs,
                                 mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  const mxnet::TShape& ishape = (*in_attrs)[0];
  CHECK_GE(ishape.ndim(), 2) << "Input array should be at least 2d";
  if (!mxnet::ndim_is_known(ishape)) {
    return false;
  }

  const NumpyDiagonalParam& param = nnvm::get<NumpyDiagonalParam>(attrs.parsed);
  mxnet::TShape oshape = NumpyDiagonalShapeImpl(ishape, param.offset, param.axis1, param.axis2);
  if (shape_is_none(oshape)) {
    LOG(FATAL) << "Diagonal does not exist.";
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  return shape_is_known(out_attrs->at(0));
}

inline bool NumpyDiagonalOpType(const nnvm::NodeAttrs& attrs,
                                std::vector<int>* in_attrs,
                                std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, (*in_attrs)[0]);
  TYPE_ASSIGN_CHECK(*in_attrs, 0, (*out_attrs)[0]);
  return (*out_attrs)[0] != -1;
}

template <int ndim, int req, bool back>
struct diag_n {
  template <typename DType>
  MSHADOW_XINLINE static void Map(index_t i,
                                  DType* out,
                                  const DType* a,
                                  mshadow::Shape<ndim> oshape,
                                  mshadow::Shape<ndim> ishape,
                                  index_t stride,
                                  index_t offset,
                                  index_t base) {
    using namespace mxnet_op;
    index_t idx = i / base;
    index_t j   = ravel(unravel(idx, oshape), ishape) + offset + stride * (i - idx * base);
    if (back) {
      KERNEL_ASSIGN(out[j], req, a[i]);
    } else {
      KERNEL_ASSIGN(out[i], req, a[j]);
    }
  }
};

template <typename xpu, bool back>
void NumpyDiagonalOpImpl(const TBlob& in_data,
                         const TBlob& out_data,
                         const mxnet::TShape& ishape,
                         const mxnet::TShape& oshape,
                         index_t dsize,
                         const NumpyDiagonalParam& param,
                         mxnet_op::Stream<xpu>* s,
                         const std::vector<OpReqType>& req) {
  using namespace mxnet_op;
  using namespace mshadow;
  int x1   = CheckAxis(param.axis1, ishape.ndim());
  int x2   = CheckAxis(param.axis2, ishape.ndim());
  int idim = ishape.ndim(), odim = oshape.ndim();
  int minx = x1, maxx = x2;
  if (minx > maxx)
    std::swap(minx, maxx);

  index_t oleading = 1, obody = 1, otrailing = 1;
  for (int i = 0; i < minx; ++i) {
    oleading *= ishape[i];
  }
  for (int i = minx + 1; i < maxx; ++i) {
    obody *= ishape[i];
  }
  for (int i = maxx + 1; i < idim; ++i) {
    otrailing *= ishape[i];
  }

  index_t ileading = oleading, ibody = obody * ishape[minx], itrailing = otrailing * ishape[maxx];

  index_t stride1 = itrailing * obody, stride2 = otrailing;
  // stride1 + stride2 is the stride for iterating over the diagonal

  if (x1 == maxx)
    std::swap(stride1, stride2);
  index_t offset;
  index_t k = param.offset;
  if (k > 0) {
    offset = stride2 * k;
  } else if (k < 0) {
    offset = stride1 * -k;
  } else {
    offset = 0;
  }  // the extra index offset introduced by k

  MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      if (back && req[0] != kAddTo && req[0] != kNullOp) {
        out_data.FlatTo1D<xpu, DType>(s) = 0;
      }
      if (ileading == 1) {
        Kernel<diag_n<2, req_type, back>, xpu>::Launch(s,
                                                       dsize,
                                                       out_data.dptr<DType>(),
                                                       in_data.dptr<DType>(),
                                                       Shape2(obody, otrailing),
                                                       Shape2(ibody, itrailing),
                                                       stride1 + stride2,
                                                       offset,
                                                       oshape[odim - 1]);
      } else {
        Kernel<diag_n<3, req_type, back>, xpu>::Launch(s,
                                                       dsize,
                                                       out_data.dptr<DType>(),
                                                       in_data.dptr<DType>(),
                                                       Shape3(oleading, obody, otrailing),
                                                       Shape3(ileading, ibody, itrailing),
                                                       stride1 + stride2,
                                                       offset,
                                                       oshape[odim - 1]);
      }
    });
  });
}

template <typename xpu>
void NumpyDiagonalOpForward(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  CHECK_EQ(req[0], kWriteTo);
  Stream<xpu>* s                  = ctx.get_stream<xpu>();
  const TBlob& in_data            = inputs[0];
  const TBlob& out_data           = outputs[0];
  const mxnet::TShape& ishape     = inputs[0].shape_;
  const mxnet::TShape& oshape     = outputs[0].shape_;
  const NumpyDiagonalParam& param = nnvm::get<NumpyDiagonalParam>(attrs.parsed);

  NumpyDiagonalOpImpl<xpu, false>(
      in_data, out_data, ishape, oshape, out_data.Size(), param, s, req);
}

template <typename xpu>
void NumpyDiagonalOpBackward(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  Stream<xpu>* s = ctx.get_stream<xpu>();

  const TBlob& in_data            = inputs[0];
  const TBlob& out_data           = outputs[0];
  const mxnet::TShape& ishape     = inputs[0].shape_;
  const mxnet::TShape& oshape     = outputs[0].shape_;
  const NumpyDiagonalParam& param = nnvm::get<NumpyDiagonalParam>(attrs.parsed);

  NumpyDiagonalOpImpl<xpu, true>(in_data, out_data, oshape, ishape, in_data.Size(), param, s, req);
}

struct NumpyDiagflatParam : public dmlc::Parameter<NumpyDiagflatParam> {
  int k;
  DMLC_DECLARE_PARAMETER(NumpyDiagflatParam) {
    DMLC_DECLARE_FIELD(k).set_default(0).describe(
        "Diagonal in question. The default is 0. "
        "Use k>0 for diagonals above the main diagonal, "
        "and k<0 for diagonals below the main diagonal. ");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream k_s;
    k_s << k;
    (*dict)["k"] = k_s.str();
  }
};

inline mxnet::TShape NumpyDiagflatShapeImpl(const mxnet::TShape& ishape, const int k) {
  if (ishape.ndim() == 1) {
    auto s = ishape[0] + std::abs(k);
    return mxnet::TShape({s, s});
  }

  if (ishape.ndim() >= 2) {
    auto s = 1;
    for (int i = 0; i < ishape.ndim(); i++) {
      if (ishape[i] >= 2) {
        s = s * ishape[i];
      }
    }
    s = s + std::abs(k);
    return mxnet::TShape({s, s});
  }
  return mxnet::TShape({-1, -1});
}

inline bool NumpyDiagflatOpShape(const nnvm::NodeAttrs& attrs,
                                 mxnet::ShapeVector* in_attrs,
                                 mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  const mxnet::TShape& ishape = (*in_attrs)[0];
  if (!mxnet::ndim_is_known(ishape)) {
    return false;
  }
  const NumpyDiagflatParam& param = nnvm::get<NumpyDiagflatParam>(attrs.parsed);

  mxnet::TShape oshape = NumpyDiagflatShapeImpl(ishape, param.k);

  if (shape_is_none(oshape)) {
    LOG(FATAL) << "Diagonal does not exist.";
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);

  return shape_is_known(out_attrs->at(0));
}

inline bool NumpyDiagflatOpType(const nnvm::NodeAttrs& attrs,
                                std::vector<int>* in_attrs,
                                std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, (*in_attrs)[0]);
  TYPE_ASSIGN_CHECK(*in_attrs, 0, (*out_attrs)[0]);
  return (*out_attrs)[0] != -1;
}

template <typename xpu, bool back>
void NumpyDiagflatOpImpl(const TBlob& in_data,
                         const TBlob& out_data,
                         const mxnet::TShape& ishape,
                         const mxnet::TShape& oshape,
                         index_t dsize,
                         const NumpyDiagflatParam& param,
                         mxnet_op::Stream<xpu>* s,
                         const std::vector<OpReqType>& req) {
  using namespace mxnet_op;
  using namespace mshadow;
  MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<diag_gen<req_type, back>, xpu>::Launch(s,
                                                    dsize,
                                                    out_data.dptr<DType>(),
                                                    in_data.dptr<DType>(),
                                                    Shape2(oshape[0], oshape[1]),
                                                    param.k);
    });
  });
}

template <typename xpu>
void NumpyDiagflatOpForward(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  CHECK_EQ(req[0], kWriteTo);
  Stream<xpu>* s                  = ctx.get_stream<xpu>();
  const TBlob& in_data            = inputs[0];
  const TBlob& out_data           = outputs[0];
  const mxnet::TShape& ishape     = inputs[0].shape_;
  const mxnet::TShape& oshape     = outputs[0].shape_;
  const NumpyDiagflatParam& param = nnvm::get<NumpyDiagflatParam>(attrs.parsed);

  NumpyDiagflatOpImpl<xpu, false>(
      in_data, out_data, ishape, oshape, out_data.Size(), param, s, req);
}

template <typename xpu>
void NumpyDiagflatOpBackward(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  Stream<xpu>* s = ctx.get_stream<xpu>();

  const TBlob& in_data            = inputs[0];
  const TBlob& out_data           = outputs[0];
  const mxnet::TShape& ishape     = inputs[0].shape_;
  const mxnet::TShape& oshape     = outputs[0].shape_;
  const NumpyDiagflatParam& param = nnvm::get<NumpyDiagflatParam>(attrs.parsed);

  NumpyDiagflatOpImpl<xpu, true>(in_data, out_data, oshape, ishape, in_data.Size(), param, s, req);
}

struct diag_indices_from {
  template <typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType* out, const int length) {
    int index = i % length;
    out[i]    = index;
  }
};

template <typename xpu>
void NumpyDiagIndicesFromForward(const nnvm::NodeAttrs& attrs,
                                 const OpContext& ctx,
                                 const std::vector<TBlob>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  CHECK_EQ(req[0], kWriteTo);
  Stream<xpu>* s        = ctx.get_stream<xpu>();
  const TBlob& in_data  = inputs[0];
  const TBlob& out_data = outputs[0];
  const int length      = in_data.shape_[0];
  if (in_data.Size() == 0)
    return;

  MSHADOW_IDX_TYPE_SWITCH(out_data.type_flag_, DType, {
    Kernel<diag_indices_from, xpu>::Launch(s, out_data.Size(), out_data.dptr<DType>(), length);
  });
}

}  // namespace op
}  // namespace mxnet

namespace std {
template <>
struct hash<mxnet::op::NumpyTransposeParam> {
  size_t operator()(const mxnet::op::NumpyTransposeParam& val) {
    size_t ret = 0;
    ret        = dmlc::HashCombine(ret, val.axes);
    return ret;
  }
};
}  // namespace std

#endif  // MXNET_OPERATOR_NUMPY_NP_MATRIX_OP_INL_H_
