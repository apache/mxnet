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
 * Copyright (c) 2019 by Contributors
 * \file np_uniform_op.h
 * \brief Operator for numpy sampling from uniform distributions
 */
#ifndef MXNET_OPERATOR_NUMPY_RANDOM_NP_UNIFORM_OP_H_
#define MXNET_OPERATOR_NUMPY_RANDOM_NP_UNIFORM_OP_H_

#include <mxnet/operator_util.h>
#include <mshadow/base.h>
#include <vector>
#include <string>
#include <algorithm>
#include "../../elemwise_op_common.h"
#include "../../tensor/elemwise_binary_broadcast_op.h"
#include "../../mshadow_op.h"
#include "../../mxnet_op.h"
#include "../../operator_common.h"

namespace mxnet {
namespace op {

struct NumpyUniformParam : public dmlc::Parameter<NumpyUniformParam> {
  int t;
  float low;
  float high;
  std::string ctx;
  int dtype;
  dmlc::optional<mxnet::Tuple<int>> size;
  DMLC_DECLARE_PARAMETER(NumpyUniformParam) {
    DMLC_DECLARE_FIELD(low);
    DMLC_DECLARE_FIELD(high);
    DMLC_DECLARE_FIELD(t)
        .describe("input type indicator, "
                  "0: array array "
                  "1: scalar array "
                  "2: array scalar "
                  "3: scalar scalar");
    DMLC_DECLARE_FIELD(size)
        .set_default(dmlc::optional<mxnet::Tuple<int>>())
        .describe("Output shape. If the given shape is, "
                  "e.g., (m, n, k), then m * n * k samples are drawn. "
                  "Default is None, in which case a single value is returned.");
    DMLC_DECLARE_FIELD(ctx)
    .set_default("")
    .describe("Context of output, in format [cpu|gpu|cpu_pinned](n)."
              " Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype)
    .add_enum("None", -1)
    .add_enum("float32", mshadow::kFloat32)
    .add_enum("float64", mshadow::kFloat64)
    .add_enum("float16", mshadow::kFloat16)
    .set_default(-1)
    .describe("DType of the output in case this can't be inferred. "
              "Defaults to float32 if not defined (dtype=None).");
  }
};


inline int FillShape(const mxnet::TShape& lshape, const mxnet::TShape& rshape,
                                       const mxnet::TShape& oshape, mxnet::TShape *new_lshape,
                                       mxnet::TShape *new_rshape, mxnet::TShape *new_oshape) {
  const int odim = std::max(oshape.ndim(), broadcast::MAX_DIM);
  *new_lshape = mxnet::TShape(odim, 1);
  *new_rshape = mxnet::TShape(odim, 1);
  *new_oshape = mxnet::TShape(odim, 1);
  int bl = oshape.ndim() - lshape.ndim();
  int br = oshape.ndim() - rshape.ndim();
  int j = 0, lprod = 1, rprod = 1, oprod = 1;
  for (int i = 0; i < oshape.ndim(); ++i) {
    int l = 1;
    int r = 1;
    int o = oshape[i];
    if (i >= bl)  l = lshape[i - bl];
    if (i >= br)  r = rshape[i - br];
    if ((lprod != rprod || lprod != oprod || l != r || l != o) &&
        (lprod * l > 1 || rprod * r > 1 || oprod * o > 1)) {
      (*new_lshape)[j] = lprod;
      (*new_rshape)[j] = rprod;
      (*new_oshape)[j] = oprod;
      lprod = rprod = oprod = 1; ++j;
    }
    lprod *= l;
    rprod *= r;
    oprod *= o;
  }
  if (lprod > 1 || rprod > 1 || oprod > 1) {
    (*new_lshape)[j] = lprod;
    (*new_rshape)[j] = rprod;
    (*new_oshape)[j] = oprod;
    ++j;
  }
  if (j <= broadcast::MAX_DIM) {
    BROADCAST_NDIM_SWITCH(j, NDim, {
      new_lshape->assign(new_lshape->begin(), new_lshape->begin() + NDim);
      new_rshape->assign(new_rshape->begin(), new_rshape->begin() + NDim);
      new_oshape->assign(new_oshape->begin(), new_oshape->begin() + NDim);
    });
  } else {
    LOG(FATAL) << "Too many broadcast dimensions with operands " << lshape << " " << rshape;
  }
  return j;
}

inline void CheckBroadcastable(const mxnet::TShape &from, const mxnet::TShape &to) {
  const int bl = to.ndim() - from.ndim();
  const int br = 0;
  for (int i = 0; i < to.ndim(); ++i) {
    int l = 1, r = 1;
    if (i >= bl)
      l = from[i - bl];
    if (i >= br)
      r = to[i - br];
    if (!mxnet::dim_size_is_known(l) || !mxnet::dim_size_is_known(r))
      continue;
    if (l != r) {
      // Make it compatible with NumPy.
      // For example, (2, 3) cannot broadcast to (2, 0, 3), but (1, 3) can
      // broadcast to (2, 0, 3).
      CHECK(l == 1 || r == 1)
          << "operands could not be broadcast together with shapes " << from
          << " " << to;
    }
  }
}

inline void InferBroadcastShape(const mxnet::TShape &lhs, const mxnet::TShape &rhs,
                         mxnet::TShape* out_ptr) {
  mxnet::TShape& out = (*out_ptr);
  const int bl = out.ndim() - lhs.ndim();
  const int br = out.ndim() - rhs.ndim();
  for (int i = 0; i < out.ndim(); ++i) {
    int l = 1, r = 1;
    if (i >= bl)
      l = lhs[i - bl];
    if (i >= br)
      r = rhs[i - br];
    if (!mxnet::dim_size_is_known(l) || !mxnet::dim_size_is_known(r))
      continue;
    if (l != r) {
      // Make it compatible with NumPy.
      // For example, (2, 3) cannot broadcast to (2, 0, 3), but (1, 3) can
      // broadcast to (2, 0, 3).
      CHECK(l == 1 || r == 1)
          << "operands could not be broadcast together with shapes " << lhs
          << " " << rhs;
      out[i] = (l == 1 ? r : l);
    } else {
      out[i] = l;
    }
  }
}

inline bool NumpyUniformOpShape(const nnvm::NodeAttrs &attrs,
                                std::vector<TShape> *in_attrs,
                                std::vector<TShape> *out_attrs) {
  const NumpyUniformParam &param = nnvm::get<NumpyUniformParam>(attrs.parsed);
  CHECK_EQ(out_attrs->size(), 1U);
  if (param.size.has_value()) {
    // Size declared.
    std::vector<dim_t> oshape_vec;
    const mxnet::Tuple<int> &size = param.size.value();
    for (int i = 0; i < size.ndim(); ++i) {
      oshape_vec.emplace_back(size[i]);
    }
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape(oshape_vec));
    for (int input_idx = 0; input_idx < in_attrs->size(); input_idx++) {
      CheckBroadcastable((*in_attrs)[input_idx], (*out_attrs)[0]);
    }
  } else {
    // Size undeclared.
    if (in_attrs->size() == 2U) {
      // Low and high both from ndarray.
      mxnet::TShape& low = (*in_attrs)[0];
      mxnet::TShape& high = (*in_attrs)[1];
      mxnet::TShape out(std::max(low.ndim(), high.ndim()), -1);
      InferBroadcastShape(low, high, &out);
      SHAPE_ASSIGN_CHECK(*out_attrs, 0, out);
    }
    if (in_attrs->size() == 1U) {
      SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0))
    }
    // Two scalar case without predefined size.
    if (in_attrs->size() == 0) {
      SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape(0, -1))
      return true;
    }
  }
  return out_attrs->at(0).ndim() != 0U;;
}

inline bool NumpyUniformOpType(const nnvm::NodeAttrs &attrs,
                                   std::vector<int> *in_attrs,
                                   std::vector<int> *out_attrs) {
  const NumpyUniformParam &param = nnvm::get<NumpyUniformParam>(attrs.parsed);
  int otype = param.dtype;
  if (otype != -1) {
    (*out_attrs)[0] = otype;
  } else {
    (*out_attrs)[0] = mshadow::kFloat32;
  }
  return true;
}

namespace mxnet_op {
template <int ndim, typename IType, typename OType>
struct uniform_kernel {
  MSHADOW_XINLINE static void Map(index_t i,
                                  const Shape <ndim> &lstride, const Shape <ndim> &hstride,
                                  const Shape <ndim> &oshape,
                                  IType *low, IType *high,
                                  float *uniform, OType *out) {
  Shape<ndim> coord = unravel(i, oshape);
  auto lidx = static_cast<index_t>(dot(coord, lstride));
  auto hidx = static_cast<index_t>(dot(coord, hstride));
  IType low_value = low[lidx];
  IType high_value = high[hidx];
  out[i] = low_value + uniform[i] * (high_value - low_value);
  }
};
}  // namespace mxnet_op

namespace mxnet_op {
template <int ndim, typename IType, typename OType>
struct uniform_one_scalar_kernel {
  MSHADOW_XINLINE static void Map(index_t i, int scalar_pos,
                                  const Shape <ndim> &stride,
                                  const Shape <ndim> &oshape,
                                  IType *array, float scalar,
                                  float *uniform, OType *out) {
  Shape<ndim> coord = unravel(i, oshape);
  auto idx = static_cast<index_t>(dot(coord, stride));
  IType low_value;
  IType high_value;
  if (scalar_pos == 0) {
    low_value = scalar;
    high_value = array[idx];
  } else {
    low_value = array[idx];
    high_value = scalar;
  }
  out[i] = low_value + uniform[i] * (high_value - low_value);
  }
};
}  // namespace mxnet_op

namespace mxnet_op {
template <typename OType>
struct uniform_two_scalar_kernel {
  MSHADOW_XINLINE static void Map(index_t i,
                                  float low, float high,
                                  float *uniform, OType *out) {
  out[i] = low + uniform[i] * (high - low);
  }
};
}  // namespace mxnet_op




template <typename xpu>
void NumpyUniformForward(const nnvm::NodeAttrs &attrs, const OpContext &ctx,
                         const std::vector<TBlob> &inputs,
                         const std::vector<OpReqType> &req,
                         const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  const NumpyUniformParam &param = nnvm::get<NumpyUniformParam>(attrs.parsed);
  CHECK_EQ(outputs.size(), 1);
  Stream<xpu> *s = ctx.get_stream<xpu>();

  // Generate base random number.
  Random<xpu, float> *prnd = ctx.requested[0].get_random<xpu, float>(s);
  Tensor<xpu, 1, float> uniform_tensor =
      ctx.requested[1].get_space_typed<xpu, 1, float>(Shape1(outputs[0].Size()), s);
  prnd->SampleUniform(&uniform_tensor, 0, 1);
  mxnet::TShape new_lshape, new_hshape, new_oshape;

  // [scalar scalar] case
  if (inputs.size() == 0U) {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, OType, {
      mxnet_op::Kernel<uniform_two_scalar_kernel<OType>, xpu>::Launch(
            s, outputs[0].Size(),
            param.low, param.high,
            uniform_tensor.dptr_, outputs[0].dptr<OType>());
    });
  }

  // [scalar tensor], [tensor scalar] case
  if (inputs.size() == 1U) {
    int ndim = FillShape(inputs[0].shape_, inputs[0].shape_, outputs[0].shape_,
                         &new_lshape, &new_lshape, &new_oshape);
    int scalar_pos;
    float scalar_value;
    int type_flag = param.t;
    if (type_flag == 1) {
      scalar_pos = 0;
      scalar_value = param.low;
    } else {
      scalar_pos = 1;
      scalar_value = param.high;
    }
    MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, IType, {
      MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, OType, {
        BROADCAST_NDIM_SWITCH(ndim, NDim, {
        mshadow::Shape<NDim> oshape = new_oshape.get<NDim>();
        mshadow::Shape<NDim> stride =
            mxnet_op::calc_stride(new_lshape.get<NDim>());
        mxnet_op::Kernel<uniform_one_scalar_kernel<NDim, IType, OType>, xpu>::Launch(
            s, outputs[0].Size(), scalar_pos, stride, oshape,
            inputs[0].dptr<IType>(), scalar_value,
            uniform_tensor.dptr_, outputs[0].dptr<OType>());
        });
      });
    });
  }

  // [tensor tensor] case
  if (inputs.size() == 2U) {
    int ndim = FillShape(inputs[0].shape_, inputs[1].shape_, outputs[0].shape_,
                         &new_lshape, &new_hshape, &new_oshape);
    MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, IType, {
      MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, OType, {
        BROADCAST_NDIM_SWITCH(ndim, NDim, {
        mshadow::Shape<NDim> oshape = new_oshape.get<NDim>();
        mshadow::Shape<NDim> lstride =
            mxnet_op::calc_stride(new_lshape.get<NDim>());
        mshadow::Shape<NDim> hstride =
            mxnet_op::calc_stride(new_hshape.get<NDim>());
        mxnet_op::Kernel<uniform_kernel<NDim, IType, OType>, xpu>::Launch(
            s, outputs[0].Size(), lstride, hstride, oshape,
            inputs[0].dptr<IType>(), inputs[1].dptr<IType>(),
            uniform_tensor.dptr_, outputs[0].dptr<OType>());
        });
      });
    });
  }
}

};  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_RANDOM_NP_UNIFORM_OP_H_
