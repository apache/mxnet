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
#include <algorithm>
#include <string>
#include <vector>
#include "../../elemwise_op_common.h"
#include "../../mshadow_op.h"
#include "../../mxnet_op.h"
#include "../../operator_common.h"
#include "../../tensor/elemwise_binary_broadcast_op.h"
#include "./dist_common.h"

namespace mxnet {
namespace op {

struct NumpyUniformParam : public dmlc::Parameter<NumpyUniformParam> {
  dmlc::optional<float> low;
  dmlc::optional<float> high;
  std::string ctx;
  int dtype;
  dmlc::optional<mxnet::Tuple<int>> size;
  DMLC_DECLARE_PARAMETER(NumpyUniformParam) {
    DMLC_DECLARE_FIELD(low);
    DMLC_DECLARE_FIELD(high);
    DMLC_DECLARE_FIELD(size)
        .set_default(dmlc::optional<mxnet::Tuple<int>>())
        .describe(
            "Output shape. If the given shape is, "
            "e.g., (m, n, k), then m * n * k samples are drawn. "
            "Default is None, in which case a single value is returned.");
    DMLC_DECLARE_FIELD(ctx).set_default("cpu").describe(
        "Context of output, in format [cpu|gpu|cpu_pinned](n)."
        " Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype)
        .add_enum("float32", mshadow::kFloat32)
        .add_enum("float64", mshadow::kFloat64)
        .add_enum("float16", mshadow::kFloat16)
        .set_default(mshadow::kFloat32)
        .describe(
            "DType of the output in case this can't be inferred. "
            "Defaults to float32 if not defined (dtype=None).");
  }
};

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
  MSHADOW_XINLINE static void Map(index_t i, const Shape<ndim> &lstride,
                                  const Shape<ndim> &hstride,
                                  const Shape<ndim> &oshape, IType *low,
                                  IType *high, float *uniform, OType *out) {
    Shape<ndim> coord = unravel(i, oshape);
    auto lidx = static_cast<index_t>(dot(coord, lstride));
    auto hidx = static_cast<index_t>(dot(coord, hstride));
    IType low_value = low[lidx];
    IType high_value = high[hidx];
    out[i] = low_value + uniform[i] * (high_value - low_value);
  }
};

template <int ndim, typename IType, typename OType>
struct uniform_one_scalar_kernel {
  MSHADOW_XINLINE static void Map(index_t i, int scalar_pos,
                                  const Shape<ndim> &stride,
                                  const Shape<ndim> &oshape, IType *array,
                                  float scalar, float *uniform, OType *out) {
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

template <typename OType>
struct uniform_two_scalar_kernel {
  MSHADOW_XINLINE static void Map(index_t i, float low, float high,
                                  float *uniform, OType *out) {
    out[i] = low + uniform[i] * (high - low);
  }
};
}  // namespace mxnet_op

template <typename xpu>
void NumpyUniformForward(const nnvm::NodeAttrs &attrs,
                         const OpContext &ctx,
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
      ctx.requested[1].get_space_typed<xpu, 1, float>(Shape1(outputs[0].Size()),
                                                      s);
  prnd->SampleUniform(&uniform_tensor, 0, 1);
  mxnet::TShape new_lshape, new_hshape, new_oshape;

  // [scalar scalar] case
  if (inputs.size() == 0U) {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, OType, {
      Kernel<uniform_two_scalar_kernel<OType>, xpu>::Launch(
          s, outputs[0].Size(), param.low.value(), param.high.value(),
          uniform_tensor.dptr_, outputs[0].dptr<OType>());
    });
  } else if (inputs.size() == 1U) {
    // [scalar tensor], [tensor scalar] case
    int ndim = FillShape(inputs[0].shape_, inputs[0].shape_, outputs[0].shape_,
                         &new_lshape, &new_lshape, &new_oshape);
    int scalar_pos;
    float scalar_value;
    // int type_flag = param.t;
    if (param.low.has_value()) {
      scalar_pos = 0;
      scalar_value = param.low.value();
    } else {
      scalar_pos = 1;
      scalar_value = param.high.value();
    }
    MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, IType, {
      MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, OType, {
        BROADCAST_NDIM_SWITCH(ndim, NDim, {
          Shape<NDim> oshape = new_oshape.get<NDim>();
          Shape<NDim> stride = calc_stride(new_lshape.get<NDim>());
          Kernel<uniform_one_scalar_kernel<NDim, IType, OType>, xpu>::Launch(
              s, outputs[0].Size(), scalar_pos, stride, oshape,
              inputs[0].dptr<IType>(), scalar_value, uniform_tensor.dptr_,
              outputs[0].dptr<OType>());
        });
      });
    });
  } else if (inputs.size() == 2U) {
    // [tensor tensor] case
    int ndim = FillShape(inputs[0].shape_, inputs[1].shape_, outputs[0].shape_,
                         &new_lshape, &new_hshape, &new_oshape);
    MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, IType, {
      MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, OType, {
        BROADCAST_NDIM_SWITCH(ndim, NDim, {
          Shape<NDim> oshape = new_oshape.get<NDim>();
          Shape<NDim> lstride = calc_stride(new_lshape.get<NDim>());
          Shape<NDim> hstride = calc_stride(new_hshape.get<NDim>());
          Kernel<uniform_kernel<NDim, IType, OType>, xpu>::Launch(
              s, outputs[0].Size(), lstride, hstride, oshape,
              inputs[0].dptr<IType>(), inputs[1].dptr<IType>(),
              uniform_tensor.dptr_, outputs[0].dptr<OType>());
        });
      });
    });
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_RANDOM_NP_UNIFORM_OP_H_
