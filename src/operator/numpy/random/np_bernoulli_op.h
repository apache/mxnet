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
 * \file np_bernoulli_op.h
 * \brief Operator for numpy sampling from bernoulli distribution.
 */
#ifndef MXNET_OPERATOR_NUMPY_RANDOM_NP_BERNOULLI_OP_H_
#define MXNET_OPERATOR_NUMPY_RANDOM_NP_BERNOULLI_OP_H_

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

struct NumpyBernoulliParam : public dmlc::Parameter<NumpyBernoulliParam> {
  dmlc::optional<float> prob;
  dmlc::optional<float> logit;
  std::string ctx;
  int dtype;
  bool is_logit;
  dmlc::optional<mxnet::Tuple<int>> size;
  DMLC_DECLARE_PARAMETER(NumpyBernoulliParam) {
    DMLC_DECLARE_FIELD(prob);
    DMLC_DECLARE_FIELD(logit);
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
        .add_enum("uint8", mshadow::kUint8)
        .add_enum("int32", mshadow::kInt32)
        .add_enum("float32", mshadow::kFloat32)
        .add_enum("float64", mshadow::kFloat64)
        .add_enum("float16", mshadow::kFloat16)
        .add_enum("bool", mshadow::kBool)
        .set_default(mshadow::kFloat32)
        .describe(
            "DType of the output in case this can't be inferred. "
            "Defaults to float32 if not defined (dtype=None).");
    DMLC_DECLARE_FIELD(is_logit);
  }
};

inline bool NumpyBernoulliOpType(const nnvm::NodeAttrs &attrs,
                               std::vector<int> *in_attrs,
                               std::vector<int> *out_attrs) {
  const NumpyBernoulliParam &param = nnvm::get<NumpyBernoulliParam>(attrs.parsed);
  int otype = param.dtype;
  (*out_attrs)[0] = otype;
  return true;
}

namespace mxnet_op {

struct prob_to_logit {
  MSHADOW_XINLINE static void Map(index_t i, float* uniforms) {
    float prob = uniforms[i];
    uniforms[i] = log(prob) - log(1 - prob);
  }
};

template <int ndim, typename IType, typename OType>
struct bernoulli_kernel {
  MSHADOW_XINLINE static void Map(index_t i,
                                  const Shape<ndim> &stride,
                                  const Shape<ndim> &oshape,
                                  IType *inputs, float* threshold, OType *out) {
    Shape<ndim> coord = unravel(i, oshape);
    auto idx = static_cast<index_t>(dot(coord, stride));
    out[i] =  inputs[idx] > threshold[i] ? OType(1) : OType(0);
  }
};

template <typename OType>
struct scalar_bernoulli_kernel {
  MSHADOW_XINLINE static void Map(index_t i, float inputs, float *threshold,
                                  OType *out) {
    out[i] = inputs > threshold[i] ? OType(1) : OType(0);
  }
};

template <typename IType>
struct check_legal_prob_kernel {
  MSHADOW_XINLINE static void Map(index_t i, IType *scalar, float* flag) {
    if (scalar[i] < 0.0 || scalar[i] > 1.0) {
      flag[0] = -1.0;
    }
  }
};

}  // namespace mxnet_op

template <typename xpu>
void NumpyBernoulliForward(const nnvm::NodeAttrs &attrs,
                         const OpContext &ctx,
                         const std::vector<TBlob> &inputs,
                         const std::vector<OpReqType> &req,
                         const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  const NumpyBernoulliParam &param = nnvm::get<NumpyBernoulliParam>(attrs.parsed);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  index_t output_len = outputs[0].Size();
  Random<xpu, float> *prnd = ctx.requested[0].get_random<xpu, float>(s);
  Tensor<xpu, 1, float> workspace =
      ctx.requested[1].get_space_typed<xpu, 1, float>(Shape1(output_len + 1), s);
  Tensor<xpu, 1, float> uniform_tensor = workspace.Slice(0, output_len);
  Tensor<xpu, 1, float> indicator_device = workspace.Slice(output_len, output_len + 1);
  float indicator_host = 1.0;
  float *indicator_device_ptr = indicator_device.dptr_;
  Kernel<set_zero, xpu>::Launch(s, 1, indicator_device_ptr);
  prnd->SampleUniform(&uniform_tensor, 0.0, 1.0);
  if (param.prob.has_value()) {
    // scalar prob input
    CHECK_LE(param.prob.value(), 1.0) << "ValueError: expect probs >= 0 && probs <= 1";
    CHECK_GE(param.prob.value(), 0.0) << "ValueError: expect probs >= 0 && probs <= 1";
    MSHADOW_TYPE_SWITCH_WITH_BOOL(outputs[0].type_flag_, OType, {
      Kernel<scalar_bernoulli_kernel<OType>, xpu>::Launch(
        s, outputs[0].Size(), param.prob.value(),
        uniform_tensor.dptr_, outputs[0].dptr<OType>());
    });
  } else if (param.logit.has_value()) {
    // scalar logit input
    // sigmoid(x) > u  <=> x > logit(u)
    Kernel<prob_to_logit, xpu>::Launch(s, outputs[0].Size(),
                                         uniform_tensor.dptr_);
    MSHADOW_TYPE_SWITCH_WITH_BOOL(outputs[0].type_flag_, OType, {
      Kernel<scalar_bernoulli_kernel<OType>, xpu>::Launch(
        s, outputs[0].Size(), param.logit.value(),
        uniform_tensor.dptr_, outputs[0].dptr<OType>());
    });
  } else {
    if (param.is_logit) {
      // tensor logit input
      Kernel<prob_to_logit, xpu>::Launch(s, outputs[0].Size(),
                                         uniform_tensor.dptr_);
    } else {
      // tensor prob input
      // sigmoid(x) > u  <=> x > logit(u)
      MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, IType, {
        Kernel<check_legal_prob_kernel<IType>, xpu>::Launch(
            s, inputs[0].Size(), inputs[0].dptr<IType>(), indicator_device_ptr);
      });
      _copy<xpu>(s, &indicator_host, indicator_device_ptr);
      CHECK_GE(indicator_host, 0.0)
          << "ValueError: expect probs >= 0 && probs <= 1";
    }
    mxnet::TShape new_lshape, new_oshape;
    int ndim = FillShape(inputs[0].shape_, inputs[0].shape_, outputs[0].shape_,
                         &new_lshape, &new_lshape, &new_oshape);
    MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, IType, {
      MSHADOW_TYPE_SWITCH_WITH_BOOL(outputs[0].type_flag_, OType, {
        BROADCAST_NDIM_SWITCH(ndim, NDim, {
          Shape<NDim> oshape = new_oshape.get<NDim>();
          Shape<NDim> stride = calc_stride(new_lshape.get<NDim>());
          Kernel<bernoulli_kernel<NDim, IType, OType>, xpu>::Launch(
              s, outputs[0].Size(), stride, oshape, inputs[0].dptr<IType>(),
              uniform_tensor.dptr_, outputs[0].dptr<OType>());
        });
      });
    });
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_RANDOM_NP_BERNOULLI_OP_H_
