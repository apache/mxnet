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
 * Copyright (c) 2017 by Contributors
 * \file sample_multinomial_op.h
 * \brief Operator for sampling from multinomial distributions
 */
#ifndef MXNET_OPERATOR_RANDOM_SAMPLE_MULTINOMIAL_OP_H_
#define MXNET_OPERATOR_RANDOM_SAMPLE_MULTINOMIAL_OP_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

struct SampleMultinomialParam : public dmlc::Parameter<SampleMultinomialParam> {
  mxnet::TShape shape;
  bool get_prob;
  int dtype;
  DMLC_DECLARE_PARAMETER(SampleMultinomialParam) {
    DMLC_DECLARE_FIELD(shape)
      .set_default(mxnet::TShape(0, 1))
      .describe("Shape to be sampled from each random distribution.");
    DMLC_DECLARE_FIELD(get_prob)
    .set_default(false)
    .describe("Whether to also return the log probability of sampled "
          "result. This is usually used for differentiating through "
          "stochastic variables, e.g. in reinforcement learning.");
    DMLC_DECLARE_FIELD(dtype)
    .add_enum("uint8", mshadow::kUint8)
    .add_enum("int32", mshadow::kInt32)
    .add_enum("float16", mshadow::kFloat16)
    .add_enum("float32", mshadow::kFloat32)
    .add_enum("float64", mshadow::kFloat64)
    .set_default(mshadow::kInt32)
    .describe("DType of the output in case this can't be inferred.");
  }
};


inline bool SampleMultinomialOpShape(const nnvm::NodeAttrs& attrs,
                                     mxnet::ShapeVector* in_attrs,
                                     mxnet::ShapeVector* out_attrs) {
  const SampleMultinomialParam& param = nnvm::get<SampleMultinomialParam>(attrs.parsed);

  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), param.get_prob ? 2U : 1U);
  const mxnet::TShape& ishape = (*in_attrs)[0];
  if (!ndim_is_known(ishape)) return false;

  if (ishape.ndim() == 1) {
    if (param.shape.ndim() > 0) {
      SHAPE_ASSIGN_CHECK(*out_attrs, 0, param.shape);
      if (param.get_prob) SHAPE_ASSIGN_CHECK(*out_attrs, 1, param.shape);
    } else {
      SHAPE_ASSIGN_CHECK(*out_attrs, 0, mxnet::TShape(1, 1));
      if (param.get_prob) SHAPE_ASSIGN_CHECK(*out_attrs, 1, mxnet::TShape(1, 1));
    }
    return true;
  }

  mxnet::TShape oshape(ishape.ndim() - 1 + param.shape.ndim(), -1);
  for (int i = 0; i < ishape.ndim() - 1; ++i) {
    oshape[i] = ishape[i];
  }
  for (int i = 0; i < param.shape.ndim(); ++i) {
    oshape[i + ishape.ndim() - 1] = param.shape[i];
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  if (param.get_prob) SHAPE_ASSIGN_CHECK(*out_attrs, 1, oshape);
  for (const auto& out_shape : *out_attrs) {
    if (!shape_is_known(out_shape)) return false;
  }
  return true;
}


inline bool SampleMultinomialOpType(const nnvm::NodeAttrs& attrs,
                                    std::vector<int>* in_attrs,
                                    std::vector<int>* out_attrs) {
  const SampleMultinomialParam& param = nnvm::get<SampleMultinomialParam>(attrs.parsed);

  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), param.get_prob ? 2U : 1U);
  int itype = (*in_attrs)[0];
  if (itype == -1) return false;

  TYPE_ASSIGN_CHECK(*out_attrs, 0, param.dtype);
  if (param.get_prob) {
    TYPE_ASSIGN_CHECK(*out_attrs, 1, itype);
  }
  return true;
}

struct SampleMultinomialKernel {
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(index_t i, index_t K, index_t M,
                                  DType* dist, float* uniform, float* cum_table,
                                  IType* out, DType* prob) {
    double acc = 0.0;
    // CDF table
    for (index_t c = 0; c < K; ++c) {
      acc += dist[i*K + c];
      cum_table[i*K + c] = static_cast<float>(acc);
    }
    for (index_t j = 0; j < M; ++j) {
      index_t left = 0, right = K;
      index_t middle = left + (right - left) / 2;
      DType loc = static_cast<DType>(uniform[i*M + j]);
      while (right - left > 0) {
        middle = left + (right - left) / 2;
        DType cum_prob = cum_table[i*K + middle];
        if (cum_prob < loc) {
          left = middle + 1;
        } else {
          right = middle;
        }
      }
      out[i*M + j] = static_cast<IType>(left);
      if (prob != nullptr) prob[i*M + j] = logf(dist[i*K + left]);
    }
  }
};


template<typename xpu>
void SampleMultinomialForward(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<TBlob>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  const SampleMultinomialParam& param = nnvm::get<SampleMultinomialParam>(attrs.parsed);

  index_t K = inputs[0].shape_[inputs[0].ndim()-1];
  index_t N = inputs[0].Size()/K;
  index_t M = outputs[0].Size()/N;

  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    Random<xpu, float> *prnd = ctx.requested[0].get_random<xpu, float>(s);
    Tensor<xpu, 1, float> workspace =
      ctx.requested[1].get_space_typed<xpu, 1, float>(Shape1(N*M + N*K), s);
    Tensor<xpu, 1, float> uniform(workspace.dptr_, Shape1(N*M));
    prnd->SampleUniform(&uniform, 0, 1);
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, IType, {
      Kernel<SampleMultinomialKernel, xpu>::Launch(
        s, N, K, M, inputs[0].dptr<DType>(), uniform.dptr_, workspace.dptr_ + N*M,
        outputs[0].dptr<IType>(),
        param.get_prob ? outputs[1].dptr<DType>() : nullptr);
    });
  });
}


template<typename kernel, typename xpu>
void SampleMultinomialBackward(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx,
                               const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  if (req[0] == kNullOp) return;

  index_t K = outputs[0].shape_[outputs[0].ndim()-1];
  index_t N = outputs[0].Size()/K;
  index_t M = inputs[0].Size()/N;

  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    if (req[0] != kAddTo) {
      Tensor<xpu, 1, DType> out = outputs[0].FlatTo1D<xpu, DType>(s);
      out = 0;
    }
    MSHADOW_TYPE_SWITCH(inputs[2].type_flag_, IType, {
      Kernel<kernel, xpu>::Launch(
        s, N, K, M, inputs[0].dptr<DType>(), inputs[1].dptr<DType>(),
        inputs[2].dptr<IType>(), outputs[0].dptr<DType>());
    });
  });
}


}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_RANDOM_SAMPLE_MULTINOMIAL_OP_H_
