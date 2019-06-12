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
 * \file np_multinomial_op.h
 * \brief Operator for sampling from multinomial distributions
 */
#ifndef MXNET_OPERATOR_NUMPY_NP_RANDOM_MULTINOMIAL_OP_H_
#define MXNET_OPERATOR_NUMPY_NP_RANDOM_MULTINOMIAL_OP_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "../../mshadow_op.h"
#include "../../mxnet_op.h"
#include "../../operator_common.h"
#include "../../elemwise_op_common.h"

namespace mxnet {
namespace op {

struct NumpyMultinomialParam : public dmlc::Parameter<NumpyMultinomialParam> {
  int n;
  mxnet::Tuple<float> pvals;
  dmlc::optional<mxnet::Tuple<int>> size;
  DMLC_DECLARE_PARAMETER(NumpyMultinomialParam) {
    DMLC_DECLARE_FIELD(n)
      .describe("Number of experiments.");
    DMLC_DECLARE_FIELD(pvals)
      .set_default(mxnet::Tuple<float>())
      .describe("Probabilities of each of the p different outcomes. "
      "These should sum to 1 ""(however, the last element is always assumed to account for the remaining probability, "
      "as long as sum(pvals[:-1]) <= 1).");
    DMLC_DECLARE_FIELD(size)
      .set_default(dmlc::optional<mxnet::Tuple<int>>())
      .describe("Output shape. If the given shape is, "
      "e.g., (m, n, k), then m * n * k samples are drawn. "
      "Default is None, in which case a single value is returned.");
  }
};


inline bool NumpyMultinomialOpShape(const nnvm::NodeAttrs& attrs,
                                     std::vector<TShape> *in_attrs,
                                     std::vector<TShape> *out_attrs) {
  const NumpyMultinomialParam& param = nnvm::get<NumpyMultinomialParam>(attrs.parsed);

  CHECK_EQ(in_attrs->size(), 0U);
  CHECK_EQ(out_attrs->size(), 1U);

  std::vector<dim_t> oshape_vec;
  if (param.size.has_value()) {
    const mxnet::Tuple<int>& size = param.size.value();
    for (int i = 0; i < size.ndim(); ++i) {
      oshape_vec.emplace_back(size[i]);
    }
  }
  oshape_vec.emplace_back(param.pvals.ndim());
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape(oshape_vec));
  return true;
}


inline bool NumpyMultinomialOpType(const nnvm::NodeAttrs& attrs,
                                    std::vector<int>* in_attrs,
                                    std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 0U);
  CHECK_EQ(out_attrs->size(), 1U);

  (*out_attrs)[0] = mshadow::kInt32;  
  return true;
}

struct MultinomialKernel {
  MSHADOW_XINLINE static void Map(int i,
                                  const int num_exp,
                                  const mxnet::Tuple<float>& pvals,
                                  float* uniform,
                                  int* out) {
    for (int j = 0; j < num_exp; ++j) {
      float loc = uniform[i * num_exp + j];
      float acc = 0.0;
      bool found = false;
      for (uint32_t k = 0; k < pvals.ndim(); ++k) {
        acc += pvals[k];
        if (acc > loc) {
          found = true;
          out[i * pvals.ndim() + k] += 1;
          break;
        }
      }
      if (!found) {
        out[i * pvals.ndim() + (pvals.ndim() - 1)] += 1;
      }
    }
  }
};


template<typename xpu>
void NumpyMultinomialForward(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<TBlob>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  const NumpyMultinomialParam& param = nnvm::get<NumpyMultinomialParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 0U);
  CHECK_EQ(outputs.size(), 1U);
  index_t prob_length = param.pvals.ndim();
  index_t num_output = outputs[0].Size() / prob_length;
  index_t num_exp = param.n;

  Stream<xpu> *s = ctx.get_stream<xpu>();
  Random<xpu, float> *prnd = ctx.requested[0].get_random<xpu, float>(s);
  Tensor<xpu, 1, float> uniform =
      ctx.requested[1].get_space_typed<xpu, 1, float>(Shape1(num_output * param.n), s);
  prnd->SampleUniform(&uniform, 0, 1);
  // set zero for the outputs
  Kernel<set_zero, xpu>::Launch(s, outputs[0].Size(), outputs[0].dptr<int>());
  Kernel<MultinomialKernel, xpu>::Launch(
        s, num_output, num_exp, param.pvals, uniform.dptr_, outputs[0].dptr<int>());
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

#endif  // MXNET_OPERATOR_NUMPY_NP_RANDOM_MULTINOMIAL_OP_H_
