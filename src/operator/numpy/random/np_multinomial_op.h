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
#ifndef MXNET_OPERATOR_NUMPY_RANDOM_NP_MULTINOMIAL_OP_H_
#define MXNET_OPERATOR_NUMPY_RANDOM_NP_MULTINOMIAL_OP_H_

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
  dmlc::optional<mxnet::Tuple<int>> size;
  DMLC_DECLARE_PARAMETER(NumpyMultinomialParam) {
    DMLC_DECLARE_FIELD(n)
      .describe("Number of experiments.");
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

  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  const TShape& ishape = (*in_attrs)[0];
  // check the input shape is only one dimension
  CHECK_GE(ishape.ndim(), 1U)
    << "object too deep for desired array";

  std::vector<dim_t> oshape_vec;
  if (param.size.has_value()) {
    const mxnet::Tuple<int>& size = param.size.value();
    for (int i = 0; i < size.ndim(); ++i) {
      oshape_vec.emplace_back(size[i]);
    }
  }
  oshape_vec.emplace_back(ishape.ndim());
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape(oshape_vec));
  return out_attrs->at(0).ndim() != 0U;;
}

inline bool NumpyMultinomialOpType(const nnvm::NodeAttrs& attrs,
                                    std::vector<int>* in_attrs,
                                    std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  (*out_attrs)[0] = mshadow::kInt64;
  return true;
}

struct multinomial_kernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i,
                                  const int num_exp,
                                  const index_t prob_length,
                                  DType* pvals,
                                  float* uniform,
                                  int64_t* out) {
    for (int j = 0; j < num_exp; ++j) {
      float loc = uniform[i * num_exp + j];
      float acc = 0.0;
      bool found = false;
      for (int k = 0; k < prob_length; ++k) {
        acc += pvals[k];
        if (acc > loc) {
          found = true;
          out[i * prob_length + k] += 1;
          break;
        }
      }
      if (!found) {
        out[i * prob_length + (prob_length - 1)] += 1;
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
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);

  // if intput is [] or size contains 0 dimension
  if (inputs[0].shape_.Size() == 0 || outputs[0].shape_.Size() == 0) return;
  index_t prob_length = inputs[0].shape_[0];
  index_t num_output = outputs[0].Size() / prob_length;
  index_t num_exp = param.n;

  Stream<xpu> *s = ctx.get_stream<xpu>();
  Random<xpu, float> *prnd = ctx.requested[0].get_random<xpu, float>(s);
  Tensor<xpu, 1, float> uniform =
      ctx.requested[1].get_space_typed<xpu, 1, float>(Shape1(num_output * param.n), s);
  prnd->SampleUniform(&uniform, 0, 1);
  MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    // check if sum of input(pvals) > 1.0
    DType sum = DType(0);
    DType* input = inputs[0].dptr<DType>();
    for (int i = 0; i < prob_length; ++i) {
      sum += input[i];
      CHECK_GE(sum, 1.0)
        << "sum(pvals[:-1]) > 1.0";
    }
    // set zero for the outputs
    Kernel<set_zero, xpu>::Launch(s, outputs[0].Size(), outputs[0].dptr<int64_t>());
    Kernel<multinomial_kernel, xpu>::Launch(
      s, num_output, num_exp, prob_length, 
      inputs[0].dptr<DType>(), uniform.dptr_, outputs[0].dptr<int64_t>());
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_RANDOM_NP_MULTINOMIAL_OP_H_
