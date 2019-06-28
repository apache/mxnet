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
  dmlc::optional<mxnet::Tuple<double>> pvals;
  dmlc::optional<mxnet::Tuple<int>> size;
  DMLC_DECLARE_PARAMETER(NumpyMultinomialParam) {
    DMLC_DECLARE_FIELD(n)
      .describe("Number of experiments.");
    DMLC_DECLARE_FIELD(pvals)
      .set_default(dmlc::optional<mxnet::Tuple<double>>())
      .describe("Probabilities of each of the p different outcomes. "
      "These should sum to 1 (however, the last element is always assumed to "
      "account for the remaining probability, as long as sum(pvals[:-1]) <= 1)"
      "Note that this is for internal usage only. "
      "This operator will only have either input mx.ndarray or this list of pvals");
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
  CHECK_EQ(out_attrs->size(), 1U);

  std::vector<dim_t> oshape_vec;
  dim_t pvals_length;
  if (param.pvals.has_value()) {
    CHECK_EQ(in_attrs->size(), 0U);
    pvals_length = param.pvals.value().ndim();
  } else {
    // pvals is from input ndarray
    CHECK_EQ(in_attrs->size(), 1U);
    const TShape& ishape = (*in_attrs)[0];
    // check the input shape is only one dimension
    CHECK_EQ(ishape.ndim(), 1U)
      << "object too deep for desired array";
    pvals_length = ishape[0];
  }
  if (param.size.has_value()) {
    const mxnet::Tuple<int>& size = param.size.value();
    for (int i = 0; i < size.ndim(); ++i) {
      oshape_vec.emplace_back(size[i]);
    }
  }
  oshape_vec.emplace_back(pvals_length);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape(oshape_vec));
  return out_attrs->at(0).ndim() != 0U;;
}

inline bool NumpyMultinomialOpType(const nnvm::NodeAttrs& attrs,
                                    std::vector<int>* in_attrs,
                                    std::vector<int>* out_attrs) {
  const NumpyMultinomialParam& param = nnvm::get<NumpyMultinomialParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), (param.pvals.has_value()) ? 0U : 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  (*out_attrs)[0] = mshadow::kInt64;
  return true;
}

struct multinomial_kernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i,
                                  const int num_exp,
                                  const int prob_length,
                                  DType* pvals,
                                  double* uniform,
                                  int64_t* out) {
    for (int j = 0; j < num_exp; ++j) {
      DType loc = static_cast<DType>(uniform[i * num_exp + j]);
      DType acc = 0.0;
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
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(inputs.size(), (param.pvals.has_value()) ? 0U : 1U);

  int prob_length = (param.pvals.has_value())
    ? param.pvals.value().ndim() : inputs[0].shape_[0];
  // if intput is [] or size contains 0 dimension
  if (prob_length == 0U || outputs[0].shape_.Size() == 0) return;
  int num_output = outputs[0].Size() / prob_length;
  int num_exp = param.n;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  Random<xpu, double> *prnd = ctx.requested[0].get_random<xpu, double>(s);
  size_t temp_space_ = (param.pvals.has_value())
                      ? num_output * param.n + prob_length : num_output * param.n;
  Tensor<xpu, 1, double> temp_tensor =
      ctx.requested[1].get_space_typed<xpu, 1, double>(Shape1(temp_space_), s);

  prnd->SampleUniform(&temp_tensor, 0, 1);
  // set zero for the outputs
  Kernel<set_zero, xpu>::Launch(s, outputs[0].Size(), outputs[0].dptr<int64_t>());
  if (param.pvals.has_value()) {
    // create a tensor to copy the param.pvals tuple to avoid
    // error: calling a __host__ function from a __host__ __device__ function is not allowed
    // reuse the uniform temp space to create pval tensor
    double* pvals_ = temp_tensor.dptr_ + num_output * param.n;
    // check if sum of input(pvals) > 1.0
    double sum = 0.0;
    for (int i = 0; i < prob_length; ++i) {
        sum += param.pvals.value()[i];
        // copy the tuple to data for later kernel usage
        pvals_[i] = param.pvals.value()[i];
        CHECK_LE(sum, 1.0)
          << "sum(pvals[:-1]) > 1.0";
    }
    Kernel<multinomial_kernel, xpu>::Launch(
      s, num_output, num_exp, prob_length, pvals_, temp_tensor.dptr_, outputs[0].dptr<int64_t>());
  } else {
    MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
      // check if sum of input(pvals) > 1.0
      DType sum = DType(0);
      DType* input = inputs[0].dptr<DType>();
      for (int i = 0; i < prob_length; ++i) {
        sum += input[i];
        CHECK_LE(sum, 1.0)
          << "sum(pvals[:-1]) > 1.0";
      }
      Kernel<multinomial_kernel, xpu>::Launch(
        s, num_output, num_exp, prob_length,
        inputs[0].dptr<DType>(), temp_tensor.dptr_, outputs[0].dptr<int64_t>());
    });
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_RANDOM_NP_MULTINOMIAL_OP_H_
