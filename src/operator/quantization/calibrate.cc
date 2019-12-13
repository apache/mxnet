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
 *  Copyright (c) 2019 by Contributors
 * \file calibrate.cc
 * \brief
 */

#include <numeric>
#include "./calibrate-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(CalibrateEntropyParam);

// Given a discrete distribution (may have not been normalized to 1),
// smooth it by replacing zeros with eps multiplied by a scaling factor and taking the
// corresponding amount off the non-zero values.
std::vector<float> SmoothDistribution(const std::vector<float>& p, const float eps = 0.0001) {
  std::vector<size_t> is_zeros(p.size());
  std::vector<size_t> is_nonzeros(p.size());
  {
    auto it = p.begin();
    std::generate(is_zeros.begin(), is_zeros.end(),
                  [&it]() { return static_cast<size_t>(*(it++) == 0.f); });
  }
  {
    auto it = p.begin();
    std::generate(is_nonzeros.begin(), is_nonzeros.end(),
                  [&it]() { return static_cast<size_t>(*(it++) != 0.f); });
  }

  size_t n_zeros = std::accumulate(is_zeros.begin(), is_zeros.end(), 0);
  size_t n_nonzeros = p.size() - n_zeros;
  if (!n_nonzeros) {
    // The discrete probability distribution is malformed. All entries are 0.
    return std::vector<float>();
  }
  float eps1 = eps * static_cast<float>(n_zeros) / static_cast<float>(n_nonzeros);
  if (eps1 >= 1.0) return std::vector<float>();
  auto ret = p;
  for (size_t i = 0; i < p.size(); i++) {
    ret[i] += eps * is_zeros[i] - eps1 * is_nonzeros[i];
  }
  return ret;
}

static float ComputeEntropy(std::vector<float>* p_ptr, std::vector<float>* q_ptr) {
  std::vector<float>& p = *p_ptr;
  std::vector<float>& q = *q_ptr;
  CHECK_EQ(p.size(), q.size());
  float p_sum = std::accumulate(p.begin(), p.end(), 0.f);
  float q_sum = std::accumulate(q.begin(), q.end(), 0.f);
  for (auto& it : p) {
    it = it / p_sum;
  }

  for (auto& it : q) {
    it = it / q_sum;
  }
  float ret = 0;
  for (size_t i = 0; i < p.size(); i++) {
    CHECK(p[i] > 0 && q[i] > 0);
    if (p[i] && q[i]) ret += p[i] * std::log(p[i] / q[i]);
  }
  return ret;
}

void CalibrateComputeCPU(const nnvm::NodeAttrs& attrs, const OpContext& ctx,
                         const std::vector<TBlob>& inputs, const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  const auto& param = nnvm::get<CalibrateEntropyParam>(attrs.parsed);
  const auto& hist = inputs[0];
  const auto& hist_ptr = hist.dptr<float>();
  const auto& hist_edges = inputs[1];
  const auto& hist_edges_ptr = hist_edges.dptr<float>();
  float* const out_threshold = outputs[0].dptr<float>();
  float* const out_divergence = outputs[1].dptr<float>();
  const auto num_bins = hist.Size();
  CHECK_EQ(num_bins + 1, hist_edges.Size());
  int num_quantized_bins = param.num_quantized_bins;

  const int zero_bin_idx = num_bins / 2;
  const int num_half_quantized_bins = num_quantized_bins / 2;
  std::vector<float> thresholds(num_bins / 2 + 1 - num_quantized_bins / 2, 0.f);
  std::vector<float> divergence(thresholds.size(), 0.f);
  #pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (index_t i = num_quantized_bins / 2; i <= zero_bin_idx; i++) {
    const size_t p_bin_idx_start = zero_bin_idx - i;
    const size_t p_bin_idx_stop = zero_bin_idx + i + 1;
    thresholds[i - num_half_quantized_bins] = hist_edges_ptr[p_bin_idx_stop];

    std::vector<size_t> sliced_nd_hist(p_bin_idx_stop - p_bin_idx_start);
    std::vector<float> p(p_bin_idx_stop - p_bin_idx_start);
    p[0] = 0;
    p.back() = 0;
    for (size_t j = 0; j < num_bins; j++) {
      if (j <= p_bin_idx_start) {
        p[0] += hist_ptr[j];
      } else if (j >= p_bin_idx_stop) {
        p.back() += hist_ptr[j];
      } else {
        sliced_nd_hist[j - p_bin_idx_start] = hist_ptr[j];
        p[j - p_bin_idx_start] = hist_ptr[j];
      }
    }
    // calculate how many bins should be merged to generate quantized distribution q
    const auto num_merged_bins = sliced_nd_hist.size() / num_quantized_bins;
    // merge hist into num_quantized_bins bins
    std::vector<float> quantized_bins(num_quantized_bins, 0);
    for (index_t j = 0; j < num_quantized_bins; j++) {
      const int start = j * num_merged_bins;
      const int stop = (j + 1) * num_merged_bins;
      quantized_bins[j] =
          std::accumulate(sliced_nd_hist.begin() + start, sliced_nd_hist.begin() + stop, 0);
    }
    quantized_bins.back() += std::accumulate(
        sliced_nd_hist.begin() + static_cast<int>(num_quantized_bins * num_merged_bins),
        sliced_nd_hist.end(), 0);
    // expand quantized_bins into p.size bins
    std::vector<float> q(sliced_nd_hist.size(), 0);
    for (index_t j = 0; j < num_quantized_bins; j++) {
      const int start = j * num_merged_bins;
      const int stop = (j == num_quantized_bins - 1) ? q.size() : ((j + 1) * num_merged_bins);
      int norm = std::count_if(sliced_nd_hist.begin() + start, sliced_nd_hist.begin() + stop,
                               [](size_t i) { return i != 0; });
      if (norm) {
        for (index_t k = start; k < stop; k++) {
          if (p[k]) q[k] = quantized_bins[j] / norm;
        }
      }
    }
    p = SmoothDistribution(p);
    q = SmoothDistribution(q);

    if (!q.size()) {
      divergence[i - num_half_quantized_bins] = std::numeric_limits<float>::infinity();
    } else {
      divergence[i - num_half_quantized_bins] = ComputeEntropy(&p, &q);
    }
  }

  size_t min_divergence_idx = 0;
  float min_divergence = mshadow::red::limits::MaxValue<float>();
  for (size_t i = 0; i < divergence.size(); i++) {
    if (divergence[i] < min_divergence) {
      min_divergence = divergence[i];
      min_divergence_idx = i;
    }
  }
  *out_divergence = min_divergence;
  *out_threshold = thresholds[min_divergence_idx];
}

static inline bool CalibrateShape(const nnvm::NodeAttrs& attrs, std::vector<TShape>* in_attrs,
                                  std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 2U);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape(1, 1));
  SHAPE_ASSIGN_CHECK(*out_attrs, 1, TShape(1, 1));
  return (!shape_is_none(in_attrs->at(0))) && (!shape_is_none(in_attrs->at(1)));
}

static inline bool CalibrateType(const nnvm::NodeAttrs& attrs, std::vector<int>* in_attrs,
                                 std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 2U);
  CHECK(in_attrs->at(0) == mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*in_attrs, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_attrs, 1, mshadow::kFloat32);
  return true;
}

NNVM_REGISTER_OP(_contrib_calibrate_entropy)
.describe(R"code(Provide calibrated min/max for input histogram.

.. Note::
    This operator only supports forward propagation. DO NOT use it in training.)code" ADD_FILELINE)
.set_attr_parser(ParamParser<CalibrateEntropyParam>)
.set_num_inputs(2)
.set_num_outputs(2)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"hist", "hist_edges"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"threshold", "divergence"};
})
.set_attr<mxnet::FInferShape>("FInferShape", CalibrateShape)
.set_attr<nnvm::FInferType>("FInferType", CalibrateType)
.set_attr<FCompute>("FCompute<cpu>", CalibrateComputeCPU)
.add_argument("hist", "NDArray-or-Symbol", "A ndarray/symbol of type `float32`")
.add_argument("hist_edges", "NDArray-or-Symbol", "A ndarray/symbol of type `float32`")
.add_arguments(CalibrateEntropyParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
