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
 *  Copyright (c) 2018 by Contributors
 * \file sample_op.h
 * \brief Elementary unique sampling operators
 */
#ifndef MXNET_OPERATOR_RANDOM_UNIQUE_SAMPLE_OP_H_
#define MXNET_OPERATOR_RANDOM_UNIQUE_SAMPLE_OP_H_

#include <mxnet/operator_util.h>
#include <mshadow/base.h>
#include <string>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "./sampler.h"

namespace mxnet {
namespace op {

struct SampleUniqueZifpianParam : public dmlc::Parameter<SampleUniqueZifpianParam> {
  int range_max;
  mxnet::TShape shape;
  DMLC_DECLARE_PARAMETER(SampleUniqueZifpianParam) {
    DMLC_DECLARE_FIELD(range_max)
    .describe("The number of possible classes.");
    DMLC_DECLARE_FIELD(shape)
    .set_default(mxnet::TShape())
    .describe("2-D shape of the output, where shape[0] is the batch size, and shape[1] "
              "is the number of candidates to sample for each batch.");
  }
};

template<typename ParamType>
inline bool SampleUniqueShape(const nnvm::NodeAttrs& attrs,
                              mxnet::ShapeVector *in_attrs,
                              mxnet::ShapeVector *out_attrs) {
  const ParamType& param = nnvm::get<ParamType>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 0U);
  CHECK_EQ(out_attrs->size(), 2U);
  // output shape is known
  if ((*out_attrs)[0].ndim() == 2 && !mxnet::ndim_is_known(param.shape)) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 1, mshadow::Shape1((*out_attrs)[0][0]));
    return true;
  }
  CHECK_EQ(param.shape.ndim(), 2U);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, param.shape);
  SHAPE_ASSIGN_CHECK(*out_attrs, 1, mshadow::Shape1(param.shape[0]));
  return true;
}

template<typename ParamType>
inline bool SampleUniqueType(const nnvm::NodeAttrs& attrs,
                             std::vector<int> *in_attrs,
                             std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 0U);
  CHECK_EQ(out_attrs->size(), 2U);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kInt64);
  TYPE_ASSIGN_CHECK(*out_attrs, 1, mshadow::kInt64);
  return true;
}

inline std::vector<ResourceRequest> UniqueSampleResource(const NodeAttrs& attrs) {
  return {ResourceRequest::kParallelRandom};
}

/*!
 * \brief Launch a generic kernel with parallel unique random generator
 * \tparam gen random generator
 * \tparam batch_size the batch size
 * \tparam num_sampled the number of unique samples per batch
 * \tparam Args Varargs type to eventually pass to the OP::Map() function
 */
template<typename GType, typename DType, typename OP, typename ...Args>
inline static void LaunchUniqueRNG(mshadow::Stream<cpu> *s,
                                   common::random::RandGenerator<cpu, GType> *gen,
                                   const int batch_size, const size_t num_sampled,
                                   std::vector<std::unordered_set<DType>> *results,
                                   Args... args) {
  // minimal check to avoid division by zero, below.
  // if `N` is zero the map operation is a no-op in any case.
  if (batch_size <= 0 || num_sampled <= 0) return;
  const int nthread = std::min(batch_size, RandGenerator<cpu>::kNumRandomStates);
  const int step = (batch_size + nthread - 1) / nthread;
  Kernel<OP, cpu>::Launch(s, nthread, *gen, batch_size, num_sampled, results, step, args...);
}

struct UniqueSampleUniformKernel {
  template<typename GType, typename DType>
  MSHADOW_XINLINE static void Map(int tid, RandGenerator<cpu, GType> gen,
                                  const int batch_size, const size_t num_sampled,
                                  std::vector<std::unordered_set<DType>> *results,
                                  const int step, const GType log_range_max,
                                  DType *samples, DType *num_tries) {
    const int begin = tid * step;
    const int end = (tid + 1) * step;
    typename RandGenerator<cpu, GType>::Impl generator(&gen, tid);
    for (int i = begin; i < end && i < batch_size; i++) {
      auto &result = results->at(i);
      const int base = i * num_sampled;
      DType tries = 0;
      while (result.size() != num_sampled) {
        const double x = generator.uniform();
        const DType value = static_cast<DType>(lround(exp(x * log_range_max)) - 1);
        // sampling without replacement
        if (result.find(value) == result.end()) {
          samples[base + result.size()] = value;
          result.emplace(value);
        }
        tries += 1;
      }
      num_tries[i] = tries;
    }
  }
};

inline void SampleUniqueZifpian(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<TBlob>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<TBlob>& outputs) {
  using DType = int64_t;
  using GType = double;
  const SampleUniqueZifpianParam& param = nnvm::get<SampleUniqueZifpianParam>(attrs.parsed);
  const int batch_size = param.shape[0];
  const size_t num_sampled = static_cast<size_t>(param.shape[1]);
  const double log_range_max = log(param.range_max);
  CHECK_EQ(outputs.size(), 2U);
  CHECK_LE(num_sampled, param.range_max)
    << "Number of samples cannot exceed the number of possible classes";
  // rand generator resource and result sets
  RandGenerator<cpu, GType> *pgen = ctx.requested[0].get_parallel_random<cpu, GType>();
  std::vector<std::unordered_set<DType>> results(batch_size);
  for (int i = 0; i < batch_size; i++) {
    results[i].reserve(num_sampled);
  }

  DType *num_tries = outputs[1].dptr<DType>();
  DType *samples = outputs[0].dptr<DType>();
  Stream<cpu> *s = ctx.get_stream<cpu>();
  LaunchUniqueRNG<GType, DType, UniqueSampleUniformKernel>(s, pgen, batch_size, num_sampled,
                                                           &results, log_range_max, samples,
                                                           num_tries);
}


}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_RANDOM_UNIQUE_SAMPLE_OP_H_
