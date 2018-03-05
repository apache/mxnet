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
 * Copyright (c) 2018 by Contributors
 * \file shuffle_op.cc
 * \brief Operator to shuffle elements of an NDArray
 */
#if (__GNUC__ > 4 && !defined(__clang__major__)) || (__clang_major__ > 4 && __linux__)
  #define USE_GNU_PARALLEL_SHUFFLE
#endif

#include <random>
#include <algorithm>
#ifdef USE_GNU_PARALLEL_SHUFFLE
  #include <parallel/algorithm>
#endif
#include "./shuffle_op.h"

namespace mxnet {
namespace op {

struct ShuffleCPUImpl {
  template<typename DType>
  static void shuffle(const OpContext& ctx,
                      mshadow::Tensor<cpu, 1, DType> out,
                      size_t n_batches,
                      size_t n_elements) {
    using namespace mxnet_op;
    auto& prnd = ctx.requested[0].get_random<cpu, size_t>(ctx.get_stream<cpu>())->GetRndEngine();
    for (size_t i = 0; i < n_batches; ++i) {
      #ifdef USE_GNU_PARALLEL_SHUFFLE
        __gnu_parallel::random_shuffle(out.dptr_ + i * n_elements,
                                       out.dptr_ + (i + 1) * n_elements,
                                       [&prnd](size_t n) {
                                         std::uniform_int_distribution<size_t> dist(0, n - 1);
                                         return dist(prnd);
                                       });
      #else
        std::shuffle(out.dptr_ + i * n_elements,
                     out.dptr_ + (i + 1) * n_elements,
                     prnd);
      #endif
    }
  }
};

// No parameter is declared.
// No backward computation is registered. Shuffling is not differentiable.

NNVM_REGISTER_OP(_shuffle)
.add_alias("shuffle")
.describe(R"code(Randomly shuffle the elements.

This shuffles the elements along the last axis, i.e., for each element,
all indices except the last one are preserved but the last one changes randomly.
)code")
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const nnvm::NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kRandom, ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int>>{{0, 0}};
  })
.set_attr<FCompute>("FCompute<cpu>", ShuffleForward<cpu, ShuffleCPUImpl>)
.add_argument("data", "NDArray-or-Symbol", "Data to be shuffled.");

}  // namespace op
}  // namespace mxnet
