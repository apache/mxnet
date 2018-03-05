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
 * \file shuffle_op.cu
 * \brief Operator to shuffle elements of an NDArray
 */
#include "./shuffle_op.h"

namespace mxnet {
namespace op {

// Note that the outcomes may not be distributed uniformly if both of the the batch size
// and the length of the last axis are very large. It is because the probability that
// there are duplicated keys is not negligible in that case.
struct ShuffleGPUImpl {
  using KeyType = double;  // `float` does not provide enough precision

  struct AddBatchIndexKernel {
    template<typename DType>
    MSHADOW_XINLINE static void Map(int i, DType* keys, size_t n_elements) {
      keys[i] += i / n_elements;
    }
  };

  template<typename DType>
  static void shuffle(const OpContext& ctx,
                      mshadow::Tensor<gpu, 1, DType> out,
                      size_t n_batches,
                      size_t n_elements) {
    using namespace mxnet_op;
    Stream<gpu> *s = ctx.get_stream<gpu>();
    size_t size = n_batches * n_elements;
    Random<gpu, KeyType> *prnd = ctx.requested[0].get_random<gpu, KeyType>(s);
    Tensor<gpu, 1, KeyType> keys =
      ctx.requested[1].get_space_typed<gpu, 1, KeyType>(Shape1(size), s);
    prnd->SampleUniform(&keys, 0, 1);
    if (n_batches > 1) {
      Kernel<AddBatchIndexKernel, gpu>::Launch(s, size, keys.dptr_, n_elements);
    }
    SortByKey(keys, out, true);
  }
};

NNVM_REGISTER_OP(_shuffle)
.set_attr<FCompute>("FCompute<gpu>", ShuffleForward<gpu, ShuffleGPUImpl>);

}  // namespace op
}  // namespace mxnet
