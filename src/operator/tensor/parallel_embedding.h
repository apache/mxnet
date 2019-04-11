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
#ifndef MXNET_OPERATOR_TENSOR_PARALLEL_EMBEDDING_H_
#define MXNET_OPERATOR_TENSOR_PARALLEL_EMBEDDING_H_
#include "indexing_op.h"
namespace mxnet {
namespace op {

struct ParallelEmbeddingParam : public dmlc::Parameter<ParallelEmbeddingParam> {
  nnvm::Tuple<int> input_dims;
  nnvm::Tuple<int> output_dims;
  nnvm::Tuple<int> dtypes;
  nnvm::Tuple<int> sparse_grads;
  int num_args;
  DMLC_DECLARE_PARAMETER(ParallelEmbeddingParam) {
    DMLC_DECLARE_FIELD(input_dims)
        .describe("Vocabulary size of the input indices.");
    DMLC_DECLARE_FIELD(output_dims)
        .describe("Dimension of the embedding vectors.");
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(1).set_default(1).describe(
        "Number of inputs to be concated.");
    DMLC_DECLARE_FIELD(dtypes).describe("Data type of weight.");
    DMLC_DECLARE_FIELD(sparse_grads)
        .describe(
            "Compute row sparse gradient in the backward calculation. If set "
            "to True, "
            "the grad's storage type is row_sparse.");
  }
};

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_TENSOR_PARALLEL_EMBEDDING_H_
