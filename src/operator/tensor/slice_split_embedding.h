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

#ifndef MXNET_OPERATOR_TENSOR_SLICE_SPLIT_EMBEDDING_H_
#define MXNET_OPERATOR_TENSOR_SLICE_SPLIT_EMBEDDING_H_

#include <mxnet/operator_util.h>
#include <algorithm>
#include <utility>
#include <string>
#include <vector>
namespace mxnet {
namespace op {

struct SliceSplitEmbeddingConcatFuseParam
    : public dmlc::Parameter<SliceSplitEmbeddingConcatFuseParam> {
  // From SliceParam, do not support step
  // Only support kWriteTo
  mxnet::Tuple<dmlc::optional<int>> cont_begin, cont_end;
  mxnet::Tuple<dmlc::optional<int>> embed_begin, embed_end;
  // From SliceChannelParam, do not support Axis
  int num_outputs;
  bool squeeze_axis;
  // From Embedding, do not support sparse_grads, dtypes is for float
  nnvm::Tuple<int> input_dims;
  nnvm::Tuple<int> output_dims;
  // concat Dim
  int concat_dim;
  DMLC_DECLARE_PARAMETER(SliceSplitEmbeddingConcatFuseParam) {
    DMLC_DECLARE_FIELD(cont_begin)
        .describe(
            "starting indices for the slice operation, just copy to final "
            "buffer");
    DMLC_DECLARE_FIELD(cont_end).describe(
        "ending indices for the slice operation, just copy to final buffer");
    DMLC_DECLARE_FIELD(embed_begin)
        .describe("starting indices for the slice operation, input to split");
    DMLC_DECLARE_FIELD(embed_end).describe(
        "ending indices for the slice operation, input to split");
    DMLC_DECLARE_FIELD(num_outputs)
        .set_lower_bound(1)
        .describe(
            "Number of splits. Note that this should evenly divide the length "
            "of the `axis`.");
    DMLC_DECLARE_FIELD(squeeze_axis).set_default(0);
    DMLC_DECLARE_FIELD(input_dims)
        .describe("Vocabulary size of the input indices.");
    DMLC_DECLARE_FIELD(output_dims)
        .describe("Dimension of the embedding vectors.");
    DMLC_DECLARE_FIELD(concat_dim)
        .set_default(1)
        .describe("the dimension to be concated.");
  }
};
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_SLICE_SPLIT_EMBEDDING_H_
