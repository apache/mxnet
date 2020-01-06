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
 * \file index_copy-inl.h
 * \brief implementation of index_copy tensor operation
 */

#ifndef MXNET_OPERATOR_CONTRIB_INDEX_COPY_INL_H_
#define MXNET_OPERATOR_CONTRIB_INDEX_COPY_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <limits>
#include <algorithm>
#include "../elemwise_op_common.h"
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../tensor/init_op.h"

namespace mxnet {
namespace op {

template<typename xpu>
void IndexCopyForward(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs);

template<typename xpu>
void IndexCopyBackward(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs);

inline bool IndexCopyShape(const nnvm::NodeAttrs& attrs,
                           mxnet::ShapeVector *in_attrs,
                           mxnet::ShapeVector *out_attrs) {
  // inputs[0]: original tensor
  // inputs[1]: index vector
  // inputs[2]: copied tensor
  CHECK_EQ(in_attrs->size(), 3U);
  // outputs[0]: a new tensor
  CHECK_EQ(out_attrs->size(), 1U);
  // inputs[1] must be a vector
  CHECK_EQ(in_attrs->at(1).ndim(), 1);
  // Shape matching
  CHECK_EQ(in_attrs->at(0).ndim(), in_attrs->at(2).ndim());
  for (int i = 0; i < in_attrs->at(0).ndim(); ++i) {
    if (i == 0) {
      CHECK_GE(in_attrs->at(0)[i], in_attrs->at(2)[i]);
    } else {
      CHECK_EQ(in_attrs->at(0)[i], in_attrs->at(2)[i]);
    }
  }
  // The the length of the first dim of copied tensor
  // must equal to the size of index vector
  CHECK_EQ(in_attrs->at(1)[0], in_attrs->at(2)[0]);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  SHAPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  return !mxnet::op::shape_is_none(out_attrs->at(0));
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_INDEX_COPY_INL_H_
