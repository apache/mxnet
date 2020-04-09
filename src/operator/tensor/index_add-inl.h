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
 * \file index_add-inl.h
 * \brief Function definition of index_add operator
*/
#ifndef MXNET_OPERATOR_TENSOR_INDEX_ADD_INL_H_
#define MXNET_OPERATOR_TENSOR_INDEX_ADD_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <dmlc/optional.h>
#include <mshadow/tensor.h>
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <vector>
#include <type_traits>
#include "./util/tensor_util-inl.h"
#include "../elemwise_op_common.h"
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"

namespace mxnet {
namespace op {

struct IndexAddParam : public dmlc::Parameter<IndexAddParam> {
    mxnet::Tuple<mxnet::Tuple<int>> ind;
    DMLC_DECLARE_PARAMETER(IndexAddParam) {
      DMLC_DECLARE_FIELD(ind)
        .describe("Index indicating where the input added values.");
    }
};

inline bool IndexAddOpShape(const nnvm::NodeAttrs& attrs,
                            mxnet::ShapeVector* in_attrs,
                            mxnet::ShapeVector* out_attrs) {
  IndexAddParam param = nnvm::get<IndexAddParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 2);
  return true;
}

inline bool IndexAddOpType(const nnvm::NodeAttrs& attrs,
                           std::vector<int>* in_attrs,
                           std::vector<int>* out_attrs) {
  IndexAddParam param = nnvm::get<IndexAddParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 2);
  for (size_t i = 0; i < in_attrs->size(); ++i) {
    if ((*in_attrs)[i] == -1)
      return false;
  }
  return true;
}

}   // namespace op
}   // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_HISTOGRAM_INL_H_
