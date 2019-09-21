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
 * \file slice-inl.h
 * \brief
 * \author Zhiyuan Huang
*/

#ifndef MXNET_OPERATOR_TENSOR_SLICE_INL_H_
#define MXNET_OPERATOR_TENSOR_SLICE_INL_H_

#include <utility>
#include <vector>
#include <string>

namespace mxnet {
namespace op {

struct SliceParam : public dmlc::Parameter<SliceParam> {
  mxnet::Tuple<dmlc::optional<index_t>> begin, end;
  mxnet::Tuple<dmlc::optional<index_t>> step;
  DMLC_DECLARE_PARAMETER(SliceParam) {
    DMLC_DECLARE_FIELD(begin)
    .describe("starting indices for the slice operation, supports negative indices.");
    DMLC_DECLARE_FIELD(end)
    .describe("ending indices for the slice operation, supports negative indices.");
    DMLC_DECLARE_FIELD(step)
    .set_default(mxnet::Tuple<dmlc::optional<index_t>>())
    .describe("step for the slice operation, supports negative values.");
  }
  bool operator==(const SliceParam& other) const {
    return this->begin == other.begin &&
           this->end == other.end &&
           this->step == other.step;
  }
};

}  // namespace op
}  // namespace mxnet

namespace std {
template<>
struct hash<mxnet::op::SliceParam> {
  size_t operator()(const mxnet::op::SliceParam& val) {
    size_t ret = 0;
    ret = dmlc::HashCombine(ret, val.begin);
    ret = dmlc::HashCombine(ret, val.end);
    ret = dmlc::HashCombine(ret, val.step);
    return ret;
  }
};
}  // namespace std

#endif  // MXNET_OPERATOR_TENSOR_SLICE_INL_H_
