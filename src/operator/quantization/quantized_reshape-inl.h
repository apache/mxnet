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
 * \file quantized_reshape-inl.h
 * \author: Adam Grabowski, adam.grabowski@intel.com
 */

#ifndef MXNET_OPERATOR_QUANTIZATION_QUANTIZED_RESHAPE_INL_H_
#define MXNET_OPERATOR_QUANTIZATION_QUANTIZED_RESHAPE_INL_H_

#include <string>
#include <vector>
#include "operator/tensor/matrix_op-inl.h"
#include "operator/numpy/np_matrix_op-inl.h"

namespace mxnet {
namespace op {

template <
    bool (*ReshapeShapeFunc)(const nnvm::NodeAttrs&, mxnet::ShapeVector*, mxnet::ShapeVector*)>
inline bool QuantizedReshapeInferShape(const nnvm::NodeAttrs& attrs,
                                       mxnet::ShapeVector* in_attrs,
                                       mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 3U);
  mxnet::ShapeVector input  = {in_attrs->at(0)};
  mxnet::ShapeVector output = {out_attrs->at(0)};

  bool ret = ReshapeShapeFunc(attrs, &input, &output);

  SHAPE_ASSIGN_CHECK(*in_attrs, 1, mxnet::TShape{1});
  SHAPE_ASSIGN_CHECK(*in_attrs, 2, mxnet::TShape{1});
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, output[0]);
  SHAPE_ASSIGN_CHECK(*out_attrs, 1, mxnet::TShape{1});
  SHAPE_ASSIGN_CHECK(*out_attrs, 2, mxnet::TShape{1});

  return ret;
}

inline bool QuantizedReshapeType(const nnvm::NodeAttrs& attrs,
                                 std::vector<int>* in_attrs,
                                 std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 3U);
  TYPE_ASSIGN_CHECK(*in_attrs, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*in_attrs, 2, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, (*in_attrs)[0]);
  TYPE_ASSIGN_CHECK(*out_attrs, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_attrs, 2, mshadow::kFloat32);
  return (*in_attrs)[0] != -1;
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_QUANTIZATION_QUANTIZED_RESHAPE_INL_H_
