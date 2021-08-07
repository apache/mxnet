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
 * \file caffe_op.cc
 * \brief caffe operator
 * \author Haoran Wang
*/
#include "./caffe_op-inl.h"
namespace mxnet {
namespace op {

template<>
Operator* CreateOp<cpu>(CaffeOpParam param, int dtype) {
  Operator *op = NULL;
  switch (dtype) {
  case mshadow::kFloat32:
    op = new CaffeOp<cpu, float>(param);
    break;
  case mshadow::kFloat64:
    op = new CaffeOp<cpu, double>(param);
    break;
  case mshadow::kFloat16:
    LOG(FATAL) << "float16 layer is not supported by caffe";
    break;
  case mshadow::kBfloat16:
    LOG(FATAL) << "bfloat16 layer is not supported by caffe";
    break;
  default:
    LOG(FATAL) << "Unsupported type " << dtype;
  }
  return op;
}

// DO_BIND_DISPATCH comes from static_operator_common.h
Operator *CaffeOpProp::CreateOperatorEx(Context ctx, mxnet::ShapeVector *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<int> out_type, aux_type;
  mxnet::ShapeVector out_shape, aux_shape;
  out_type.resize(this->ListOutputs().size());
  out_shape.resize(this->ListOutputs().size());
  aux_type.resize(this->ListAuxiliaryStates().size());
  aux_shape.resize(this->ListAuxiliaryStates().size());
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(CaffeOpParam);

MXNET_REGISTER_OP_PROPERTY(CaffeOp, CaffeOpProp)
.describe("Apply caffe operator")
.add_argument("data", "Symbol[]", "List of tensors")
.add_arguments(CaffeOpParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
