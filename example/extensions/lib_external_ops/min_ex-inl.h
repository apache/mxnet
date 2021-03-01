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
 * Copyright (c) 2020 by Contributors
 * \file min_ex-inl.h
 * \brief example external operator header file
 */

#ifndef MXNET_OPERATOR_TENSOR_MIN_EX_OP_INL_H_
#define MXNET_OPERATOR_TENSOR_MIN_EX_OP_INL_H_

#include <dmlc/parameter.h>
#include <vector>
#include <algorithm>
#include "operator/mxnet_op.h"
#include "operator/operator_common.h"
#include "operator/elemwise_op_common.h"

namespace mxnet {
namespace op {

template<typename xpu>
void MinExForward(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
  //do nothing                                                                                                                                                                         
}


inline bool MinExOpShape(const nnvm::NodeAttrs& attrs,
                         mxnet::ShapeVector* in_attrs,
                         mxnet::ShapeVector* out_attrs) {
    //do nothing                                                                                                                                                                       
    return true;
}

inline bool MinExOpType(const nnvm::NodeAttrs& attrs,
                        std::vector<int> *in_attrs,
                        std::vector<int> *out_attrs) {
  //do nothing                                                                                                                                                                         
  return true;
}

}  // namespace op                                                                                                                                                                     
}  // namespace mxnet                                                                                                                                                                  

#endif  // MXNET_OPERATOR_TENSOR_MIN_EX_OP_INL_H_
