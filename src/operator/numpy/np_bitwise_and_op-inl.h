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
 *  Copyright (c) 2019 by Contributors
 * \file np_bitwise_and_op-inl.h
 * \brief Function definition of element-wise binary operator: bitwise_and
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_BITWISE_AND_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_BITWISE_AND_OP_INL_H_

#include <vector>
#include "../mxnet_op.h"
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"  //ElemenwiseShape, ElemenwiseType
#include "../operator_common.h"

namespace mxnet{
namespace op {

/*!
 * \brief kernel struct
 */
template<int req>
struct bitwise_and_forward {
    template<typename DType>
    MSHADOW_XINLINE static void Map(int i, DType* out, const DType* lhs, const DType* rhs) {
        KERNEL_ASSIGN(out[i], req, lhs[i] & rhs[i]);
    }
};

} // namespace op
} // namespace mxnet
#endif
