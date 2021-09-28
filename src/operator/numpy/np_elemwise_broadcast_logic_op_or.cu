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
 * \file np_elemwise_broadcast_logic_op_or.cu
 * \brief GPU Implementation of basic functions for logical_or operation.
 */

#include "./np_elemwise_broadcast_logic_op.h"

namespace mxnet {
namespace op {

MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_GPU(logical_or);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_GPU(logical_or);

}  // namespace op
}  // namespace mxnet
