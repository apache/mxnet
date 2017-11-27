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
 * Copyright (c) 2015 by Contributors
 * \file dropout.cc
 * \brief
 * \author Bing Xu, Da Zheng
*/

#include "./dropout-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(Dropout)
.set_attr<FCompute>("FCompute<gpu>", DropoutCompute<gpu>);

NNVM_REGISTER_OP(_backward_Dropout)
.set_attr<FCompute>("FCompute<gpu>", DropoutGradCompute<gpu>);

}  // namespace op
}  // namespace mxnet


