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
 * \file min_ex.cc
 * \brief example external operator source file
 */

#include "min_ex-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(min_ex)
.describe("some description")
.set_num_inputs(0)
.set_num_outputs(0)
.set_attr<mxnet::FInferShape>("FInferShape", MinExOpShape)
.set_attr<nnvm::FInferType>("FInferType", MinExOpType)
.set_attr<FCompute>("FCompute<cpu>", MinExForward<cpu>);

}  // namespace op                                                                                                                                                                     
}  // namespace mxnet 
