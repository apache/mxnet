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
  * \file index_copy.cc
  * \brief
  */
 #include "./index_copy-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_contrib_index_copy)
.describe(R"code(Implementation of index_copy)code" ADD_FILELINE)
.set_num_inputs(3)
.set_num_outputs(0)
.set_attr<nnvm::FInferShape>("FInferShape", IndexCopyShape)
.set_attr<nnvm::FInferType>("FInferType", IndexCopyType)
.set_attr<FCompute>("FCompute<cpu>", IndexCopyCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_index_copy"})
.add_argument("old_tensor", "NDArray-or-Symbol", "Old tensor")
.add_argument("index_tensor", "NDArray-or-Symbol", "Index vector")
.add_argument("new_tensor", "NDArray-or-Symbol", "New tensor");

}  // namespace op
}  // namespace mxnet