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
 * Copyright (c) 2019 by Contributors
 * \file np_norm_backward.cc
 * \brief CPU registration of np.linalg.norm
 */

#include "./np_norm-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_backward_npi_norm)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NumpyNormParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
     return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_num_inputs(2 * 4 + 1)
.set_attr<FCompute>("FCompute<cpu>", NumpyNormComputeBackward<cpu>);

}  // namespace op
}  // namespace mxnet
