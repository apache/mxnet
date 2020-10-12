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
 * \file np_unpackbits_op.cc
 * \brief CPU implementation of numpy-compatible unpackbits operator
 */

#include "./np_unpackbits_op-inl.h"

namespace mxnet {
namespace op {


DMLC_REGISTER_PARAMETER(NumpyUnpackbitsParam);

NNVM_REGISTER_OP(_npi_unpackbits)
.set_attr_parser(ParamParser<NumpyUnpackbitsParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.add_argument("a", "NDArray-or-Symbol", "Input array")
.set_attr<nnvm::FListInputNames>("FListInputNames",
[](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a"};
})
.set_attr<mxnet::FInferShape>("FInferShape", NumpyUnpackbitsShape)
.set_attr<nnvm::FInferType>("FInferType", NumpyUnpackbitsDType)
.set_attr<FCompute>("FCompute<cpu>", NumpyUnpackbitsForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_arguments(NumpyUnpackbitsParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
