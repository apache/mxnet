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
 *  Copyright (c) 2016 by Contributors
 * \file sample_op.cc
 * \brief CPU Implementation of sample op
 */

#include "./unique_sample_op.h"
#include "../tensor/init_op.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(SampleUniqueZifpianParam);

#define MXNET_OPERATOR_REGISTER_UNIQUE_SAMPLE(name, ParamType)             \
  NNVM_REGISTER_OP(name)                                                   \
  .set_num_inputs(0)                                                       \
  .set_num_outputs(2)                                                      \
  .set_attr_parser(ParamParser<ParamType>)                                 \
  .set_attr<FResourceRequest>("FResourceRequest", UniqueSampleResource) \
  .add_arguments(ParamType::__FIELDS__())

MXNET_OPERATOR_REGISTER_UNIQUE_SAMPLE(_sample_unique_log_uniform,
                                      SampleUniqueZifpianParam)
.describe(R"code(Draw unique random samples from a uniform distribution.

Samples are uniformly distributed over the half-open interval *[low, high)*
(includes *low*, but excludes *high*).

Example::

   uniform(low=0, high=1, shape=(2,2)) = [[ 0.60276335,  0.85794562],
                                          [ 0.54488319,  0.84725171]]

)code" ADD_FILELINE)
.set_attr<nnvm::FInferShape>("FInferShape", SampleUniqueShape<SampleUniqueZifpianParam>)
.set_attr<nnvm::FInferType>("FInferType", SampleUniqueType<SampleUniqueZifpianParam>)
.set_attr<FCompute>("FCompute<cpu>", SampleUniqueZifpian);

}  // namespace op
}  // namespace mxnet
