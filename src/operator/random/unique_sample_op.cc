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
 * \brief CPU Implementation of unique sample op
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
  .set_attr<FResourceRequest>("FResourceRequest", UniqueSampleResource)    \
  .add_arguments(ParamType::__FIELDS__())

MXNET_OPERATOR_REGISTER_UNIQUE_SAMPLE(_sample_unique_zipfian,
                                      SampleUniqueZifpianParam)
.describe(R"code(Draw random samples from an an approximately log-uniform
or Zipfian distribution without replacement.

This operation takes a 2-D shape `(batch_size, num_sampled)`,
and randomly generates *num_sampled* samples from the range of integers [0, range_max)
for each instance in the batch.

The elements in each instance are drawn without replacement from the base distribution.
The base distribution for this operator is an approximately log-uniform or Zipfian distribution:

  P(class) = (log(class + 2) - log(class + 1)) / log(range_max + 1)

Additionaly, it also returns the number of trials used to obtain `num_sampled` samples for
each instance in the batch.

Example::

   samples, trials = _sample_unique_zipfian(750000, shape=(4, 8192))
   unique(samples[0]) = 8192
   unique(samples[3]) = 8192
   trials[0] = 16435

)code" ADD_FILELINE)
.set_attr<mxnet::FInferShape>("FInferShape", SampleUniqueShape<SampleUniqueZifpianParam>)
.set_attr<nnvm::FInferType>("FInferType", SampleUniqueType<SampleUniqueZifpianParam>)
.set_attr<FCompute>("FCompute<cpu>", SampleUniqueZifpian);

}  // namespace op
}  // namespace mxnet
