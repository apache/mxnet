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
 * \file sample_op.cu
 * \brief GPU Implementation of sample op
 */
#include "./sample_op.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_random_uniform)
.set_attr<FCompute>("FCompute<gpu>", Sample_<gpu, UniformSampler<gpu>>)
.set_attr<FComputeEx>("FComputeEx<gpu>", SampleEx_<gpu, UniformSampler<gpu>>);

NNVM_REGISTER_OP(_random_normal)
.set_attr<FCompute>("FCompute<gpu>", Sample_<gpu, NormalSampler<gpu>>)
.set_attr<FComputeEx>("FComputeEx<gpu>", SampleEx_<gpu, NormalSampler<gpu>>);

NNVM_REGISTER_OP(_random_gamma)
.set_attr<FCompute>("FCompute<gpu>", Sample_<gpu, GammaSampler<gpu>>)
.set_attr<FComputeEx>("FComputeEx<gpu>", SampleEx_<gpu, GammaSampler<gpu>>);

NNVM_REGISTER_OP(_random_exponential)
.set_attr<FCompute>("FCompute<gpu>", Sample_<gpu, ExponentialSampler<gpu>>)
.set_attr<FComputeEx>("FComputeEx<gpu>", SampleEx_<gpu, ExponentialSampler<gpu>>);

NNVM_REGISTER_OP(_random_poisson)
.set_attr<FCompute>("FCompute<gpu>", Sample_<gpu, PoissonSampler<gpu>>)
.set_attr<FComputeEx>("FComputeEx<gpu>", SampleEx_<gpu, PoissonSampler<gpu>>);

NNVM_REGISTER_OP(_random_negative_binomial)
.set_attr<FCompute>("FCompute<gpu>", Sample_<gpu, NegativeBinomialSampler<gpu>>)
.set_attr<FComputeEx>("FComputeEx<gpu>", SampleEx_<gpu, NegativeBinomialSampler<gpu>>);

NNVM_REGISTER_OP(_random_generalized_negative_binomial)
.set_attr<FCompute>("FCompute<gpu>", Sample_<gpu, GeneralizedNegativeBinomialSampler<gpu>>)
.set_attr<FComputeEx>("FComputeEx<gpu>", SampleEx_<gpu, GeneralizedNegativeBinomialSampler<gpu>>);

}  // namespace op
}  // namespace mxnet
