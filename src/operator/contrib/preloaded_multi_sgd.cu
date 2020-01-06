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
 * \file preloaded_multi_sgd.cu
 * \brief Multi-sgd optimizers with lrs and wds as mxnet inputs
 * \author Clement Fuji Tsang
 */
#include "./preloaded_multi_sgd-inl.h"
#include <cub/cub.cuh>

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(preloaded_multi_sgd_update)
.set_attr<FCompute>("FCompute<gpu>", PreloadedMultiSGDUpdate<gpu, preloaded_type_identity, 2>);
NNVM_REGISTER_OP(preloaded_multi_sgd_mom_update)
.set_attr<FCompute>("FCompute<gpu>", PreloadedMultiSGDMomUpdate<gpu, preloaded_type_identity, 3>);
NNVM_REGISTER_OP(preloaded_multi_mp_sgd_update)
.set_attr<FCompute>("FCompute<gpu>", PreloadedMultiSGDUpdate<gpu, preloaded_single_precision, 3>);
NNVM_REGISTER_OP(preloaded_multi_mp_sgd_mom_update)
.set_attr<FCompute>("FCompute<gpu>",
                    PreloadedMultiSGDMomUpdate<gpu, preloaded_single_precision, 4>);

}  // namespace op
}  // namespace mxnet
