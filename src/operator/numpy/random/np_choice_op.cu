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
 * \file np_choice_op.cu
 * \brief Operator for random subset sampling
 */

#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/swap.h>
#include "./np_choice_op.h"

namespace mxnet {
namespace op {

template <>
void _sort<gpu>(float* key, int64_t* data, index_t length) {
  thrust::device_ptr<float> dev_key(key);
  thrust::device_ptr<int64_t> dev_data(data);
  thrust::sort_by_key(dev_key, dev_key + length, dev_data,
                      thrust::greater<float>());
}

NNVM_REGISTER_OP(_npi_choice)
.set_attr<FCompute>("FCompute<gpu>", NumpyChoiceForward<gpu>);

}  // namespace op
}  // namespace mxnet
