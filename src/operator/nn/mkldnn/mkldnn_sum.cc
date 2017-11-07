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
 * \file mkldnn_sum.cc
 * \brief
 * \author Da Zheng
*/
#include <iostream>

#include "./mkldnn_ops-inl.h"
#include "./mkldnn_base-inl.h"

#if MXNET_USE_MKLDNN == 1
namespace mxnet {
namespace op {

void Sum(const mkldnn::memory &arr1, const mkldnn::memory &arr2,
    const mkldnn::memory &out) {
  std::vector<mkldnn::memory::primitive_desc> input_pds(2);
  std::vector<double> scales(2);
  std::vector<mkldnn::primitive::at> inputs;
  input_pds[0] = arr1.get_primitive_desc();
  input_pds[1] = arr2.get_primitive_desc();
  CHECK(input_pds[0] == input_pds[1]);
  scales[0] = 1;
  scales[1] = 1;
  inputs.push_back(arr1);
  inputs.push_back(arr2);
  mkldnn::sum::primitive_desc sum_pd(scales, input_pds);
  MKLDNNStream::Instance().RegisterPrim(mkldnn::sum(sum_pd, inputs, out));
}

}
}
#endif
