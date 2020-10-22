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
 * Copyright (c) 2015-2020 by Contributors
 * \file np_broadcast_reduce-inl.cuh
 * \brief GPU implementations for numpy binary broadcast ops
 * \author Zhaoqi Zhu
*/
#ifndef MXNET_OPERATOR_NUMPY_NP_BROADCAST_REDUCE_OP_CUH_
#define MXNET_OPERATOR_NUMPY_NP_BROADCAST_REDUCE_OP_CUH_

using namespace mshadow::cuda;
using namespace mshadow;
using namespace broadcast;

template<typename Reducer, int NDim, typename DType, typename OType>
void NumpyArgMinMaxReduce(Stream<gpu> *s, const TBlob& in_data, const TBlob& out_data,
                          const Tensor<gpu, 1, char>& workspace) {
  cudaStream_t stream = Stream<gpu>::GetStream(s);
  ReduceImplConfig config(out_data.shape_, in_data.shape_, nullptr, nullptr, sizeof(OType));

  ReduceImpl<Reducer, NDim, OType, DType, OType, mxnet::op::mshadow_op::identity,
             mxnet::op::mshadow_op::arg_min_max_set_index<OType, int>>
            (stream, out_data, kWriteTo, in_data, workspace, config);
}

#endif // MXNET_OPERATOR_NUMPY_NP_BROADCAST_REDUCE_OP_CUH_
