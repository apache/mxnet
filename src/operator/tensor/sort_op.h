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
 *  Copyright (c) 2017 by Contributors
 * \file sort_op.h
 * \brief SortByKey function
 */
#ifndef MXNET_OPERATOR_TENSOR_SORT_OP_H_
#define MXNET_OPERATOR_TENSOR_SORT_OP_H_

#include <dmlc/logging.h>
#include <mshadow/tensor.h>
#include <vector>
#include <type_traits>

namespace mxnet {
namespace op {
/*!
 * \brief CPU/GPU: Sort key-value pairs stored in separate places. (Stable sort is performed!)
 * \param keys the keys to sort
 * \param values the values that sorts w.r.t the key
 * \param is_ascend whether to sort key in ascending order
 */
template<typename KDType, typename VDType>
inline void SortByKey(mshadow::Tensor<cpu, 1, KDType> keys, mshadow::Tensor<cpu, 1, VDType> values,
                      bool is_ascend = true, mshadow::Tensor<cpu, 1, char>* workspace = NULL,
                      const int begin_bit = 0, const int end_bit = sizeof(KDType)*8) {
  CHECK_EQ(keys.CheckContiguous(), true);
  CHECK_EQ(values.CheckContiguous(), true);
  CHECK_EQ(keys.size(0), values.size(0))
    << "The sizes of key/value are not equal! keys_size: " << keys.size(0)
    << "values_size: " << values.size(0);
  std::vector<size_t> idx(keys.size(0));
  std::vector<KDType> keys_vec(keys.size(0));
  std::vector<VDType> values_vec(values.size(0));
  for (index_t i = 0; i < keys.size(0); i++) {
    idx[i] = i;
    keys_vec[i] = keys[i];
    values_vec[i] = values[i];
  }
  if (is_ascend) {
    std::stable_sort(idx.begin(), idx.end(),
                     [&keys_vec](size_t i1, size_t i2)
                       {return keys_vec[i1] < keys_vec[i2]; });
  } else {
    std::stable_sort(idx.begin(), idx.end(),
                     [&keys_vec](size_t i1, size_t i2)
                       {return keys_vec[i1] > keys_vec[i2]; });
  }
  for (index_t i = 0; i < values.size(0); i++) {
    keys[i] = keys_vec[idx[i]];
    values[i] = values_vec[idx[i]];
  }
}

/*!
 * \brief CPU/GPU: Return the amount of temporary storage in bytes required for SortByKey
 * \param num_keys number of keys to sort
 */
template <typename KDType, typename VDType, typename xpu>
inline typename std::enable_if<std::is_same<xpu, cpu>::value, size_t>::type
SortByKeyWorkspaceSize(const size_t num_keys) {
  return 0;
}

/*!
 * \brief CPU/GPU: Sort key-value pairs stored in separate places. (Stable sort is performed!)
 * \param keys the keys to sort
 * \param values the values that sorts w.r.t the key
 * \param is_ascend whether to sort key in ascending order
 */
template<typename KDType, typename VDType>
inline void SortByKey(mshadow::Tensor<gpu, 1, KDType> keys, mshadow::Tensor<gpu, 1, VDType> values,
                      bool is_ascend = true, mshadow::Tensor<gpu, 1, char>* workspace = NULL,
                      const int begin_bit = 0, const int end_bit = sizeof(KDType)*8);
/*!
 * \brief CPU/GPU: Return the amount of temporary storage in bytes required for SortByKey
 * \param num_keys number of keys to sort
 */
template <typename KDType, typename VDType, typename xpu>
inline typename std::enable_if<std::is_same<xpu, gpu>::value, size_t>::type
SortByKeyWorkspaceSize(const size_t num_keys);

}  // namespace op
}  // namespace mxnet
#ifdef __CUDACC__
#include "./sort_op-inl.cuh"
#endif
#endif  // MXNET_OPERATOR_TENSOR_SORT_OP_H_
