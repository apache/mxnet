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

#ifndef MXNET_COMMON_CUDA_RTC_REDUCER_INL_H_
#define MXNET_COMMON_CUDA_RTC_REDUCER_INL_H_

#if MXNET_USE_CUDA

namespace mxnet {
namespace common {
namespace cuda {
namespace rtc {

const char reducer[] = R"code(

namespace red {

/*! \brief sum reducer */
struct sum {
  /*! \brief do reduction into dst */
  template<typename DType, typename DType2>
  __device__ inline static void Reduce(volatile DType& dst,  volatile DType2 src) {
    dst = op::add(dst, src);
  }

  /*! \brief do stable reduction into dst */
  template<typename DType, typename DType2>
  __device__ inline static void Reduce(volatile DType& dst,  volatile DType2 src,
                                       volatile DType& residual) {
    DType y = op::sub(src, residual);
    DType t = dst + y;
    if (util::isinf(t)) {
      residual = 0;
    } else {
      residual = (t - dst) - y;
    }
    dst = t;
  }
  /*! \brief combine the results of two reducers */
  template<typename DType>
  __device__ inline static void Merge(volatile DType& dst_val, volatile DType& src_val) {
    Reduce(dst_val, src_val);
  }
  /*! \brief combine the results of two reducers */
  template<typename DType>
  __device__ inline static void Merge(volatile DType& dst_val, volatile DType& dst_residual,
                                      volatile DType& src_val, volatile DType& src_residual) {
    DType t1 = dst_val + src_val;
    if (util::isinf(t1)) {
      dst_val = t1;
      dst_residual = 0;
    } else {
      DType e = t1 - dst_val;
      DType t2 = ((src_val - e) + (dst_val - (t1 - e))) + dst_residual + src_residual;
      dst_val = t1 + t2;
      dst_residual = t2 - (dst_val - t1);
    }
  }
  /*! \brief finalize reduction result */
  template<typename DType>
  __device__ inline static void Finalize(volatile DType& dst) {}
  /*! \brief finalize reduction result */
  template<typename DType>
  __device__ inline static void Finalize(volatile DType& dst, volatile DType& none) {}
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  __device__ inline static void SetInitValue(DType &initv) {
    initv = 0;
  }
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  __device__ inline static void SetInitValue(DType &initv, DType &residual) {
    SetInitValue(initv);
    residual = 0;
  }
};
}  // namespace red

)code";

}  // namespace rtc
}  // namespace cuda
}  // namespace common
}  // namespace mxnet

#endif  // MXNET_USE_CUDA

#endif  // MXNET_COMMON_CUDA_RTC_REDUCER_INL_H_

