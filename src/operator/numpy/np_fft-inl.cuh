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
 * Copyright (c) 2015-2017 by Contributors
 * \file np_fft-inl.cuh
 * \brief CUDA implementations of numpy-compatible fft operator
 * \author Zhiqiang Xie
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_FFT_INL_CUH_
#define MXNET_OPERATOR_NUMPY_NP_FFT_INL_CUH_

#include <cufft.h>

struct resize_and_cast;

inline void cuFFTPlan(cufftHandle* plan, int len_fft, int batch_size,
                      const mshadow::complex::complex64& indicator) {
  cufftPlanMany(plan, 1, &len_fft, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, batch_size);
}

inline void cuFFTPlan(cufftHandle* plan, int len_fft, int batch_size,
                      const mshadow::complex::complex128& indicator) {
  cufftPlanMany(plan, 1, &len_fft, nullptr, 0, 0, nullptr, 0, 0, CUFFT_Z2Z, batch_size);
}

template <typename IType, typename OType>
inline void cuFFTExec(
    const OpContext& ctx, const cufftHandle& plan,
    const mshadow::Tensor<gpu, 2, IType>& in_data,
    const mshadow::Tensor<gpu, 2, OType>& out_data,
    const mshadow::Tensor<gpu, 2, mshadow::complex::complex64>& input_buffer,
    size_t offset, int batch_size, int len_fft, int grad_dim) {
  using namespace mxnet_op;
  Stream<gpu>* s = ctx.get_stream<gpu>();

  Kernel<resize_and_cast, gpu>::Launch(
      s, batch_size * len_fft, input_buffer.dptr_,
      in_data.Slice(offset, offset + batch_size).dptr_,
      in_data.shape_[1], len_fft);
  cufftComplex* in_tmp = const_cast<cufftComplex*>(
      reinterpret_cast<const cufftComplex*>(input_buffer.dptr_));
  if (grad_dim == 0) {
    cufftComplex* out_tmp = reinterpret_cast<cufftComplex*>(out_data.dptr_ + offset * len_fft);
    CHECK_EQ(cufftExecC2C(plan, in_tmp, out_tmp, CUFFT_FORWARD), CUFFT_SUCCESS);
  } else {
    CHECK_EQ(cufftExecC2C(plan, in_tmp, in_tmp, CUFFT_INVERSE), CUFFT_SUCCESS);
    Kernel<resize_and_cast, gpu>::Launch(
        s, batch_size * grad_dim,
        out_data.Slice(offset, offset + batch_size).dptr_, input_buffer.dptr_,
        len_fft, grad_dim, len_fft);
  }
}

template <typename IType, typename OType>
inline void cuFFTExec(
    const OpContext& ctx, const cufftHandle& plan,
    const mshadow::Tensor<gpu, 2, IType>& in_data,
    const mshadow::Tensor<gpu, 2, OType>& out_data,
    const mshadow::Tensor<gpu, 2, mshadow::complex::complex128>& input_buffer,
    size_t offset, int batch_size, int len_fft, int grad_dim) {
  using namespace mxnet_op;
  Stream<gpu>* s = ctx.get_stream<gpu>();

  Kernel<resize_and_cast, gpu>::Launch(
      s, batch_size * len_fft, input_buffer.dptr_,
      in_data.Slice(offset, offset + batch_size).dptr_,
      in_data.shape_[1], len_fft);
  cufftDoubleComplex* in_tmp = const_cast<cufftDoubleComplex*>(
      reinterpret_cast<const cufftDoubleComplex*>(input_buffer.dptr_));
  if (grad_dim == 0) {
    cufftDoubleComplex* out_tmp = reinterpret_cast<cufftDoubleComplex*>(
      out_data.dptr_ + offset * len_fft);
    CHECK_EQ(cufftExecZ2Z(plan, in_tmp, out_tmp, CUFFT_FORWARD), CUFFT_SUCCESS);
  } else {
    CHECK_EQ(cufftExecZ2Z(plan, in_tmp, in_tmp, CUFFT_INVERSE), CUFFT_SUCCESS);
    Kernel<resize_and_cast, gpu>::Launch(
        s, batch_size * grad_dim,
        out_data.Slice(offset, offset + batch_size).dptr_, input_buffer.dptr_,
        len_fft, grad_dim, len_fft);
  }
}

#endif  // MXNET_OPERATOR_NUMPY_NP_FFT_INL_CUH_
