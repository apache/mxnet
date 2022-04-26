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
 * \file softmax-inl.h
 * \brief
 */
#ifndef MXNET_OPERATOR_NN_SOFTMAX_INL_H_
#define MXNET_OPERATOR_NN_SOFTMAX_INL_H_

#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include <type_traits>

#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../tensor/broadcast_reduce_op.h"

using mshadow::red::limits::MinValue;

namespace mxnet {
namespace op {
namespace mxnet_op {

struct softmax_fwd {
  template <typename AType>
  MSHADOW_XINLINE static AType Map(float a, AType b) {
    return AType(expf(a) / b);
  }

  template <typename AType>
  MSHADOW_XINLINE static AType Map(double a, AType b) {
    return AType(exp(a) / b);
  }
};

struct log_softmax_fwd {
  template <typename DType>
  MSHADOW_XINLINE static float Map(DType a, float b) {
    return a - logf(b);
  }

  template <typename DType>
  MSHADOW_XINLINE static double Map(DType a, double b) {
    return a - log(b);
  }
};

template <typename OP,
          bool negate,
          typename AType,
          typename DType,
          typename OType,
          typename IType,
          int ndim>
inline void Softmax(Stream<cpu>* s,
                    DType* in,
                    OType* out,
                    IType* length,
                    Shape<ndim> shape,
                    int axis,
                    const DType temperature) {
  index_t M = shape[axis];
  if (M == 0)
    return;
  index_t N          = shape.Size() / M;
  Shape<ndim> stride = calc_stride(shape);
  Shape<ndim> sshape = shape;
  sshape[axis]       = 1;
  index_t sa         = stride[axis];

  if (length == nullptr) {
#pragma omp parallel for
    for (index_t i = 0; i < N; ++i) {
      index_t base = unravel_dot(i, sshape, stride);

      DType mmax = negate ? -in[base] : in[base];
      DType val;
      for (index_t j = 1; j < M; ++j) {
        val = negate ? -in[base + j * sa] : in[base + j * sa];
        if (mmax < val)
          mmax = val;
      }

      AType sum = AType(0);
      DType in_val;
      // By default temperature is 1.0.
      // Adding a branch here to save the CPU 'divide-by-1' computation at runtime
      if (temperature == 1.0) {
        for (index_t j = 0; j < M; ++j) {
          in_val = negate ? -in[base + j * sa] : in[base + j * sa];
          sum += std::exp(in_val - mmax);
        }

        for (index_t j = 0; j < M; ++j) {
          in_val             = negate ? -in[base + j * sa] : in[base + j * sa];
          out[base + j * sa] = OP::Map(in_val - mmax, sum);
        }
      } else {
        for (index_t j = 0; j < M; ++j) {
          in_val = negate ? -in[base + j * sa] : in[base + j * sa];
          sum += std::exp((in_val - mmax) / temperature);
        }

        for (index_t j = 0; j < M; ++j) {
          in_val             = negate ? -in[base + j * sa] : in[base + j * sa];
          out[base + j * sa] = OP::Map((in_val - mmax) / temperature, sum);
        }
      }
    }
  } else {
#pragma omp parallel for
    for (index_t i = 0; i < N; ++i) {
      index_t len  = static_cast<index_t>(length[i]);
      index_t base = unravel_dot(i, sshape, stride);

      DType mmax = negate ? -in[base] : in[base];
      DType val;
      for (index_t j = 1; j < len; ++j) {
        val = negate ? -in[base + j * sa] : in[base + j * sa];
        if (mmax < val)
          mmax = val;
      }
      for (index_t j = len; j < M; ++j) {
        out[base + j * sa] = OType(0.0f);
      }

      AType sum = AType(0);
      DType in_val;
      // By default temperature is 1.0.
      // Adding a branch here to save the CPU 'divide-by-1' computation at runtime
      if (temperature == 1.0) {
        for (index_t j = 0; j < len; ++j) {
          in_val = negate ? -in[base + j * sa] : in[base + j * sa];
          sum += std::exp(in_val - mmax);
        }

        for (index_t j = 0; j < len; ++j) {
          in_val             = negate ? -in[base + j * sa] : in[base + j * sa];
          out[base + j * sa] = OP::Map(in_val - mmax, sum);
        }
      } else {
        for (index_t j = 0; j < len; ++j) {
          in_val = negate ? -in[base + j * sa] : in[base + j * sa];
          sum += std::exp((in_val - mmax) / temperature);
        }

        for (index_t j = 0; j < len; ++j) {
          in_val             = negate ? -in[base + j * sa] : in[base + j * sa];
          out[base + j * sa] = OP::Map((in_val - mmax) / temperature, sum);
        }
      }
    }
  }
}

struct masked_softmax_where {
  template <typename DType, int ndim>
  MSHADOW_XINLINE static void Map(index_t id,
                                  DType* out,
                                  const bool* cond,
                                  const DType* x,
                                  const double y,
                                  Shape<ndim> data_shape,
                                  Shape<ndim> mask_shape) {
    index_t mask_pos = 0;
    index_t stride   = 1;
    for (index_t i = ndim - 1, j = id; i >= 0; --i) {
      auto tmp = j / data_shape[i];
      if (mask_shape[i] != 1) {
        mask_pos += (j - tmp * mask_shape[i]) * stride;
      }
      stride *= mask_shape[i];
      j = tmp;
    }
    KERNEL_ASSIGN(out[id], kWriteTo, (cond[mask_pos] ? x[id] : static_cast<DType>(y)));
  }
};

template <typename OP, bool masked_neg_inf, bool negate, typename AType, typename DType, int ndim>
inline void MaskedSoftmax(Stream<cpu>* s,
                          DType* in,
                          DType* out,
                          bool* mask,
                          Shape<ndim> data_shape,
                          Shape<ndim> mask_shape,
                          int axis,
                          const double temperature,
                          bool normalize,
                          const OpContext& ctx) {
  Tensor<cpu, 1, DType> workspace =
      ctx.requested[0].get_space_typed<cpu, 1, DType>(Shape1(data_shape.Size()), s);
  DType* masked_input = TBlob(workspace).dptr<DType>();

  double neg = MinValue<DType>();
  Kernel<masked_softmax_where, cpu>::Launch(
      s, data_shape.Size(), masked_input, mask, in, neg, data_shape, mask_shape);
  int* max_lenghts    = nullptr;
  double masked_value = 0.0;
  if (masked_neg_inf)
    masked_value = -INFINITY;
  Softmax<OP, negate, AType, DType>(
      s, masked_input, out, max_lenghts, data_shape, axis, temperature);
  Kernel<masked_softmax_where, cpu>::Launch(
      s, data_shape.Size(), out, mask, out, masked_value, data_shape, mask_shape);
}

struct softmax_bwd {
  template <typename DType, typename AType>
  MSHADOW_XINLINE static AType Map(DType ograd, DType out, AType sum) {
    return AType(out * (ograd - sum));
  }
};

struct log_softmax_bwd {
  template <typename AType>
  MSHADOW_XINLINE static AType Map(float ograd, float out, AType sum) {
    return AType(ograd - expf(out) * sum);
  }

  template <typename AType>
  MSHADOW_XINLINE static AType Map(double ograd, double out, AType sum) {
    return AType(ograd - exp(out) * sum);
  }
};

template <typename OP1,
          typename OP2,
          int Req,
          bool negate,
          typename AType,
          typename DType,
          typename OType,
          typename IType,
          int ndim>
inline void SoftmaxGrad(Stream<cpu>* s,
                        OType* out,
                        OType* ograd,
                        DType* igrad,
                        IType* length,
                        Shape<ndim> shape,
                        int axis,
                        const DType temperature) {
  index_t M = shape[axis];
  if (M == 0)
    return;
  index_t N          = shape.Size() / M;
  Shape<ndim> stride = calc_stride(shape);
  Shape<ndim> sshape = shape;
  sshape[axis]       = 1;
  index_t sa         = stride[axis];

  if (length != nullptr) {
#pragma omp parallel for
    for (index_t i = 0; i < N; ++i) {
      index_t base = unravel_dot(i, sshape, stride);
      index_t len  = static_cast<index_t>(length[i]);

      AType sum = AType(0);
      for (index_t j = 0; j < len; ++j) {
        sum += OP1::Map(ograd[base + j * sa], out[base + j * sa]);
      }

      // By default temperature is 1.0.
      // Adding a branch here to save the CPU 'divide-by-1' computation at runtime
      DType final_result;
      if (temperature == 1.0) {
        for (index_t j = 0; j < M; ++j) {
          final_result = negate ? -OP2::Map(ograd[base + j * sa], out[base + j * sa], sum) :
                                  OP2::Map(ograd[base + j * sa], out[base + j * sa], sum);
          final_result = (j < len) ? final_result : DType(0.0f);
          KERNEL_ASSIGN(igrad[base + j * sa], Req, final_result);
        }
      } else {
        for (index_t j = 0; j < M; ++j) {
          final_result =
              negate ? -OP2::Map(ograd[base + j * sa], out[base + j * sa], sum) / temperature :
                       OP2::Map(ograd[base + j * sa], out[base + j * sa], sum) / temperature;
          final_result = (j < len) ? final_result : DType(0.0f);
          KERNEL_ASSIGN(igrad[base + j * sa], Req, final_result);
        }
      }
    }
  } else {
#pragma omp parallel for
    for (index_t i = 0; i < N; ++i) {
      index_t base = unravel_dot(i, sshape, stride);

      AType sum = AType(0);
      for (index_t j = 0; j < M; ++j) {
        sum += OP1::Map(ograd[base + j * sa], out[base + j * sa]);
      }

      // By default temperature is 1.0.
      // Adding a branch here to save the CPU 'divide-by-1' computation at runtime
      DType final_result;
      if (temperature == 1.0) {
        for (index_t j = 0; j < M; ++j) {
          final_result = negate ? -OP2::Map(ograd[base + j * sa], out[base + j * sa], sum) :
                                  OP2::Map(ograd[base + j * sa], out[base + j * sa], sum);
          KERNEL_ASSIGN(igrad[base + j * sa], Req, final_result);
        }
      } else {
        for (index_t j = 0; j < M; ++j) {
          final_result =
              negate ? -OP2::Map(ograd[base + j * sa], out[base + j * sa], sum) / temperature :
                       OP2::Map(ograd[base + j * sa], out[base + j * sa], sum) / temperature;
          KERNEL_ASSIGN(igrad[base + j * sa], Req, final_result);
        }
      }
    }
  }
}

template <typename OP1,
          typename OP2,
          int Req,
          bool negate,
          typename AType,
          int ndim,
          typename DType>
inline void MaskedSoftmaxGrad(Stream<cpu>* s,
                              DType* out,
                              DType* ograd,
                              DType* igrad,
                              bool* mask,
                              Shape<ndim> data_shape,
                              Shape<ndim> mask_shape,
                              int axis,
                              const double temperature,
                              const OpContext& ctx) {
  Tensor<cpu, 1, DType> workspace =
      ctx.requested[0].get_space_typed<cpu, 1, DType>(Shape1(data_shape.Size()), s);
  DType* masked_ograd = TBlob(workspace).dptr<DType>();
  Kernel<masked_softmax_where, cpu>::Launch(
      s, data_shape.Size(), masked_ograd, mask, ograd, 0.0, data_shape, mask_shape);
  int* max_lenghts = nullptr;
  SoftmaxGrad<OP1, OP2, Req, negate, AType, DType, DType, int, ndim>(
      s, out, masked_ograd, igrad, max_lenghts, data_shape, axis, temperature);
  Kernel<masked_softmax_where, cpu>::Launch(
      s, data_shape.Size(), igrad, mask, igrad, 0.0, data_shape, mask_shape);
}

#ifdef __CUDACC__
const int softmax_threads_per_block = 512;

template <int ndim>
MSHADOW_XINLINE index_t get_mask_position(const index_t idx,
                                          const Shape<ndim>& data_shape,
                                          const Shape<ndim>& mask_shape,
                                          int axis,
                                          index_t* stride_axis) {
  index_t ret    = 0;
  index_t stride = 1;
  *stride_axis   = 1;
#pragma unroll
  for (index_t i = ndim - 1, j = idx; i >= 0; --i) {
    auto tmp = j / data_shape[i];
    if (i != axis && mask_shape[i] != 1) {
      ret += (j - tmp * mask_shape[i]) * stride;
      if (i > axis)
        *stride_axis *= mask_shape[i];
    }
    stride *= mask_shape[i];
    j = tmp;
  }
  return ret;
}

template <bool normalize,
          int x_bits,
          typename OP,
          bool masked_neg_inf,
          bool negate,
          typename AType,
          int ndim,
          typename DType>
__global__ void masked_softmax_kernel(DType* in,
                                      DType* out,
                                      bool* in_mask,
                                      index_t M,
                                      int axis,
                                      Shape<ndim> sshape,
                                      Shape<ndim> stride,
                                      Shape<ndim> mask_shape,
                                      const double temperature) {
  extern __shared__ double shared[];
  AType* smem = reinterpret_cast<AType*>(shared);  // x_size

  const unsigned x_size = 1 << x_bits;
  index_t sa            = stride[axis];
  index_t base          = unravel_dot(blockIdx.x, sshape, stride);
  index_t sa_mask       = 0;
  index_t base_mask     = get_mask_position(blockIdx.x, sshape, mask_shape, axis, &sa_mask);
  bool bcst_mask_axis   = (mask_shape[axis] == 1);
  index_t x             = threadIdx.x;

  DType smax = 0.0;
  if (normalize) {
    red::maximum::SetInitValue(smem[x]);
    for (index_t i = x; i < M; i += x_size) {
      bool mask_value = bcst_mask_axis ? in_mask[base_mask] : in_mask[base_mask + i * sa_mask];
      if (mask_value)
        smem[x] = ::max(smem[x], negate ? -in[base + i * sa] : in[base + i * sa]);
    }
    __syncthreads();
    cuda::Reduce1D<red::maximum, x_bits>(smem);
    __syncthreads();
    smax = smem[0];
    __syncthreads();
  }

  red::sum::SetInitValue(smem[x]);
  DType val;
  for (index_t i = x; i < M; i += x_size) {
    bool mask_value = bcst_mask_axis ? in_mask[base_mask] : in_mask[base_mask + i * sa_mask];
    if (mask_value) {
      val = (negate ? -in[base + i * sa] : in[base + i * sa]);
      smem[x] += static_cast<AType>(expf((val - smax) / static_cast<AType>(temperature)));
    }
  }
  __syncthreads();
  cuda::Reduce1D<red::sum, x_bits>(smem);
  __syncthreads();
  AType ssum = smem[0];
  __syncthreads();

  double masked_value = 0.0;
  if (masked_neg_inf)
    masked_value = -INFINITY;
  for (index_t i = x; i < M; i += x_size) {
    val                = (negate ? -in[base + i * sa] : in[base + i * sa]);
    bool mask_value    = bcst_mask_axis ? in_mask[base_mask] : in_mask[base_mask + i * sa_mask];
    out[base + i * sa] = mask_value ?
                             DType(OP::Map((val - smax) / static_cast<DType>(temperature), ssum)) :
                             DType(masked_value);
  }
}

template <bool normalize,
          typename OP,
          bool masked_neg_inf,
          bool negate,
          typename AType,
          typename LType,
          typename LTypeMask,
          typename DType,
          int ndim>
__global__ void masked_softmax_stride1_kernel(const DType* in,
                                              DType* out,
                                              bool* in_mask,
                                              const index_t M,
                                              int axis,
                                              Shape<ndim> sshape,
                                              Shape<ndim> mask_shape,
                                              const double temperature,
                                              const int rows_per_block,
                                              const index_t total_rows,
                                              const size_t size_input_shared,
                                              const size_t size_mask_shared) {
  const int entries_per_load      = sizeof(LType) / sizeof(DType);
  const int entries_per_load_mask = sizeof(LTypeMask) / sizeof(bool);
  const int row_length            = entries_per_load > 0 ? M / entries_per_load : 0;
  const int row_length_mask       = entries_per_load > 0 ? M / entries_per_load_mask : 0;
  extern __shared__ double shared[];
  LType* persistent_storage = reinterpret_cast<LType*>(shared);
  // rows_per_block * M (DType), aligned to double
  LTypeMask* mask_shared = reinterpret_cast<LTypeMask*>(&shared[size_input_shared]);
  // rows_per_block * M (bool), aligned to double
  AType* scratch = reinterpret_cast<AType*>(&shared[size_input_shared + size_mask_shared]);
  // softmax_threads_per_block

  const int warp_size       = 32;
  const int threads_per_row = softmax_threads_per_block / rows_per_block;
  const int my_local_row    = threadIdx.x / threads_per_row;
  const int my_row          = blockIdx.x * rows_per_block + my_local_row;
  if (my_row >= total_rows)
    return;
  const int my_id  = threadIdx.x % threads_per_row;
  size_t base      = my_row * row_length;
  index_t pos_mask = 0;
  index_t stride   = mask_shape[axis];
#pragma unroll
  for (index_t i = axis - 1, j = my_row; i >= 0; --i) {
    auto tmp = j / sshape[i];
    if (mask_shape[i] != 1) {
      pos_mask += (j - tmp * mask_shape[i]) * stride;
      stride *= mask_shape[i];
    }
    j = tmp;
  }

  const LType* in_aligned = reinterpret_cast<const LType*>(in);
  for (index_t i = my_id; i < row_length; i += threads_per_row) {
    persistent_storage[my_local_row * row_length + i] = in_aligned[base + i];
  }
  const LTypeMask* in_mask_aligned = reinterpret_cast<const LTypeMask*>(&in_mask[pos_mask]);
  for (index_t i = my_id; i < row_length_mask; i += threads_per_row) {
    mask_shared[my_local_row * row_length_mask + i] =
        (mask_shape[axis] > 1) ? in_mask_aligned[i] : in_mask_aligned[0];
  }
  DType* row     = reinterpret_cast<DType*>(persistent_storage + my_local_row * row_length);
  bool* row_mask = reinterpret_cast<bool*>(mask_shared + my_local_row * row_length_mask);
  __syncthreads();

  DType smax = 0.0;
  if (normalize) {
    DType my_max_value;
    red::maximum::SetInitValue(my_max_value);
    for (index_t i = my_id; i < M; i += threads_per_row) {
      if (row_mask[i])
        my_max_value = ::max(my_max_value, negate ? -row[i] : row[i]);
    }
    scratch[threadIdx.x] = my_max_value;
    __syncthreads();
    for (int size = threads_per_row / 2; size >= warp_size; size /= 2) {
      if (my_id < size) {
        scratch[threadIdx.x] = ::max(scratch[threadIdx.x], scratch[threadIdx.x + size]);
      }
      __syncthreads();
    }
    if (my_id < warp_size) {
      AType my_value       = common::cuda::warp_reduce(scratch[threadIdx.x],
                                                 [](AType x, AType y) { return ::max(x, y); });
      scratch[threadIdx.x] = my_value;
    }
    __syncthreads();
    smax = scratch[threadIdx.x - threadIdx.x % threads_per_row];
    __syncthreads();
  }

  AType my_sum;
  red::sum::SetInitValue(my_sum);
  for (index_t i = my_id; i < M; i += threads_per_row) {
    if (row_mask[i]) {
      const DType val = (negate ? -row[i] : row[i]);
      my_sum += static_cast<AType>(expf((val - smax) / static_cast<AType>(temperature)));
    }
  }
  scratch[threadIdx.x] = my_sum;
  __syncthreads();
  for (int size = threads_per_row / 2; size >= warp_size; size /= 2) {
    if (my_id < size) {
      scratch[threadIdx.x] += scratch[threadIdx.x + size];
    }
    __syncthreads();
  }
  if (my_id < warp_size) {
    AType my_value =
        common::cuda::warp_reduce(scratch[threadIdx.x], [](AType x, AType y) { return x + y; });
    scratch[threadIdx.x] = my_value;
  }
  __syncthreads();

  AType ssum = scratch[threadIdx.x - threadIdx.x % threads_per_row];
  __syncthreads();

  double masked_value = 0.0;
  if (masked_neg_inf)
    masked_value = -INFINITY;
  for (index_t i = my_id; i < M; i += threads_per_row) {
    const DType val = (negate ? -row[i] : row[i]);
    row[i] = row_mask[i] ? DType(OP::Map((val - smax) / static_cast<DType>(temperature), ssum)) :
                           DType(masked_value);
  }
  __syncthreads();

  LType* out_aligned = reinterpret_cast<LType*>(out);

  for (index_t i = my_id; i < row_length; i += threads_per_row) {
    out_aligned[base + i] = persistent_storage[my_local_row * row_length + i];
  }
}

template <typename OP,
          bool masked_neg_inf,
          bool negate,
          typename AType,
          typename DType,
          typename OType,
          int ndim>
inline void MaskedSoftmax(Stream<gpu>* s,
                          DType* in,
                          OType* out,
                          bool* mask,
                          Shape<ndim> data_shape,
                          Shape<ndim> mask_shape,
                          int axis,
                          const double temperature,
                          bool normalize,
                          const OpContext& ctx) {
  const int x_bits = 7;
  const int x_size = 1 << x_bits;
  index_t M        = data_shape[axis];
  if (M == 0 || data_shape.Size() == 0)
    return;
  index_t N          = data_shape.Size() / M;
  Shape<ndim> stride = calc_stride(data_shape);
  Shape<ndim> sshape = data_shape;
  sshape[axis]       = 1;

  const size_t DSize = sizeof(DType);
  // Using max of 20 kB of shared memory for InputData in the optimized case
  const size_t max_opt_M = 20 * 1024 / DSize;
  if (stride[axis] == 1 && static_cast<size_t>(M) <= max_opt_M &&
      std::is_same<DType, OType>::value) {
    int ltype      = mxnet::common::cuda::get_load_type(M * sizeof(DType));
    int ltype_mask = mxnet::common::cuda::get_load_type(mask_shape[axis] * sizeof(bool));
    MXNET_LOAD_TYPE_SWITCH(ltype, LType, {
      CHECK_LE(sizeof(DType), sizeof(LType));
      MXNET_LOAD_TYPE_SWITCH(ltype_mask, LTypeMask, {
        CHECK_LE(sizeof(bool), sizeof(LTypeMask));
        int rows_per_block = mxnet::common::cuda::get_rows_per_block(
            M * sizeof(DType) / sizeof(LType), softmax_threads_per_block);
        // calculate amount shared memory (slots aligned to double)
        int entries_per_load = entries_per_load = sizeof(LType) / sizeof(DType);
        int entries_per_load_mask               = sizeof(LTypeMask) / sizeof(bool);
        size_t size_input_shared = entries_per_load > 0 ? rows_per_block * M / entries_per_load : 0;
        size_t size_mask_shared =
            entries_per_load_mask > 0 ? rows_per_block * M / entries_per_load_mask : 0;
        size_input_shared =
            ((size_input_shared * sizeof(LType) + sizeof(double) - 1) / sizeof(double));
        size_mask_shared =
            ((size_mask_shared * sizeof(LTypeMask) + sizeof(double) - 1) / sizeof(double));
        size_t amount_shared = size_input_shared * sizeof(double) +
                               size_mask_shared * sizeof(double) +
                               softmax_threads_per_block * sizeof(AType);

        int nblocks = (N + rows_per_block - 1) / rows_per_block;
        if (normalize) {
          masked_softmax_stride1_kernel<true, OP, masked_neg_inf, negate, AType, LType, LTypeMask>
              <<<nblocks,
                 softmax_threads_per_block,
                 amount_shared,
                 mshadow::Stream<gpu>::GetStream(s)>>>(in,
                                                       out,
                                                       mask,
                                                       M,
                                                       axis,
                                                       sshape,
                                                       mask_shape,
                                                       temperature,
                                                       rows_per_block,
                                                       N,
                                                       size_input_shared,
                                                       size_mask_shared);
        } else {
          masked_softmax_stride1_kernel<false, OP, masked_neg_inf, negate, AType, LType, LTypeMask>
              <<<nblocks,
                 softmax_threads_per_block,
                 amount_shared,
                 mshadow::Stream<gpu>::GetStream(s)>>>(in,
                                                       out,
                                                       mask,
                                                       M,
                                                       axis,
                                                       sshape,
                                                       mask_shape,
                                                       temperature,
                                                       rows_per_block,
                                                       N,
                                                       size_input_shared,
                                                       size_mask_shared);
        }
      });
    });
    MSHADOW_CUDA_POST_KERNEL_CHECK(masked_softmax_stride1_kernel);
  } else {
    size_t amount_shared = x_size * sizeof(AType);
    if (normalize) {
      masked_softmax_kernel<true, x_bits, OP, masked_neg_inf, negate, AType, ndim>
          <<<N, x_size, amount_shared, mshadow::Stream<gpu>::GetStream(s)>>>(
              in, out, mask, M, axis, sshape, stride, mask_shape, temperature);
    } else {
      masked_softmax_kernel<false, x_bits, OP, masked_neg_inf, negate, AType, ndim>
          <<<N, x_size, amount_shared, mshadow::Stream<gpu>::GetStream(s)>>>(
              in, out, mask, M, axis, sshape, stride, mask_shape, temperature);
    }
    MSHADOW_CUDA_POST_KERNEL_CHECK(masked_softmax_kernel);
  }
}

template <typename OP1,
          typename OP2,
          int Req,
          bool negate,
          typename AType,
          typename LType,
          typename LTypeMask,
          typename DType,
          typename OType,
          int ndim>
__global__ void masked_softmax_stride1_grad_kernel(const OType* out,
                                                   const OType* ograd,
                                                   DType* igrad,
                                                   const bool* in_mask,
                                                   const index_t M,
                                                   int axis,
                                                   Shape<ndim> sshape,
                                                   Shape<ndim> mask_shape,
                                                   const double temperature,
                                                   const int rows_per_block,
                                                   const index_t total_rows,
                                                   const size_t size_input_shared,
                                                   const size_t size_mask_shared) {
  const int entries_per_load      = sizeof(LType) / sizeof(DType);
  const int entries_per_load_mask = sizeof(LTypeMask) / sizeof(bool);
  const int row_length            = entries_per_load > 0 ? M / entries_per_load : 0;
  const int row_length_mask       = entries_per_load > 0 ? M / entries_per_load_mask : 0;
  extern __shared__ double shared[];
  LType* persistent_storage = reinterpret_cast<LType*>(shared);
  // 2 * rows_per_block * M (DType), aligned to double
  LTypeMask* mask_shared = reinterpret_cast<LTypeMask*>(&shared[size_input_shared]);
  // rows_per_block * M (bool), aligned to double
  AType* scratch = reinterpret_cast<AType*>(&shared[size_input_shared + size_mask_shared]);
  // softmax_threads_per_block

  const int warp_size       = 32;
  const int threads_per_row = softmax_threads_per_block / rows_per_block;
  const int my_local_row    = threadIdx.x / threads_per_row;
  const int my_row          = blockIdx.x * rows_per_block + my_local_row;
  if (my_row >= total_rows)
    return;
  const int my_id  = threadIdx.x % threads_per_row;
  size_t base      = my_row * row_length;
  index_t pos_mask = 0;
  index_t stride   = mask_shape[axis];
#pragma unroll
  for (index_t i = axis - 1, j = my_row; i >= 0; --i) {
    auto tmp = j / sshape[i];
    if (mask_shape[i] != 1) {
      pos_mask += (j - tmp * mask_shape[i]) * stride;
      stride *= mask_shape[i];
    }
    j = tmp;
  }

  const LType* out_aligned   = reinterpret_cast<const LType*>(out);
  const LType* ograd_aligned = reinterpret_cast<const LType*>(ograd);
  for (index_t i = my_id; i < row_length; i += threads_per_row) {
    persistent_storage[my_local_row * row_length * 2 + i]              = out_aligned[base + i];
    persistent_storage[my_local_row * row_length * 2 + row_length + i] = ograd_aligned[base + i];
  }
  const LTypeMask* in_mask_aligned = reinterpret_cast<const LTypeMask*>(&in_mask[pos_mask]);
  for (index_t i = my_id; i < row_length_mask; i += threads_per_row) {
    mask_shared[my_local_row * row_length_mask + i] =
        (mask_shape[axis] > 1) ? in_mask_aligned[i] : in_mask_aligned[0];
  }
  DType* row     = reinterpret_cast<DType*>(persistent_storage + my_local_row * row_length * 2);
  bool* row_mask = reinterpret_cast<bool*>(mask_shared + my_local_row * row_length_mask);
  __syncthreads();

  AType my_sum_value;
  red::sum::SetInitValue(my_sum_value);

  for (index_t i = my_id; i < M; i += threads_per_row) {
    if (row_mask[i])
      my_sum_value += OP1::Map(row[i + M], row[i]);
  }
  scratch[threadIdx.x] = my_sum_value;
  __syncthreads();
  for (int size = threads_per_row / 2; size >= warp_size; size /= 2) {
    if (my_id < size) {
      scratch[threadIdx.x] = scratch[threadIdx.x] + scratch[threadIdx.x + size];
    }
    __syncthreads();
  }
  if (my_id < warp_size) {
    AType my_value =
        common::cuda::warp_reduce(scratch[threadIdx.x], [](AType x, AType y) { return x + y; });
    scratch[threadIdx.x] = my_value;
  }
  __syncthreads();
  AType ssum = scratch[threadIdx.x - threadIdx.x % threads_per_row];
  __syncthreads();

  for (index_t i = my_id; i < M; i += threads_per_row) {
    const DType val =
        negate ? -OP2::Map(row[i + M], row[i], ssum) : OP2::Map(row[i + M], row[i], ssum);
    row[i] = row_mask[i] ? DType(val / static_cast<DType>(temperature)) : DType(0.0f);
    if (Req == kAddTo) {
      row[i] += igrad[my_row * M + i];
    }
  }
  __syncthreads();

  LType* igrad_aligned = reinterpret_cast<LType*>(igrad);

  for (index_t i = my_id; i < row_length; i += threads_per_row) {
    igrad_aligned[base + i] = persistent_storage[my_local_row * row_length * 2 + i];
  }
}

template <int x_bits,
          typename OP1,
          typename OP2,
          int Req,
          bool negate,
          typename AType,
          int ndim,
          typename DType,
          typename OType>
__global__ void masked_softmax_grad_kernel(OType* out,
                                           OType* ograd,
                                           DType* igrad,
                                           const bool* in_mask,
                                           index_t M,
                                           int axis,
                                           Shape<ndim> sshape,
                                           Shape<ndim> stride,
                                           Shape<ndim> mask_shape,
                                           const double temperature) {
  const unsigned x_size = 1 << x_bits;
  __shared__ AType smem[x_size];
  index_t sa          = stride[axis];
  index_t base        = unravel_dot(blockIdx.x, sshape, stride);
  index_t sa_mask     = 0;
  index_t base_mask   = get_mask_position(blockIdx.x, sshape, mask_shape, axis, &sa_mask);
  bool bcst_mask_axis = (mask_shape[axis] == 1);
  index_t x           = threadIdx.x;

  red::sum::SetInitValue(smem[x]);
  for (index_t i = x; i < M; i += x_size) {
    bool mask_value = bcst_mask_axis ? in_mask[base_mask] : in_mask[base_mask + i * sa_mask];
    if (mask_value)
      smem[x] += OP1::Map(ograd[base + i * sa], out[base + i * sa]);
  }
  __syncthreads();
  cuda::Reduce1D<red::sum, x_bits>(smem);
  __syncthreads();
  AType ssum = smem[0];
  __syncthreads();

  DType final_result;
  for (index_t i = x; i < M; i += x_size) {
    bool mask_value = bcst_mask_axis ? in_mask[base_mask] : in_mask[base_mask + i * sa_mask];
    final_result    = negate ? -OP2::Map(ograd[base + i * sa], out[base + i * sa], ssum) :
                            OP2::Map(ograd[base + i * sa], out[base + i * sa], ssum);
    final_result = mask_value ? final_result / static_cast<DType>(temperature) : DType(0.0f);
    KERNEL_ASSIGN(igrad[base + i * sa], Req, final_result);
  }
}

template <typename OP1,
          typename OP2,
          int Req,
          bool negate,
          typename AType,
          int ndim,
          typename DType,
          typename OType>
inline void MaskedSoftmaxGrad(Stream<gpu>* s,
                              OType* out,
                              OType* ograd,
                              DType* igrad,
                              bool* mask,
                              Shape<ndim> data_shape,
                              Shape<ndim> mask_shape,
                              int axis,
                              const double temperature,
                              const OpContext& ctx) {
  const int x_bits = 7;
  const int x_size = 1 << x_bits;
  index_t M        = data_shape[axis];
  if (M == 0 || data_shape.Size() == 0)
    return;
  index_t N          = data_shape.Size() / M;
  Shape<ndim> stride = calc_stride(data_shape);
  Shape<ndim> sshape = data_shape;
  sshape[axis]       = 1;

  const size_t DSize = sizeof(DType);
  // Using max of 20 kB of shared memory for InputData in the optimized case
  const size_t max_opt_M = 20 * 1024 / DSize;
  if (stride[axis] == 1 && static_cast<size_t>(M) <= max_opt_M &&
      std::is_same<DType, OType>::value) {
    int ltype      = mxnet::common::cuda::get_load_type(M * sizeof(DType));
    int ltype_mask = mxnet::common::cuda::get_load_type(mask_shape[axis] * sizeof(bool));
    MXNET_LOAD_TYPE_SWITCH(ltype, LType, {
      CHECK_LE(sizeof(DType), sizeof(LType));
      MXNET_LOAD_TYPE_SWITCH(ltype_mask, LTypeMask, {
        CHECK_LE(sizeof(bool), sizeof(LTypeMask));
        int rows_per_block = mxnet::common::cuda::get_rows_per_block(
            M * sizeof(DType) / sizeof(LType), softmax_threads_per_block);
        // calculate amount shared memory (slots aligned to double)
        int entries_per_load = entries_per_load = sizeof(LType) / sizeof(DType);
        int entries_per_load_mask               = sizeof(LTypeMask) / sizeof(bool);
        size_t size_input_shared = entries_per_load > 0 ? rows_per_block * M / entries_per_load : 0;
        size_t size_mask_shared =
            entries_per_load_mask > 0 ? rows_per_block * M / entries_per_load_mask : 0;
        size_input_shared =
            ((2 * size_input_shared * sizeof(LType) + sizeof(double) - 1) / sizeof(double));
        size_mask_shared =
            ((size_mask_shared * sizeof(LTypeMask) + sizeof(double) - 1) / sizeof(double));
        size_t amount_shared = size_input_shared * sizeof(double) +
                               size_mask_shared * sizeof(double) +
                               softmax_threads_per_block * sizeof(AType);

        int nblocks = (N + rows_per_block - 1) / rows_per_block;
        masked_softmax_stride1_grad_kernel<OP1, OP2, Req, negate, AType, LType, LTypeMask>
            <<<nblocks,
               softmax_threads_per_block,
               amount_shared,
               mshadow::Stream<gpu>::GetStream(s)>>>(out,
                                                     ograd,
                                                     igrad,
                                                     mask,
                                                     M,
                                                     axis,
                                                     sshape,
                                                     mask_shape,
                                                     temperature,
                                                     rows_per_block,
                                                     N,
                                                     size_input_shared,
                                                     size_mask_shared);
      });
    });
    MSHADOW_CUDA_POST_KERNEL_CHECK(masked_softmax_stride1_grad_kernel);
  } else {
    masked_softmax_grad_kernel<x_bits, OP1, OP2, Req, negate, AType, ndim>
        <<<N, x_size, 0, mshadow::Stream<gpu>::GetStream(s)>>>(
            out, ograd, igrad, mask, M, axis, sshape, stride, mask_shape, temperature);
    MSHADOW_CUDA_POST_KERNEL_CHECK(masked_softmax_grad_kernel);
  }
}
#endif

}  // namespace mxnet_op

struct SoftmaxParam : public dmlc::Parameter<SoftmaxParam> {
  int axis;
  dmlc::optional<double> temperature;
  dmlc::optional<int> dtype;
  dmlc::optional<bool> use_length;
  DMLC_DECLARE_PARAMETER(SoftmaxParam) {
    DMLC_DECLARE_FIELD(axis).set_default(-1).describe("The axis along which to compute softmax.");
    DMLC_DECLARE_FIELD(temperature)
        .set_default(dmlc::optional<double>())
        .describe("Temperature parameter in softmax");
    DMLC_DECLARE_FIELD(dtype)
        .add_enum("float16", mshadow::kFloat16)
        .add_enum("float32", mshadow::kFloat32)
        .add_enum("float64", mshadow::kFloat64)
        .set_default(dmlc::optional<int>())
        .describe(
            "DType of the output in case this can't be inferred. "
            "Defaults to the same as input's dtype if not defined (dtype=None).");
    DMLC_DECLARE_FIELD(use_length)
        .set_default(dmlc::optional<bool>(false))
        .describe("Whether to use the length input as a mask over the data input.");
  }

  bool operator==(const SoftmaxParam& other) const {
    return this->axis == other.axis && this->temperature == other.temperature &&
           this->dtype == other.dtype && this->use_length == other.use_length;
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream axis_s, temperature_s, dtype_s, use_length_s;
    axis_s << axis;
    temperature_s << temperature;
    dtype_s << dtype;
    use_length_s << use_length;
    (*dict)["axis"]        = axis_s.str();
    (*dict)["temperature"] = temperature_s.str();
    if (dtype.has_value()) {
      (*dict)["dtype"] = MXNetTypeWithBool2String(dtype.value());
    } else {
      (*dict)["dtype"] = dtype_s.str();
    }
    (*dict)["use_length"] = use_length_s.str();
  }
};

struct MaskedSoftmaxParam : public dmlc::Parameter<MaskedSoftmaxParam> {
  int axis;
  dmlc::optional<double> temperature;
  dmlc::optional<bool> normalize;
  DMLC_DECLARE_PARAMETER(MaskedSoftmaxParam) {
    DMLC_DECLARE_FIELD(axis).set_default(-1).describe("The axis along which to compute softmax.");
    DMLC_DECLARE_FIELD(temperature)
        .set_default(dmlc::optional<double>())
        .describe("Temperature parameter in softmax");
    DMLC_DECLARE_FIELD(normalize)
        .set_default(dmlc::optional<bool>(true))
        .describe("Whether to normalize input data x: x = x - max(x)");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream axis_s, temperature_s, normalize_s;
    axis_s << axis;
    temperature_s << temperature;
    normalize_s << normalize;
    (*dict)["axis"]        = axis_s.str();
    (*dict)["temperature"] = temperature_s.str();
    (*dict)["normalize"]   = normalize_s.str();
  }

  bool operator==(const MaskedSoftmaxParam& other) const {
    return this->axis == other.axis && this->temperature == other.temperature &&
           this->normalize == other.normalize;
  }
};

static inline bool softmax_has_dtype_override(const nnvm::NodeAttrs& attrs) {
  const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
  return param.dtype.has_value() && param.dtype.value() != -1;
}

static inline bool softmax_use_length(const nnvm::NodeAttrs& attrs) {
  const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
  return param.use_length.value();
}

static inline bool SoftmaxOpType(const nnvm::NodeAttrs& attrs,
                                 std::vector<int>* in_attrs,
                                 std::vector<int>* out_attrs) {
  CHECK_EQ(out_attrs->size(), 1);
  const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), softmax_use_length(attrs) ? 2U : 1U);

  if (softmax_has_dtype_override(attrs)) {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, param.dtype.value());
    type_assign(&(*in_attrs)[0], (*out_attrs)[0]);
    return true;
  } else {
    std::vector<int> tmp = {in_attrs->at(0)};
    return ElemwiseType<1, 1>(attrs, &tmp, out_attrs);
  }
}

static inline bool SoftmaxOpShape(const nnvm::NodeAttrs& attrs,
                                  mxnet::ShapeVector* in_attrs,
                                  mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(out_attrs->size(), 1U);
  const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), param.use_length.value() ? 2U : 1U);

  if (param.use_length.value()) {
    mxnet::TShape& dshape = in_attrs->at(0);
    mxnet::TShape tmp_shape((dshape.ndim() == 1) ? 1U : dshape.ndim() - 1, 1);
    int j    = 0;
    int axis = param.axis != -1 ? param.axis : dshape.ndim() - 1;
    for (int i = 0; i < dshape.ndim(); ++i) {
      if (i != axis) {
        tmp_shape[j++] = dshape[i];
      }
    }
    SHAPE_ASSIGN_CHECK(*in_attrs, 1, tmp_shape);
  }
  mxnet::ShapeVector tmp = {in_attrs->at(0)};
  return ElemwiseShape<1, 1>(attrs, &tmp, out_attrs);
}

static inline bool SoftmaxGradOpShape(const nnvm::NodeAttrs& attrs,
                                      mxnet::ShapeVector* in_attrs,
                                      mxnet::ShapeVector* out_attrs) {
  if (softmax_has_dtype_override(attrs) || softmax_use_length(attrs)) {
    if (softmax_use_length(attrs)) {
      mxnet::ShapeVector ins   = {in_attrs->at(0), in_attrs->at(1), in_attrs->at(3)};
      mxnet::ShapeVector dgrad = {out_attrs->at(0)};
      bool res                 = ElemwiseShape<3, 1>(attrs, &ins, &dgrad);
      SHAPE_ASSIGN_CHECK(*in_attrs, 0, ins[0]);
      SHAPE_ASSIGN_CHECK(*in_attrs, 1, ins[1]);
      SHAPE_ASSIGN_CHECK(*in_attrs, 3, ins[2]);
      SHAPE_ASSIGN_CHECK(*out_attrs, 0, dgrad[0]);
      mxnet::ShapeVector length = {in_attrs->at(2)};
      mxnet::ShapeVector lgrad  = {out_attrs->at(1)};
      res                       = (res && ElemwiseShape<1, 1>(attrs, &length, &lgrad));
      SHAPE_ASSIGN_CHECK(*in_attrs, 2, length[0]);
      SHAPE_ASSIGN_CHECK(*out_attrs, 1, lgrad[0]);
      return res;
    } else {
      return ElemwiseShape<3, 1>(attrs, in_attrs, out_attrs);
    }
  } else {
    return ElemwiseShape<2, 1>(attrs, in_attrs, out_attrs);
  }
}

static inline bool SoftmaxGradOpType(const nnvm::NodeAttrs& attrs,
                                     std::vector<int>* in_attrs,
                                     std::vector<int>* out_attrs) {
  CHECK_EQ(out_attrs->size(), softmax_use_length(attrs) ? 2U : 1U);
  if (softmax_has_dtype_override(attrs) || softmax_use_length(attrs)) {
    CHECK_EQ(in_attrs->size(), softmax_use_length(attrs) ? 4U : 3U);
    int in_dtype  = (*in_attrs)[1];
    int out_dtype = (*in_attrs)[softmax_use_length(attrs) ? 3 : 2];
    TYPE_ASSIGN_CHECK(*in_attrs, 0, out_dtype);
    TYPE_ASSIGN_CHECK(*out_attrs, 0, in_dtype);
    if (softmax_use_length(attrs)) {
      TYPE_ASSIGN_CHECK(*out_attrs, 1, in_attrs->at(2));
    }

    return (*out_attrs)[0] != -1 && (*in_attrs)[0] != -1 &&
           (!softmax_use_length(attrs) || ((*out_attrs)[1] != -1 && (*in_attrs)[1] != -1));
  } else {
    CHECK_EQ(in_attrs->size(), 2U);
    int out_dtype = (*in_attrs)[1];
    TYPE_ASSIGN_CHECK(*out_attrs, 0, out_dtype);
    TYPE_ASSIGN_CHECK(*in_attrs, 0, out_dtype);

    return (*out_attrs)[0] != -1 && (*in_attrs)[0] != -1;
  }
}

static inline std::vector<std::pair<int, int>> SoftmaxGradOpInplaceOption(
    const nnvm::NodeAttrs& attrs) {
  if (softmax_has_dtype_override(attrs) || softmax_use_length(attrs)) {
    if (softmax_use_length(attrs)) {
      return std::vector<std::pair<int, int>>{{0, 0}, {1, 0}, {2, 1}, {3, 0}};
    } else {
      return std::vector<std::pair<int, int>>{{0, 0}, {1, 0}, {2, 0}};
    }
  } else {
    return std::vector<std::pair<int, int>>{{0, 0}, {1, 0}};
  }
}

static inline uint32_t SoftmaxGradOpNumInputs(const nnvm::NodeAttrs& attrs) {
  if (softmax_has_dtype_override(attrs) || softmax_use_length(attrs)) {
    return softmax_use_length(attrs) ? 4 : 3;
  }
  return 2;
}

static inline std::vector<std::string> SoftmaxGradOpInputNames(const nnvm::NodeAttrs& attrs) {
  if (softmax_has_dtype_override(attrs) || softmax_use_length(attrs)) {
    if (softmax_use_length(attrs)) {
      return std::vector<std::string>{"ograd", "data", "length", "output"};
    } else {
      return std::vector<std::string>{"ograd", "data", "output"};
    }
  } else {
    return std::vector<std::string>{"ograd", "output"};
  }
}

struct SoftmaxFGradient {
  const char* op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::ObjectPtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    if (softmax_has_dtype_override(n->attrs) || softmax_use_length(n->attrs)) {
      return ElemwiseGradUseInOut{op_name}(n, ograds);  // NOLINT
    } else {
      return ElemwiseGradUseOut{op_name}(n, ograds);  // NOLINT
    }
  }
};

static inline bool MaskedSoftmaxOpType(const nnvm::NodeAttrs& attrs,
                                       std::vector<int>* in_attrs,
                                       std::vector<int>* out_attrs) {
  CHECK_EQ(out_attrs->size(), 1);
  CHECK_EQ(in_attrs->size(), 2U);

  std::vector<int> tmp = {in_attrs->at(0)};
  return ElemwiseType<1, 1>(attrs, &tmp, out_attrs);
}

static inline bool MaskedSoftmaxOpShape(const nnvm::NodeAttrs& attrs,
                                        mxnet::ShapeVector* in_shape,
                                        mxnet::ShapeVector* out_shape) {
  CHECK_EQ(out_shape->size(), 1U);
  CHECK_EQ(in_shape->size(), 2U);

  mxnet::TShape& data_shape = (*in_shape)[0];
  mxnet::TShape& mask_shape = (*in_shape)[1];

  if (!mxnet::ndim_is_known(data_shape) || !mxnet::ndim_is_known(mask_shape)) {
    return false;
  }
  CHECK(data_shape.ndim() == mask_shape.ndim())
      << "Number of dimensions in data and mask does not match";
  CHECK(data_shape.ndim() > 0) << "Empty tuple is not allowed";

  for (int i = 0; i < data_shape.ndim(); ++i) {
    CHECK(data_shape[i] == mask_shape[i] || mask_shape[i] == 1)
        << "Mask cannot be broadcasted from " << mask_shape << " to " << data_shape;
  }
  SHAPE_ASSIGN_CHECK(*out_shape, 0, in_shape->at(0));
  SHAPE_ASSIGN_CHECK(*in_shape, 0, out_shape->at(0));
  return true;
}

static inline bool MaskedSoftmaxGradOpShape(const nnvm::NodeAttrs& attrs,
                                            mxnet::ShapeVector* in_shape,
                                            mxnet::ShapeVector* out_shape) {
  CHECK_EQ(out_shape->size(), 1U);
  CHECK_EQ(in_shape->size(), 3U);

  mxnet::TShape& ograd_shape = (*in_shape)[0];
  mxnet::TShape& mask_shape  = (*in_shape)[1];

  if (!mxnet::ndim_is_known(ograd_shape) || !mxnet::ndim_is_known(mask_shape)) {
    return false;
  }
  CHECK(ograd_shape.ndim() == mask_shape.ndim())
      << "Number of dimensions in data and mask does not match";
  CHECK(ograd_shape.ndim() > 0) << "Empty tuple is not allowed";

  for (int i = 0; i < ograd_shape.ndim(); ++i) {
    CHECK(ograd_shape[i] == mask_shape[i] || mask_shape[i] == 1)
        << "Mask cannot be broadcasted from " << mask_shape << " to " << ograd_shape;
  }

  SHAPE_ASSIGN_CHECK(*out_shape, 0, in_shape->at(0));
  SHAPE_ASSIGN_CHECK(*out_shape, 0, in_shape->at(2));
  SHAPE_ASSIGN_CHECK(*in_shape, 0, out_shape->at(0));
  SHAPE_ASSIGN_CHECK(*in_shape, 2, out_shape->at(0));
  return true;
}

static inline bool MaskedSoftmaxGradOpType(const nnvm::NodeAttrs& attrs,
                                           std::vector<int>* in_attrs,
                                           std::vector<int>* out_attrs) {
  CHECK_EQ(out_attrs->size(), 1U);
  CHECK_EQ(in_attrs->size(), 3U);
  int data_dtype = (*in_attrs)[0];
  TYPE_ASSIGN_CHECK(*in_attrs, 2, data_dtype);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, data_dtype);
  data_dtype = (*out_attrs)[0];
  TYPE_ASSIGN_CHECK(*in_attrs, 0, data_dtype);

  return true;
}

static inline std::vector<std::pair<int, int>> MaskedSoftmaxGradOpInplaceOption(
    const nnvm::NodeAttrs& attrs) {
  return std::vector<std::pair<int, int>>{{0, 0}, {1, 0}, {2, 1}, {3, 0}};
}

template <typename xpu, typename OP, bool negate = false>
void SoftmaxCompute(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  if (req[0] == kNullOp || inputs[0].Size() == 0U)
    return;
  CHECK_NE(req[0], kAddTo);
  const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
  int axis                  = CheckAxis(param.axis, inputs[0].ndim());
  const double temperature  = param.temperature.has_value() ? param.temperature.value() : 1.0;
  mxnet::TShape shape       = AxisShapeCompact(inputs[0].shape_, &axis, true);
  bool safe_acc             = dmlc::GetEnv("MXNET_SAFE_ACCUMULATION", true);
  if (!safe_acc && inputs[0].type_flag_ == mshadow::kFloat16) {
    common::LogOnce(
        "MXNET_SAFE_ACCUMULATION=1 is recommended for softmax with float16 inputs. "
        "See https://mxnet.apache.org/api/faq/env_var "
        "for more details.");
  }

  MXNET_REAL_ACC_TYPE_SWITCH(inputs[0].type_flag_, DType, AType, {
    MSHADOW_REAL_TYPE_SWITCH(
        outputs[0].type_flag_, OType, {
          int type = kInt32;
          if (param.use_length.value()) {
            CHECK(inputs.size() > 1)
                << "Mask needs to be provided when using softmax with use_length=True.";
            type = inputs[1].type_flag_;
          }
          MXNET_INT32_INT64_TYPE_SWITCH(type, IType, {
            IType* mask_ptr = nullptr;
            if (param.use_length.value()) {
              mask_ptr = inputs[1].dptr<IType>();
            }
            if (safe_acc) {
              if (shape.ndim() == 2) {
                Softmax<OP, negate, AType>(ctx.get_stream<xpu>(),
                                           inputs[0].dptr<DType>(),
                                           outputs[0].dptr<OType>(),
                                           mask_ptr,
                                           shape.get<2>(),
                                           axis,
                                           static_cast<DType>(temperature));
              } else {
                Softmax<OP, negate, AType>(ctx.get_stream<xpu>(),
                                           inputs[0].dptr<DType>(),
                                           outputs[0].dptr<OType>(),
                                           mask_ptr,
                                           shape.get<3>(),
                                           axis,
                                           static_cast<DType>(temperature));
              }
            } else {
              if (shape.ndim() == 2) {
                Softmax<OP, negate, DType>(ctx.get_stream<xpu>(),
                                           inputs[0].dptr<DType>(),
                                           outputs[0].dptr<OType>(),
                                           mask_ptr,
                                           shape.get<2>(),
                                           axis,
                                           static_cast<DType>(temperature));
              } else {
                Softmax<OP, negate, DType>(ctx.get_stream<xpu>(),
                                           inputs[0].dptr<DType>(),
                                           outputs[0].dptr<OType>(),
                                           mask_ptr,
                                           shape.get<3>(),
                                           axis,
                                           static_cast<DType>(temperature));
              }
            }
          });
        });
  });
}

template <typename xpu, typename OP, bool masked_neg_inf, bool negate = false>
void MaskedSoftmaxCompute(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  if (req[0] == kNullOp || inputs[0].Size() == 0U)
    return;
  CHECK_NE(req[0], kAddTo);
  const MaskedSoftmaxParam& param = nnvm::get<MaskedSoftmaxParam>(attrs.parsed);
  int axis                        = CheckAxis(param.axis, inputs[0].ndim());
  const double temperature        = param.temperature.has_value() ? param.temperature.value() : 1.0;
  bool safe_acc                   = dmlc::GetEnv("MXNET_SAFE_ACCUMULATION", true);
  if (!safe_acc && inputs[0].type_flag_ == mshadow::kFloat16) {
    common::LogOnce(
        "MXNET_SAFE_ACCUMULATION=1 is recommended for masked_softmax with "
        "float16 inputs. "
        "See https://mxnet.apache.org/api/faq/env_var "
        "for more details.");
  }
  MXNET_REAL_ACC_TYPE_SWITCH(inputs[0].type_flag_, DType, AType, {
    MXNET_NDIM_SWITCH(inputs[0].ndim(), ndim, {
      bool* mask_ptr = inputs[1].dptr<bool>();
      if (safe_acc) {
        MaskedSoftmax<OP, masked_neg_inf, negate, AType>(ctx.get_stream<xpu>(),
                                                         inputs[0].dptr<DType>(),
                                                         outputs[0].dptr<DType>(),
                                                         mask_ptr,
                                                         inputs[0].shape_.get<ndim>(),
                                                         inputs[1].shape_.get<ndim>(),
                                                         axis,
                                                         temperature,
                                                         param.normalize.value(),
                                                         ctx);
      } else {
        MaskedSoftmax<OP, masked_neg_inf, negate, DType>(ctx.get_stream<xpu>(),
                                                         inputs[0].dptr<DType>(),
                                                         outputs[0].dptr<DType>(),
                                                         mask_ptr,
                                                         inputs[0].shape_.get<ndim>(),
                                                         inputs[1].shape_.get<ndim>(),
                                                         axis,
                                                         temperature,
                                                         param.normalize.value(),
                                                         ctx);
      }
    });
  });
}

#if MXNET_USE_CUDA

struct SoftmaxRTCCompute {
  std::string OP;
  bool negate = false;

  void operator()(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs);
};

struct SoftmaxRTCGradCompute {
  std::string OP1;
  std::string OP2;
  bool negate = false;

  void operator()(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs);
};

#endif

template <typename xpu, typename OP1, typename OP2, bool negate = false>
void SoftmaxGradCompute(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  if (softmax_use_length(attrs)) {
    MXNET_INT32_INT64_TYPE_SWITCH(inputs[2].type_flag_, IType, {
      if (req[1] != kNullOp) {
        mxnet_op::Kernel<mxnet_op::set_zero, xpu>::Launch(
            ctx.get_stream<xpu>(), outputs[1].Size(), outputs[1].dptr<IType>());
      }
    });
  }
  if (req[0] == kNullOp)
    return;
  const int itype           = softmax_use_length(attrs) ? inputs[2].type_flag_ : kInt32;
  const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
  int axis                  = CheckAxis(param.axis, inputs[0].ndim());
  const double temperature  = param.temperature.has_value() ? param.temperature.value() : 1.0;
  mxnet::TShape shape       = AxisShapeCompact(inputs[0].shape_, &axis, true);

  int out_idx   = softmax_has_dtype_override(attrs) ? 2 : 1;
  out_idx       = softmax_use_length(attrs) ? 3 : out_idx;
  bool safe_acc = dmlc::GetEnv("MXNET_SAFE_ACCUMULATION", true);

  MXNET_REAL_ACC_TYPE_SWITCH(inputs[0].type_flag_, OType, AType, {
    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        MXNET_INT32_INT64_TYPE_SWITCH(itype, IType, {
          IType* length_ptr = nullptr;
          if (softmax_use_length(attrs)) {
            length_ptr = inputs[2].dptr<IType>();
          }
          if (safe_acc) {
            if (shape.ndim() == 2) {
              SoftmaxGrad<OP1, OP2, Req, negate, AType>(ctx.get_stream<xpu>(),
                                                        inputs[out_idx].dptr<OType>(),
                                                        inputs[0].dptr<OType>(),
                                                        outputs[0].dptr<DType>(),
                                                        length_ptr,
                                                        shape.get<2>(),
                                                        axis,
                                                        static_cast<DType>(temperature));
            } else {
              SoftmaxGrad<OP1, OP2, Req, negate, AType>(ctx.get_stream<xpu>(),
                                                        inputs[out_idx].dptr<OType>(),
                                                        inputs[0].dptr<OType>(),
                                                        outputs[0].dptr<DType>(),
                                                        length_ptr,
                                                        shape.get<3>(),
                                                        axis,
                                                        static_cast<DType>(temperature));
            }
          } else {
            if (shape.ndim() == 2) {
              SoftmaxGrad<OP1, OP2, Req, negate, DType>(ctx.get_stream<xpu>(),
                                                        inputs[out_idx].dptr<OType>(),
                                                        inputs[0].dptr<OType>(),
                                                        outputs[0].dptr<DType>(),
                                                        length_ptr,
                                                        shape.get<2>(),
                                                        axis,
                                                        static_cast<DType>(temperature));
            } else {
              SoftmaxGrad<OP1, OP2, Req, negate, DType>(ctx.get_stream<xpu>(),
                                                        inputs[out_idx].dptr<OType>(),
                                                        inputs[0].dptr<OType>(),
                                                        outputs[0].dptr<DType>(),
                                                        length_ptr,
                                                        shape.get<3>(),
                                                        axis,
                                                        static_cast<DType>(temperature));
            }
          }
        });
      });
    });
  });
}

template <typename xpu, typename OP1, typename OP2, bool negate = false>
void MaskedSoftmaxGradCompute(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<TBlob>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;

  if (req[0] == kNullOp)
    return;
  const MaskedSoftmaxParam& param = nnvm::get<MaskedSoftmaxParam>(attrs.parsed);
  int axis                        = CheckAxis(param.axis, inputs[0].ndim());
  const double temperature        = param.temperature.has_value() ? param.temperature.value() : 1.0;

  bool safe_acc = dmlc::GetEnv("MXNET_SAFE_ACCUMULATION", true);
  MXNET_REAL_ACC_TYPE_SWITCH(inputs[0].type_flag_, DType, AType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
      MXNET_NDIM_SWITCH(inputs[0].ndim(), ndim, {
        DType* ograd_ptr = inputs[0].dptr<DType>();
        DType* out_ptr   = inputs[2].dptr<DType>();
        bool* mask_ptr   = inputs[1].dptr<bool>();
        DType* grad_data = outputs[0].dptr<DType>();
        if (safe_acc) {
          MaskedSoftmaxGrad<OP1, OP2, Req, negate, AType>(ctx.get_stream<xpu>(),
                                                          out_ptr,
                                                          ograd_ptr,
                                                          grad_data,
                                                          mask_ptr,
                                                          inputs[0].shape_.get<ndim>(),
                                                          inputs[1].shape_.get<ndim>(),
                                                          axis,
                                                          static_cast<DType>(temperature),
                                                          ctx);
        } else {
          MaskedSoftmaxGrad<OP1, OP2, Req, negate, DType>(ctx.get_stream<xpu>(),
                                                          out_ptr,
                                                          ograd_ptr,
                                                          grad_data,
                                                          mask_ptr,
                                                          inputs[0].shape_.get<ndim>(),
                                                          inputs[1].shape_.get<ndim>(),
                                                          axis,
                                                          static_cast<DType>(temperature),
                                                          ctx);
        }
      });
    });
  });
}

}  // namespace op
}  // namespace mxnet

namespace std {
template <>
struct hash<mxnet::op::SoftmaxParam> {
  size_t operator()(const mxnet::op::SoftmaxParam& val) {
    size_t ret = 0;
    ret        = dmlc::HashCombine(ret, val.axis);
    ret        = dmlc::HashCombine(ret, val.temperature);
    ret        = dmlc::HashCombine(ret, val.dtype);
    ret        = dmlc::HashCombine(ret, val.use_length);
    return ret;
  }
};

template <>
struct hash<mxnet::op::MaskedSoftmaxParam> {
  size_t operator()(const mxnet::op::MaskedSoftmaxParam& val) {
    size_t ret = 0;
    ret        = dmlc::HashCombine(ret, val.axis);
    ret        = dmlc::HashCombine(ret, val.temperature);
    ret        = dmlc::HashCombine(ret, val.normalize);
    return ret;
  }
};
}  // namespace std

#endif  // MXNET_OPERATOR_NN_SOFTMAX_INL_H_
