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
 * Copyright (c) 2017 by Contributors
 * \file indexing_op-inl.cuh
 * \brief CUDA implementations for indexing_op.h
 * \author Antti-Pekka Hynninen
*/
#ifndef MXNET_OPERATOR_TENSOR_INDEXING_OP_CUH_
#define MXNET_OPERATOR_TENSOR_INDEXING_OP_CUH_
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include "../mxnet_op.h"

#if CUDA_VERSION >= 9000
#define FULLMASK 0xFFFFFFFF
#define __ballot(x) __ballot_sync(FULLMASK, (x))
#define __all(x) __all_sync(FULLMASK, (x))
#endif

namespace mxnet {
namespace op {
const int kWarpSize = 32;

template<int SZ, typename DType, typename IdxType>
__global__ void AddTakeGradLargeBatchKernel(DType* dst,
                                           // If idx_start == NULL, then in-kernel edge
                                           // detection is used
                                           const IdxType *idx_start,
                                           // idx_start_size_ptr ignored if idx_start == NULL
                                           const int* idx_start_size_ptr,
                                           const IdxType *sorted, const IdxType *index,
                                           const DType *src,
                                           int ymax, int xmax) {
  // Size of the shared memory is [blockDim.x*SZ*blockDim.y]*sizeof(DType)
  extern __shared__ char sh_grad_weight_char[];
  DType* sh_grad_weight = (DType*)sh_grad_weight_char;

  int iidx_end = (idx_start == NULL) ? ymax : *idx_start_size_ptr;

  for (int iidx = blockIdx.y;iidx < iidx_end;iidx += gridDim.y) {

    // Thread block sums up elements in the range [idx_begin, idx_end-1]
    int idx_begin, idx_end;
    int sorted_value;
    if (idx_start == NULL) {
      idx_begin = iidx;
      sorted_value = static_cast<int>(sorted[idx_begin]);
      if (idx_begin > 0 && sorted_value == static_cast<int>(sorted[idx_begin - 1])) continue;
      // Algorithm is explained using an example:
      //   blockDim.x = 32
      //   blockDim.y = 4
      //   sorted[idx_begin:] = [4 4 4 9]
      //   (3,4) denotes threadIdx.x=3, threadIdx.y=4, ":" is used for ranges
      //   (0:31,0:3) sorted_value = 4
      idx_end = idx_begin + 1;
      unsigned int* sh_ballot = (unsigned int*)sh_grad_weight_char;
      int no_edge = 0;
      do {
        int idx = idx_end + threadIdx.x + threadIdx.y*blockDim.x;
        // Example:
        //   (0:1,0) sorted_idx = 4
        //   (rest)  sorted_idx = -1
        int sorted_idx = (idx < ymax) ? static_cast<int>(sorted[idx]) : -1;
        // Example:
        //   (0:31,0) sh_ballot[0]     = 0b100
        //   (rest)   sh_ballot[1...3] = 0
        // sh_ballot[] tells us which thread within the warp found the edge
        sh_ballot[threadIdx.y] = __ballot(sorted_value != sorted_idx);
        __syncthreads();
        // No edge if sh_ballot[threadIdx.x] == 0
        // NOTE: All warps have the same value for no_edge
        // Example:
        //   (0,:)  no_edge = 0
        //   (rest) no_edge = 1
        no_edge = (threadIdx.x < blockDim.y) ? (sh_ballot[threadIdx.x] == 0) : 1;
        idx_end += blockDim.x*blockDim.y;
        // Example:
        //   __all(no_edge) = 0 since no_edge = 0 for threadIdx.x = 0, hence we leave the loop
      } while (__all(no_edge));
      idx_end -= blockDim.x*blockDim.y;
      // Find the first edge
      // Example:
      //   (0,:)  val = 1
      //   (rest) val = 0
      unsigned int val = (threadIdx.x < blockDim.y && sh_ballot[threadIdx.x] != 0) ?
        1 : 0;
      // NOTE: Set nth bit if thread n in the warp has val = 1
      // Example:
      //   (all) val = 1
      val = __ballot( val );
      // __ffs() returns the position of first set bit, 1...32. __ffs(1) = 1
      // j will be the warp index where edge was found
      // Example:
      //   (all) j = 1 - 1 = 0
      int j = __ffs(val) - 1;
      // j = warp index where the edge was found
      // __ffs(sh_ballot[j]) - 1 = warp lane where the edge was found
      // idx_end points to the one over the last value.
      // Example:
      //  idx_end += 0*blockDim.x + _ffs(0b100) - 1 = 0 + 3 - 1 = 2
      //  sorted[idx_end] = 9
      idx_end += j*blockDim.x + __ffs(sh_ballot[j]) - 1;
      __syncthreads();
    } else {
      idx_begin = idx_start[iidx];
      idx_end   = ((iidx + 1) < iidx_end) ? idx_start[iidx + 1] : ymax;
      sorted_value = static_cast<int>(sorted[idx_begin]);
    }

    const int start_feature = threadIdx.x + blockIdx.x * blockDim.x * SZ;
    const int dst_row = sorted_value * xmax;

    int num_idx = idx_end - idx_begin;
    int idx0 = idx_begin + threadIdx.y*num_idx/blockDim.y;
    int idx1 = idx_begin + (threadIdx.y + 1)*num_idx/blockDim.y;

    // Read and sum data into grad_weight[]
    DType grad_weight[SZ];
    #pragma unroll
    for (int ii = 0; ii < SZ; ii++) {
      grad_weight[ii] = (DType)0;
    }
    for (int idx=idx0; idx < idx1;idx++) {
      const int src_row = static_cast<int>(index[idx]) * xmax;
      #pragma unroll
      for (int ii = 0; ii < SZ; ii++)
      {
        int feature_dim = start_feature + ii * blockDim.x;
        if (feature_dim < xmax)
        {
          grad_weight[ii] += src[src_row + feature_dim];
        }
      }
    }
    #pragma unroll
    for (int ii = 0; ii < SZ; ii++) {
      sh_grad_weight[threadIdx.x + ii*blockDim.x + threadIdx.y*blockDim.x*SZ] = grad_weight[ii];
    }
    __syncthreads();
    // We now have grad_weight[] values, reduce within thread block
    for (int t=1;t < blockDim.y;t <<= 1) {
      DType tmp[SZ];
      #pragma unroll
      for (int ii = 0; ii < SZ; ii++) {
        tmp[ii] = (threadIdx.y + t < blockDim.y) ?
          sh_grad_weight[threadIdx.x + ii*blockDim.x + (threadIdx.y + t)*blockDim.x*SZ] : (DType)0;
      }
      __syncthreads();
      #pragma unroll
      for (int ii = 0; ii < SZ; ii++) {
        sh_grad_weight[threadIdx.x + ii*blockDim.x + threadIdx.y*blockDim.x*SZ] += tmp[ii];
      }
      __syncthreads();
    }
    // Result is in sh_grad_weight[threadIdx.x + ii*blockDim.x]
    if (threadIdx.y == 0) {
      #pragma unroll
      for (int ii = 0; ii < SZ; ii++) {
        int feature_dim = start_feature + ii * blockDim.x;
        if (feature_dim < xmax) {
          dst[dst_row + feature_dim] += sh_grad_weight[threadIdx.x + ii*blockDim.x];
        }
      }
    }

  }
}

template <typename IndexType, typename xpu>
inline typename std::enable_if<std::is_same<xpu, gpu>::value, size_t>::type
AddTakeGradLargeBatchWorkspaceSize(size_t num_keys) {
  size_t encode_bytes = 0;
  cub::DeviceRunLengthEncode::Encode<IndexType*, IndexType*, IndexType*, int*>
    (NULL, encode_bytes, NULL, NULL, NULL, NULL, num_keys);
  size_t exclusivesum_bytes = 0;
  cub::DeviceScan::ExclusiveSum<IndexType*, IndexType*>(NULL, exclusivesum_bytes,
    NULL, NULL, num_keys);
  size_t temporary_bytes = std::max(encode_bytes, exclusivesum_bytes);
  size_t unique_bytes = num_keys*sizeof(IndexType);
  size_t counts_bytes = num_keys*sizeof(IndexType);
  size_t num_runs_bytes = 1*sizeof(int);
  return (unique_bytes + counts_bytes + num_runs_bytes + temporary_bytes);
}

template<typename IndexType, typename DType>
inline void AddTakeGradLargeBatchKernelLaunch(mshadow::Tensor<gpu, 2, DType> dst,
                                              const mshadow::Tensor<gpu, 1, IndexType>& sorted,
                                              const mshadow::Tensor<gpu, 1, IndexType>& index,
                                              const mshadow::Tensor<gpu, 2, DType> &src,
                                              IndexType* sum_counts_ptr,
                                              int* num_runs_ptr,
                                              const mshadow::index_t num_rows) {
  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(dst.stream_);
  const int num_unique_est = min(num_rows, src.size(0));
  const int max_nthread = 128;
  const int num_y = max(src.size(0)/num_unique_est, 1);
  const int block_dim_x = kWarpSize;
  const int block_dim_y = min(num_y, max_nthread/block_dim_x);
  const int SZ = min((src.size(1) + block_dim_x - 1) / block_dim_x, 4);
  const int grid_dim_x = (src.size(1) + block_dim_x * SZ - 1) / (block_dim_x * SZ);
  const int grid_dim_y = min(num_unique_est, mshadow::cuda::kBaseGridNum);
  dim3 dimBlock(block_dim_x, block_dim_y);
  dim3 dimGrid(grid_dim_x, grid_dim_y);
  // Maximum shared memory usage: 128*4*sizeof(DType), which is 4K for 64bit DType elements
  int shmem_size = dimBlock.x*SZ*dimBlock.y*sizeof(DType);

  CHECK_EQ(dst.size(1), src.size(1)) << "AddTakeGradLargeBatch: shape mismatch";
  CHECK_EQ(index.size(0), src.size(0)) << "AddTakeGradLargeBatch: shape mismatch";
  mshadow::cuda::CheckLaunchParam(dimGrid, dimBlock, "AddTakeGradLargeBatch");

  switch (SZ) {
    case 1:
    AddTakeGradLargeBatchKernel<1, DType>
        <<<dimGrid, dimBlock, shmem_size, stream>>>
        (dst.dptr_, sum_counts_ptr, num_runs_ptr,
         sorted.dptr_, index.dptr_, src.dptr_,
         static_cast<int>(src.size(0)),
         static_cast<int>(src.size(1)));
    break;
    case 2:
    AddTakeGradLargeBatchKernel<2, DType>
        <<<dimGrid, dimBlock, shmem_size, stream>>>
        (dst.dptr_, sum_counts_ptr, num_runs_ptr,
         sorted.dptr_, index.dptr_, src.dptr_,
         static_cast<int>(src.size(0)),
         static_cast<int>(src.size(1)));
    break;
    case 3:
    AddTakeGradLargeBatchKernel<3, DType>
        <<<dimGrid, dimBlock, shmem_size, stream>>>
        (dst.dptr_, sum_counts_ptr, num_runs_ptr,
         sorted.dptr_, index.dptr_, src.dptr_,
         static_cast<int>(src.size(0)),
         static_cast<int>(src.size(1)));
    break;
    case 4:
    AddTakeGradLargeBatchKernel<4, DType>
        <<<dimGrid, dimBlock, shmem_size, stream>>>
        (dst.dptr_, sum_counts_ptr, num_runs_ptr,
         sorted.dptr_, index.dptr_, src.dptr_,
         static_cast<int>(src.size(0)),
         static_cast<int>(src.size(1)));
    break;
    default:
    LOG(FATAL) << "AddTakeGradLargeBatch, incorrect value SZ " << SZ;
    break;
  }
  MSHADOW_CUDA_POST_KERNEL_CHECK(AddTakeGradLargeBatchKernel);
}


template<typename IndexType, typename DType>
inline void AddTakeGradLargeBatch(mshadow::Tensor<gpu, 2, DType> dst,
                                  const mshadow::Tensor<gpu, 1, IndexType>& sorted,
                                  const mshadow::Tensor<gpu, 1, IndexType>& index,
                                  const mshadow::Tensor<gpu, 2, DType> &src,
                                  mshadow::Tensor<gpu, 1, char>* workspace) {
  CHECK_EQ(dst.CheckContiguous(), true);
  CHECK_EQ(sorted.CheckContiguous(), true);
  CHECK_EQ(index.CheckContiguous(), true);
  CHECK_EQ(src.CheckContiguous(), true);
  // const int kWarpBits = kMemUnitBits;
  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(dst.stream_);
  IndexType* sum_counts_ptr = NULL;
  int* num_runs_ptr = NULL;
  if (dst.size(0)*4 < src.size(0) && workspace != NULL) {
    // Workspace given and potentially loops at least 4 times, use CUB to create sum_counts
    CHECK_EQ(workspace->CheckContiguous(), true);
    // workspace = [unique_out, counts_out, temporary_storage]
    size_t unique_bytes = sorted.size(0)*sizeof(IndexType);
    size_t counts_bytes = sorted.size(0)*sizeof(IndexType);
    size_t num_runs_bytes = 1*sizeof(int);

    size_t encode_bytes = 0;
    cub::DeviceRunLengthEncode::Encode<IndexType*, IndexType*, IndexType*, int*>
      (NULL, encode_bytes, NULL, NULL, NULL, NULL, sorted.size(0), stream);
    size_t exclusivesum_bytes = 0;
    cub::DeviceScan::ExclusiveSum<IndexType*, IndexType*>
      (NULL, exclusivesum_bytes, NULL, NUsrc_indices_bytesLL, sorted.size(0), stream);
    size_t temporary_bytes = std::max(encode_bytes, exclusivesum_bytes);

    // Check that we have enough storage
    CHECK_GE(workspace->size(0), unique_bytes + counts_bytes +
      num_runs_bytes + temporary_bytes);

    IndexType* unique_out_ptr = reinterpret_cast<IndexType*>(workspace->dptr_);
    IndexType* counts_out_ptr = reinterpret_cast<IndexType*>(workspace->dptr_ + unique_bytes);
    num_runs_ptr = reinterpret_cast<int*>(workspace->dptr_ + unique_bytes +
      counts_bytes);
    void* temporary_storage = reinterpret_cast<void *>(workspace->dptr_ + unique_bytes +
      counts_bytes + num_runs_bytes);

    cub::DeviceRunLengthEncode::Encode<IndexType*, IndexType*, IndexType*, int*>
    (temporary_storage, temporary_bytes, sorted.dptr_, unique_out_ptr, counts_out_ptr,
      num_runs_ptr, sorted.size(0), stream);

    sum_counts_ptr = unique_out_ptr;
    cub::DeviceScan::ExclusiveSum<IndexType*, IndexType*>
    (temporary_storage, temporary_bytes, counts_out_ptr, sum_counts_ptr,
      sorted.size(0), stream);
  }
  AddTakeGradLargeBatchKernelLaunch(dst, sorted, index, src, sum_counts_ptr,
                                    num_runs_ptr, dst.size(0));
}

template<bool clip = true>
struct TakeGradGeneralKernel {
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int tid, DType* arr_grad, const DType* ograd,
                                  const IType* src_indptr, const IType* original_idx,
                                  mshadow::Shape<10> in_strides, mshadow::Shape<10> out_strides,
                                  const int in_ndims, const int out_ndims, const int idx_ndims, const int axis) {
    const int in_head_index = (axis == 0) ? 0 : tid / in_strides[axis - 1];
    const int in_rest_index = (axis == 0) ? tid : tid % in_strides[axis - 1];
    const int in_mid_index = in_rest_index / in_stride[axis];
    const int in_tail_index = (axis == in_ndims - 1) ?
                              0 : (in_rest_index % in_stride[axis]);
    for (int i = src_indptr[in_mid_index]; i < src_indptr[in_mid_index + 1]; ++i) {
      const int out_mid_index = original_idx[i];
      int target = in_tail_index + out_mid_index * out_stride[axis + idx_ndims - 1];
      target += (axis == 0) ? 0 : in_head_index * out_strides[axis - 1];
      arr_grad[tid] += ograd[target];
    }
  }
}

template<bool clip = true>
inline void TakeOpBackwardImpl(mshadow::Stream<gpu>* s,
                               const OpContext& ctx,
                               const TBlob& arr,
                               const TBlob& idx,
                               const TBlob& ograd,
                               const int axis) {
  using namespace mxnet_op;
  using namespace mshadow;
  CHECK(axis != 0) << "axis == 0 case should be dispatched to the legacy implementation";
  const TShape& arrshape = arr.shape_;
  const TShape& idxshape = idx.shape_;
  const TShape& oshape = ograd.shape_;
  // get size of temporary storage for sort
  char* temp_storage_ptr = nullptr;
  size_t scan_temp_storage_bytes = 0;
  IType* src_indptr_bytes = nullptr;
  cub::DeviceScan::ExclusiveSum(temp_storage_ptr,
                                scan_temp_storage_bytes,
                                src_indptr_bytes,
                                src_indptr_bytes,
                                arrshape[axis] + 1,
                                mshadow::Stream<gpu>::GetStream(s));
  size_t sort_temp_storage_bytes = SortByKeyWorkspaceSize<IType, IType, xpu>(idxshape.Size());
  size_t temp_storage_bytes = max(scan_temp_storage_bytes, sort_temp_storage_bytes);
  size_t original_idx_bytes = idxshape.Size() * sizeof(IType);
  size_t src_indptr_bytes = (arrshape[actual_axis] + 1) * sizeof(IType);
  size_t workspace_bytes = src_indptr_bytes + 2 * original_idx_bytes + temp_storage_bytes;
  Tensor<gpu, 1, char> workspace =
    ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_bytes), s);
  IType* sorted_idx_ptr = reinterpret_cast<IType*>(workspace.dptr_);
  IType* original_idx_ptr = reinterpret_cast<IType*>(workspace.dptr_ + original_idx_bytes);
  src_indptr_ptr = reinterpret_cast<IType*>(workspace.dptr_ + 2 * original_idx_bytes);
  char* temp_storage_ptr = workspace.dptr_ + 2 * original_idx_bytes + src_indptr_bytes;
  // Reset indptr to zero
  Kernel<set_zero, gpu>::Launch(s, arrshape[actual_axis] + 1, src_indptr_ptr);
  // Fill original_idx
  Kernel<range_fwd, gpu>::Launch(
    s, idxshape.Size(), 1, IType(0), IType(1), kWriteTo, original_idx_ptr);
  // Fill sorted_idx_ptr with unsorted copy of idx
  Kernel<op_with_req<mshadow_op::identity, kWriteTo>, gpu>::Launch(
    s, idxshape.Size(), sorted_idx_ptr, idx.dptr<IType>());
  if (clip) {
    Kernel<op_with_req<clip, kWriteTo>, gpu>::Launch(s, idxshape.Size(), sorted_idx_ptr,
                                                     sorted_idx_ptr, IType(0), IType(arrshape[axis]));
  } else {
    Kernel<op_with_req<mod, kWriteTo>, gpu>::Launch(s, idxshape.Size(), sorted_idx_ptr,
                                                    sorted_idx_ptr, IType(arrshape[axis]));
  }
  Tensor<gpu, 1, IType> original_idx(original_idx_ptr, Shape1(idxshape.Size()), s);
  Tensor<gpu, 1, char> temp_storage(temp_storage_ptr, Shape1(temp_storage_bytes), s);
  int num_bits = ilog2(static_cast<unsigned int>(idxshape.Size()) - 1);
  Tensor<gpu, 1, IType> sorted_idx(sorted_idx_ptr, Shape1(idxshape.Size()), s);
  SortByKey(sorted_idx, original_idx, true, &temp_storage, 0, num_bits);
  Kernel<HistogramKernel, gpu>::Launch(
    s, idxshape.Size(), src_indptr_ptr, idx.dptr<IType>(), idxshape.Size());
  cub::DeviceScan::ExclusiveSum(temp_storage_ptr,
                                temp_storage_bytes,
                                src_indptr_bytes,
                                src_indptr_bytes,
                                arrshape[actual_axis] + 1,
                                mshadow::Stream<gpu>::GetStream(s));

  Shape<10> in_strides;
  int stride = 1;
  for (int i = arrshape.ndim() - 1; i > 0; stride *= arrshape[i], --i) {
    in_strides[i] = stride;
  }
  Shape<10> out_strides;
  stride = 1;
  for (int i = oshape.ndim() - 1; i > 0; stride *= oshape[i], --i) {
    out_strides[i] = stride;
  }
  MSHADOW_TYPE_SWITCH(arr.type_flag_, DType, {
    Kernel<TakeGradGeneralKernel, gpu>::Launch(
      s, arrshape.Size(), arr.dptr<DType>(), ograd.dptr<DType>(), src_indptr_ptr, original_idx_ptr,
      in_strides, out_strides, arrshape.ndim(), oshape.ndim(), idxshape.ndim(), actual_axis);
  });
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_TENSOR_INDEXING_OP_CUH_
