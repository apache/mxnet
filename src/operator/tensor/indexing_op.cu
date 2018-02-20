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
 * \file indexing_op.cu
 * \brief
 * \author Siyi Li, Chi Zhang
*/

#include "./indexing_op.h"
#include "./util/tensor_util-inl.cuh"
#include <chrono>

namespace mxnet {
namespace op {

/*! \brief If there are out-of-bound indices, out will be assigned to 1.
 */

struct is_valid_check {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, int32_t* out, const DType* data,
                                  const DType min, const DType max) {
    if (data[i] < min || data[i] > max) *out = 1;
  }
};


struct AddTakeGradRspGPUKernel {
  template<typename DType, typename IType>
  __device__ __forceinline__ static void Map(int tid,
                                             DType* out,
                                             const nnvm::dim_t* prefix_sum,
                                             const IType* data,
                                             const DType* ograd,
                                             const nnvm::dim_t row_length) {
    using nnvm::dim_t;
    const dim_t data_i = tid / row_length;
    const dim_t grad_i = tid % row_length;
    const dim_t irow = static_cast<dim_t>(data[data_i]);
    const dim_t rsp_row = prefix_sum[irow] - 1;
    const DType val = ograd[data_i * row_length + grad_i];
    atomicAdd(static_cast<DType *>(&(out[rsp_row*row_length+grad_i])), val);
  }
};

template<>
void SparseEmbeddingOpForwardRspImpl<gpu>(const OpContext& ctx,
                                          const TBlob& data,
                                          const NDArray& weight,
                                          const OpReqType req,
                                          const TBlob& output) {
  if (req == kNullOp) return;
  using namespace rowsparse;
  using namespace mxnet_op;
  mshadow::Stream<gpu>* s = ctx.get_stream<gpu>();
  // zeros weight
  if (req == kWriteTo && !weight.storage_initialized()) {
    size_t out_size = output.shape_.Size();
    MSHADOW_TYPE_SWITCH(output.type_flag_, DType, {
      Fill<false>(s, TBlob(output.dptr<DType>(), mshadow::Shape1(out_size),
          gpu::kDevMask), kWriteTo, 0);
    })
    return;
  }
  // check out-of-bound indices
  int32_t is_valid = 0;
  MSHADOW_TYPE_SWITCH(data.type_flag_, DType, {
    DType min = 0;
    DType max = static_cast<DType>(weight.shape()[0] - 1);
    DType* data_ptr = data.dptr<DType>();
    size_t data_size = data.shape_.Size();
    Tensor<gpu, 1, char> workspace = ctx.requested[0]
        .get_space_typed<gpu, 1, char>(Shape1(sizeof(int32_t)), s);
    int32_t* is_valid_ptr = reinterpret_cast<int32_t*>(workspace.dptr_);
    Kernel<set_zero, gpu>::Launch(s, 1, is_valid_ptr);
    Kernel<is_valid_check, gpu>::Launch(s, data_size, is_valid_ptr, data_ptr, min, max);
    CUDA_CALL(cudaMemcpy(&is_valid, is_valid_ptr, sizeof(int32_t),
              cudaMemcpyDeviceToHost));
  })
  CHECK_EQ(is_valid, 0) << "SparseEmbedding input contains data out of bound";
  // the weight is actually dense
  if (weight.aux_shape(kIdx)[0] == weight.shape()[0]) {
    EmbeddingOpForwardDnsImpl<gpu>(s, data, weight.data(), req, output);
  } else {
    EmbeddingOpForwardRspImpl<gpu>(s, data, weight, req, output);
  }
}

inline void SparseEmbeddingOpBackwardRspImpl3(const OpContext& ctx,
                                              const TBlob& ograd,
                                              const TBlob& data,
                                              const OpReqType req,
                                              const NDArray& output) {
  using namespace mshadow;
  using namespace mxnet_op;
  using namespace mshadow::expr;
  using namespace rowsparse;
  using nnvm::dim_t;
  if (req == kNullOp) return;
  CHECK_EQ(req, kWriteTo) << "SparseEmbedding layer doesn't support "
                          << "weight gradient calculation with req != write";

  // Request temporary storage for marking non-zero rows and prefix sum
  Stream<gpu> *s = ctx.get_stream<gpu>();
  dim_t num_rows = output.shape()[0];
  dim_t row_length = output.shape()[1];
  dim_t data_size = static_cast<dim_t>(data.shape_.Size());
  dim_t num_threads;
  if (data_size == 0) {
    FillZerosRspImpl(s, output);
    return;
  }
   s->Wait();
  auto t0 = std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::high_resolution_clock::now().time_since_epoch()).count();

  MSHADOW_TYPE_SWITCH(data.type_flag_, IType, {
    MSHADOW_TYPE_SWITCH(ograd.type_flag_, DType, {
      MSHADOW_IDX_TYPE_SWITCH(output.aux_type(kIdx), RType, {
        dim_t* prefix_sum = NULL;
        void* temp_storage = NULL;
        dim_t* sorted_data = NULL;
        dim_t* original_idx = NULL;
        // calculate resource bytes
        size_t row_flg_storage_bytes = num_rows * sizeof(dim_t);
        size_t sorted_data_storage_bytes = data_size * sizeof(dim_t);
        size_t original_idx_storage_bytes = data_size * sizeof(dim_t);
        size_t sum_workspace_bytes = 0;
        size_t sort_workspace_size = SortByKeyWorkspaceSize<dim_t, dim_t, gpu>(data_size);
        cub::DeviceScan::InclusiveSum(temp_storage,
                                      sum_workspace_bytes,
                                      prefix_sum,
                                      prefix_sum,
                                      num_rows,
                                      Stream<gpu>::GetStream(s));
        // temp_workspace is shared by inclusive sum and sort
        size_t temp_workspace_bytes = std::max(sum_workspace_bytes, sort_workspace_size);
        size_t total_storage_bytes = row_flg_storage_bytes + sorted_data_storage_bytes +
                                     original_idx_storage_bytes + temp_workspace_bytes;

        // request resource and split it. layout =
        // row_flg/prefixsum, sorted_data, original_idx, temp_storage
        Tensor<gpu, 1, char> workspace = ctx.requested[0]
            .get_space_typed<gpu, 1, char>(Shape1(total_storage_bytes), s);
        prefix_sum = reinterpret_cast<dim_t*>(workspace.dptr_);
        sorted_data = reinterpret_cast<dim_t*>(workspace.dptr_ + row_flg_storage_bytes);
        original_idx = reinterpret_cast<dim_t*>(workspace.dptr_ + row_flg_storage_bytes +
                                                sorted_data_storage_bytes);
        temp_storage = workspace.dptr_ + total_storage_bytes - temp_workspace_bytes;
        // compute row flags and prefix sum
        num_threads = num_rows;
   s->Wait();
  auto t1 = std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        Fill<false>(s, TBlob(prefix_sum, Shape1(num_threads), gpu::kDevMask), kWriteTo, 0);
        Kernel<MarkRowFlgKernel, gpu>::Launch(s, data_size, prefix_sum, data.dptr<IType>());
        cub::DeviceScan::InclusiveSum(temp_storage,
                                      temp_workspace_bytes,
                                      prefix_sum,
                                      prefix_sum,
                                      num_rows,
                                      mshadow::Stream<gpu>::GetStream(s));
        // retrieve nnr and allocate output
   s->Wait();
  auto t2 = std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        dim_t nnr = 0;
        CUDA_CALL(cudaMemcpy(&nnr, &prefix_sum[num_rows-1], sizeof(dim_t),
            cudaMemcpyDeviceToHost));
        output.CheckAndAlloc({Shape1(nnr)});
        // fill row_idx array of output matrix, using the row_flg values
        RType* grad_row_idx = output.aux_data(kIdx).dptr<RType>();
        Kernel<FillRspRowIdxKernel, gpu>::Launch(s, num_rows,
            grad_row_idx, prefix_sum, num_rows);

        // make a copy of the data, to be sorted
        TBlob sorted_data_blob(sorted_data, Shape1(data_size), gpu::kDevMask);
        auto sorted_data_tensor = sorted_data_blob.FlatTo1D<gpu, dim_t>(s);
        mxnet_op::copy(s, sorted_data_blob, data);

        // generate original idx
        Tensor<gpu, 1, dim_t> original_idx_tensor(original_idx, Shape1(data_size), s);
        Kernel<range_fwd, gpu>::Launch(s, data_size, 1, static_cast<dim_t>(0), static_cast<dim_t>(1),
                                       kWriteTo, original_idx);
        // sort data with its original idx
        int num_bits = ilog2(num_rows - 1);
        char* temp_storage_ptr = reinterpret_cast<char*>(temp_storage);
        Tensor<gpu, 1, char> temp_storage_tensor(temp_storage_ptr,
                                                 Shape1(sort_workspace_size), s);
   s->Wait();
  auto t3 = std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        SortByKey(sorted_data_tensor, original_idx_tensor, true,
                  &temp_storage_tensor, 0, num_bits);
   s->Wait();
  auto t4 = std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::high_resolution_clock::now().time_since_epoch()).count();

        // accumulate gradients
        DType* grad_data = output.data().dptr<DType>();
        Fill<false>(s, TBlob(grad_data, Shape1(nnr * row_length), gpu::kDevMask),
            kWriteTo, 0);
        //const int SZ = 4;
        //int num_threads_per_data = (row_length + SZ - 1) / SZ;

        //Kernel<AddTakeGradRspGPUKernel2, gpu>::Launch(s, data_size * num_threads_per_data,
        //       grad_data, prefix_sum, sorted_data, data_size, original_idx,
        //       ograd.dptr<DType>(), row_length, num_threads_per_data, SZ);

{
  //using IndexType = IType;
  dim_t* sum_counts_ptr = NULL;
  int* num_runs_ptr = NULL;
  mshadow::Tensor<gpu, 2, DType> dst = output.data().get<gpu, 2, DType>(s);
  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(dst.stream_);
  mshadow::Tensor<gpu, 1, dim_t> sorted = sorted_data_tensor;
  mshadow::Tensor<gpu, 1, dim_t> index = original_idx_tensor;
  auto arrshape = ograd.shape_;
  mshadow::Tensor<gpu, 2, DType> src = ograd.get_with_shape<gpu, 2, DType>(
          Shape2(arrshape[0], arrshape.ProdShape(1, arrshape.ndim())), s);

  const int num_unique_est = min(dst.size(0), src.size(0));
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
  nnvm::dim_t* lookup_table = prefix_sum;

  switch (SZ) {
    case 1:
    AddTakeGradLargeBatchKernel<1, true, DType>
        <<<dimGrid, dimBlock, shmem_size, stream>>>
        (dst.dptr_, sum_counts_ptr, num_runs_ptr,
         sorted.dptr_, index.dptr_, src.dptr_,
         static_cast<int>(src.size(0)),
         static_cast<int>(src.size(1)), lookup_table);
    break;
    case 2:
    AddTakeGradLargeBatchKernel<2, true, DType>
        <<<dimGrid, dimBlock, shmem_size, stream>>>
        (dst.dptr_, sum_counts_ptr, num_runs_ptr,
         sorted.dptr_, index.dptr_, src.dptr_,
         static_cast<int>(src.size(0)),
         static_cast<int>(src.size(1)), lookup_table);
    break;
    case 3:
    AddTakeGradLargeBatchKernel<3, true, DType>
        <<<dimGrid, dimBlock, shmem_size, stream>>>
        (dst.dptr_, sum_counts_ptr, num_runs_ptr,
         sorted.dptr_, index.dptr_, src.dptr_,
         static_cast<int>(src.size(0)),
         static_cast<int>(src.size(1)), lookup_table);
    break;
    case 4:
    AddTakeGradLargeBatchKernel<4, true, DType>
        <<<dimGrid, dimBlock, shmem_size, stream>>>
        (dst.dptr_, sum_counts_ptr, num_runs_ptr,
         sorted.dptr_, index.dptr_, src.dptr_,
         static_cast<int>(src.size(0)),
         static_cast<int>(src.size(1)), lookup_table);
    break;
    default:
    LOG(FATAL) << "AddTakeGradLargeBatch, incorrect value SZ " << SZ;
    break;
  }
  MSHADOW_CUDA_POST_KERNEL_CHECK(AddTakeGradLargeBatchKernel);


}


   s->Wait();
  auto t5 = std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::high_resolution_clock::now().time_since_epoch()).count();
bool log = dmlc::GetEnv("LOG_TIME", true);
if (log) {
LOG(INFO) << t1-t0 << "\t"
          << t2-t1 << "\t"
          << t3-t2 << "\t"
          << t4-t3 << "\t"
          << t5-t4 << "\t"
          << t5-t0 << "\t";
}
      });
    });
  });
}

template<>
inline void SparseEmbeddingOpBackwardRspImpl<gpu>(const OpContext& ctx,
                                                  const TBlob& ograd,
                                                  const TBlob& data,
                                                  const OpReqType req,
                                                  const NDArray& output) {
  
  int deterministic = dmlc::GetEnv("MXNET_DETERM", 0);
  if (deterministic == 1) {
    LOG(FATAL) << "NOT REAHCED";
    return;
  } else if (deterministic == 2) {
    SparseEmbeddingOpBackwardRspImpl3(ctx, ograd, data, req, output);
    return;
  }
  using namespace mshadow;
  using namespace mxnet_op;
  using namespace mshadow::expr;
  using namespace rowsparse;
  using nnvm::dim_t;
  if (req == kNullOp) return;
  CHECK_EQ(req, kWriteTo) << "SparseEmbedding layer doesn't support "
                          << "weight gradient calculation with req != write";

  // Request temporary storage for marking non-zero rows and prefix sum
  Stream<gpu> *s = ctx.get_stream<gpu>();
  dim_t num_rows = output.shape()[0];
  dim_t row_length = output.shape()[1];
  dim_t data_size = static_cast<dim_t>(data.shape_.Size());
  dim_t num_threads;

  MSHADOW_TYPE_SWITCH(data.type_flag_, IType, {
    MSHADOW_SGL_DBL_TYPE_SWITCH(ograd.type_flag_, DType, {
      MSHADOW_IDX_TYPE_SWITCH(output.aux_type(kIdx), RType, {
        dim_t* prefix_sum = NULL;
        void* d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(d_temp_storage,
                                      temp_storage_bytes,
                                      prefix_sum,
                                      prefix_sum,
                                      num_rows,
                                      Stream<gpu>::GetStream(s));
        Tensor<gpu, 1, char> workspace = ctx.requested[0]
            .get_space_typed<gpu, 1, char>(Shape1(num_rows * sizeof(dim_t) +
                                           temp_storage_bytes), s);
        prefix_sum = reinterpret_cast<dim_t*>(workspace.dptr_);
        d_temp_storage = workspace.dptr_ + num_rows*sizeof(dim_t);
        num_threads = num_rows;
        Fill<false>(s, TBlob(prefix_sum, Shape1(num_threads), gpu::kDevMask), kWriteTo, 0);
        Kernel<MarkRowFlgKernel, gpu>::Launch(s, data_size, prefix_sum, data.dptr<IType>());

        cub::DeviceScan::InclusiveSum(d_temp_storage,
                                      temp_storage_bytes,
                                      prefix_sum,
                                      prefix_sum,
                                      num_rows,
                                      mshadow::Stream<gpu>::GetStream(s));
        dim_t nnr = 0;
        CUDA_CALL(cudaMemcpy(&nnr, &prefix_sum[num_rows-1], sizeof(dim_t),
            cudaMemcpyDeviceToHost));

        if (nnr == 0) {
          FillZerosRspImpl(s, output);
          return;
        }
        output.CheckAndAlloc({Shape1(nnr)});
        RType* grad_row_idx = output.aux_data(kIdx).dptr<RType>();
        // fill row_idx array of output matrix, using the row_flg values
        Kernel<FillRspRowIdxKernel, gpu>::Launch(s, num_rows,
            grad_row_idx, prefix_sum, num_rows);
        // prefill with zeros
        DType* grad_data = output.data().dptr<DType>();
        Fill<false>(s, TBlob(grad_data, Shape1(nnr * row_length), gpu::kDevMask),
            kWriteTo, 0);
        // add the final gradients
        num_threads = row_length * data_size;
        Kernel<AddTakeGradRspGPUKernel, gpu>::Launch(s, num_threads, grad_data, prefix_sum,
            data.dptr<IType>(), ograd.dptr<DType>(), row_length);
      });
    });
  });
}

struct backward_gather_nd_gpu {
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i, int N, int M, int K,
                                  const mshadow::Shape<10> strides,
                                  DType* out, const DType* data,
                                  const IType* indices) {
    int offset = 0;
    for (int j = 0; j < M; ++j) {
      offset += strides[j] * static_cast<int>(indices[j*N + i]);
    }
    for (int j = 0; j < K; ++j) {
      atomicAdd(out + (offset + j), data[i * K + j]);
    }
  }
};

template<typename DType, typename IType>
inline void GatherNDBackwardImpl(int N, int M, int K,
                                 const mshadow::Shape<10> strides,
                                 DType* out,
                                 const DType* data,
                                 const IType* indices,
                                 mshadow::Stream<gpu> *s) {
  mxnet_op::Kernel<backward_gather_nd_gpu, gpu>::Launch(s, N, N, M, K, strides, out, data, indices);
}

NNVM_REGISTER_OP(Embedding)
.set_attr<FCompute>("FCompute<gpu>", EmbeddingOpForward<gpu>);

NNVM_REGISTER_OP(_contrib_SparseEmbedding)
.set_attr<FComputeEx>("FComputeEx<gpu>", SparseEmbeddingOpForwardEx<gpu>);

NNVM_REGISTER_OP(_backward_Embedding)
.set_attr<FCompute>("FCompute<gpu>", EmbeddingOpBackward<gpu>);

NNVM_REGISTER_OP(_backward_SparseEmbedding)
.set_attr<FComputeEx>("FComputeEx<gpu>", SparseEmbeddingOpBackwardEx<gpu>);

NNVM_REGISTER_OP(take)
.set_attr<FCompute>("FCompute<gpu>", TakeOpForward<gpu>);

NNVM_REGISTER_OP(_backward_take)
.set_attr<FCompute>("FCompute<gpu>", TakeOpBackward<gpu>);

NNVM_REGISTER_OP(batch_take)
.set_attr<FCompute>("FCompute<gpu>", BatchTakeOpForward<gpu>);

NNVM_REGISTER_OP(one_hot)
.set_attr<FCompute>("FCompute<gpu>", OneHotOpForward<gpu>);

NNVM_REGISTER_OP(gather_nd)
.set_attr<FCompute>("FCompute<gpu>", GatherNDForward<gpu>);

NNVM_REGISTER_OP(scatter_nd)
.set_attr<FCompute>("FCompute<gpu>", ScatterNDForward<gpu>);

NNVM_REGISTER_OP(_backward_gather_nd)
.set_attr<FCompute>("FCompute<gpu>", GatherNDBackward<gpu>);

NNVM_REGISTER_OP(_scatter_set_nd)
.set_attr<FCompute>("FCompute<gpu>", ScatterSetNDForward<gpu>);
}  // namespace op
}  // namespace mxnet
