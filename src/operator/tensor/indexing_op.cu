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
 * \file indexing_op.cu
 * \brief GPU implementation of indexing operator
 * \author Siyi Li, Chi Zhang
 */

#include "./indexing_op.h"
#include "./util/tensor_util-inl.cuh"
#include "./util/tensor_util-inl.h"

namespace mxnet {
namespace op {

/*! \brief If there are out-of-bound indices, out will be assigned to 1.
 */
struct is_valid_check {
  template <typename DType>
  MSHADOW_XINLINE static void Map(int i,
                                  char* out,
                                  const DType* data,
                                  const DType min,
                                  const DType max) {
    if (data[i] < min || data[i] > max)
      *out = 1;
  }
};

struct AddTakeGradRspGPUKernel {
  template <typename DType, typename IType>
  __device__ __forceinline__ static void Map(int tid,
                                             DType* out,
                                             const nnvm::dim_t* prefix_sum,
                                             const IType* data,
                                             const DType* ograd,
                                             const nnvm::dim_t row_length) {
    using nnvm::dim_t;
    const dim_t data_i  = tid / row_length;
    const dim_t grad_i  = tid % row_length;
    const dim_t irow    = static_cast<dim_t>(data[data_i]);
    const dim_t rsp_row = prefix_sum[irow] - 1;
    const DType val     = ograd[data_i * row_length + grad_i];
    atomicAdd(static_cast<DType*>(&(out[rsp_row * row_length + grad_i])), val);
  }
};

/*
 * \brief kernel for backward computation for take, executed with deterministic order
 * \param thread_id the thread id
 * \param out the output gradient data
 * \param lookup_table the table to lookup the position of an id in gradient array
 * \param sorted_data the sorted data input
 * \param original_idx the original indices of the sorted data input
 * \param ograd head gradient
 * \param row_length the output dimension
 * \param num_threads_per_row the number of threads to process a row together
 * \param SZ the number of features a thread is responsible for
 */
template <int SZ>
struct AddTakeGradRspDeterministicKernel {
  template <typename DType>
  __device__ __forceinline__ static void Map(int thread_id,
                                             DType* out,
                                             const nnvm::dim_t* lookup_table,
                                             const nnvm::dim_t* sorted_data,
                                             const nnvm::dim_t data_size,
                                             const nnvm::dim_t* original_idx,
                                             const DType* ograd,
                                             const nnvm::dim_t row_length,
                                             const nnvm::dim_t num_threads_per_row) {
    using nnvm::dim_t;
    int tid                 = thread_id / num_threads_per_row;
    const int feature_start = thread_id % num_threads_per_row * SZ;
    int num_features        = SZ;
    if (feature_start + num_features > row_length) {
      num_features = row_length - feature_start;
    }
    if (tid == 0 || sorted_data[tid - 1] != sorted_data[tid]) {
      DType acc[SZ];
#pragma unroll
      for (int i = 0; i < SZ; i++) {
        acc[i] = 0;
      }
      const dim_t data       = sorted_data[tid];
      const dim_t row_id     = lookup_table[data];
      const dim_t out_offset = row_id * row_length + feature_start;
      do {
        const dim_t idx          = original_idx[tid];
        const dim_t ograd_offset = idx * row_length + feature_start;
        for (int i = 0; i < num_features; i++) {
          acc[i] += ograd[ograd_offset + i];
        }
        tid++;
      } while (tid < data_size && sorted_data[tid - 1] == sorted_data[tid]);
      for (int i = 0; i < num_features; i++) {
        out[out_offset + i] += acc[i];
      }
    }
  }
};

template <bool clip = true>
struct TakeZeroAxisGPU {
  // assume that idx have been flattened to a 1-D tensor (N,)
  // assume that out_data and in_data have been flattened to 2-D tensors, (N, M) and (K, M)
  // M is the number of columns of in_data and out_data
  // K is the number of rows of in_data
  // i is the index of out_data
  template <typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i,
                                  DType* out_data,
                                  const DType* in_data,
                                  const IType* idx,
                                  const int64_t M,
                                  const int64_t K) {
    int64_t j = static_cast<int64_t>(idx[i / M]);
    if (clip) {
      if (j <= 0)
        j = 0;
      else if (j >= K)
        j = K - 1;
    } else {
      j = j % K;
      j += (j < 0) ? K : 0;
    }
    out_data[i] = in_data[j * M + i % M];
  }
};

/*
 * \brief returns true if all indices are between [min, max]
 * \param s the stream
 * \param data_ptr the indices on the stream
 * \param data_size the number of indices to examine
 * \param min the expected min value for indices
 * \param max the expected max value for indices
 * \param is_valid_ptr the temparary workspace
 */
template <typename DType>
bool CheckIndexOutOfBound(mshadow::Stream<gpu>* s,
                          const DType* data_ptr,
                          size_t data_size,
                          const DType min,
                          const DType max,
                          char* is_valid_ptr) {
  using namespace mxnet_op;
  int32_t is_valid = 0;
  Kernel<set_zero, gpu>::Launch(s, 1, is_valid_ptr);
  Kernel<is_valid_check, gpu>::Launch(s, data_size, is_valid_ptr, data_ptr, min, max);
  CUDA_CALL(cudaMemcpyAsync(&is_valid,
                            is_valid_ptr,
                            sizeof(char),
                            cudaMemcpyDeviceToHost,
                            mshadow::Stream<gpu>::GetStream(s)));
  CUDA_CALL(cudaStreamSynchronize(mshadow::Stream<gpu>::GetStream(s)));
  return is_valid == 0;
}

// Embedding forward implementation with dense weight
template <>
void EmbeddingOpForwardDnsImpl<gpu>(mshadow::Stream<gpu>* s,
                                    const TBlob& data,
                                    const TBlob& weight,
                                    const OpReqType req,
                                    const TBlob& output) {
  using namespace mxnet_op;
  const mxnet::TShape& ishape = data.shape_;
  const mxnet::TShape& oshape = output.shape_;

  MSHADOW_TYPE_SWITCH(output.type_flag_, DType, {
    MSHADOW_TYPE_SWITCH(data.type_flag_, IType, {
      Tensor<gpu, 1, IType> idx =
          data.get_with_shape<gpu, 1, IType>(Shape1(ishape.ProdShape(0, ishape.ndim())), s);
      Tensor<gpu, 2, DType> wmat = weight.get<gpu, 2, DType>(s);
      Tensor<gpu, 2, DType> out  = output.get_with_shape<gpu, 2, DType>(
          Shape2(oshape.ProdShape(0, oshape.ndim() - 1), oshape[oshape.ndim() - 1]), s);
      Kernel<TakeZeroAxisGPU<true>, gpu>::Launch(
          s, oshape.Size(), out.dptr_, wmat.dptr_, idx.dptr_, wmat.shape_[1], wmat.shape_[0]);
    });
  });
}

template <>
void SparseEmbeddingOpForwardRspImpl<gpu>(const OpContext& ctx,
                                          const TBlob& data,
                                          const NDArray& weight,
                                          const OpReqType req,
                                          const TBlob& output) {
  if (req == kNullOp)
    return;
  using namespace rowsparse;
  using namespace mxnet_op;
  mshadow::Stream<gpu>* s = ctx.get_stream<gpu>();
  // zeros weight
  if (req == kWriteTo && !weight.storage_initialized()) {
    size_t out_size = output.shape_.Size();
    MSHADOW_TYPE_SWITCH(output.type_flag_, DType, {
      Fill<false>(
          s, TBlob(output.dptr<DType>(), mshadow::Shape1(out_size), gpu::kDevMask), kWriteTo, 0);
    })
    return;
  }
  // check out-of-bound indices
  MSHADOW_TYPE_SWITCH(data.type_flag_, DType, {
    DType min                      = 0;
    DType max                      = static_cast<DType>(weight.shape()[0] - 1);
    DType* data_ptr                = data.dptr<DType>();
    size_t data_size               = data.shape_.Size();
    Tensor<gpu, 1, char> workspace = ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(1), s);
    char* is_valid_ptr             = reinterpret_cast<char*>(workspace.dptr_);
    bool is_valid = CheckIndexOutOfBound(s, data_ptr, data_size, min, max, is_valid_ptr);
    CHECK(is_valid) << "SparseEmbedding input contains data out of bound";
  })
  // the weight is actually dense
  if (weight.aux_shape(kIdx)[0] == weight.shape()[0]) {
    EmbeddingOpForwardDnsImpl<gpu>(s, data, weight.data(), req, output);
  } else {
    EmbeddingOpForwardRspImpl<gpu>(s, data, weight, req, output);
  }
}

template <typename IType, typename DType, typename RType>
void SparseEmbeddingDeterministicKernelLaunch(const OpContext& ctx,
                                              const TBlob& ograd,
                                              const TBlob& data,
                                              const OpReqType req,
                                              const NDArray& output) {
  using namespace mshadow;
  using namespace mxnet_op;
  using namespace expr;
  using namespace rowsparse;
  using nnvm::dim_t;
  mshadow::Stream<gpu>* s = ctx.get_stream<gpu>();
  const dim_t num_rows    = output.shape()[0];
  const dim_t row_length  = output.shape()[1];
  const dim_t data_size   = static_cast<dim_t>(data.shape_.Size());
  // temp resource declarations
  dim_t* lookup_table = nullptr;
  void* temp_storage  = nullptr;
  dim_t* sorted_data  = nullptr;
  dim_t* original_idx = nullptr;
  // calculate number of bytes for temp resources
  size_t lookup_table_bytes         = num_rows * sizeof(dim_t);
  size_t sorted_data_storage_bytes  = data_size * sizeof(dim_t);
  size_t original_idx_storage_bytes = data_size * sizeof(dim_t);
  size_t sort_workspace_size        = SortByKeyWorkspaceSize<dim_t, dim_t, gpu>(data_size);
  size_t unique_workspace_bytes     = 0;
  // estimate unique temp space
  IType* data_ptr  = data.dptr<IType>();
  size_t* null_ptr = nullptr;
  // unique operations will be applied on sorted data
  cub::DeviceSelect::Unique(nullptr,
                            unique_workspace_bytes,
                            sorted_data,
                            sorted_data,
                            null_ptr,
                            data_size,
                            Stream<gpu>::GetStream(s));
  // One more space reserved for unique count
  size_t temp_workspace_bytes = std::max(unique_workspace_bytes, sort_workspace_size);
  size_t total_storage_bytes  = lookup_table_bytes + sorted_data_storage_bytes +
                               original_idx_storage_bytes + temp_workspace_bytes;

  // request resource and split it. layout is:
  // lookup_table, sorted_data, original_idx, temp_storage
  Tensor<gpu, 1, char> workspace =
      ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(total_storage_bytes), s);
  lookup_table = reinterpret_cast<dim_t*>(workspace.dptr_);
  sorted_data  = reinterpret_cast<dim_t*>(workspace.dptr_ + lookup_table_bytes);
  original_idx =
      reinterpret_cast<dim_t*>(workspace.dptr_ + lookup_table_bytes + sorted_data_storage_bytes);
  temp_storage = workspace.dptr_ + total_storage_bytes - temp_workspace_bytes;

  // check out-of-bound indices
  {
    IType min        = 0;
    IType max        = static_cast<IType>(output.shape()[0] - 1);
    IType* data_ptr  = data.dptr<IType>();
    size_t data_size = data.shape_.Size();
    bool is_valid    = CheckIndexOutOfBound(
        s, data_ptr, data_size, min, max, reinterpret_cast<char*>(temp_storage));
    CHECK(is_valid) << "Embedding input contains data out of bound";
  }

  // make a copy of the data, to be sorted
  TBlob sorted_data_blob(sorted_data, Shape1(data_size), gpu::kDevMask);
  auto sorted_data_tensor = sorted_data_blob.FlatTo1D<gpu, dim_t>(s);
  mxnet_op::copy(s, sorted_data_blob, data);

  // generate original idx
  Tensor<gpu, 1, dim_t> original_idx_tensor(original_idx, Shape1(data_size), s);
  Kernel<range_fwd, gpu>::Launch(
      s, data_size, 1, static_cast<dim_t>(0), static_cast<dim_t>(1), kWriteTo, original_idx);
  // sort data with its original idx
  int num_bits           = common::ilog2ui(num_rows - 1);
  char* temp_storage_ptr = reinterpret_cast<char*>(temp_storage);
  Tensor<gpu, 1, char> temp_storage_tensor(temp_storage_ptr, Shape1(sort_workspace_size), s);
  SortByKey(sorted_data_tensor, original_idx_tensor, true, &temp_storage_tensor, 0, num_bits);

  // compute unique row ids based on sorted values.
  output.CheckAndAllocAuxData(kIdx, Shape1(data_size + 1));

  // fill row_idx array of output matrix, using the row_flg values
  RType* grad_row_idx = output.aux_data(kIdx).dptr<RType>();
  cub::DeviceSelect::Unique(temp_storage_ptr,
                            unique_workspace_bytes,
                            sorted_data,
                            grad_row_idx,
                            grad_row_idx + data_size,
                            data_size,
                            Stream<gpu>::GetStream(s));

  dim_t nnr = 0;
  CUDA_CALL(cudaMemcpyAsync(&nnr,
                            grad_row_idx + data_size,
                            sizeof(RType),
                            cudaMemcpyDeviceToHost,
                            mshadow::Stream<gpu>::GetStream(s)));
  CUDA_CALL(cudaStreamSynchronize(mshadow::Stream<gpu>::GetStream(s)));
  CHECK_EQ(output.shape().ndim(), 2) << "Unexcepted ndim";
  output.CheckAndAllocData(Shape2(nnr, output.shape()[1]));
  output.set_aux_shape(kIdx, Shape1(nnr));

  // generate lookup table
  Kernel<MarkLookupTable, gpu>::Launch(s, nnr, lookup_table, grad_row_idx);

  // accumulate gradients
  DType* grad_data = output.data().dptr<DType>();
  Fill<false>(s, TBlob(grad_data, Shape1(nnr * row_length), gpu::kDevMask), kWriteTo, 0);
  const int SZ                          = 4;
  const nnvm::dim_t num_threads_per_row = (row_length + SZ - 1) / SZ;
  Kernel<AddTakeGradRspDeterministicKernel<SZ>, gpu>::Launch(s,
                                                             data_size * num_threads_per_row,
                                                             grad_data,
                                                             lookup_table,
                                                             sorted_data,
                                                             data_size,
                                                             original_idx,
                                                             ograd.dptr<DType>(),
                                                             row_length,
                                                             num_threads_per_row);
}

inline void SparseEmbeddingOpBackwardDeterministicRspImpl(const OpContext& ctx,
                                                          const TBlob& ograd,
                                                          const TBlob& data,
                                                          const OpReqType req,
                                                          const NDArray& output) {
  using nnvm::dim_t;
  if (req == kNullOp)
    return;
  CHECK_EQ(req, kWriteTo) << "SparseEmbedding layer doesn't support "
                          << "weight gradient calculation with req != write";

  mshadow::Stream<gpu>* s = ctx.get_stream<gpu>();
  const dim_t data_size   = static_cast<dim_t>(data.shape_.Size());
  if (data_size == 0) {
    FillZerosRspImpl(s, output);
    return;
  }

  MSHADOW_TYPE_SWITCH(data.type_flag_, IType, {
    MSHADOW_TYPE_SWITCH(ograd.type_flag_, DType, {
      MSHADOW_IDX_TYPE_SWITCH(output.aux_type(rowsparse::kIdx), RType, {
        SparseEmbeddingDeterministicKernelLaunch<IType, DType, RType>(
            ctx, ograd, data, req, output);
      });
    });
  });
}

template <>
inline void SparseEmbeddingOpBackwardRspImpl<gpu>(const bool deterministic,
                                                  const OpContext& ctx,
                                                  const TBlob& ograd,
                                                  const TBlob& data,
                                                  const OpReqType req,
                                                  const NDArray& output) {
  if (deterministic) {
    SparseEmbeddingOpBackwardDeterministicRspImpl(ctx, ograd, data, req, output);
    return;
  }
  using namespace mshadow;
  using namespace mxnet_op;
  using namespace mshadow::expr;
  using namespace rowsparse;
  using nnvm::dim_t;
  if (req == kNullOp)
    return;
  CHECK_EQ(req, kWriteTo) << "SparseEmbedding layer doesn't support "
                          << "weight gradient calculation with req != write";

  // Request temporary storage for marking non-zero rows and prefix sum
  Stream<gpu>* s   = ctx.get_stream<gpu>();
  dim_t num_rows   = output.shape()[0];
  dim_t row_length = output.shape()[1];
  dim_t data_size  = static_cast<dim_t>(data.shape_.Size());
  dim_t num_threads;

  MSHADOW_TYPE_SWITCH(data.type_flag_, IType, {
    MSHADOW_SGL_DBL_TYPE_SWITCH(ograd.type_flag_, DType, {
      MSHADOW_IDX_TYPE_SWITCH(output.aux_type(kIdx), RType, {
        dim_t* prefix_sum         = nullptr;
        void* d_temp_storage      = nullptr;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(d_temp_storage,
                                      temp_storage_bytes,
                                      prefix_sum,
                                      prefix_sum,
                                      num_rows,
                                      Stream<gpu>::GetStream(s));
        Tensor<gpu, 1, char> workspace = ctx.requested[0].get_space_typed<gpu, 1, char>(
            Shape1(num_rows * sizeof(dim_t) + temp_storage_bytes), s);
        prefix_sum     = reinterpret_cast<dim_t*>(workspace.dptr_);
        d_temp_storage = workspace.dptr_ + num_rows * sizeof(dim_t);
        num_threads    = num_rows;
        Fill<false>(s, TBlob(prefix_sum, Shape1(num_threads), gpu::kDevMask), kWriteTo, 0);
        Kernel<MarkRowFlgKernel, gpu>::Launch(s, data_size, prefix_sum, data.dptr<IType>());

        cub::DeviceScan::InclusiveSum(d_temp_storage,
                                      temp_storage_bytes,
                                      prefix_sum,
                                      prefix_sum,
                                      num_rows,
                                      mshadow::Stream<gpu>::GetStream(s));
        dim_t nnr = 0;
        CUDA_CALL(cudaMemcpyAsync(&nnr,
                                  &prefix_sum[num_rows - 1],
                                  sizeof(dim_t),
                                  cudaMemcpyDeviceToHost,
                                  mshadow::Stream<gpu>::GetStream(s)));
        CUDA_CALL(cudaStreamSynchronize(mshadow::Stream<gpu>::GetStream(s)));
        if (nnr == 0) {
          FillZerosRspImpl(s, output);
          return;
        }
        output.CheckAndAlloc({Shape1(nnr)});
        RType* grad_row_idx = output.aux_data(kIdx).dptr<RType>();
        // fill row_idx array of output matrix, using the row_flg values
        Kernel<FillRspRowIdxKernel, gpu>::Launch(s, num_rows, grad_row_idx, prefix_sum, num_rows);
        // prefill with zeros
        DType* grad_data = output.data().dptr<DType>();
        Fill<false>(s, TBlob(grad_data, Shape1(nnr * row_length), gpu::kDevMask), kWriteTo, 0);
        // add the final gradients
        num_threads = row_length * data_size;
        Kernel<AddTakeGradRspGPUKernel, gpu>::Launch(s,
                                                     num_threads,
                                                     grad_data,
                                                     prefix_sum,
                                                     data.dptr<IType>(),
                                                     ograd.dptr<DType>(),
                                                     row_length);
      });
    });
  });
}

/*
 * \brief check if any of the indices is out of bound
 * \param s the stream
 * \param idx_ptr the indices on the stream
 * \param N the number of indices in an axis
 * \param M the number of axises to exmaine
 * \param mshape the array that stores shape for each dimension
 * \param is_valid_dim_ptr the temparary workspace that contains out-of-bound indices
 */
template <typename DType>
void GatherNDCheckBoundGPU(mshadow::Stream<gpu>* s,
                           const DType* idx_ptr,
                           index_t N,
                           index_t M,
                           const mshadow::Shape<10> mshape,
                           DType* is_valid_dim_ptr) {
  using namespace mxnet_op;
  Kernel<set_zero, gpu>::Launch(s, M, is_valid_dim_ptr);
  Kernel<is_valid_check_gather_nd, gpu>::Launch(s, M, is_valid_dim_ptr, idx_ptr, N, mshape);

  std::vector<DType> is_valid_dim(M);
  CUDA_CALL(cudaMemcpyAsync(is_valid_dim.data(),
                            is_valid_dim_ptr,
                            sizeof(DType) * M,
                            cudaMemcpyDeviceToHost,
                            mshadow::Stream<gpu>::GetStream(s)));
  CUDA_CALL(cudaStreamSynchronize(mshadow::Stream<gpu>::GetStream(s)));
  for (int m = 0; m < M; m++) {
    if (is_valid_dim[m] > mshape[m] - 1 || is_valid_dim[m] < -mshape[m]) {
      LOG(FATAL) << "IndexError: index " << is_valid_dim[m] << " is out of bounds for axis " << m
                 << " with size " << mshape[m];
    }
  }
}

void GatherNDForwardGPU(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  if (req[0] == kNullOp)
    return;
  mshadow::Stream<gpu>* s     = ctx.get_stream<gpu>();
  const mxnet::TShape& dshape = inputs[0].shape_;
  const mxnet::TShape& ishape = inputs[1].shape_;
  int M                       = ishape[0];
  int N                       = ishape.Size() / M;
  int K                       = dshape.ProdShape(M, dshape.ndim());
  mshadow::Shape<10> strides;
  mshadow::Shape<10> mshape;
  for (int i = M - 1, stride = K; i >= 0; stride *= dshape[i], --i) {
    strides[i] = stride;
    mshape[i]  = dshape[i];
  }
  MSHADOW_TYPE_SWITCH_WITH_BOOL(inputs[0].type_flag_, DType, {  // output data type switch
    MSHADOW_TYPE_SWITCH(inputs[1].type_flag_, IType, {          // indices data type switch
      // check whether indices are out of bound
      IType* idx_ptr = inputs[1].dptr<IType>();
      Tensor<gpu, 1, IType> workspace =
          ctx.requested[0].get_space_typed<gpu, 1, IType>(Shape1(M), s);
      IType* is_valid_dim_ptr = reinterpret_cast<IType*>(workspace.dptr_);
      GatherNDCheckBoundGPU(s, idx_ptr, N, M, mshape, is_valid_dim_ptr);
      Kernel<gather_nd, gpu>::Launch(s,
                                     N,
                                     req[0],
                                     N,
                                     M,
                                     K,
                                     strides,
                                     mshape,
                                     outputs[0].dptr<DType>(),
                                     inputs[0].dptr<DType>(),
                                     inputs[1].dptr<IType>());
    });
  });
}

struct backward_gather_nd_gpu {
  template <typename DType, typename IType>
  MSHADOW_XINLINE static void Map(index_t i,
                                  index_t N,
                                  index_t M,
                                  index_t K,
                                  const mshadow::Shape<10> strides,
                                  DType* out,
                                  const DType* data,
                                  const IType* indices) {
    index_t offset = 0;
    for (index_t j = 0; j < M; ++j) {
      offset += strides[j] * static_cast<int>(indices[j * N + i]);
    }
    for (index_t j = 0; j < K; ++j) {
      atomicAdd(out + (offset + j), data[i * K + j]);
    }
  }
};

template <typename DType, typename IType>
inline void GatherNDBackwardImpl(index_t N,
                                 index_t M,
                                 index_t K,
                                 const mshadow::Shape<10> strides,
                                 DType* out,
                                 const DType* data,
                                 const IType* indices,
                                 mshadow::Stream<gpu>* s) {
  mxnet_op::Kernel<backward_gather_nd_gpu, gpu>::Launch(s, N, N, M, K, strides, out, data, indices);
}

template <>
void TakeOpForward<gpu>(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  if (req[take_::kOut] == kNullOp)
    return;
  const TakeParam& param = nnvm::get<TakeParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);

  const mxnet::TShape& idxshape = inputs[take_::kIdx].shape_;
  const mxnet::TShape& arrshape = inputs[take_::kArr].shape_;
  const mxnet::TShape& oshape   = outputs[take_::kOut].shape_;

  if (idxshape.Size() == 0) {
    return;
  }

  Stream<gpu>* s        = ctx.get_stream<gpu>();
  const int actual_axis = param.axis + ((param.axis < 0) ? arrshape.ndim() : 0);

  MSHADOW_TYPE_SWITCH_WITH_BOOL(outputs[take_::kOut].type_flag_, DType, {   // output data type
    MSHADOW_TYPE_SWITCH_WITH_BOOL(inputs[take_::kIdx].type_flag_, IType, {  // index data type
      if (param.mode == take_::kRaise) {
        // check out-of-bound indices
        IType min       = 0;
        IType max       = static_cast<IType>(arrshape[actual_axis] - 1);
        IType* idx_ptr  = inputs[take_::kIdx].dptr<IType>();
        size_t idx_size = idxshape.Size();
        Tensor<gpu, 1, char> workspace =
            ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(1), s);
        char* is_valid_ptr = reinterpret_cast<char*>(workspace.dptr_);
        bool is_valid      = CheckIndexOutOfBound(s, idx_ptr, idx_size, min, max, is_valid_ptr);
        CHECK(is_valid) << "Take indices contains indices out of bound";
      }
      if (actual_axis == 0) {
        if (param.mode == take_::kClip) {
          Kernel<TakeZeroAxisGPU<true>, gpu>::Launch(s,
                                                     oshape.Size(),
                                                     outputs[take_::kOut].dptr<DType>(),
                                                     inputs[take_::kArr].dptr<DType>(),
                                                     inputs[take_::kIdx].dptr<IType>(),
                                                     oshape.Size() / idxshape.Size(),
                                                     arrshape[0]);
        } else {
          Kernel<TakeZeroAxisGPU<false>, gpu>::Launch(s,
                                                      oshape.Size(),
                                                      outputs[take_::kOut].dptr<DType>(),
                                                      inputs[take_::kArr].dptr<DType>(),
                                                      inputs[take_::kIdx].dptr<IType>(),
                                                      oshape.Size() / idxshape.Size(),
                                                      arrshape[0]);
        }
      } else {
        mshadow::Shape<10> in_strides;
        int stride = 1;
        for (int i = arrshape.ndim() - 1; i >= 0; stride *= arrshape[i], --i) {
          in_strides[i] = stride;
        }
        mshadow::Shape<10> out_strides;
        stride = 1;
        for (int i = oshape.ndim() - 1; i >= 0; stride *= oshape[i], --i) {
          out_strides[i] = stride;
        }
        if (param.mode == take_::kClip) {
          Kernel<TakeNonzeroAxis<true>, gpu>::Launch(s,
                                                     oshape.Size(),
                                                     outputs[take_::kOut].dptr<DType>(),
                                                     inputs[take_::kArr].dptr<DType>(),
                                                     inputs[take_::kIdx].dptr<IType>(),
                                                     out_strides[actual_axis - 1],
                                                     in_strides[actual_axis - 1],
                                                     in_strides[actual_axis],
                                                     arrshape.ndim(),
                                                     oshape.ndim(),
                                                     idxshape.ndim(),
                                                     arrshape[actual_axis],
                                                     actual_axis);
        } else {
          Kernel<TakeNonzeroAxis<false>, gpu>::Launch(s,
                                                      oshape.Size(),
                                                      outputs[take_::kOut].dptr<DType>(),
                                                      inputs[take_::kArr].dptr<DType>(),
                                                      inputs[take_::kIdx].dptr<IType>(),
                                                      out_strides[actual_axis - 1],
                                                      in_strides[actual_axis - 1],
                                                      in_strides[actual_axis],
                                                      arrshape.ndim(),
                                                      oshape.ndim(),
                                                      idxshape.ndim(),
                                                      arrshape[actual_axis],
                                                      actual_axis);
        }
      }
    });
  });
}

namespace {
/*
 * \brief returns integer log2(a) rounded up
 */
inline int ilog2(unsigned int a) {
  int k = 1;
  while (a >>= 1)
    k++;
  return k;
}
}  // namespace

/*
 * \brief finds the lower and upper-bound positions of each unique element within
 * a sorted input array
 *
 * \param sorted_data input elements previously sorted
 * \param bounds output containing all lower-bound followed by all upper-bound positions
 * \param data_dim total number of elements in the input array
 * \param vocab_dim maximum number of unique elements
 */
template <typename IType>
__global__ void EmbeddingFindBounds(const IType* sorted_data,
                                    IType* bounds,
                                    const index_t data_dim,
                                    const index_t vocab_dim) {
  const index_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= vocab_dim)
    return;

  // Binary search to find lower bound: stored at bounds[0..vocab_dim-1]
  IType lower_bound = 0;
  IType upper_bound = data_dim - 1;
  IType mean;
  while (lower_bound < upper_bound) {
    mean = (lower_bound + upper_bound) / 2;
    if (id <= sorted_data[mean])
      upper_bound = mean;
    else
      lower_bound = mean + 1;
  }
  bool found_row = (sorted_data[lower_bound] == id);
  if (!found_row) {
    bounds[id]             = -1;
    bounds[vocab_dim + id] = -2;
    return;
  } else {
    bounds[id] = lower_bound;
  }

  // Binary search to find upper bound: stored at bounds[vocab_dim..2*vocab_dim-1]
  lower_bound = 0;
  upper_bound = data_dim - 1;
  while (lower_bound < upper_bound) {
    mean = (lower_bound + upper_bound + 1) / 2;
    if (id >= sorted_data[mean])
      lower_bound = mean;
    else
      upper_bound = mean - 1;
  }
  bounds[vocab_dim + id] = upper_bound;
}

/*
 * \brief kernel to compute gradient of EmbeddingOp
 * \param grad_in input gradient data
 * \param original_index reference to the position at original input data for each index
 * \param index_bounds lower and upper-bounds positions of each unique index
 * \param grad_out output gradient data
 * \param embbedding_dim dimension of the dense embedding
 * \param vocab_dim maximum number of unique indices in the data array: tokens vocabulary size
 * \param nelems_per_load number of elements per each load based on (LType / DType)
 * \param req write/add/null
 */
template <typename AType, typename LType, typename DType, typename IType>
__global__ void EmbeddingGradKernel(DType* grad_in,
                                    const IType* original_index,
                                    const IType* index_bounds,
                                    const DType* grad_out,
                                    const index_t embbedding_dim,
                                    const index_t vocab_dim,
                                    const int nelems_per_load,
                                    const int req) {
  extern __shared__ int sharedmem[];
  AType* grad_in_row            = reinterpret_cast<AType*>(sharedmem);
  const LType* aligned_grad_out = reinterpret_cast<const LType*>(grad_out);
  LType* aligned_grad_in        = reinterpret_cast<LType*>(grad_in);
  const index_t aligned_emb_dim = embbedding_dim / nelems_per_load;
  LType load_value[1];
  DType* data_values = reinterpret_cast<DType*>(load_value);

  IType my_row = blockIdx.x;
  if (my_row < vocab_dim) {
    // Read lower and upper bounds for current row
    IType lower_bound = index_bounds[my_row];
    IType upper_bound = index_bounds[vocab_dim + my_row];
    int nOccurrences  = upper_bound - lower_bound + 1;

    for (index_t emb_id = threadIdx.x; emb_id < aligned_emb_dim; emb_id += blockDim.x) {
      // Initialize grad_in
      if (req == kAddTo) {
        *load_value = aligned_grad_in[my_row * aligned_emb_dim + emb_id];
        for (index_t val_id = 0; val_id < nelems_per_load; val_id++) {
          grad_in_row[val_id * blockDim.x + threadIdx.x] = static_cast<AType>(data_values[val_id]);
        }
      } else {
        for (index_t val_id = 0; val_id < nelems_per_load; val_id++) {
          grad_in_row[val_id * blockDim.x + threadIdx.x] = static_cast<AType>(0.0);
        }
      }
      // Add all rows from grad_out according to indices in data
      for (index_t data_idx = lower_bound; data_idx < (lower_bound + nOccurrences); ++data_idx) {
        *load_value = aligned_grad_out[original_index[data_idx] * aligned_emb_dim + emb_id];
        for (index_t val_id = 0; val_id < nelems_per_load; val_id++) {
          grad_in_row[val_id * blockDim.x + threadIdx.x] += static_cast<AType>(data_values[val_id]);
        }
      }
      // Save results
      for (index_t val_id = 0; val_id < nelems_per_load; val_id++) {
        data_values[val_id] = static_cast<DType>(grad_in_row[val_id * blockDim.x + threadIdx.x]);
      }
      aligned_grad_in[my_row * aligned_emb_dim + emb_id] = *load_value;
    }
  }
}

template <typename AType, typename IType, typename DType>
void EmbeddingGradKernelCaller(const OpContext& ctx,
                               mshadow::Tensor<gpu, 2, DType> grad_in,
                               const mshadow::Tensor<gpu, 1, IType>& index,
                               const mshadow::Tensor<gpu, 2, DType>& grad_out,
                               const std::vector<OpReqType>& req) {
  using namespace mxnet_op;
  using namespace mshadow::expr;

  Stream<gpu>* s               = ctx.get_stream<gpu>();
  const index_t data_dim       = index.shape_[0];
  const index_t vocab_dim      = grad_in.shape_[0];
  const index_t embbedding_dim = grad_in.shape_[1];

  // Calculate amount of temporary storage
  size_t sort_workspace_size = mxnet::op::SortByKeyWorkspaceSize<int, int, gpu>(data_dim);
  size_t workspace_size =
      2 * data_dim * sizeof(int) + 2 * vocab_dim * sizeof(int) + sort_workspace_size;

  // Request temporary storage
  Tensor<gpu, 1, char> workspace =
      ctx.requested[embedding::kTempSpace].get_space_typed<gpu, 1, char>(Shape1(workspace_size), s);

  // Create tensors
  size_t pos = 0;
  Tensor<gpu, 1, int> sorted_data(reinterpret_cast<int*>(&workspace[pos]), Shape1(data_dim), s);
  pos += data_dim * sizeof(int);
  // Reference to input data positions for each element of sorted_data
  Tensor<gpu, 1, int> original_index(reinterpret_cast<int*>(&workspace[pos]), Shape1(data_dim), s);
  pos += data_dim * sizeof(int);
  // lower and upper bound positions of each index within sorted_data
  Tensor<gpu, 1, int> bounds_index(
      reinterpret_cast<int*>(&workspace[pos]), Shape1(2 * vocab_dim), s);
  pos += 2 * vocab_dim * sizeof(int);
  Tensor<gpu, 1, char> Sort_temp_storage(&workspace[pos], Shape1(sort_workspace_size), s);

  // Clip indices [0, vocab_dim-1]
  Kernel<tcast_clip, gpu>::Launch(
      s, data_dim, sorted_data.dptr_, index.dptr_, static_cast<int>(vocab_dim));

  Kernel<range_fwd, gpu>::Launch(s, data_dim, 1, 0, 1, kWriteTo, original_index.dptr_);

  // Sort indices array
  int num_bits = ilog2((vocab_dim - 1));
  mxnet::op::SortByKey(sorted_data, original_index, true, &Sort_temp_storage, 0, num_bits);

  // Find lower & upper bounds of each possible index
  const int threads_block_bounds = 128;
  const int nblocks_bounds       = (vocab_dim + threads_block_bounds - 1) / threads_block_bounds;
  EmbeddingFindBounds<<<nblocks_bounds, threads_block_bounds, 0, Stream<gpu>::GetStream(s)>>>(
      sorted_data.dptr_, bounds_index.dptr_, data_dim, vocab_dim);

  // Compute Gradient
  int ltype = mxnet::common::cuda::get_load_type(embbedding_dim * sizeof(DType));

  MXNET_LOAD_TYPE_SWITCH(ltype, LType, {
    CHECK_LE(sizeof(DType), sizeof(LType));
    int nelems_per_load    = sizeof(LType) / sizeof(DType);
    int threads_block_grad = 32;
    int maxThreads         = 1024;
    while (threads_block_grad < (embbedding_dim / nelems_per_load) &&
           (threads_block_grad < maxThreads))
      threads_block_grad += 32;
    size_t required_shared = threads_block_grad * nelems_per_load * sizeof(AType);
    dim3 blocks(vocab_dim, 1);
    EmbeddingGradKernel<AType, LType>
        <<<blocks, threads_block_grad, required_shared, Stream<gpu>::GetStream(s)>>>(
            grad_in.dptr_,
            original_index.dptr_,
            bounds_index.dptr_,
            grad_out.dptr_,
            embbedding_dim,
            vocab_dim,
            nelems_per_load,
            req[embedding::kWeight]);
  });
}

template <>
void EmbeddingOpBackward<gpu>(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<TBlob>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 2U);
  CHECK_EQ(req[embedding::kData], kNullOp)
      << "Embedding layer doesn't support calculate data gradient";
  if (req[embedding::kWeight] == kNullOp) {
    return;
  }
  CHECK_EQ(outputs[1].type_flag_, inputs[0].type_flag_);

  const mxnet::TShape& ishape = inputs[1].shape_;
  const mxnet::TShape& oshape = inputs[0].shape_;

  Stream<gpu>* s = ctx.get_stream<gpu>();

  CHECK_NE(req[embedding::kWeight], kWriteInplace)
      << "Backward of Embedding does not support writing in place.";
  bool safe_acc = dmlc::GetEnv("MXNET_SAFE_ACCUMULATION", true);
  if (!safe_acc && outputs[1].type_flag_ == mshadow::kFloat16) {
    common::LogOnce(
        "MXNET_SAFE_ACCUMULATION=1 is recommended for EmbeddingOpBackward "
        "with float16 inputs. "
        "See https://mxnet.apache.org/api/faq/env_var "
        "for more details.");
  }
  MXNET_REAL_ACC_TYPE_SWITCH(outputs[1].type_flag_, DType, AType, {
    MSHADOW_TYPE_SWITCH(inputs[1].type_flag_, IType, {
      Tensor<gpu, 1, IType> data =
          inputs[1].get_with_shape<gpu, 1, IType>(Shape1(ishape.ProdShape(0, ishape.ndim())), s);
      Tensor<gpu, 2, DType> grad_out = inputs[0].get_with_shape<gpu, 2, DType>(
          Shape2(oshape.ProdShape(0, oshape.ndim() - 1), oshape[oshape.ndim() - 1]), s);
      Tensor<gpu, 2, DType> grad_in = outputs[1].get<gpu, 2, DType>(s);

      if (req[embedding::kWeight] == kWriteTo || req[embedding::kWeight] == kAddTo) {
        if (safe_acc)
          EmbeddingGradKernelCaller<AType>(ctx, grad_in, data, grad_out, req);
        else
          EmbeddingGradKernelCaller<DType>(ctx, grad_in, data, grad_out, req);
      } else {
        LOG(FATAL) << "wrong req";
      }
    });
  });
}

NNVM_REGISTER_OP(Embedding).set_attr<FCompute>("FCompute<gpu>", EmbeddingOpForward<gpu>);

NNVM_REGISTER_OP(_backward_Embedding)
    .set_attr<FCompute>("FCompute<gpu>", EmbeddingOpBackward<gpu>)
    .set_attr<FComputeEx>("FComputeEx<gpu>", EmbeddingOpBackwardEx<gpu>);

NNVM_REGISTER_OP(take).set_attr<FCompute>("FCompute<gpu>", TakeOpForward<gpu>);

NNVM_REGISTER_OP(_backward_take).set_attr<FCompute>("FCompute<gpu>", TakeOpBackward<gpu>);

NNVM_REGISTER_OP(batch_take).set_attr<FCompute>("FCompute<gpu>", BatchTakeOpForward<gpu>);

NNVM_REGISTER_OP(one_hot).set_attr<FCompute>("FCompute<gpu>", OneHotOpForward<gpu>);

NNVM_REGISTER_OP(gather_nd)
    .set_attr<FIsCUDAGraphsCompatible>("FIsCUDAGraphsCompatible",
                                       [](const NodeAttrs&, const bool) { return false; })
    .set_attr<FCompute>("FCompute<gpu>", GatherNDForwardGPU);

NNVM_REGISTER_OP(scatter_nd).set_attr<FCompute>("FCompute<gpu>", ScatterNDForward<gpu>);

NNVM_REGISTER_OP(_backward_gather_nd).set_attr<FCompute>("FCompute<gpu>", GatherNDBackward<gpu>);

NNVM_REGISTER_OP(_scatter_set_nd).set_attr<FCompute>("FCompute<gpu>", ScatterSetNDForward<gpu>);
}  // namespace op
}  // namespace mxnet
