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
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, char* out, const DType* data,
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
template<int SZ>
struct AddTakeGradRspDeterministicKernel {
  template<typename DType>
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
    int tid = thread_id / num_threads_per_row;
    const int feature_start = thread_id % num_threads_per_row * SZ;
    int num_features = SZ;
    if (feature_start + num_features > row_length) {
      num_features = row_length - feature_start;
    }
    if (tid == 0 || sorted_data[tid - 1] != sorted_data[tid]) {
      DType acc[SZ];
      #pragma unroll
      for (int i = 0; i < SZ; i++) {
        acc[i] = 0;
      }
      const dim_t data = sorted_data[tid];
      const dim_t row_id = lookup_table[data];
      const dim_t out_offset = row_id * row_length + feature_start;
      do {
        const dim_t idx = original_idx[tid];
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

/*! \brief name the struct Take instead of take
 * to avoid conflict with the take function in mshadow
 */
template<bool clip = true>
struct TakeGPU {
  // assume that idx have been flattened to a 1-D tensor (N,)
  // assume that out_data and in_data have been flattened to 2-D tensors, (N, M) and (K, M)
  // M is the number of columns of in_data and out_data
  // K is the number of rows of in_data
  // i is the index of out_data
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data, const DType* in_data,
                                  const IType* idx, const int64_t M, const int64_t K) {
    int64_t j = static_cast<int64_t>(idx[i/M]);
    if (clip) {
      if (j <= 0) j = 0;
      else if (j >= K) j = K - 1;
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
template<typename DType>
bool CheckIndexOutOfBound(mshadow::Stream<gpu> *s, const DType* data_ptr, size_t data_size,
                          const DType min, const DType max, char* is_valid_ptr) {
  using namespace mxnet_op;
  int32_t is_valid = 0;
  Kernel<set_zero, gpu>::Launch(s, 1, is_valid_ptr);
  Kernel<is_valid_check, gpu>::Launch(s, data_size, is_valid_ptr, data_ptr, min, max);
  CUDA_CALL(cudaMemcpy(&is_valid, is_valid_ptr, sizeof(char),
            cudaMemcpyDeviceToHost));
  return is_valid == 0;
}

// Embedding forward implementation with dense weight
template<>
void EmbeddingOpForwardDnsImpl<gpu>(mshadow::Stream<gpu>* s,
                                    const TBlob& data,
                                    const TBlob& weight,
                                    const OpReqType req,
                                    const TBlob& output) {
  using namespace mxnet_op;
  const TShape& ishape = data.shape_;
  const TShape& oshape = output.shape_;

  MSHADOW_TYPE_SWITCH(output.type_flag_, DType, {
    MSHADOW_TYPE_SWITCH(data.type_flag_, IType, {
      Tensor<gpu, 1, IType> idx = data.get_with_shape<gpu, 1, IType>(
        Shape1(ishape.ProdShape(0, ishape.ndim())), s);
      Tensor<gpu, 2, DType> wmat = weight.get<gpu, 2, DType>(s);
      Tensor<gpu, 2, DType> out = output.get_with_shape<gpu, 2, DType>(
        Shape2(oshape.ProdShape(0, oshape.ndim()-1), oshape[oshape.ndim()-1]), s);
      Kernel<TakeGPU<true>, gpu>::Launch(s, oshape.Size(), out.dptr_, wmat.dptr_,
                                         idx.dptr_, wmat.shape_[1], wmat.shape_[0]);
    });
  });
}

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
  MSHADOW_TYPE_SWITCH(data.type_flag_, DType, {
    DType min = 0;
    DType max = static_cast<DType>(weight.shape()[0] - 1);
    DType* data_ptr = data.dptr<DType>();
    size_t data_size = data.shape_.Size();
    Tensor<gpu, 1, char> workspace = ctx.requested[0]
        .get_space_typed<gpu, 1, char>(Shape1(1), s);
    char* is_valid_ptr = reinterpret_cast<char*>(workspace.dptr_);
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

template<typename IType, typename DType, typename RType>
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
  mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
  const dim_t num_rows = output.shape()[0];
  const dim_t row_length = output.shape()[1];
  const dim_t data_size = static_cast<dim_t>(data.shape_.Size());
  // temp resource declarations
  dim_t* lookup_table = NULL;
  void* temp_storage = NULL;
  dim_t* sorted_data = NULL;
  dim_t* original_idx = NULL;
  // calculate number of bytes for temp resources
  size_t lookup_table_bytes = num_rows * sizeof(dim_t);
  size_t sorted_data_storage_bytes = data_size * sizeof(dim_t);
  size_t original_idx_storage_bytes = data_size * sizeof(dim_t);
  size_t sort_workspace_size = SortByKeyWorkspaceSize<dim_t, dim_t, gpu>(data_size);
  size_t unique_workspace_bytes = 0;
  // estimate unique temp space
  IType* data_ptr = data.dptr<IType>();
  size_t *null_ptr = nullptr;
  // unique operations will be applied on sorted data
  cub::DeviceSelect::Unique(NULL, unique_workspace_bytes, sorted_data, sorted_data,
    null_ptr, data_size, Stream<gpu>::GetStream(s));
  // One more space reserved for unique count
  size_t temp_workspace_bytes = std::max(unique_workspace_bytes,
                                         sort_workspace_size);
  size_t total_storage_bytes = lookup_table_bytes + sorted_data_storage_bytes +
                               original_idx_storage_bytes + temp_workspace_bytes;

  // request resource and split it. layout is:
  // lookup_table, sorted_data, original_idx, temp_storage
  Tensor<gpu, 1, char> workspace = ctx.requested[0]
      .get_space_typed<gpu, 1, char>(Shape1(total_storage_bytes), s);
  lookup_table = reinterpret_cast<dim_t*>(workspace.dptr_);
  sorted_data = reinterpret_cast<dim_t*>(workspace.dptr_ + lookup_table_bytes);
  original_idx = reinterpret_cast<dim_t*>(workspace.dptr_ + lookup_table_bytes +
                                          sorted_data_storage_bytes);
  temp_storage = workspace.dptr_ + total_storage_bytes - temp_workspace_bytes;

  // check out-of-bound indices
  {
    IType min = 0;
    IType max = static_cast<IType>(output.shape()[0] - 1);
    IType* data_ptr = data.dptr<IType>();
    size_t data_size = data.shape_.Size();
    bool is_valid = CheckIndexOutOfBound(s, data_ptr, data_size, min, max,
                                         reinterpret_cast<char*>(temp_storage));
    CHECK(is_valid) << "Embedding input contains data out of bound";
  }

  // make a copy of the data, to be sorted
  TBlob sorted_data_blob(sorted_data, Shape1(data_size), gpu::kDevMask);
  auto sorted_data_tensor = sorted_data_blob.FlatTo1D<gpu, dim_t>(s);
  mxnet_op::copy(s, sorted_data_blob, data);

  // generate original idx
  Tensor<gpu, 1, dim_t> original_idx_tensor(original_idx, Shape1(data_size), s);
  Kernel<range_fwd, gpu>::Launch(s, data_size, 1, static_cast<dim_t>(0),
                                 static_cast<dim_t>(1), kWriteTo, original_idx);
  // sort data with its original idx
  int num_bits = common::ilog2ui(num_rows - 1);
  char* temp_storage_ptr = reinterpret_cast<char*>(temp_storage);
  Tensor<gpu, 1, char> temp_storage_tensor(temp_storage_ptr,
                                           Shape1(sort_workspace_size), s);
  SortByKey(sorted_data_tensor, original_idx_tensor, true,
            &temp_storage_tensor, 0, num_bits);

  // compute unique row ids based on sorted values.
  output.CheckAndAllocAuxData(kIdx, Shape1(data_size + 1));

  // fill row_idx array of output matrix, using the row_flg values
  RType* grad_row_idx = output.aux_data(kIdx).dptr<RType>();
  cub::DeviceSelect::Unique(temp_storage_ptr, unique_workspace_bytes, sorted_data,
      grad_row_idx, grad_row_idx + data_size, data_size, Stream<gpu>::GetStream(s));

  dim_t nnr = 0;
  CUDA_CALL(cudaMemcpy(&nnr, grad_row_idx + data_size, sizeof(RType),
      cudaMemcpyDeviceToHost));
  CHECK_EQ(output.shape().ndim(), 2) << "Unexcepted ndim";
  output.CheckAndAllocData(Shape2(nnr, output.shape()[1]));
  output.set_aux_shape(kIdx, Shape1(nnr));

  // generate lookup table
  Kernel<MarkLookupTable, gpu>::Launch(s, nnr, lookup_table, grad_row_idx);

  // accumulate gradients
  DType* grad_data = output.data().dptr<DType>();
  Fill<false>(s, TBlob(grad_data, Shape1(nnr * row_length), gpu::kDevMask),
              kWriteTo, 0);
  const int SZ = 4;
  const nnvm::dim_t num_threads_per_row = (row_length + SZ - 1) / SZ;
  Kernel<AddTakeGradRspDeterministicKernel<SZ>, gpu>::Launch(s, data_size * num_threads_per_row,
                     grad_data, lookup_table, sorted_data, data_size, original_idx,
                     ograd.dptr<DType>(), row_length, num_threads_per_row);
}

inline void SparseEmbeddingOpBackwardDeterministicRspImpl(const OpContext& ctx,
                                                          const TBlob& ograd,
                                                          const TBlob& data,
                                                          const OpReqType req,
                                                          const NDArray& output) {
  using nnvm::dim_t;
  if (req == kNullOp) return;
  CHECK_EQ(req, kWriteTo) << "SparseEmbedding layer doesn't support "
                          << "weight gradient calculation with req != write";

  mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
  const dim_t data_size = static_cast<dim_t>(data.shape_.Size());
  if (data_size == 0) {
    FillZerosRspImpl(s, output);
    return;
  }

  MSHADOW_TYPE_SWITCH(data.type_flag_, IType, {
    MSHADOW_TYPE_SWITCH(ograd.type_flag_, DType, {
      MSHADOW_IDX_TYPE_SWITCH(output.aux_type(rowsparse::kIdx), RType, {
        SparseEmbeddingDeterministicKernelLaunch<IType, DType, RType>(ctx, ograd, data,
                                                                      req, output);
      });
    });
  });
}


template<>
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

template<>
void TakeOpForward<gpu>(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  if (req[take_::kOut] == kNullOp) return;
  const TakeParam& param = nnvm::get<TakeParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);

  const TShape& idxshape = inputs[take_::kIdx].shape_;
  const TShape& arrshape = inputs[take_::kArr].shape_;
  const TShape& oshape = outputs[take_::kOut].shape_;

  Stream<gpu> *s = ctx.get_stream<gpu>();
  const int actual_axis = param.axis + ((param.axis < 0) ? arrshape.ndim() : 0);

  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {  // output data type
    MSHADOW_TYPE_SWITCH(inputs[1].type_flag_, IType, {  // index data type
      if (actual_axis == 0) {
        if (param.mode == take_::kClip) {
          Kernel<TakeGPU<true>, gpu>::Launch(s, oshape.Size(),
                                             outputs[take_::kOut].dptr<DType>(),
                                             inputs[take_::kArr].dptr<DType>(),
                                             inputs[take_::kIdx].dptr<IType>(),
                                             oshape.Size()/idxshape.Size(), arrshape[0]);
        } else {
          Kernel<TakeGPU<false>, gpu>::Launch(s, oshape.Size(),
                                              outputs[take_::kOut].dptr<DType>(),
                                              inputs[take_::kArr].dptr<DType>(),
                                              inputs[take_::kIdx].dptr<IType>(),
                                              oshape.Size()/idxshape.Size(), arrshape[0]);
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
          Kernel<Take<true>, gpu>::Launch(s, oshape.Size(),
                                          outputs[take_::kOut].dptr<DType>(),
                                          inputs[take_::kArr].dptr<DType>(),
                                          inputs[take_::kIdx].dptr<IType>(),
                                          in_strides, out_strides, arrshape.ndim(), oshape.ndim(),
                                          idxshape.ndim(), arrshape[actual_axis], actual_axis);
        } else if (param.mode == take_::kWrap) {
          Kernel<Take<false>, gpu>::Launch(s, oshape.Size(),
                                           outputs[take_::kOut].dptr<DType>(),
                                           inputs[take_::kArr].dptr<DType>(),
                                           inputs[take_::kIdx].dptr<IType>(),
                                           in_strides, out_strides, arrshape.ndim(), oshape.ndim(),
                                           idxshape.ndim(), arrshape[actual_axis], actual_axis);
        }
      }
    });
  });
}

NNVM_REGISTER_OP(Embedding)
.set_attr<FCompute>("FCompute<gpu>", EmbeddingOpForward<gpu>)
.set_attr<FComputeEx>("FComputeEx<gpu>", SparseEmbeddingOpForwardEx<gpu>);

NNVM_REGISTER_OP(_contrib_SparseEmbedding)
.set_attr<FComputeEx>("FComputeEx<gpu>", SparseEmbeddingOpForwardEx<gpu>);

NNVM_REGISTER_OP(_backward_Embedding)
.set_attr<FCompute>("FCompute<gpu>", EmbeddingOpBackward<gpu>)
.set_attr<FComputeEx>("FComputeEx<gpu>", EmbeddingOpBackwardEx<gpu>);

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
