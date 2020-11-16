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
 * Copyright (c) 2018 by Contributors
 * \file np_indexing_op.cu
*/

#include "./np_indexing_op.h"
#include <cub/cub.cuh>

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

template<typename DType>
bool CheckIndexOutOfBound(mshadow::Stream<gpu> *s, const DType* data_ptr, size_t data_size,
                            const DType min, const DType max, char* is_valid_ptr) {
    using namespace mxnet_op;
    int32_t is_valid = 0;
    Kernel<set_zero, gpu>::Launch(s, 1, is_valid_ptr);
    Kernel<is_valid_check, gpu>::Launch(s, data_size, is_valid_ptr, data_ptr, min, max);
    CUDA_CALL(cudaMemcpyAsync(&is_valid, is_valid_ptr, sizeof(char),
                            cudaMemcpyDeviceToHost, mshadow::Stream<gpu>::GetStream(s)));
    CUDA_CALL(cudaStreamSynchronize(mshadow::Stream<gpu>::GetStream(s)));
    return is_valid == 0;
}

struct AdvancedIndexingTakeGPU {
    // assume that idx have been flattened to a 1-D tensor (N,)
    // assume that out_data and in_data have been flattened to 2-D tensors, (N, M) and (K, M)
    // M is the number of columns of in_data and out_data
    // K is the number of rows of in_data
    // i is the index of out_data
    template<typename DType, typename IType>
    MSHADOW_XINLINE static void Map(int i, DType* out_data, const DType* in_data,
                                    const IType* idx, const int64_t M, const int64_t K) {
      int64_t j = static_cast<int64_t>(idx[i]);
      j = j % K;
      j += (j < 0) ? K : 0;

      for (int64_t k = 0; k < M; k++){
        out_data[i * M + k] = in_data[j * M + k];
      }
    }
};

struct AdvancedIndexingTakeMultiDimensionGPU {
    // assume that idx have been flattened to a 1-D tensor (N,)
    // assume that out_data and in_data have been flattened to 2-D tensors, (N, M) and (K, M)
    // M is the number of columns of in_data and out_data
    // K is the number of rows of in_data
    // i is the index of out_data
    template<typename DType, typename IType>
    MSHADOW_XINLINE static void Map(int i, DType* out_data, const DType* in_data,
                                    const IType* idx, const int64_t M, const int64_t K) {
      int64_t j = static_cast<int64_t>(idx[i]);
      j = j % K;
      j += (j < 0) ? K : 0;

      for (int64_t k = 0; k < M; k++){
        out_data[i * M + k] = in_data[(i * k + j) * M + k];
      }
    }
};

template<>
inline void AdvancedIndexingOpForward<gpu>(const nnvm::NodeAttrs& attrs,
                                    const OpContext &ctx,
                                    const std::vector<NDArray> &inputs,
                                    const std::vector<OpReqType> &req,
                                    const std::vector<NDArray> &outputs) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);

  if (inputs[np_indexing_::kIdx].dtype() == mshadow::kBool) {
    CHECK(req[0] == kWriteTo || req[0] == kWriteInplace);
    const int axis = 0;
    const NDArray &data = inputs[0];
    const NDArray &idx = inputs[1];
    const NDArray &out = outputs[0];
    CHECK_EQ(axis, 0) << "Not supported yet";
    CHECK_EQ(data.shape()[axis], idx.shape()[0]);
    CHECK_EQ(idx.shape().ndim(), 1U);
    Stream<gpu>* s = ctx.get_stream<gpu>();
    cudaStream_t stream = Stream<gpu>::GetStream(s);
    // count the number of 1s in `idx`, so that we could know the output dimension
    size_t idx_size = idx.shape()[0];
    int32_t valid_num = 0;
    int32_t* prefix_sum = nullptr;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    // Calculate total temporary memory size
    cub::DeviceScan::InclusiveSum(d_temp_storage,
                                    temp_storage_bytes,
                                    prefix_sum,
                                    prefix_sum,
                                    idx_size,
                                    stream);
    size_t buffer_size = idx_size * sizeof(int32_t);
    temp_storage_bytes += buffer_size;
    // Allocate memory on GPU and allocate pointer
    Tensor<gpu, 1, char> workspace =
        ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(temp_storage_bytes), s);
    prefix_sum = reinterpret_cast<int32_t*>(workspace.dptr_);
    d_temp_storage = workspace.dptr_ + buffer_size;
    mxnet_op::Kernel<mshadow_op::identity_with_cast, gpu>::Launch(
    s, idx.shape()[0], prefix_sum, idx.data().dptr<bool>());
    // Calculate prefix sum
    cub::DeviceScan::InclusiveSum(d_temp_storage,
                                    temp_storage_bytes,
                                    prefix_sum,
                                    prefix_sum,
                                    idx_size,
                                    stream);
    CUDA_CALL(cudaMemcpyAsync(&valid_num, &prefix_sum[idx_size - 1], sizeof(int32_t),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CALL(cudaStreamSynchronize(stream));

    // Set the output shape forcefully
    mxnet::TShape data_shape = data.shape();
    data_shape[axis] = valid_num;
    const_cast<NDArray &>(out).Init(data_shape);
    size_t input_size = data.shape().Size();
    size_t col_size = input_size / idx.shape()[0];
    // Do the copy
    MSHADOW_TYPE_SWITCH_WITH_BOOL(out.dtype(), DType, {
        if (valid_num > 0) {
        mxnet_op::Kernel<BooleanMaskForwardKernel, gpu>::Launch(
            s, input_size, out.data().dptr<DType>(),
            data.data().dptr<DType>(), prefix_sum, col_size);
        }
    });
} else if (inputs[np_indexing_::kIdx].dtype() == mshadow::kInt8 ||
           inputs[np_indexing_::kIdx].dtype() == mshadow::kInt16 ||
           inputs[np_indexing_::kIdx].dtype() == mshadow::kInt32 ||
           inputs[np_indexing_::kIdx].dtype() == mshadow::kInt64) {
    using namespace mxnet_op;
    const mxnet::TShape& idxshape = inputs[np_indexing_::kIdx].shape();
    const mxnet::TShape& arrshape = inputs[np_indexing_::kArr].shape();

    if (idxshape.Size() == 0) {
        return;
    }

    mxnet::TShape oshape(idxshape.ndim() + arrshape.ndim() - 1, -1);
    for (index_t i = 0; i < idxshape.ndim(); ++i) {
        oshape[i] = idxshape[i];
    }
    for (index_t i = 0; i < arrshape.ndim(); i++) {
        if (i < 0) {
        oshape[i] = arrshape[i];
        } else if (i > 0) {
        oshape[i + idxshape.ndim() - 1] = arrshape[i];
        }
    }

    const NDArray &out = outputs[0];
    const_cast<NDArray &>(out).Init(oshape);

    Stream<gpu> *s = ctx.get_stream<gpu>();

    MSHADOW_TYPE_SWITCH(outputs[np_indexing_::kOut].dtype(), DType, {  // output data type
        MSHADOW_TYPE_SWITCH(inputs[np_indexing_::kIdx].dtype(), IType, {
        IType min = 0;
        IType max = static_cast<IType>(arrshape[0] - 1);
        // check with single thread is faster since data is small
        IType* idx_ptr = inputs[np_indexing_::kIdx].data().dptr<IType>();
        size_t idx_size = idxshape.Size();
        Tensor<gpu, 1, char> workspace =
          ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(1), s);
        char* is_valid_ptr = reinterpret_cast<char*>(workspace.dptr_);
        bool is_valid = CheckIndexOutOfBound(s, idx_ptr, idx_size, min, max, is_valid_ptr);
        CHECK(is_valid) << "take operator contains indices out of bound";
        Kernel<AdvancedIndexingTakeGPU, gpu>::Launch(s, idxshape.Size(),
                        outputs[np_indexing_::kOut].data().dptr<DType>(),
                        inputs[np_indexing_::kArr].data().dptr<DType>(),
                        inputs[np_indexing_::kIdx].data().dptr<IType>(),
                        oshape.Size()/idxshape.Size(), arrshape[0]);
        });
    });
  } else {
    LOG(FATAL)
    << "arrays used as indices must be explictly declared as integer (or boolean) type. "
    << "Use np.astype() to cast indices to integer or boolean.";
  }
}

template<>
inline void AdvancedIndexingOpBackward<gpu>(const nnvm::NodeAttrs& attrs,
                                     const OpContext &ctx,
                                     const std::vector<NDArray> &inputs,
                                     const std::vector<OpReqType> &req,
                                     const std::vector<NDArray> &outputs) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 2U);
  if (req[0] == kNullOp) return;

  if (inputs[np_indexing_::kIdx+1].dtype() == mshadow::kBool) {
    // inputs: {ograd, data, idx}
    // outputs: {igrad_data, igrad_idx}
    const NDArray& ograd = inputs[0];
    const NDArray& idx = inputs[2];
    const NDArray& igrad_data = outputs[0];
    Stream<gpu>* s = ctx.get_stream<gpu>();
    cudaStream_t stream = Stream<gpu>::GetStream(s);
    // Count the number of 1s in `idx`, so that we could know the output dimension
    size_t idx_size = idx.shape()[0];
    int32_t* prefix_sum = nullptr;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    // Calculate total temporary memory size
    cub::DeviceScan::InclusiveSum(d_temp_storage,
                                    temp_storage_bytes,
                                    prefix_sum,
                                    prefix_sum,
                                    idx_size,
                                    stream);
    size_t buffer_size = idx_size * sizeof(int32_t);
    temp_storage_bytes += buffer_size;
    // Allocate memory on GPU and allocate pointer
    Tensor<gpu, 1, char> workspace =
        ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(temp_storage_bytes), s);
    prefix_sum = reinterpret_cast<int32_t*>(workspace.dptr_);
    d_temp_storage = workspace.dptr_ + buffer_size;
    MSHADOW_TYPE_SWITCH_WITH_BOOL(idx.dtype(), IType, {
        mxnet_op::Kernel<mshadow_op::identity_with_cast, gpu>::Launch(
        s, idx.shape()[0], prefix_sum, idx.data().dptr<IType>());
    });
    // Calculate prefix sum
    cub::DeviceScan::InclusiveSum(d_temp_storage,
                                    temp_storage_bytes,
                                    prefix_sum,
                                    prefix_sum,
                                    idx_size,
                                    stream);
    size_t input_size = igrad_data.shape().Size();
    size_t col_size = input_size / idx_size;
    // Backward pass
    MSHADOW_TYPE_SWITCH(igrad_data.dtype(), DType, {
        if (input_size > 0) {
        mxnet_op::Kernel<BooleanMaskBackwardKernel, gpu>::Launch(
            s, input_size, igrad_data.data().dptr<DType>(), req[0], ograd.data().dptr<DType>(),
            prefix_sum, col_size);
        }
    });
  } else if (inputs[np_indexing_::kIdx+1].dtype() == mshadow::kInt8 ||
             inputs[np_indexing_::kIdx+1].dtype() == mshadow::kInt16 ||
             inputs[np_indexing_::kIdx+1].dtype() == mshadow::kInt32 ||
             inputs[np_indexing_::kIdx+1].dtype() == mshadow::kInt64) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_NE(req[np_indexing_::kIdx], kAddTo)
        << "take layer doesn't support gradient of req type kAddTo to index";

    // grad_out is the gradient of the outputs in the feed-forward
    // grad_in is the gradient of the inputs in the feed-forward
    Stream<gpu> *s = ctx.get_stream<gpu>();

    MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {  // output data type
        MSHADOW_TYPE_SWITCH(inputs[2].dtype(), IType, {  // index data type
          // inputs are specified in the .cc file, which are the gradients from
          // the upper layer and the input index
          // outputs are the gradients of inputs in the feed-forward pass
          const mxnet::TShape& idxshape = inputs[2].shape();
          const mxnet::TShape& arrshape = outputs[0].shape();
          const mxnet::TShape& oshape = inputs[0].shape();

          if (idxshape.Size() == 0) {
              return;
          }

          if (req[np_indexing_::kIdx] != kNullOp) {
              mxnet_op::Kernel<mxnet_op::set_zero, gpu>::Launch(
              s, idxshape.Size(), outputs[np_indexing_::kIdx].data().dptr<IType>());
          }

          int idxndim = idxshape.ndim();
          Tensor<gpu, 1, IType> idx = inputs[2].data().get_with_shape<gpu, 1, IType>(
              Shape1(idxshape.ProdShape(0, idxndim)), s);
          Tensor<gpu, 2, DType> grad_out = inputs[0].data().get_with_shape<gpu, 2, DType>(
              Shape2(oshape.ProdShape(0, idxndim), oshape.ProdShape(idxndim, oshape.ndim())), s);
          Tensor<gpu, 2, DType> grad_in = outputs[0].data().get_with_shape<gpu, 2, DType>(
              Shape2(arrshape[0], arrshape.ProdShape(1, arrshape.ndim())), s);

          // re-using the previous code for axis = 0 case
          if (req[np_indexing_::kArr] == kWriteTo || req[np_indexing_::kArr] == kAddTo) {
              if (req[np_indexing_::kArr] == kWriteTo) {
                  grad_in = scalar<DType>(0.0f);
              }
              AddTakeGrad<false>(grad_in, idx, grad_out);
          } else {
              LOG(FATAL) << "wrong req";
          }
        });
    });
  } else {
    LOG(FATAL)
    << "arrays used as indices must be explictly declared as integer (or boolean) type. "
    << "Use np.astype() to cast indices to integer or boolean.";
  }
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
template<typename DType>
void GatherNDCheckBoundGPU(mshadow::Stream<gpu> *s, const DType* idx_ptr, index_t N,
                          index_t M, const mshadow::Shape<10> mshape, DType* is_valid_dim_ptr) {
  using namespace mxnet_op;
  Kernel<set_zero, gpu>::Launch(s, M, is_valid_dim_ptr);
  Kernel<is_valid_check_gather_nd, gpu>::Launch(s, M, is_valid_dim_ptr, idx_ptr, N, mshape);

  std::vector<DType> is_valid_dim(M);
  CUDA_CALL(cudaMemcpyAsync(is_valid_dim.data(), is_valid_dim_ptr, sizeof(DType)*M,
                            cudaMemcpyDeviceToHost, mshadow::Stream<gpu>::GetStream(s)));
  CUDA_CALL(cudaStreamSynchronize(mshadow::Stream<gpu>::GetStream(s)));
  for (int m = 0; m < M; m++) {
    if (is_valid_dim[m] > mshape[m] - 1 || is_valid_dim[m] < - mshape[m]) {
      LOG(FATAL)<< "IndexError: index " << is_valid_dim[m] << " is out of bounds for axis "
        << m << " with size " << mshape[m];
    }
  }
}

void AdvancedIndexingMultipleForwardGPU(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  if (req[0] == kNullOp) return;

  if (inputs[np_indexing_::kIdx].type_flag_ == mshadow::kBool) {
    LOG(FATAL)
    << "Multi-dimension boolean indexing is not supported.";
  } else if (inputs[np_indexing_::kIdx].type_flag_ == mshadow::kInt8 ||
          inputs[np_indexing_::kIdx].type_flag_ == mshadow::kInt16 ||
          inputs[np_indexing_::kIdx].type_flag_ == mshadow::kInt32 ||
          inputs[np_indexing_::kIdx].type_flag_ == mshadow::kInt64) {
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    const mxnet::TShape& dshape = inputs[0].shape_;
    const mxnet::TShape& ishape = inputs[1].shape_;
    int M = ishape[0];
    int N = ishape.Size() / M;
    int K = dshape.ProdShape(M, dshape.ndim());
    mshadow::Shape<10> strides;
    mshadow::Shape<10> mshape;
    for (int i = M-1, stride = K; i >= 0; stride *= dshape[i], --i) {
      strides[i] = stride;
      mshape[i] = dshape[i];
    }
    MSHADOW_TYPE_SWITCH_WITH_BOOL(inputs[0].type_flag_, DType, {  // output data type switch
      MSHADOW_TYPE_SWITCH(inputs[1].type_flag_, IType, {  // indices data type switch
        // check whether indices are out of bound
        IType* idx_ptr = inputs[1].dptr<IType>();
        Tensor<gpu, 1, IType> workspace =
          ctx.requested[0].get_space_typed<gpu, 1, IType>(Shape1(M), s);
        IType* is_valid_dim_ptr = reinterpret_cast<IType*>(workspace.dptr_);
        GatherNDCheckBoundGPU(s, idx_ptr, N, M, mshape, is_valid_dim_ptr);
        Kernel<gather_nd, gpu>::Launch(
          s, N, req[0], N, M, K, strides, mshape, outputs[0].dptr<DType>(),
          inputs[0].dptr<DType>(), inputs[1].dptr<IType>());
      });
    });
  } else {
    LOG(FATAL)
    << "arrays used as indices must be explictly declared as integer (or boolean) type."
    << "Use np.astype() to cast indices to integer or boolean.";
  }
}

struct backward_gather_nd_gpu {
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(index_t i, index_t N, index_t M, index_t K,
                                  const mshadow::Shape<10> strides,
                                  DType* out, const DType* data,
                                  const IType* indices) {
    index_t offset = 0;
    for (index_t j = 0; j < M; ++j) {
      offset += strides[j] * static_cast<int>(indices[j*N + i]);
    }
    for (index_t j = 0; j < K; ++j) {
      atomicAdd(out + (offset + j), data[i * K + j]);
    }
  }
};

template<typename DType, typename IType>
inline void GatherNDBackwardImpl(index_t N, index_t M, index_t K,
                                const mshadow::Shape<10> strides,
                                DType* out,
                                const DType* data,
                                const IType* indices,
                                mshadow::Stream<gpu> *s) {
  mxnet_op::Kernel<backward_gather_nd_gpu, gpu>::Launch(s, N, N, M, K, strides, out, data, indices);
}

NNVM_REGISTER_OP(_npi_advanced_indexing)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.set_attr<FComputeEx>("FComputeEx<gpu>", AdvancedIndexingOpForward<gpu>);

NNVM_REGISTER_OP(_backward_np_advanced_indexing)
.set_attr<FComputeEx>("FComputeEx<gpu>", AdvancedIndexingOpBackward<gpu>);

NNVM_REGISTER_OP(_npi_advanced_indexing_multiple)
.set_attr<FCompute>("FCompute<gpu>", AdvancedIndexingMultipleForwardGPU);

NNVM_REGISTER_OP(_backward_np_advanced_indexing_multiple)
.set_attr<FCompute>("FCompute<gpu>", AdvancedIndexingMultipleBackward<gpu>);

}  // namespace op
}  // namespace mxnet
