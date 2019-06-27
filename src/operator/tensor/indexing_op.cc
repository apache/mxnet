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
 * \file indexing_op.cc
 * \brief CPU implementation of indexing operator
 * \author Siyi Li, Chi Zhang
*/

#include "./indexing_op.h"
namespace mxnet {
namespace op {

template<bool clip = true>
struct TakeCPU {
  // assume that idx have been flattened to a 1-D tensor (N,)
  // assume that out_data and in_data have been flattened to 2-D tensors, (N, M) and (K, M)
  // M is the number of columns of in_data and out_data
  // K is the number of rows of in_data
  // i is the index of out_data
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(index_t i, DType* out_data, const DType* in_data,
                                  const IType* idx, const size_t M, const int64_t K) {
    int64_t j = static_cast<int64_t>(idx[i]);
    if (clip) {
      if (j <= 0) j = 0;
      else if (j >= K) j = K - 1;
    } else {
      j = j % K;
      j += (j < 0) ? K : 0;
    }
    std::memcpy(out_data + i * M, in_data + j * M, M * sizeof(DType));
  }
};

/*
 * \brief returns true if all indices are between [min, max]
 * \param data_ptr the indices to check
 * \param data_size the number of indices to examine
 * \param min the expected min value for indices
 * \param max the expected max value for indices
 */
template<typename DType>
bool CheckIndexOutOfBound(const DType* data_ptr, size_t data_size,
                          const DType min, const DType max) {
  bool is_valid = true;
  for (size_t i = 0; i < data_size; i++) {
    if (data_ptr[i] > max || data_ptr[i] < min) {
      is_valid = false;
      break;
    }
  }
  return is_valid;
}

// Embedding forward implementation with dense weight
template<>
void EmbeddingOpForwardDnsImpl<cpu>(mshadow::Stream<cpu>* s,
                                    const TBlob& data,
                                    const TBlob& weight,
                                    const OpReqType req,
                                    const TBlob& output) {
  using namespace mxnet_op;
  const mxnet::TShape& ishape = data.shape_;
  const mxnet::TShape& oshape = output.shape_;

  MSHADOW_TYPE_SWITCH(output.type_flag_, DType, {
    MSHADOW_TYPE_SWITCH(data.type_flag_, IType, {
      Tensor<cpu, 1, IType> idx = data.get_with_shape<cpu, 1, IType>(
        Shape1(ishape.ProdShape(0, ishape.ndim())), s);
      Tensor<cpu, 2, DType> wmat = weight.get<cpu, 2, DType>(s);
      Tensor<cpu, 2, DType> out = output.get_with_shape<cpu, 2, DType>(
        Shape2(oshape.ProdShape(0, oshape.ndim()-1), oshape[oshape.ndim()-1]), s);
      Kernel<TakeCPU<true>, cpu>::Launch(s, oshape.Size() / wmat.shape_[1], out.dptr_, wmat.dptr_,
                                         idx.dptr_, wmat.shape_[1], wmat.shape_[0]);
    });
  });
}

template<>
void SparseEmbeddingOpForwardRspImpl<cpu>(const OpContext& ctx,
                                          const TBlob& data,
                                          const NDArray& weight,
                                          const OpReqType req,
                                          const TBlob& output) {
  if (req == kNullOp) return;
  using namespace rowsparse;
  using namespace mxnet_op;
  mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
  // zeros weight
  if (req == kWriteTo && !weight.storage_initialized()) {
    size_t out_size = output.shape_.Size();
    MSHADOW_TYPE_SWITCH(output.type_flag_, DType, {
      Fill<false>(s, TBlob(output.dptr<DType>(), mshadow::Shape1(out_size),
          cpu::kDevMask), kWriteTo, 0);
    })
    return;
  }
  // check out-of-bound indices
  MSHADOW_TYPE_SWITCH(data.type_flag_, DType, {
    DType min = 0;
    DType max = static_cast<DType>(weight.shape()[0] - 1);
    // check with single thread is faster since data is small
    DType* data_ptr = data.dptr<DType>();
    size_t data_size = data.shape_.Size();
    bool is_valid = CheckIndexOutOfBound(data_ptr, data_size,
                                         min, max);
    CHECK(is_valid) << "SparseEmbedding input contains data out of bound";
  })
  // the weight is actually dense
  if (weight.aux_shape(kIdx)[0] == weight.shape()[0]) {
    EmbeddingOpForwardDnsImpl<cpu>(s, data, weight.data(), req, output);
  } else {
    EmbeddingOpForwardRspImpl<cpu>(s, data, weight, req, output);
  }
}

template<bool clip>
struct CsrTakeDataKernel {
  /*!
   * \brief Map function for general case of take grad
   * \param tid           global thread id
   * \param out_idx       ptr to out idx
   * \param out_data      ptr to out data
   * \param out_indptr    ptr to out indptr
   * \param src_data      ptr to original csr data
   * \param src_idx       ptr to original csr idx
   * \param idx_ptr       ptr to indices
   * \param num_rows      maximum number of rows in src array
   */
  template<typename IType, typename DType, typename RType>
  MSHADOW_XINLINE static void Map(int tid, RType* out_idx, DType* out_data,
                                  const RType* out_indptr, const RType* src_idx,
                                  const DType* src_data, const RType* src_indptr,
                                  const IType* idx_ptr, const nnvm::dim_t num_rows) {
    nnvm::dim_t idx = static_cast<nnvm::dim_t>(idx_ptr[tid]);
    // clip mode
    if (clip) {
      if (idx < 0) idx = 0;
      if (idx >= num_rows) idx = num_rows - 1;
    } else {
      // wrap mode
      idx = idx % num_rows;
      idx += (idx < 0) ? num_rows : 0;
    }
    int row_nnz = src_indptr[idx + 1] - src_indptr[idx];
    for (int i = 0; i < row_nnz; i++) {
        out_data[out_indptr[tid] + i] = src_data[src_indptr[idx] + i];
        out_idx[out_indptr[tid] + i] = src_idx[src_indptr[idx] + i];
    }
  }
};

template<bool clip>
struct CsrTakeRowCountKernel {
  /*!
   * \brief Map function for general case of take grad
   * \param tid           global thread id
   * \param out_indptr    ptr to out indptr
   * \param src_indptr    ptr to original csr indptr
   * \param idx_ptr       ptr to indices
   * \param num_rows      maximum number of rows in src array
   */
  template<typename IType, typename RType>
  MSHADOW_XINLINE static void Map(int tid, RType* out_indptr,
                                  const RType* src_indptr, const IType* idx_ptr,
                                  const nnvm::dim_t num_rows) {
    if (tid == 0) {
      out_indptr[0] = 0;
      return;
    }
    nnvm::dim_t idx = static_cast<nnvm::dim_t>(idx_ptr[tid - 1]);
    // clip mode
    if (clip) {
      if (idx < 0) idx = 0;
      if (idx >= num_rows) idx = num_rows - 1;
    } else {
      // wrap mode
      idx = idx % num_rows;
      idx += (idx < 0) ? num_rows : 0;
    }
    out_indptr[tid] = src_indptr[idx + 1] - src_indptr[idx];
  }
};

template<>
void TakeOpForwardCsrImpl<cpu>(const TakeParam& params,
                               const OpContext& ctx,
                               const TBlob& idx,
                               const NDArray& arr,
                               OpReqType req,
                               const NDArray& out) {
  using namespace csr;
  using namespace mxnet_op;
  using nnvm::dim_t;
  Stream<cpu> *s = ctx.get_stream<cpu>();
  if (req == kNullOp) return;
  if (!arr.storage_initialized()) {
    FillZerosCsrImpl(s, out);
    return;
  }
  CHECK_EQ(idx.shape_.ndim(), 1U)
          << "Take with CSR array only supports one-dimensional indices. "
          << idx.shape_.ndim() << " dimensional input is given instead";
  CHECK_EQ(req, kWriteTo) << "req = " << req << " is not supported for take(csr)";
  auto axis = params.axis;
  CHECK_EQ(axis, 0) << "axis = " << axis << " is not supported for take(csr)";
  CHECK(params.mode == take_::kClip || params.mode == take_::kWrap)
    << "mode = " << params.mode << " is not supported";
  const dim_t num_rows = out.shape()[0];
  const dim_t max_num_rows = arr.shape()[0];
  out.CheckAndAllocAuxData(kIndPtr, {Shape1(num_rows + 1)});

  MSHADOW_TYPE_SWITCH(idx.type_flag_, IType, {
    MSHADOW_TYPE_SWITCH(arr.dtype(), DType, {
      MSHADOW_IDX_TYPE_SWITCH(out.aux_type(kIdx), RType, {
        RType* out_indptr = out.aux_data(kIndPtr).dptr<RType>();
        const RType* src_indptr = arr.aux_data(kIndPtr).dptr<RType>();
        const IType* idx_ptr = idx.dptr<IType>();
        // gather per row nnz information for output
        bool clip = params.mode == take_::kClip;
        if (clip) {
          Kernel<CsrTakeRowCountKernel<true>, cpu>::Launch(s, num_rows + 1,
              out_indptr, src_indptr, idx_ptr, max_num_rows);
        } else {
          Kernel<CsrTakeRowCountKernel<false>, cpu>::Launch(s, num_rows + 1,
              out_indptr, src_indptr, idx_ptr, max_num_rows);
        }
        // calculate prefix sum with single thread
        for (dim_t i = 0; i < num_rows; i++) {
           out_indptr[i + 1] += out_indptr[i];
        }
        // total number of non-zero rows
        const dim_t nnz = out_indptr[num_rows];
        if (nnz == 0) {
          FillZerosCsrImpl(s, out);
          return;
        }
        out.CheckAndAllocAuxData(kIdx, {Shape1(nnz)});
        out.CheckAndAllocData(Shape1(nnz));
        RType* out_idx = out.aux_data(kIdx).dptr<RType>();
        DType* out_data = out.data().dptr<DType>();
        const RType* src_idx = arr.aux_data(kIdx).dptr<RType>();
        const DType* src_data = arr.data().dptr<DType>();
        // copy indices and data for output
        if (clip) {
          Kernel<CsrTakeDataKernel<true>, cpu>::Launch(s, num_rows, out_idx,
              out_data, out_indptr, src_idx, src_data, src_indptr, idx_ptr, max_num_rows);
        } else {
          Kernel<CsrTakeDataKernel<false>, cpu>::Launch(s, num_rows, out_idx,
              out_data, out_indptr, src_idx, src_data, src_indptr, idx_ptr, max_num_rows);
        }
      });
    });
  });
}

template<>
void TakeOpForward<cpu>(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  if (req[take_::kOut] == kNullOp) return;
  const TakeParam& param = nnvm::get<TakeParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);

  const mxnet::TShape& idxshape = inputs[take_::kIdx].shape_;
  const mxnet::TShape& arrshape = inputs[take_::kArr].shape_;
  const mxnet::TShape& oshape = outputs[take_::kOut].shape_;

  Stream<cpu> *s = ctx.get_stream<cpu>();
  const int actual_axis = param.axis + ((param.axis < 0) ? arrshape.ndim() : 0);

  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {  // output data type
    MSHADOW_TYPE_SWITCH(inputs[1].type_flag_, IType, {  // index data type
      if (actual_axis == 0) {
        if (param.mode == take_::kClip) {
          Kernel<TakeCPU<true>, cpu>::Launch(s, idxshape.Size(),
                                             outputs[take_::kOut].dptr<DType>(),
                                             inputs[take_::kArr].dptr<DType>(),
                                             inputs[take_::kIdx].dptr<IType>(),
                                             oshape.Size()/idxshape.Size(), arrshape[0]);
        } else {
          Kernel<TakeCPU<false>, cpu>::Launch(s, idxshape.Size(),
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
          Kernel<Take<true>, cpu>::Launch(s, oshape.Size(),
                                          outputs[take_::kOut].dptr<DType>(),
                                          inputs[take_::kArr].dptr<DType>(),
                                          inputs[take_::kIdx].dptr<IType>(),
                                          in_strides, out_strides, arrshape.ndim(),
                                          oshape.ndim(), idxshape.ndim(),
                                          arrshape[actual_axis], actual_axis);
        } else if (param.mode == take_::kWrap) {
          Kernel<Take<false>, cpu>::Launch(s, oshape.Size(),
                                           outputs[take_::kOut].dptr<DType>(),
                                           inputs[take_::kArr].dptr<DType>(),
                                           inputs[take_::kIdx].dptr<IType>(),
                                           in_strides, out_strides, arrshape.ndim(),
                                           oshape.ndim(), idxshape.ndim(),
                                           arrshape[actual_axis], actual_axis);
        }
      }
    });
  });
}

template<>
inline void SparseEmbeddingOpBackwardRspImpl<cpu>(const bool deterministic,
                                                  const OpContext& ctx,
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
  Stream<cpu> *s = ctx.get_stream<cpu>();
  dim_t num_rows = output.shape()[0];
  dim_t row_length = output.shape()[1];
  size_t workspace_size = num_rows * sizeof(dim_t);
  Tensor<cpu, 1, char> workspace =
    ctx.requested[embedding::kTempSpace].get_space_typed<cpu, 1, char>(
      Shape1(workspace_size), s);
  dim_t* row_flg = reinterpret_cast<dim_t*>(workspace.dptr_);
  // prefix sum array re-uses the row_flg array temp space
  dim_t* prefix_sum = row_flg;
  dim_t data_size = static_cast<dim_t>(data.shape_.Size());

  MSHADOW_TYPE_SWITCH(data.type_flag_, IType, {
    MSHADOW_SGL_DBL_TYPE_SWITCH(ograd.type_flag_, DType, {
      MSHADOW_IDX_TYPE_SWITCH(output.aux_type(kIdx), RType, {
        // check out of bound indices
        {
          IType min = 0;
          IType max = static_cast<IType>(output.shape()[0] - 1);
          // check with single thread is faster since data is small
          IType* data_ptr = data.dptr<IType>();
          bool is_valid = CheckIndexOutOfBound(data_ptr, data.shape_.Size(), min, max);
          CHECK(is_valid) << "Embedding input contains data out of bound";
        }
        // mark row flags
        Fill<false>(s, TBlob(row_flg, Shape1(num_rows), cpu::kDevMask), kWriteTo, 0);
        Kernel<MarkRowFlgKernel, cpu>::Launch(s, data_size, row_flg, data.dptr<IType>());
        // calculate inclusive prefix sum
        // TODO(haibin) ideally this is should be done in parallel
        prefix_sum[0] = row_flg[0];
        for (dim_t i = 1; i < num_rows; i++) {
          prefix_sum[i] = prefix_sum[i - 1] + row_flg[i];
        }
        // total number of non-zero rows
        dim_t nnr = prefix_sum[num_rows - 1];
        if (nnr == 0) {
          FillZerosRspImpl(s, output);
          return;
        }
        output.CheckAndAlloc({Shape1(nnr)});
        RType* grad_row_idx = output.aux_data(kIdx).dptr<RType>();
        // fill row_idx array of output matrix, using the row_flg values
        Kernel<FillRspRowIdxKernel, cpu>::Launch(s, num_rows,
            grad_row_idx, prefix_sum, num_rows);
        // prefill with zeros
        DType* grad_data = output.data().dptr<DType>();
        Fill<false>(s, TBlob(grad_data, Shape1(nnr * row_length),
            cpu::kDevMask), kWriteTo, 0);
        // add the final gradients
        const int num_threads = engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
        dim_t segment_len = (nnr + num_threads - 1) / num_threads;
        Kernel<AddTakeGradRspKernel, cpu>::Launch(s, num_threads, grad_data, prefix_sum,
                                                  ograd.dptr<DType>(), row_length,
                                                  data.dptr<IType>(), data_size, segment_len,
                                                  num_rows);
      });
    });
  });
}


template<typename DType, typename IType>
inline typename std::enable_if<(!std::is_same<DType, mshadow::half::half_t>::value), void>::type
GatherNDBackwardImpl(index_t N, index_t M, index_t K,
                     const mshadow::Shape<10> strides,
                     DType* out,
                     const DType* data,
                     const IType* indices,
                     mshadow::Stream<cpu> *s) {
#pragma omp parallel for
  for (index_t i = 0; i < N; i++) {
    index_t offset = 0;
    for (index_t j = 0; j < M; ++j) {
      offset += strides[j] * static_cast<index_t>(indices[j*N + i]);
    }
    for (index_t j = 0; j < K; ++j) {
#pragma omp atomic
      out[offset + j] += data[i * K + j];
    }
  }
}

template<typename DType, typename IType>
inline typename std::enable_if<std::is_same<DType, mshadow::half::half_t>::value, void>::type
GatherNDBackwardImpl(index_t N, index_t M, index_t K,
                     const mshadow::Shape<10> strides,
                     DType* out,
                     const DType* data,
                     const IType* indices,
                     mshadow::Stream<cpu> *s) {
  for (index_t i = 0; i < N; i++) {
    index_t offset = 0;
    for (index_t j = 0; j < M; ++j) {
      offset += strides[j] * static_cast<index_t>(indices[j*N + i]);
    }
    for (index_t j = 0; j < K; ++j) {
      out[offset + j] += data[i * K + j];
    }
  }
}

DMLC_REGISTER_PARAMETER(EmbeddingParam);
DMLC_REGISTER_PARAMETER(SparseEmbeddingParam);
DMLC_REGISTER_PARAMETER(TakeParam);
DMLC_REGISTER_PARAMETER(OneHotParam);
DMLC_REGISTER_PARAMETER(ScatterNDParam);

NNVM_REGISTER_OP(Embedding)
MXNET_ADD_SPARSE_OP_ALIAS(Embedding)
.add_alias("_npx_embedding")
.describe(R"code(Maps integer indices to vector representations (embeddings).

This operator maps words to real-valued vectors in a high-dimensional space,
called word embeddings. These embeddings can capture semantic and syntactic properties of the words.
For example, it has been noted that in the learned embedding spaces, similar words tend
to be close to each other and dissimilar words far apart.

For an input array of shape (d1, ..., dK),
the shape of an output array is (d1, ..., dK, output_dim).
All the input values should be integers in the range [0, input_dim).

If the input_dim is ip0 and output_dim is op0, then shape of the embedding weight matrix must be
(ip0, op0).

By default, if any index mentioned is too large, it is replaced by the index that addresses
the last vector in an embedding matrix.

Examples::

  input_dim = 4
  output_dim = 5

  // Each row in weight matrix y represents a word. So, y = (w0,w1,w2,w3)
  y = [[  0.,   1.,   2.,   3.,   4.],
       [  5.,   6.,   7.,   8.,   9.],
       [ 10.,  11.,  12.,  13.,  14.],
       [ 15.,  16.,  17.,  18.,  19.]]

  // Input array x represents n-grams(2-gram). So, x = [(w1,w3), (w0,w2)]
  x = [[ 1.,  3.],
       [ 0.,  2.]]

  // Mapped input x to its vector representation y.
  Embedding(x, y, 4, 5) = [[[  5.,   6.,   7.,   8.,   9.],
                            [ 15.,  16.,  17.,  18.,  19.]],

                           [[  0.,   1.,   2.,   3.,   4.],
                            [ 10.,  11.,  12.,  13.,  14.]]]


The storage type of weight can be either row_sparse or default.

.. Note::

    If "sparse_grad" is set to True, the storage type of gradient w.r.t weights will be
    "row_sparse". Only a subset of optimizers support sparse gradients, including SGD, AdaGrad
    and Adam. Note that by default lazy updates is turned on, which may perform differently
    from standard updates. For more details, please check the Optimization API at:
    https://mxnet.incubator.apache.org/api/python/optimization/optimization.html

)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<EmbeddingParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "weight"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", EmbeddingOpShape<EmbeddingParam>)
.set_attr<nnvm::FInferType>("FInferType", EmbeddingOpType<EmbeddingParam>)
.set_attr<FInferStorageType>("FInferStorageType", EmbeddingOpForwardStorageType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", EmbeddingOpForward<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", SparseEmbeddingOpForwardEx<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    return MakeNonlossGradNode("_backward_Embedding", n, ograds,
                               {n->inputs[0]}, n->attrs.dict);
  })
.add_argument("data", "NDArray-or-Symbol", "The input array to the embedding operator.")
.add_argument("weight", "NDArray-or-Symbol", "The embedding weight matrix.")
.add_arguments(EmbeddingParam::__FIELDS__());

NNVM_REGISTER_OP(_contrib_SparseEmbedding)
.describe(R"code(Maps integer indices to vector representations (embeddings).

note:: ``contrib.SparseEmbedding`` is deprecated, use ``Embedding`` instead.

This operator maps words to real-valued vectors in a high-dimensional space,
called word embeddings. These embeddings can capture semantic and syntactic properties of the words.
For example, it has been noted that in the learned embedding spaces, similar words tend
to be close to each other and dissimilar words far apart.

For an input array of shape (d1, ..., dK),
the shape of an output array is (d1, ..., dK, output_dim).
All the input values should be integers in the range [0, input_dim).

If the input_dim is ip0 and output_dim is op0, then shape of the embedding weight matrix must be
(ip0, op0).

The storage type of the gradient will be `row_sparse`.

.. Note::

    `SparseEmbedding` is designed for the use case where `input_dim` is very large (e.g. 100k).
    The operator is available on both CPU and GPU.
    When `deterministic` is set to `True`, the accumulation of gradients follows a
    deterministic order if a feature appears multiple times in the input. However, the
    accumulation is usually slower when the order is enforced on GPU.
    When the operator is used on the GPU, the recommended value for `deterministic` is `True`.

Examples::

  input_dim = 4
  output_dim = 5

  // Each row in weight matrix y represents a word. So, y = (w0,w1,w2,w3)
  y = [[  0.,   1.,   2.,   3.,   4.],
       [  5.,   6.,   7.,   8.,   9.],
       [ 10.,  11.,  12.,  13.,  14.],
       [ 15.,  16.,  17.,  18.,  19.]]

  // Input array x represents n-grams(2-gram). So, x = [(w1,w3), (w0,w2)]
  x = [[ 1.,  3.],
       [ 0.,  2.]]

  // Mapped input x to its vector representation y.
  SparseEmbedding(x, y, 4, 5) = [[[  5.,   6.,   7.,   8.,   9.],
                                 [ 15.,  16.,  17.,  18.,  19.]],

                                [[  0.,   1.,   2.,   3.,   4.],
                                 [ 10.,  11.,  12.,  13.,  14.]]]

)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<SparseEmbeddingParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "weight"};
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<mxnet::FInferShape>("FInferShape", EmbeddingOpShape<SparseEmbeddingParam>)
.set_attr<nnvm::FInferType>("FInferType", EmbeddingOpType<SparseEmbeddingParam>)
.set_attr<FInferStorageType>("FInferStorageType", SparseEmbeddingOpForwardStorageType)
.set_attr<FComputeEx>("FComputeEx<cpu>", SparseEmbeddingOpForwardEx<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    return MakeNonlossGradNode("_backward_SparseEmbedding", n, ograds,
                               {n->inputs[0]}, n->attrs.dict);
  })
.add_argument("data", "NDArray-or-Symbol", "The input array to the embedding operator.")
.add_argument("weight", "NDArray-or-Symbol", "The embedding weight matrix.")
.add_arguments(EmbeddingParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_Embedding)
.set_num_inputs(2)
.set_num_outputs(2)
.set_attr_parser(ParamParser<EmbeddingParam>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FInferStorageType>("FInferStorageType", EmbeddingOpBackwardStorageType)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", EmbeddingOpBackward<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", EmbeddingOpBackwardEx<cpu>);

NNVM_REGISTER_OP(_backward_SparseEmbedding)
.set_attr_parser(ParamParser<SparseEmbeddingParam>)
.set_num_inputs(2)
.set_num_outputs(2)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FInferStorageType>("FInferStorageType", SparseEmbeddingOpBackwardStorageType)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FComputeEx>("FComputeEx<cpu>", SparseEmbeddingOpBackwardEx<cpu>);

NNVM_REGISTER_OP(take)
.describe(R"code(Takes elements from an input array along the given axis.

This function slices the input array along a particular axis with the provided indices.

Given data tensor of rank r >= 1, and indices tensor of rank q, gather entries of the axis
dimension of data (by default outer-most one as axis=0) indexed by indices, and concatenates them
in an output tensor of rank q + (r - 1).

Examples::

  x = [4.  5.  6.]

  // Trivial case, take the second element along the first axis.

  take(x, [1]) = [ 5. ]

  // The other trivial case, axis=-1, take the third element along the first axis

  take(x, [3], axis=-1, mode='clip') = [ 6. ]

  x = [[ 1.,  2.],
       [ 3.,  4.],
       [ 5.,  6.]]

  // In this case we will get rows 0 and 1, then 1 and 2. Along axis 0

  take(x, [[0,1],[1,2]]) = [[[ 1.,  2.],
                             [ 3.,  4.]],

                            [[ 3.,  4.],
                             [ 5.,  6.]]]

  // In this case we will get rows 0 and 1, then 1 and 2 (calculated by wrapping around).
  // Along axis 1

  take(x, [[0, 3], [-1, -2]], axis=1, mode='wrap') = [[[ 1.  2.]
                                                       [ 2.  1.]]

                                                      [[ 3.  4.]
                                                       [ 4.  3.]]

                                                      [[ 5.  6.]
                                                       [ 6.  5.]]]

The storage type of ``take`` output depends upon the input storage type:

   - take(default, default) = default
   - take(csr, default, axis=0) = csr

)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<TakeParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a", "indices"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", TakeOpShape)
.set_attr<nnvm::FInferType>("FInferType", TakeOpType)
.set_attr<FInferStorageType>("FInferStorageType", TakeOpForwardStorageType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", TakeOpForward<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", TakeOpForwardEx<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n,  const std::vector<nnvm::NodeEntry>& ograds) {
    return MakeNonlossGradNode("_backward_take", n, ograds,
                               {n->inputs[1]}, n->attrs.dict);
  })
.add_argument("a", "NDArray-or-Symbol", "The input array.")
.add_argument("indices", "NDArray-or-Symbol", "The indices of the values to be extracted.")
.add_arguments(TakeParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_take)
.set_num_inputs(2)
.set_num_outputs(2)
.set_attr_parser(ParamParser<TakeParam>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", TakeOpBackward<cpu>);


NNVM_REGISTER_OP(batch_take)
.describe(R"code(Takes elements from a data batch.

.. note::
  `batch_take` is deprecated. Use `pick` instead.

Given an input array of shape ``(d0, d1)`` and indices of shape ``(i0,)``, the result will be
an output array of shape ``(i0,)`` with::

  output[i] = input[i, indices[i]]

Examples::

  x = [[ 1.,  2.],
       [ 3.,  4.],
       [ 5.,  6.]]

  // takes elements with specified indices
  batch_take(x, [0,1,0]) = [ 1.  4.  5.]

)code" ADD_FILELINE)
.set_num_outputs(1)
.set_num_inputs(2)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a", "indices"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", BatchTakeOpShape)
.set_attr<nnvm::FInferType>("FInferType", BatchTakeOpType)
.set_attr<FCompute>("FCompute<cpu>", BatchTakeOpForward<cpu>)
.add_argument("a", "NDArray-or-Symbol", "The input array")
.add_argument("indices", "NDArray-or-Symbol", "The index array");

NNVM_REGISTER_OP(one_hot)
.add_alias("_npx_one_hot")
.describe(R"code(Returns a one-hot array.

The locations represented by `indices` take value `on_value`, while all
other locations take value `off_value`.

`one_hot` operation with `indices` of shape ``(i0, i1)`` and `depth`  of ``d`` would result
in an output array of shape ``(i0, i1, d)`` with::

  output[i,j,:] = off_value
  output[i,j,indices[i,j]] = on_value

Examples::

  one_hot([1,0,2,0], 3) = [[ 0.  1.  0.]
                           [ 1.  0.  0.]
                           [ 0.  0.  1.]
                           [ 1.  0.  0.]]

  one_hot([1,0,2,0], 3, on_value=8, off_value=1,
          dtype='int32') = [[1 8 1]
                            [8 1 1]
                            [1 1 8]
                            [8 1 1]]

  one_hot([[1,0],[1,0],[2,0]], 3) = [[[ 0.  1.  0.]
                                      [ 1.  0.  0.]]

                                     [[ 0.  1.  0.]
                                      [ 1.  0.  0.]]

                                     [[ 0.  0.  1.]
                                      [ 1.  0.  0.]]]
)code" ADD_FILELINE)
.set_num_outputs(1)
.set_num_inputs(1)
.set_attr_parser(ParamParser<OneHotParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"indices"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", OneHotOpShape)
.set_attr<nnvm::FInferType>("FInferType", OneHotOpType)
.set_attr<FCompute>("FCompute<cpu>", OneHotOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("indices", "NDArray-or-Symbol", "array of locations where to set on_value")
.add_arguments(OneHotParam::__FIELDS__());


NNVM_REGISTER_OP(gather_nd)
.describe(R"code(Gather elements or slices from `data` and store to a tensor whose
shape is defined by `indices`.

Given `data` with shape `(X_0, X_1, ..., X_{N-1})` and indices with shape
`(M, Y_0, ..., Y_{K-1})`, the output will have shape `(Y_0, ..., Y_{K-1}, X_M, ..., X_{N-1})`,
where `M <= N`. If `M == N`, output shape will simply be `(Y_0, ..., Y_{K-1})`.

The elements in output is defined as follows::

  output[y_0, ..., y_{K-1}, x_M, ..., x_{N-1}] = data[indices[0, y_0, ..., y_{K-1}],
                                                      ...,
                                                      indices[M-1, y_0, ..., y_{K-1}],
                                                      x_M, ..., x_{N-1}]

Examples::

  data = [[0, 1], [2, 3]]
  indices = [[1, 1, 0], [0, 1, 0]]
  gather_nd(data, indices) = [2, 3, 0]

  data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
  indices = [[0, 1], [1, 0]]
  gather_nd(data, indices) = [[3, 4], [5, 6]]

)code")
.set_num_outputs(1)
.set_num_inputs(2)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "indices"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", GatherNDShape)
.set_attr<nnvm::FInferType>("FInferType", GatherNDType)
.set_attr<FCompute>("FCompute<cpu>", GatherNDForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    auto p = nnvm::Node::Create();
    p->attrs.op = nnvm::Op::Get("_backward_gather_nd");
    p->attrs.name = n->attrs.name + "_backward";
    p->inputs.push_back(ograds[0]);
    p->inputs.push_back(n->inputs[1]);
    p->control_deps.emplace_back(n);
    auto zero = MakeNode("zeros_like", n->attrs.name + "_backward_indices",
                         {n->inputs[1]}, nullptr, &n);

    std::vector<nnvm::NodeEntry> ret;
    ret.emplace_back(p);
    ret.emplace_back(zero);
    return ret;
  })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.add_argument("data", "NDArray-or-Symbol", "data")
.add_argument("indices", "NDArray-or-Symbol", "indices");

NNVM_REGISTER_OP(scatter_nd)
.describe(R"code(Scatters data into a new tensor according to indices.

Given `data` with shape `(Y_0, ..., Y_{K-1}, X_M, ..., X_{N-1})` and indices with shape
`(M, Y_0, ..., Y_{K-1})`, the output will have shape `(X_0, X_1, ..., X_{N-1})`,
where `M <= N`. If `M == N`, data shape should simply be `(Y_0, ..., Y_{K-1})`.

The elements in output is defined as follows::

  output[indices[0, y_0, ..., y_{K-1}],
         ...,
         indices[M-1, y_0, ..., y_{K-1}],
         x_M, ..., x_{N-1}] = data[y_0, ..., y_{K-1}, x_M, ..., x_{N-1}]

all other entries in output are 0.

.. warning::

    If the indices have duplicates, the result will be non-deterministic and
    the gradient of `scatter_nd` will not be correct!!


Examples::

  data = [2, 3, 0]
  indices = [[1, 1, 0], [0, 1, 0]]
  shape = (2, 2)
  scatter_nd(data, indices, shape) = [[0, 0], [2, 3]]

  data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
  indices = [[0, 1], [1, 1]]
  shape = (2, 2, 2, 2)
  scatter_nd(data, indices, shape) = [[[[0, 0],
                                        [0, 0]],

                                       [[1, 2],
                                        [3, 4]]],

                                      [[[0, 0],
                                        [0, 0]],

                                       [[5, 6],
                                        [7, 8]]]]

)code")
.set_num_outputs(1)
.set_num_inputs(2)
.set_attr_parser(ParamParser<ScatterNDParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "indices"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", ScatterNDShape)
.set_attr<nnvm::FInferType>("FInferType", ScatterNDType)
.set_attr<FCompute>("FCompute<cpu>", ScatterNDForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    auto p = nnvm::Node::Create();
    p->attrs.op = nnvm::Op::Get("gather_nd");
    p->attrs.name = n->attrs.name + "_backward";
    p->inputs.push_back(ograds[0]);
    p->inputs.push_back(n->inputs[1]);
    p->control_deps.emplace_back(n);
    auto zero = MakeNode("zeros_like", n->attrs.name + "_backward_indices",
                         {n->inputs[1]}, nullptr, &n);
    std::vector<nnvm::NodeEntry> ret;
    ret.emplace_back(p);
    ret.emplace_back(zero);
    return ret;
  })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.add_argument("data", "NDArray-or-Symbol", "data")
.add_argument("indices", "NDArray-or-Symbol", "indices")
.add_arguments(ScatterNDParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_gather_nd)
.describe(R"code(Accumulates data according to indices and get the result. It's the backward of
`gather_nd`.

Given `data` with shape `(Y_0, ..., Y_{K-1}, X_M, ..., X_{N-1})` and indices with shape
`(M, Y_0, ..., Y_{K-1})`, the output will have shape `(X_0, X_1, ..., X_{N-1})`,
where `M <= N`. If `M == N`, data shape should simply be `(Y_0, ..., Y_{K-1})`.

The elements in output is defined as follows::

  output[indices[0, y_0, ..., y_{K-1}],
         ...,
         indices[M-1, y_0, ..., y_{K-1}],
         x_M, ..., x_{N-1}] += data[y_0, ..., y_{K-1}, x_M, ..., x_{N-1}]

all other entries in output are 0 or the original value if AddTo is triggered.

Examples::

  data = [2, 3, 0]
  indices = [[1, 1, 0], [0, 1, 0]]
  shape = (2, 2)
  _backward_gather_nd(data, indices, shape) = [[0, 0], [2, 3]] # Same as scatter_nd

  # The difference between scatter_nd and scatter_nd_acc is the latter will accumulate
  #  the values that point to the same index.

  data = [2, 3, 0]
  indices = [[1, 1, 0], [1, 1, 0]]
  shape = (2, 2)
  _backward_gather_nd(data, indices, shape) = [[0, 0], [0, 5]]

)code")
.set_num_outputs(1)
.set_num_inputs(2)
.set_attr_parser(ParamParser<ScatterNDParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "indices"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", ScatterNDShape)
.set_attr<nnvm::FInferType>("FInferType", ScatterNDType)
.set_attr<FCompute>("FCompute<cpu>", GatherNDBackward<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    auto p = nnvm::Node::Create();
    p->attrs.op = nnvm::Op::Get("gather_nd");
    p->attrs.name = n->attrs.name + "_backward";
    p->inputs.push_back(ograds[0]);
    p->inputs.push_back(n->inputs[1]);
    p->control_deps.emplace_back(n);
    auto zero = MakeNode("zeros_like", n->attrs.name + "_backward_indices",
                         {n->inputs[1]}, nullptr, &n);
    std::vector<nnvm::NodeEntry> ret;
    ret.emplace_back(p);
    ret.emplace_back(zero);
    return ret;
  })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.add_argument("data", "NDArray-or-Symbol", "data")
.add_argument("indices", "NDArray-or-Symbol", "indices")
.add_arguments(ScatterNDParam::__FIELDS__());

NNVM_REGISTER_OP(_scatter_set_nd)
.describe(R"code(This operator has the same functionality as scatter_nd
except that it does not reset the elements not indexed by the input
index `NDArray` in the input data `NDArray`. output should be explicitly
given and be the same as lhs.

.. note:: This operator is for internal use only.

Examples::

  data = [2, 3, 0]
  indices = [[1, 1, 0], [0, 1, 0]]
  out = [[1, 1], [1, 1]]
  _scatter_set_nd(lhs=out, rhs=data, indices=indices, out=out)
  out = [[0, 1], [2, 3]]

)code")
.set_num_outputs(1)
.set_num_inputs(3)
.set_attr_parser(ParamParser<ScatterNDParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"lhs", "rhs", "indices"};
  })
.set_attr<mxnet::FInferShape>("FInferShape",
  [](const nnvm::NodeAttrs& attrs,
     mxnet::ShapeVector *in_attrs,
     mxnet::ShapeVector *out_attrs) {
    CHECK_EQ(in_attrs->size(), 3U);
    CHECK_EQ(out_attrs->size(), 1U);
    SHAPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
    mxnet::ShapeVector tmp_in_attrs = {in_attrs->at(1), in_attrs->at(2)};
    if (!ScatterNDShape(attrs, &tmp_in_attrs, out_attrs)) {
      return false;
    }
    SHAPE_ASSIGN_CHECK(*in_attrs, 1, tmp_in_attrs[0]);
    SHAPE_ASSIGN_CHECK(*in_attrs, 2, tmp_in_attrs[1]);
    SHAPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
    return true;
  })
.set_attr<nnvm::FInferType>("FInferType",
  [](const nnvm::NodeAttrs& attrs,
     std::vector<int> *in_attrs,
     std::vector<int> *out_attrs) {
    CHECK_EQ(in_attrs->size(), 3U);
    CHECK_EQ(out_attrs->size(), 1U);
    TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
    TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
    std::vector<int> tmp_in_attrs = {in_attrs->at(1), in_attrs->at(2)};
    if (!ScatterNDType(attrs, &tmp_in_attrs, out_attrs)) {
      return false;
    }
    TYPE_ASSIGN_CHECK(*in_attrs, 1, tmp_in_attrs[0]);
    TYPE_ASSIGN_CHECK(*in_attrs, 2, tmp_in_attrs[1]);
    TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
    return true;
  })
.set_attr<FCompute>("FCompute<cpu>", ScatterSetNDForward<cpu>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
  [](const NodeAttrs& attrs){
    return std::vector<bool>{true};
  })
.add_argument("lhs", "NDArray-or-Symbol", "source input")
.add_argument("rhs", "NDArray-or-Symbol", "value to assign")
.add_argument("indices", "NDArray-or-Symbol", "indices")
.add_arguments(ScatterNDParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
