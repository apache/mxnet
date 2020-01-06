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
 * \file ndarray_function.cc
 * \brief CPU Implementation of ndarray function.
 */

// this will be invoked by gcc and compile CPU version
#include "./ndarray_function.h"
#include "./ndarray_function-inl.h"
#include "../common/utils.h"
#include "../operator/mxnet_op.h"
#include "../operator/tensor/elemwise_binary_op-inl.h"
#include "../operator/tensor/elemwise_sum.h"

namespace mxnet {
namespace ndarray {
template<>
void Copy<cpu, cpu>(const TBlob &from, TBlob *to,
                    Context from_ctx, Context to_ctx,
                    RunContext ctx) {
  MSHADOW_TYPE_SWITCH_WITH_BOOL(to->type_flag_, DType, {
    if (to->type_flag_ == from.type_flag_) {
      if (!features::is_enabled(features::INT64_TENSOR_SIZE)) {
        CHECK_LT(from.Size(), (int64_t{1} << 31) - 1) <<
                  "Size of tensor you are trying to allocate is larger than "
                  "2^31 elements. Please build with flag USE_INT64_TENSOR_SIZE=1";
      }
      const index_t size = static_cast<index_t>(from.Size());
      CHECK_EQ(size, to->Size()) << "copying size mismatch, from: " << size * sizeof(DType)
               << " bytes, to: " << to->Size() * sizeof(DType) << " bytes.";
      common::ParallelCopy(to->dptr<DType>(), from.dptr<DType>(), size);
    } else {
      MSHADOW_TYPE_SWITCH_WITH_BOOL(from.type_flag_, SrcDType, {
          to->FlatTo1D<cpu, DType>() =
              mshadow::expr::tcast<DType>(from.FlatTo1D<cpu, SrcDType>());
      })
    }
  })
}

template<typename DType, typename IType>
void ElementwiseSumRspImpl(mshadow::Stream<cpu>* s,
                           const std::vector<NDArray>& nds,
                           const std::vector<IType>& uniq_row_idx,
                           NDArray* out,
                           const int nthreads = 4) {
#pragma omp parallel num_threads(nthreads)
  {
    const size_t nnr = uniq_row_idx.size();
    const int num_threads = omp_get_num_threads();
    size_t row_block_len = (nnr + num_threads  - 1) / num_threads;
    const size_t row_block_start = omp_get_thread_num() * row_block_len;
    if (row_block_start < nnr) {
      const size_t row_block_end = std::min(row_block_start+row_block_len, nnr);

      const size_t row_length = out->data().shape_.ProdShape(1, out->data().shape_.ndim());
      auto out_values = out->data().get_with_shape<cpu, 2, DType>(
          mshadow::Shape2(out->storage_shape()[0], row_length), s);
      auto out_indices = out->aux_data(rowsparse::kIdx).FlatTo1D<cpu, IType>();
      for (size_t i = row_block_start; i < row_block_end; ++i) {
        out_indices[i] = uniq_row_idx[i];
      }
      for (const auto& nd : nds) {
        if (nd.storage_initialized()) {
          const auto nd_indices = nd.aux_data(rowsparse::kIdx).FlatTo1D<cpu, IType>();
          const auto nd_values = nd.data().get_with_shape<cpu, 2, DType>(
              mshadow::Shape2(nd.storage_shape()[0], row_length), s);
          const auto nd_num_rows = nd.aux_shape(rowsparse::kIdx).Size();
          const IType* nd_indices_start = &nd_indices[0];
          const IType* nd_indices_end = nd_indices_start + nd_num_rows;
          const IType* row_idx_ptr = std::lower_bound(nd_indices_start, nd_indices_end,
                                                      out_indices[row_block_start]);
          // skip this nd if all of its row indices are smaller than out_indices[row_block_start]
          // or current row block is not covered by [*row_idx_ptr, nd_indices_end).
          if (nd_indices_end == row_idx_ptr || *row_idx_ptr > out_indices[row_block_end-1]) {
            continue;
          }
          for (size_t irow = row_block_start;
               irow < row_block_end && row_idx_ptr != nd_indices_end;) {
            if (out_indices[irow] == *row_idx_ptr) {
              auto out_value_cur_row = out_values[irow];
              const auto offset = row_idx_ptr - nd_indices_start;
              auto nd_value_cur_row = nd_values[offset];
              for (index_t j = 0; j < nd_value_cur_row.shape_[0]; ++j) {
                out_value_cur_row[j] += nd_value_cur_row[j];
              }
              ++irow;
              ++row_idx_ptr;
            } else if (out_indices[irow] < *row_idx_ptr) {
              ++irow;
            } else {
              ++row_idx_ptr;
            }
          }
        }
      }
    }
  }
}

/*!
 * \brief Given a vector of ndarrays, generate a index vector containing
 * all the unique row indices of the ndarrays.
 */
template<typename IType>
void GetUniqueRspRowIdx(const std::vector<NDArray>& nds,
                        std::vector<IType>* uniq_row_idx) {
  using namespace rowsparse;
  size_t total_num_rows = 0;
  for (const auto& nd : nds) {
    CHECK_EQ(nd.storage_type(), kRowSparseStorage);
    if (nd.storage_initialized()) {
      total_num_rows += nd.aux_shape(kIdx).Size();
    }
  }

  uniq_row_idx->resize(total_num_rows);
  int nthreads = omp_get_max_threads();
  int offset = 0;
  for (const auto& nd : nds) {
    if (nd.storage_initialized()) {
      const IType* nd_row_idx = nd.aux_data(kIdx).dptr<IType>();
      const int num_rows = nd.aux_shape(kIdx).Size();
#pragma omp parallel for num_threads(nthreads)
      for (int i = 0; i < num_rows; ++i) {
        (*uniq_row_idx)[offset+i] = nd_row_idx[i];
      }
      offset += num_rows;
    }
  }

  common::ParallelSort(uniq_row_idx->begin(), uniq_row_idx->end(), nthreads);
  auto it = std::unique(uniq_row_idx->begin(), uniq_row_idx->end());
  uniq_row_idx->resize(it - uniq_row_idx->begin());
}

void ElementwiseSumRsp(mshadow::Stream<cpu>* s,
                       const Resource& rsc,
                       const std::vector<NDArray>& nds,
                       NDArray* out) {
  if (nds.empty()) return;
  using namespace rowsparse;
  CHECK_EQ(out->storage_type(), kRowSparseStorage)
    << "Expected row sparse storage type ("
    << out->storage_type() << " given)";

  MSHADOW_TYPE_SWITCH(out->dtype(), DType, {
    MSHADOW_IDX_TYPE_SWITCH(out->aux_type(kIdx), IType, {
      // TODO(Jun): Use resource rsc for temporary vector instead of
      //            allocating it directly in GetUniqueRspRowIdx
      std::vector<IType> uniq_row_idx;
      GetUniqueRspRowIdx(nds, &uniq_row_idx);
      out->CheckAndAlloc({mshadow::Shape1(uniq_row_idx.size())});
      out->data().FlatTo2D<cpu, DType>() = static_cast<DType>(0);
      ElementwiseSumRspImpl<DType, IType>(s, nds, uniq_row_idx, out, omp_get_max_threads());
    });
  });
}

void ElementwiseSumDnsCsrDnsImpl(mshadow::Stream<cpu>* s,
                                 const Resource& rsc,
                                 const std::vector<NDArray>& nds,
                                 NDArray* out) {
  using namespace mxnet::op;
  using namespace mxnet::op::mxnet_op;
  const TBlob& out_data = out->data();
  MSHADOW_TYPE_SWITCH(out->dtype(), DType, {  // data type
    Kernel<Sum, cpu>::Launch(
      s, out_data.Size(), out_data.dptr<DType>(), kWriteTo, nds[0].data().dptr<DType>(),
      nds[2].data().dptr<DType>());
    const TBlob& csr_data = nds[1].data();
    const TBlob& csr_indices = nds[1].aux_data(csr::kIdx);
    const TBlob& csr_indptr = nds[1].aux_data(csr::kIndPtr);
    const nnvm::dim_t num_rows = nds[1].shape()[0];
    const nnvm::dim_t num_cols = nds[1].shape()[1];
    MSHADOW_IDX_TYPE_SWITCH(csr_indices.type_flag_, IType, {  // indices type
      MSHADOW_IDX_TYPE_SWITCH(csr_indptr.type_flag_, CType, {  // indptr type
        if (nds[1].storage_initialized()) {
          Kernel<ElemwiseDnsCsrDnsKernel<kWriteTo, mshadow_op::plus>, cpu>::Launch(
            s, num_rows, out_data.dptr<DType>(), out_data.dptr<DType>(),
            csr_data.dptr<DType>(), csr_indices.dptr<IType>(),
            csr_indptr.dptr<CType>(), num_rows, num_cols);
        }
      });
    });
  });
}

void ElementwiseSumContainsDnsImpl(mshadow::Stream<cpu>* s,
                                   const Resource& rsc,
                                   const std::vector<NDArray>& nds,
                                   NDArray* out) {
  using namespace mxnet::op;
  using namespace mxnet::op::mxnet_op;
  const TBlob& out_data = out->data();
  MSHADOW_TYPE_SWITCH(out->dtype(), DType, {  // data type
    // Do not set_zero when output mem inplace with input[0] mem
    // Now for add_n OP, output mem can be in-placed with the first input
    if (nds[0].data().dptr<DType>() != out_data.dptr<DType>()) {
      Kernel<set_zero, cpu>::Launch(s, out_data.Size(), out_data.dptr<DType>());
    }
    for (size_t i = 0; i < nds.size(); ++i) {
      const NDArray& nd = nds[i];
      const TBlob& nd_data = nd.data();

      if (i == 0) {
        if (nd.storage_type() == kDefaultStorage) {
          Kernel<op_with_req<mshadow_op::identity, kWriteTo>, cpu>::Launch(
            s, out_data.Size(), out_data.dptr<DType>(), nd_data.dptr<DType>());
          continue;
        } else {
          Kernel<set_zero, cpu>::Launch(s, out_data.Size(), out_data.dptr<DType>());
        }
      }

      switch (nd.storage_type()) {
        case kDefaultStorage: {
          Kernel<op_with_req<mshadow_op::plus, kWriteTo>, cpu>::Launch(
            s, out_data.Size(), out_data.dptr<DType>(), out_data.dptr<DType>(),
            nd_data.dptr<DType>());
          break;
        }
        case kCSRStorage: {
          const TBlob& nd_indices = nd.aux_data(csr::kIdx);
          const TBlob& nd_indptr = nd.aux_data(csr::kIndPtr);
          const nnvm::dim_t num_rows = nd.shape()[0];
          const nnvm::dim_t num_cols = nd.shape()[1];
          MSHADOW_IDX_TYPE_SWITCH(nd_indices.type_flag_, IType, {  // indices type
            MSHADOW_IDX_TYPE_SWITCH(nd_indptr.type_flag_, CType, {  // indptr type
              if (nd.storage_initialized()) {
                Kernel<ElemwiseDnsCsrDnsKernel<kWriteTo, mshadow_op::plus>, cpu>::Launch(
                  s, num_rows, out_data.dptr<DType>(), out_data.dptr<DType>(),
                  nd_data.dptr<DType>(), nd_indices.dptr<IType>(),
                  nd_indptr.dptr<CType>(), num_rows, num_cols);
              }
            });
          });
          break;
        }
        case kRowSparseStorage: {
          const TBlob& nd_indices = nd.aux_data(rowsparse::kIdx);
          const nnvm::dim_t num_rows = nd.shape()[0];
          const nnvm::dim_t num_cols = nd.shape()[1];
          MSHADOW_IDX_TYPE_SWITCH(nd_indices.type_flag_, IType, {  // indices type
            if (nd.storage_initialized()) {
              const nnvm::dim_t nz_rows = nd_indices.Size();
              Kernel<ElemwiseDnsRspDnsKernel<kWriteTo, mshadow_op::plus>, cpu>::Launch(
                s, nz_rows * num_cols, out_data.dptr<DType>(),
                out_data.dptr<DType>(), nd_data.dptr<DType>(), nd_indices.dptr<IType>(),
                num_rows, nz_rows, num_cols);
            }
          });
          break;
        }
        default:
          LOG(FATAL) << "unknown storage type " << nd.storage_type() << "encountered...";
      }
    }
  });
}

/*!
 * \brief Parallel cpu impl of elemwise sum for sparse tensors.
 * Currently only support row sparse sum.
 */
template<>
void ElementwiseSum<cpu>(mshadow::Stream<cpu>* s,
                         const Resource& rsc,
                         const std::vector<NDArray>& nds,
                         NDArray* out) {
  if (nds.empty()) return;
  if (common::ContainsOnlyStorage(nds, kRowSparseStorage)) {
    ElementwiseSumRsp(s, rsc, nds, out);
  } else if (nds.size() == 3U && nds[0].storage_type() == kDefaultStorage &&
             nds[1].storage_type() == kCSRStorage && nds[2].storage_type() == kDefaultStorage &&
             out->storage_type() == kDefaultStorage) {
    ElementwiseSumDnsCsrDnsImpl(s, rsc, nds, out);
  } else if (nds.size() > 4U && common::ContainsStorageType(nds, kDefaultStorage) &&
             out->storage_type() == kDefaultStorage) {
    ElementwiseSumContainsDnsImpl(s, rsc, nds, out);
  } else {
    LOG(FATAL) << "ElementwiseSum<cpu> has not been implemented for storage_type = << "
               << nds[0].storage_type();
  }
}


template<>
void Eval<cpu>(mshadow::Stream<cpu> *s,
               const real_t val, const NDArray& dst) {
  NDArray temp = dst;
  const NDArrayStorageType stype = temp.storage_type();
  if (stype == kRowSparseStorage) {
    SetValueRspImpl(s, val, &temp);
  } else {
    LOG(FATAL) << "Not implemented for storage type" << stype;
  }
}

}  // namespace ndarray
}  // namespace mxnet
