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
 * \file cast_storage-inl.h
 * \brief CPU implementation of cast_storage operator for dense and sparse tensors
 */
#ifndef MXNET_OPERATOR_TENSOR_CAST_STORAGE_INL_H_
#define MXNET_OPERATOR_TENSOR_CAST_STORAGE_INL_H_

#include <dmlc/timer.h>
#include <mxnet/ndarray.h>
#include <vector>
#include <algorithm>
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "./init_op.h"
#ifdef __CUDACC__
#include "./cast_storage-inl.cuh"
#endif  // __CUDACC__
#if MXNET_USE_ONEDNN == 1
#include "../nn/dnnl/dnnl_base-inl.h"
#endif

namespace mxnet {
namespace op {

/*!
 * \brief CPU Kernel for marking row_idx of a RSP tensor per row.
 */
struct MarkRspRowIdx {
  // i represents the row index of the tensor data
  template <typename DType, typename RType>
  MSHADOW_CINLINE static void Map(int i,
                                  RType* row_idx,
                                  const DType* data,
                                  const nnvm::dim_t row_length) {
    using nnvm::dim_t;
    dim_t j      = 0;
    dim_t offset = i * row_length;
    for (; j < row_length; ++j) {
      if (data[offset + j] != 0) {
        break;
      }
    }
    if (row_length == j) {
      row_idx[i] = 0;  // mark as zero for zero row
    } else {
      row_idx[i] = 1;  // mark as one for non-zero row
    }
  }
};

/*!
 * \brief CPU implementation of casting a dns tensor to rsp type.
 */
inline void CastStorageDnsRspImpl(const OpContext& ctx,
                                  const cpu& cpu_dev,
                                  const TBlob& dns,
                                  NDArray* rsp) {
  using namespace rowsparse;
  using namespace mshadow;
  using nnvm::dim_t;
  CHECK(rsp != nullptr);
  CHECK_EQ(rsp->storage_type(), kRowSparseStorage);
  CHECK_EQ(dns.shape_, rsp->shape());
  mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
  MSHADOW_TYPE_SWITCH(dns.type_flag_, DType, {             // data type
    MSHADOW_IDX_TYPE_SWITCH(rsp->aux_type(kIdx), RType, {  // row idx type
      const dim_t num_rows   = dns.shape_[0];
      const dim_t row_length = dns.shape_.ProdShape(1, dns.shape_.ndim());
      rsp->CheckAndAllocAuxData(kIdx, Shape1(num_rows));
      TBlob row_idx_blob = rsp->aux_data(kIdx);
      RType* row_idx     = row_idx_blob.dptr<RType>();
      dim_t num_threads  = num_rows;
      mxnet_op::Kernel<MarkRspRowIdx, cpu>::Launch(
          s, num_threads, row_idx, dns.dptr<DType>(), row_length);
      dim_t nnr = 0;
      nnr       = common::ParallelAccumulate(row_idx, num_rows, nnr);
      rsp->set_aux_shape(kIdx, Shape1(nnr));
      if (0 == nnr)
        return;
      auto storage_shape = dns.shape_;
      storage_shape[0]   = nnr;
      rsp->CheckAndAllocData(storage_shape);
      auto dns_data = dns.get_with_shape<cpu, 2, DType>(Shape2(num_rows, row_length), s);
      auto rsp_data = rsp->data().get_with_shape<cpu, 2, DType>(Shape2(nnr, row_length), s);
      dim_t idx     = 0;
      for (dim_t i = 0; i < num_rows; ++i) {
        if (row_idx[i] > 0) {
          row_idx[idx] = i;
          Copy(rsp_data[idx], dns_data[i], s);
          ++idx;
        }
      }
    });
  });
}

// TODO(haibin) Use memcopy instead will be much faster than assigning each individual element
struct CastStorageRspDnsKernel {
  template <typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i,
                                  const nnvm::dim_t row_length,
                                  const IType* idx,
                                  const DType* data,
                                  DType* dns) {
    using nnvm::dim_t;
    IType rid        = idx[i];
    dim_t dns_offset = rid * row_length;
    dim_t rsp_offset = i * row_length;
    for (dim_t col = 0; col < row_length; col++) {
      dns[dns_offset + col] = data[rsp_offset + col];
    }
  }
};

/*!
 * \brief This function assumes that the memory for dns has been allocated already
 * since the shape is known at binding stage.
 */
template <typename xpu>
void CastStorageRspDnsImpl(const OpContext& ctx, const NDArray& rsp, TBlob* dns) {
  mshadow::Stream<xpu>* s = ctx.get_stream<xpu>();
  CHECK_EQ(rsp.storage_type(), kRowSparseStorage);
  using nnvm::dim_t;
  MSHADOW_TYPE_SWITCH(dns->type_flag_, DType, {
    MSHADOW_IDX_TYPE_SWITCH(rsp.aux_type(rowsparse::kIdx), IType, {
      // assign zeros
      mxnet_op::Kernel<mxnet_op::set_zero, xpu>::Launch(s, dns->Size(), dns->dptr<DType>());
      if (rsp.storage_initialized()) {
        // copy over row by row
        auto in_idx             = rsp.aux_data(rowsparse::kIdx).FlatTo1D<xpu, IType>(s).dptr_;
        auto in_data            = rsp.data().dptr<DType>();
        auto out_data           = dns->dptr<DType>();
        auto shape              = rsp.shape();
        const dim_t num_rows    = rsp.aux_shape(rowsparse::kIdx).Size();
        const dim_t row_length  = shape.ProdShape(1, shape.ndim());
        const dim_t num_threads = num_rows;
        mxnet_op::Kernel<CastStorageRspDnsKernel, xpu>::Launch(
            s, num_threads, row_length, in_idx, in_data, out_data);
      }
    });
  });
}

/*!
 * \brief CPU kernel for initializing the indptr in a csr matrix.
 */
struct FillCsrIndPtr {
  /*!
   * \brief
   * \param i         the i-th row of the dns tensor
   * \param indptr    the indptr of the csr tensor
   * \param dns       the dns tensor
   * \param num_rows  number of rows of the dns tensor
   * \param num_cols  number of columns of the dns tensor
   */
  template <typename DType, typename IType>
  MSHADOW_CINLINE static void Map(int i,
                                  IType* indptr,
                                  const DType* dns,
                                  const nnvm::dim_t num_rows,
                                  const nnvm::dim_t num_cols) {
    using nnvm::dim_t;
    indptr[i + 1]      = 0;
    const dim_t offset = i * num_cols;
    for (dim_t j = 0; j < num_cols; ++j) {
      if (dns[offset + j] != 0) {
        ++indptr[i + 1];
      }
    }
  }
};

/*!
 * \brief CPU kernel for initializing the col_idx and value array of the csr matrix.
 */
struct FillCsrColIdxAndVals {
  /*!
   * \brief
   * \param i         the i-th row of the dns tensor
   * \param val       value array of the csr tensor
   * \param col_idx   column idx array of the csr tensor
   * \param indptr    indptr array of the csr tensor
   * \param dns       dns tensor
   * \param num_rows  number of rows of the dns tensor
   * \param num_cols  number of columns of the dns tensor
   */
  template <typename DType, typename IType, typename CType>
  MSHADOW_CINLINE static void Map(int i,
                                  DType* val,
                                  CType* col_idx,
                                  const IType* indptr,
                                  const DType* dns,
                                  const nnvm::dim_t num_rows,
                                  const nnvm::dim_t num_cols) {
    using nnvm::dim_t;
    const dim_t offset = i * num_cols;
    IType k            = indptr[i];
    for (dim_t j = 0; j < num_cols; ++j) {
      if (dns[offset + j] != 0) {
        val[k]     = dns[offset + j];
        col_idx[k] = j;
        ++k;
      }
    }
  }
};

/*!
 * \brief CPU implementation of casting a dns matrix to csr type.
 */
inline void CastStorageDnsCsrImpl(const OpContext& ctx,
                                  const cpu& cpu_dev,
                                  const TBlob& dns,
                                  NDArray* csr) {
  CHECK(csr != nullptr);
  CHECK_EQ(csr->storage_type(), kCSRStorage);
  CHECK_EQ(dns.shape_.ndim(), 2);
  CHECK_EQ(dns.shape_, csr->shape());
  using mshadow::Shape1;
  using nnvm::dim_t;
  mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
  MSHADOW_TYPE_SWITCH(dns.type_flag_, DType, {                     // data type
    MSHADOW_IDX_TYPE_SWITCH(csr->aux_type(csr::kIndPtr), IType, {  // indptr type
      MSHADOW_IDX_TYPE_SWITCH(csr->aux_type(csr::kIdx), CType, {   // col idx type
        const dim_t num_rows = dns.shape_[0];
        const dim_t num_cols = dns.shape_[1];
        csr->CheckAndAllocAuxData(csr::kIndPtr, mshadow::Shape1(num_rows + 1));
        IType* indptr     = csr->aux_data(csr::kIndPtr).dptr<IType>();
        DType* dns_data   = dns.dptr<DType>();
        dim_t num_threads = num_rows;
        mxnet_op::Kernel<FillCsrIndPtr, cpu>::Launch(
            s, num_threads, indptr, dns_data, num_rows, num_cols);
        // single thread to accumulate indptr
        // indptr[num_rows] indicates the number of non-zero elements
        indptr[0] = 0;
        for (dim_t i = 0; i < num_rows; ++i) {
          indptr[i + 1] += indptr[i];
        }
        // allocate column idx array and value array
        csr->CheckAndAllocAuxData(csr::kIdx, Shape1(static_cast<index_t>(indptr[num_rows])));
        csr->CheckAndAllocData(Shape1(static_cast<index_t>(indptr[num_rows])));
        // fill col_idx and value arrays of the csr
        mxnet_op::Kernel<FillCsrColIdxAndVals, cpu>::Launch(s,
                                                            num_threads,
                                                            csr->data().dptr<DType>(),
                                                            csr->aux_data(csr::kIdx).dptr<CType>(),
                                                            indptr,
                                                            dns_data,
                                                            num_rows,
                                                            num_cols);
      });
    });
  });
}

/*!
 * \brief This is the kernel for copying csr.data to its corresponding dns matrix.
 */
struct CopyCsrDataToDns {
  /*!
   * \brief
   * \param i         the i-th row of the dns tensor
   * \param dns_data  data blob of the dns tensor
   * \param col_idx   column idx array of the csr tensor
   * \param indptr    indptr array of the csr tensor
   * \param csr_data  data blob of the csr tensor
   * \param num_cols  number of columns of the dns tensor
   */
  template <typename DType, typename IType, typename CType>
  MSHADOW_XINLINE static void Map(index_t i,
                                  DType* dns_data,
                                  const CType* col_idx,
                                  const IType* indptr,
                                  const DType* csr_data,
                                  const nnvm::dim_t num_cols) {
    const nnvm::dim_t offset = i * num_cols;
    for (IType j = indptr[i]; j < indptr[i + 1]; ++j) {
      dns_data[offset + col_idx[j]] = csr_data[j];
    }
  }
};

/*!
 * \brief Casts a csr matrix to dns format.
 */
template <typename xpu>
void CastStorageCsrDnsImpl(const OpContext& ctx, const NDArray& csr, TBlob* dns) {
  CHECK(dns != nullptr);
  CHECK_EQ(csr.storage_type(), kCSRStorage);
  CHECK_EQ(dns->shape_.ndim(), 2);
  CHECK_EQ(dns->shape_, csr.shape());
  using nnvm::dim_t;
  mshadow::Stream<xpu>* s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(dns->type_flag_, DType, {                   // data type
    MSHADOW_IDX_TYPE_SWITCH(csr.aux_type(csr::kIndPtr), IType, {  // indptr type
      MSHADOW_IDX_TYPE_SWITCH(csr.aux_type(csr::kIdx), CType, {   // col idx type
        const dim_t num_rows = dns->shape_[0];
        const dim_t num_cols = dns->shape_[1];
        DType* dns_data      = dns->dptr<DType>();
        dim_t num_threads    = dns->shape_.Size();
        mxnet_op::Kernel<mxnet_op::set_zero, xpu>::Launch(s, num_threads, dns_data);
        if (!csr.storage_initialized())
          return;
        const IType* indptr   = csr.aux_data(csr::kIndPtr).dptr<IType>();
        const CType* col_idx  = csr.aux_data(csr::kIdx).dptr<CType>();
        const DType* csr_data = csr.data().dptr<DType>();
        num_threads           = num_rows;
        mxnet_op::Kernel<CopyCsrDataToDns, xpu>::Launch(
            s, num_threads, dns_data, col_idx, indptr, csr_data, num_cols);
      });
    });
  });
}

/*!
 * \brief Casts a csr matrix to another csr.
 */
template <typename xpu>
void CastStorageCsrCsrImpl(const OpContext& ctx, const NDArray& csr, NDArray* output) {
  mshadow::Stream<xpu>* s = ctx.get_stream<xpu>();
  if (!csr.storage_initialized()) {
    FillZerosCsrImpl(s, *output);
    return;
  }
  mxnet::ShapeVector aux_shapes({csr.aux_shape(csr::kIndPtr), csr.aux_shape(csr::kIdx)});
  output->CheckAndAlloc(aux_shapes);
  const TBlob& val    = output->data();
  const TBlob& indptr = output->aux_data(csr::kIndPtr);
  const TBlob& idx    = output->aux_data(csr::kIdx);
  mxnet_op::copy(s, val, csr.data());
  mxnet_op::copy(s, indptr, csr.aux_data(csr::kIndPtr));
  mxnet_op::copy(s, idx, csr.aux_data(csr::kIdx));
}

/*!
 * \brief Casts a rsp matrix to another rsp.
 */
template <typename xpu>
void CastStorageRspRspImpl(const OpContext& ctx, const NDArray& rsp, NDArray* output) {
  CHECK_EQ(rsp.storage_type(), output->storage_type()) << "Copying with different storage type";
  mshadow::Stream<xpu>* s = ctx.get_stream<xpu>();
  if (!rsp.storage_initialized()) {
    FillZerosRspImpl(s, *output);
    return;
  }
  auto aux_shape = rsp.aux_shape(rowsparse::kIdx);
  output->CheckAndAlloc({aux_shape});
  const TBlob& val      = output->data();
  const TBlob& idx      = output->aux_data(rowsparse::kIdx);
  const TBlob& from_val = rsp.data();
  const TBlob& from_idx = rsp.aux_data(rowsparse::kIdx);
  mxnet_op::copy(s, val, from_val);
  mxnet_op::copy(s, idx, from_idx);
}

template <typename xpu>
void CastStorageComputeImpl(const OpContext& ctx, const NDArray& input, const NDArray& output) {
  const auto src_stype = input.storage_type();
  const auto dst_stype = output.storage_type();
  if (src_stype == kRowSparseStorage && dst_stype == kDefaultStorage) {
    TBlob ret = output.data();
    CastStorageRspDnsImpl<xpu>(ctx, input, &ret);
  } else if (src_stype == kDefaultStorage && dst_stype == kRowSparseStorage) {
    NDArray ret = output;  // get rid of the const qualifer
    CastStorageDnsRspImpl(ctx, xpu(), input.data(), &ret);
  } else if (src_stype == kDefaultStorage && dst_stype == kCSRStorage) {
    NDArray ret = output;  // get rid of the const qualifer
    CastStorageDnsCsrImpl(ctx, xpu(), input.data(), &ret);
  } else if (src_stype == kCSRStorage && dst_stype == kDefaultStorage) {
    TBlob ret = output.data();
    CastStorageCsrDnsImpl<xpu>(ctx, input, &ret);
  } else if (src_stype == kCSRStorage && dst_stype == kCSRStorage) {
    NDArray ret = output;
    CastStorageCsrCsrImpl<xpu>(ctx, input, &ret);
  } else if (src_stype == kRowSparseStorage && dst_stype == kRowSparseStorage) {
    NDArray ret = output;
    CastStorageRspRspImpl<xpu>(ctx, input, &ret);
#if MXNET_USE_ONEDNN == 1
  } else if (src_stype == kDefaultStorage && dst_stype == kDefaultStorage) {
    CHECK_EQ(output.ctx().dev_type, input.ctx().dev_type);
    // If one of them uses the DNNL layout.
    if (input.IsDNNLData() || output.IsDNNLData()) {
      NDArray tmp_input = input;
      // If the input data is DNNL and is a view, we need to reorder the input
      // data first.
      if (input.IsDNNLData() && input.IsView())
        tmp_input = input.Reorder2Default();
      const dnnl::memory* in_mem = tmp_input.GetDNNLData();
      const_cast<NDArray&>(output).CopyFrom(*in_mem);
      DNNLStream::Get()->Submit();
    } else {
      mxnet_op::copy(ctx.get_stream<xpu>(), output.data(), input.data());
    }
#endif
  } else {
    LOG(FATAL) << "Not implemented from " << src_stype << " to " << dst_stype;
  }
}

struct CastStorageParam : public dmlc::Parameter<CastStorageParam> {
  int stype;
  DMLC_DECLARE_PARAMETER(CastStorageParam) {
    DMLC_DECLARE_FIELD(stype)
        .add_enum("default", kDefaultStorage)
        .add_enum("row_sparse", kRowSparseStorage)
        .add_enum("csr", kCSRStorage)
        .describe("Output storage type.");
  }
};

inline bool CastStorageInferStorageType(const nnvm::NodeAttrs& attrs,
                                        const int dev_mask,
                                        DispatchMode* dispatch_mode,
                                        std::vector<int>* in_attrs,
                                        std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  CHECK_NE(in_attrs->at(0), kUndefinedStorage) << "src ndarray's storage type must be specified";
  const CastStorageParam& param = nnvm::get<CastStorageParam>(attrs.parsed);
  CHECK_NE(param.stype, kUndefinedStorage) << "dst ndarray's storage type must be specified";
  const auto& in_stype    = in_attrs->at(0);
  const auto& param_stype = static_cast<NDArrayStorageType>(param.stype);
  bool dispatched         = false;
  // dns -> dns, dns -> rsp, dns -> csr
  if (!dispatched && in_stype == kDefaultStorage && param_stype == kDefaultStorage) {
    // dns -> dns
    DispatchMode mode = DispatchMode::kFCompute;
#if MXNET_USE_ONEDNN == 1
    // If we use oneDNN and the arrays are in CPU memory, the array may store
    // oneDNN layout, we should convert its layout explicitly.
    if (dev_mask == kCPU)
      mode = DispatchMode::kFComputeEx;
#endif
    dispatched = storage_type_assign(out_attrs, kDefaultStorage, dispatch_mode, mode);
  }
  if (!dispatched && in_stype == kDefaultStorage &&
      (param_stype == kRowSparseStorage || param_stype == kCSRStorage)) {
    // dns -> rsp, dns -> csr
    dispatched =
        storage_type_assign(out_attrs, param_stype, dispatch_mode, DispatchMode::kFComputeEx);
  }
  if (!dispatched && in_stype == kRowSparseStorage &&
      (param_stype == kRowSparseStorage || param_stype == kDefaultStorage)) {
    // rsp -> rsp, rsp -> dns
    dispatched =
        storage_type_assign(out_attrs, param_stype, dispatch_mode, DispatchMode::kFComputeEx);
  }
  if (!dispatched && in_stype == kCSRStorage &&
      (param_stype == kCSRStorage || param_stype == kDefaultStorage)) {
    // csr -> csr, csr -> dns
    dispatched =
        storage_type_assign(out_attrs, param_stype, dispatch_mode, DispatchMode::kFComputeEx);
  }
  return dispatched;
}

template <typename xpu>
void CastStorageComputeEx(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<NDArray>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(outputs.size(), 1);
  if (req[0] == kNullOp)
    return;
  CHECK_EQ(req[0], kWriteTo) << "CastStorageComputeEx expects req[0] == kWriteTo";
  CastStorageComputeImpl<xpu>(ctx, inputs[0], outputs[0]);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_CAST_STORAGE_INL_H_
