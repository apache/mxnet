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
 * \file sparse_retain-inl.h
 * \brief
*/
#ifndef MXNET_OPERATOR_TENSOR_SPARSE_RETAIN_INL_H_
#define MXNET_OPERATOR_TENSOR_SPARSE_RETAIN_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <utility>
#include "./init_op.h"
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "../mxnet_op.h"

namespace mxnet {
namespace op {

/*!
 * \brief sparse retain namespace
 */
namespace sr {
enum SparseRetainOpInputs {kArr, kIdx};
enum SparseRetainOpOutputs {kOut};
}  // namespace sr

inline bool SparseRetainOpShape(const nnvm::NodeAttrs& attrs,
                                std::vector<TShape> *in_attrs,
                                std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U)
    << "sparse_retain operator takes 2 arguments (" << in_attrs->size() << " given)";
  CHECK_EQ(out_attrs->size(), 1U);

  TShape tshape((*in_attrs)[sr::kArr]);
  shape_assign(&tshape, (*out_attrs)[sr::kOut]);
  SHAPE_ASSIGN_CHECK(*in_attrs, sr::kArr, tshape);
  SHAPE_ASSIGN_CHECK(*out_attrs, sr::kOut, tshape);
  return true;
}

inline bool SparseRetainOpType(const nnvm::NodeAttrs& attrs,
                               std::vector<int> *in_attrs,
                               std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  CHECK_NE((*in_attrs)[sr::kIdx], -1) << "Index type must be set for sparse_retain operator";

  TYPE_ASSIGN_CHECK(*out_attrs, 0, (*in_attrs)[sr::kArr]);
  TYPE_ASSIGN_CHECK(*in_attrs, 0, (*out_attrs)[sr::kOut]);
  return (*in_attrs)[0] != -1;
}

inline bool SparseRetainForwardInferStorageType(const nnvm::NodeAttrs& attrs,
                                                const int dev_mask,
                                                DispatchMode* dispatch_mode,
                                                std::vector<int> *in_attrs,
                                                std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  bool dispatched = false;
  auto &arr_stype = in_attrs->at(sr::kArr);
  auto &idx_stype = in_attrs->at(sr::kIdx);
  auto &out_stype = out_attrs->at(sr::kOut);
  if (!dispatched && arr_stype == kRowSparseStorage && idx_stype == kDefaultStorage) {
    // rsp, dns -> rsp
    dispatched = storage_type_assign(&out_stype, kRowSparseStorage,
                                     dispatch_mode, DispatchMode::kFComputeEx);
  }
  return dispatched;
}

inline bool SparseRetainBackwardInferStorageType(const nnvm::NodeAttrs& attrs,
                                                 const int dev_mask,
                                                 DispatchMode* dispatch_mode,
                                                 std::vector<int> *in_attrs,
                                                 std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 2U);
  bool dispatched = false;
  const auto &ograd_stype = in_attrs->at(sr::kOut);
  const auto &idx_stype = in_attrs->at(sr::kArr);
  auto &arr_grad_stype = out_attrs->at(sr::kArr);
  auto &idx_grad_stype = out_attrs->at(sr::kIdx);
  if (!dispatched && ograd_stype == kDefaultStorage && idx_stype == kDefaultStorage) {
    if (type_assign(&arr_grad_stype, kRowSparseStorage) &&
        type_assign(&idx_grad_stype, kDefaultStorage)) {
      DISPATCH_MODE_ASSIGN_CHECK(dispatch_mode, 0, DispatchMode::kFComputeEx);
      dispatched = true;
    }
  }
  return dispatched;
}

/*!
 * \brief Each thread searches for a user input index in the input
 * row sparse ndarray alternatively. This ensures each thread
 * has the almost the same workload. The overhead is the binary
 * search. If all the indices of the idx array are contained
 * in the in_idx, one should use SparseRetainRspRowBlockKernel instead,
 * where each thread only perform binary search once.
 */
struct SparseRetainRspThreadKernel {
  template<typename DType, typename RType, typename IType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data, RType* out_idx,
                                  const DType* in_data, const RType* in_idx,
                                  const IType* idx, const size_t nnr,
                                  const size_t row_length) {
    const RType irow = idx[i];
    int j = -1, left = 0, right = nnr - 1;
    while (left <= right) {
      int m = left + (right - left) / 2;
      const auto in_idx_m = in_idx[m];
      if (in_idx_m == irow) {
        j = m;
        break;
      } else if (in_idx_m < irow) {
        left = m + 1;
      } else {
        right = m - 1;
      }
    }
    out_idx[i] = idx[i];
    if (j >= 0) {
      const size_t in_offset = j * row_length;
      const size_t out_offset = i * row_length;
      for (size_t k = 0; k < row_length; ++k) {
        out_data[out_offset+k] = in_data[in_offset+k];
      }
    }
  }
};

/*!
 * \brief This kernel should be invoked when the row indices
 * to be retained are all in the input rsp.
 * Each thread searches for a subarray of indices of
 * the user-input idx array for retain. The first index
 * in the subarray will be searched for using binary search.
 * The rest of the indices will be searched for starting from
 * the lower bound of the binary search. This kernel assumes
 * that idx has been sorted in ascending order.
 */
struct SparseRetainRspRowBlockKernel {
  template<typename DType, typename RType, typename IType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data, RType* out_idx,
                                  const DType* in_data, const RType* in_idx,
                                  const IType* idx, const size_t num_indices,
                                  const size_t nnr, const size_t row_length,
                                  const size_t seg_len) {
    const size_t seg_start = i * seg_len;
    if (seg_start >= num_indices) return;
    const size_t seg_end = (seg_start+seg_len < num_indices? seg_start+seg_len : num_indices);
    for (size_t j = seg_start; j < seg_end; ++j) {
      out_idx[j] = idx[j];
    }
    // use binary search to find the lower bound of idx[seg_start] in in_idx
    const RType* first = in_idx;
    const RType* last = in_idx + nnr;
    const auto val = idx[seg_start];
    const RType* it;
    int count = last - first, step;
    while (count > 0) {
      it = first;
      step = count / 2;
      it += step;
      if (*it < val) {
        first = ++it;
        count -= step + 1;
      } else {
        count = step;
      }
    }
    size_t cur_row_idx = first - in_idx;
    // end of binary search
    if (cur_row_idx == nnr ||  in_idx[cur_row_idx] > idx[seg_end-1]) {
      return;
    }
    size_t cur_idx = seg_start;
    while (cur_row_idx < nnr && cur_idx < seg_end) {
      if (in_idx[cur_row_idx] == idx[cur_idx]) {
        const size_t in_offset = cur_row_idx * row_length;
        const size_t out_offset = cur_idx * row_length;
        for (size_t k = 0; k < row_length; ++k) {
          out_data[out_offset+k] = in_data[in_offset+k];
        }
        ++cur_row_idx;
        ++cur_idx;
      } else if (in_idx[cur_row_idx] < idx[cur_idx]) {
        ++cur_row_idx;
      } else {
        ++cur_idx;
      }
    }
  }
};

/*!
 * Copy input indices to output indices.
 * Only used when input rsp is dense.
 */
struct SparseRetainCopyIndices {
  template<typename RType, typename IType>
  MSHADOW_XINLINE static void Map(int i, RType* out_idx, IType* idx) {
    out_idx[i] = idx[i];
  }
};

/*!
 * Copy input retained rows to output rows.
 * Only used when input rsp is dense.
 * This kernel is only used when ctx is on GPU.
 * So it's parallelized by out_rows' elements,
 * instead of rows.
 */
struct SparseRetainCopyRetainedRowsFromDnsPerElem {
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i, DType* out_rows, const DType* in_rows,
                                  const IType* idx, const size_t row_length) {
    const size_t irow = i / row_length;
    const size_t icol = i % row_length;
    out_rows[i] = in_rows[static_cast<size_t>(idx[irow]) * row_length + icol];
  }
};

/*!
 * Copy input retained rows to output rows.
 * Only used when input rsp is dense.
 * This kernel is only used when ctx is on CPU.
 * So it's parallelized by out_rows' rows instead of elements.
 */
struct SparseRetainCopyRetainedRowsFromDnsPerRow {
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i, DType* out_rows, const DType* in_rows,
                                  const IType* idx, const size_t row_length) {
    const size_t dst_offset = i * row_length;
    const size_t src_offset = static_cast<size_t>(idx[i]) * row_length;
    for (size_t j = 0; j < row_length; j++) {
      out_rows[dst_offset + j] = in_rows[src_offset + j];
    }
  }
};

template<typename xpu>
void SparseRetainOpForwardRspImpl(mshadow::Stream<xpu> *s,
                                  const NDArray& input_nd,
                                  const TBlob& idx_data,
                                  const OpReqType req,
                                  NDArray* output_nd) {
  if (req == kNullOp) return;
  CHECK_EQ(req, kWriteTo) << "SparseRetainOpForwardRspImpl only support req = kWriteTo now";
  CHECK_EQ(input_nd.storage_type(), kRowSparseStorage)
    << "SparseRetainOpForwardRspImpl operator only takes row sparse NDArray as input";
  CHECK_EQ(output_nd->storage_type(), kRowSparseStorage)
    << "SparseRetainOpForwardRspImpl operator only outputs row sparse NDArray";

  if (!input_nd.storage_initialized() || idx_data.Size() == 0U || input_nd.shape()[0] == 0) {
    FillZerosRspImpl(s, *output_nd);
    return;
  }

  const TBlob input_data = input_nd.data();
  const TBlob input_idx = input_nd.aux_data(rowsparse::kIdx);

  output_nd->CheckAndAlloc({mshadow::Shape1(idx_data.Size())});
  TBlob output_data = output_nd->data();
  TBlob output_idx = output_nd->aux_data(rowsparse::kIdx);
  const auto row_length = input_data.shape_.ProdShape(1, input_data.shape_.ndim());

  using namespace mxnet_op;
  MSHADOW_TYPE_SWITCH(output_data.type_flag_, DType, {  // output data type
    Kernel<set_zero, xpu>::Launch(s, output_data.Size(), output_data.dptr<DType>());
    MSHADOW_IDX_TYPE_SWITCH(output_idx.type_flag_, RType, {  // row index data type
      MSHADOW_TYPE_SWITCH(idx_data.type_flag_, IType, {  // index array data type
        if (input_idx.Size() == input_nd.shape()[0]) {  // input rsp is dense
          using namespace mshadow;
          // copy indices
          Tensor<xpu, 1, RType> output_idx_tensor = output_idx.FlatTo1D<xpu, RType>(s);
          const size_t num_rows_retained = output_idx.Size();
          if (output_idx.type_flag_ == idx_data.type_flag_) {  // same type, use Copy
            const Tensor<xpu, 1, RType> idx_tensor = idx_data.FlatTo1D<xpu, RType>(s);
            Copy(output_idx_tensor, idx_tensor, s);
          } else {  // different index types, use Kernel::Launch
            Kernel<SparseRetainCopyIndices, xpu>::Launch(s, num_rows_retained,
                output_idx.dptr<RType>(), idx_data.dptr<IType>());
          }
          // copy data
          if (std::is_same<xpu, cpu>::value) {  // For cpu, parallelize by rows
            Kernel<SparseRetainCopyRetainedRowsFromDnsPerRow, xpu>::Launch(s, idx_data.Size(),
              output_data.dptr<DType>(), input_data.dptr<DType>(),
              idx_data.dptr<IType>(), row_length);
          } else {  // For gpu, parallelize by elements
            Kernel<SparseRetainCopyRetainedRowsFromDnsPerElem, xpu>::Launch(s, output_data.Size(),
              output_data.dptr<DType>(), input_data.dptr<DType>(),
              idx_data.dptr<IType>(), row_length);
          }
        } else {  // input rsp is not dense
          Kernel<SparseRetainRspThreadKernel, xpu>::Launch(s, idx_data.Size(),
              output_data.dptr<DType>(), output_idx.dptr<RType>(), input_data.dptr<DType>(),
              input_idx.dptr<RType>(), idx_data.dptr<IType>(), input_data.shape_[0], row_length);
        }
      });
    });
  });
}

template<typename xpu>
void SparseRetainOpForwardEx(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<NDArray>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  if (req[sr::kOut] == kNullOp) return;
  CHECK_EQ(req[sr::kOut], kWriteTo) << "sparse_retain only supports req=\'write\'";
  CHECK_EQ(inputs[sr::kIdx].storage_type(), kDefaultStorage)
    << "sparse_retain operator only takes default NDArray as its index array";
  if (inputs[sr::kArr].storage_type() == kRowSparseStorage) {
    NDArray output_nd = outputs[sr::kOut];
    SparseRetainOpForwardRspImpl<xpu>(ctx.get_stream<xpu>(), inputs[sr::kArr],
        inputs[sr::kIdx].data(), req[sr::kOut], &output_nd);
  } else {
    LOG(FATAL) << "sparse_retain op only supports row-sparse ndarrays as input";
  }
}

template<int req>
struct SparseRetainRspGradKernel {
  template<typename DType, typename RType, typename IType>
  MSHADOW_XINLINE static void Map(int i, DType* in_grad, RType* in_grad_idx,
                                  const DType* out_grad, const IType* idx,
                                  const size_t row_length) {
    const RType irow = idx[i];
    in_grad_idx[i] = irow;
    const size_t out_offset = irow * row_length;
    const size_t in_offset = i * row_length;
    for (size_t j = 0; j < row_length; ++j) {
      KERNEL_ASSIGN(in_grad[in_offset+j], req, out_grad[out_offset+j]);
    }
  }
};

template<typename xpu>
void SparseRetainOpBackwardEx(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<NDArray>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<NDArray>& outputs) {
  CHECK_EQ(req.size(), 2U);
  CHECK_EQ(req[sr::kIdx], kNullOp);
  if (req[sr::kArr] == kNullOp) return;
  CHECK_EQ(req[sr::kArr], kWriteTo);

  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 2U)
    << "sparse_retain does not support calculating gradients of indices";

  CHECK_EQ(inputs[sr::kOut].storage_type(), kDefaultStorage)
    << "sparse_retain backward only takes default NDArray as ograd";
  CHECK_EQ(inputs[sr::kIdx].storage_type(), kDefaultStorage)
    << "sparse_retain backward only takes default NDArray as its index array";
  CHECK_EQ(outputs[sr::kArr].storage_type(), kRowSparseStorage)
    << "sparse_retain backward only outputs row sparse NDArray as grad of input";

  using namespace mxnet_op;
  using namespace mshadow;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob idx_data = inputs[sr::kIdx].data();
  if (idx_data.Size() == 0U) {
    FillZerosRspImpl(s, outputs[sr::kArr]);
    return;
  }

  const TBlob out_grad_data = inputs[sr::kOut].data();

  NDArray in_grad_nd = outputs[sr::kArr];
  in_grad_nd.CheckAndAlloc({mshadow::Shape1(idx_data.Size())});
  TBlob in_grad_data = in_grad_nd.data();
  TBlob in_grad_idx = in_grad_nd.aux_data(rowsparse::kIdx);
  const auto row_length = out_grad_data.shape_.ProdShape(1, out_grad_data.shape_.ndim());

  MSHADOW_TYPE_SWITCH(out_grad_data.type_flag_, DType, {  // output data type
    MSHADOW_IDX_TYPE_SWITCH(in_grad_idx.type_flag_, RType, {  // row index data type
      MSHADOW_TYPE_SWITCH(idx_data.type_flag_, IType, {  // index array data type
        MXNET_ASSIGN_REQ_SWITCH(req[sr::kArr], req_type, {
          Kernel<SparseRetainRspGradKernel<req_type>, xpu>::Launch(
              s, in_grad_idx.Size(), in_grad_data.dptr<DType>(), in_grad_idx.dptr<RType>(),
              out_grad_data.dptr<DType>(), idx_data.dptr<IType>(), row_length);
        });
      });
    });
  });
}


}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_SPARSE_RETAIN_INL_H_
