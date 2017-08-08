/*!
 * Copyright (c) 2017 by Contributors
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
                                                const Context& ctx,
                                                std::vector<int> *in_attrs,
                                                std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  type_assign(&(in_attrs->at(sr::kArr)), kRowSparseStorage);
  type_assign(&(in_attrs->at(sr::kIdx)), kDefaultStorage);
  type_assign(&(out_attrs->at(sr::kOut)), kRowSparseStorage);
  return true;
}

inline bool SparseRetainBackwardInferStorageType(const nnvm::NodeAttrs& attrs,
                                                 const Context& ctx,
                                                 std::vector<int> *in_attrs,
                                                 std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 2U);

  type_assign(&(in_attrs->at(sr::kOut)), kDefaultStorage);
  type_assign(&(in_attrs->at(sr::kIdx)), kDefaultStorage);
  type_assign(&(out_attrs->at(sr::kArr)), kRowSparseStorage);
  type_assign(&(out_attrs->at(sr::kIdx)), kDefaultStorage);
  return true;
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
 * \brief This kernel is invoked when the input row-sparse
 * is actually dense.
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

template<typename xpu>
void SparseRetainOpForwardRspImpl(mshadow::Stream<xpu> *s,
                                  const NDArray& input_nd,
                                  const TBlob& idx_data,
                                  const OpReqType req,
                                  NDArray* output_nd) {
  if (req == kNullOp) return;
  CHECK_EQ(input_nd.storage_type(), kRowSparseStorage)
    << "SparseRetainOpForwardRspImpl operator only takes row sparse NDArray as input";
  CHECK_EQ(output_nd->storage_type(), kRowSparseStorage)
    << "SparseRetainOpForwardRspImpl operator only outputs row sparse NDArray";

  if (!input_nd.storage_initialized()
      || idx_data.Size() == 0U
      || input_nd.shape()[0] == 0) {
    FillZerosRspImpl(s, output_nd);
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
          int num_threads = get_num_threads<xpu>(idx_data.Size());
          size_t seg_len = (idx_data.Size() + num_threads - 1) / num_threads;
          Kernel<SparseRetainRspRowBlockKernel, xpu>::Launch(s, num_threads,
              output_data.dptr<DType>(), output_idx.dptr<RType>(), input_data.dptr<DType>(),
              input_idx.dptr<RType>(), idx_data.dptr<IType>(), idx_data.Size(),
              input_data.shape_[0], row_length, seg_len);
        } else {
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
    NDArray output = outputs[sr::kArr];
    FillZerosRspImpl<xpu>(s, &output);
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
