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
 *  Copyright (c) 2016 by Contributors
 * \file ordering_op-inl.h
 * \brief Function definition of ordering operators
 */
#ifndef MXNET_OPERATOR_TENSOR_ORDERING_OP_INL_H_
#define MXNET_OPERATOR_TENSOR_ORDERING_OP_INL_H_

#include <mxnet/operator_util.h>
#include <dmlc/optional.h>
#include <mshadow/tensor.h>
#include <algorithm>
#include <vector>
#include <type_traits>
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "./sort_op.h"
#include "./indexing_op.h"

namespace mshadow {
template<typename xpu, int src_dim, typename DType, int dst_dim>
inline Tensor<xpu, dst_dim, DType> inplace_reshape(Tensor<xpu, src_dim, DType> src,
                                                   Shape<dst_dim> target_shape) {
  CHECK_EQ(src.CheckContiguous(), true);
  return Tensor<xpu, dst_dim, DType>(src.dptr_, target_shape, src.stream_);
}
};


namespace mxnet {
namespace op {
// These enums are only visible within this header
namespace topk_enum {
enum TopKReturnType {kReturnValue, kReturnIndices, kReturnMask, kReturnBoth};
}  // topk_enum

struct TopKParam : public dmlc::Parameter<TopKParam> {
  dmlc::optional<int> axis;
  int k;
  int ret_typ;
  bool is_ascend;
  int dtype;
  DMLC_DECLARE_PARAMETER(TopKParam) {
    DMLC_DECLARE_FIELD(axis).set_default(dmlc::optional<int>(-1))
    .describe("Axis along which to choose the top k indices."
              " If not given, the flattened array is used. Default is -1.");
    DMLC_DECLARE_FIELD(k).set_default(1)
    .describe("Number of top elements to select,"
              " should be always smaller than or equal to the element number in the given axis."
              " A global sort is performed if set k < 1.");
    DMLC_DECLARE_FIELD(ret_typ).set_default(topk_enum::kReturnIndices)
    .add_enum("value", topk_enum::kReturnValue)
    .add_enum("indices", topk_enum::kReturnIndices)
    .add_enum("mask", topk_enum::kReturnMask)
    .add_enum("both", topk_enum::kReturnBoth)
    .describe("The return type.\n"
        " \"value\" means to return the top k values,"
        " \"indices\" means to return the indices of the top k values,"
        " \"mask\" means to return a mask array containing 0 and 1. 1 means the top k values."
        " \"both\" means to return a list of both values and indices of top k elements.");
    DMLC_DECLARE_FIELD(is_ascend).set_default(false)
      .describe("Whether to choose k largest or k smallest elements."
                " Top K largest elements will be chosen if set to false.");
    DMLC_DECLARE_FIELD(dtype)
    // TODO(srivrohi): remove support for real data type in mxnet-2.0
    .add_enum("uint8", mshadow::kUint8)
    .add_enum("int32", mshadow::kInt32)
    .add_enum("int64", mshadow::kInt64)
    .add_enum("float16", mshadow::kFloat16)
    .add_enum("float32", mshadow::kFloat32)
    .add_enum("float64", mshadow::kFloat64)
    .set_default(mshadow::kFloat32)
    .describe("DType of the output indices when ret_typ is \"indices\" or \"both\". "
              "An error will be raised if the selected data type cannot precisely represent the "
              "indices.");
  }
};

struct SortParam : public dmlc::Parameter<SortParam> {
  dmlc::optional<int> axis;
  bool is_ascend;
  DMLC_DECLARE_PARAMETER(SortParam) {
    DMLC_DECLARE_FIELD(axis).set_default(dmlc::optional<int>(-1))
    .describe("Axis along which to choose sort the input tensor."
              " If not given, the flattened array is used. Default is -1.");
    DMLC_DECLARE_FIELD(is_ascend).set_default(true)
      .describe("Whether to sort in ascending or descending order.");
  }
};

struct ArgSortParam : public dmlc::Parameter<ArgSortParam> {
  dmlc::optional<int> axis;
  bool is_ascend;
  int dtype;
  DMLC_DECLARE_PARAMETER(ArgSortParam) {
    DMLC_DECLARE_FIELD(axis).set_default(dmlc::optional<int>(-1))
    .describe("Axis along which to sort the input tensor."
              " If not given, the flattened array is used. Default is -1.");
    DMLC_DECLARE_FIELD(is_ascend).set_default(true)
      .describe("Whether to sort in ascending or descending order.");
    DMLC_DECLARE_FIELD(dtype)
    // TODO(srivrohi): remove support for real data type in mxnet-2.0
    .add_enum("uint8", mshadow::kUint8)
    .add_enum("int32", mshadow::kInt32)
    .add_enum("int64", mshadow::kInt64)
    .add_enum("float16", mshadow::kFloat16)
    .add_enum("float32", mshadow::kFloat32)
    .add_enum("float64", mshadow::kFloat64)
    .set_default(mshadow::kFloat32)
    .describe("DType of the output indices. It is only valid when ret_typ is \"indices\" or"
              " \"both\". An error will be raised if the selected data type cannot precisely "
              "represent the indices.");
  }
};

inline void ParseTopKParam(const TShape& src_shape,
                           const TopKParam& param,
                           TShape *target_shape,
                           size_t *batch_size,
                           index_t *element_num,
                           int *axis,
                           index_t *k,
                           bool *do_transpose,
                           bool *is_ascend) {
  *do_transpose = false;
  *k = param.k;
  *is_ascend = param.is_ascend;
  // get batch_size, axis and element_num
  if (!static_cast<bool>(param.axis)) {  // No axis given
    *axis = 0;
    *batch_size = 1;
    *element_num = src_shape.Size();
  } else {
    *axis = param.axis.value();
    if (*axis < 0) {
      *axis += src_shape.ndim();
    }
    CHECK(*axis >= 0 && *axis < static_cast<int>(src_shape.ndim()))
                                                  << "Invalid axis! axis should be between 0 and "
                                                  << src_shape.ndim() << ", found axis=" << *axis;
    *batch_size = src_shape.Size() / src_shape[*axis];
    *element_num = src_shape[*axis];
    if (*axis != src_shape.ndim() - 1) {
      *do_transpose = true;
    }
  }
  // get k
  if (param.k <= 0) {
    *k = *element_num;
  }
  // get target_shape
  if (!static_cast<bool>(param.axis)) {
    if (param.ret_typ != topk_enum::kReturnMask) {
      *target_shape = mshadow::Shape1(*k);
    } else {
      *target_shape = src_shape;
    }
  } else {
    *target_shape = src_shape;
    if (param.ret_typ != topk_enum::kReturnMask) {
      (*target_shape)[*axis] = *k;
    }
  }
  CHECK(*k >= 1 && *k <= *element_num) << "k must be smaller than "
                                      << *element_num << ", get k = " << *k;
}

using namespace mshadow;


struct fill_ind_to_one {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, const index_t* indices, DType* out) {
    out[indices[i]] = static_cast<DType>(1);
  }
};

struct fill_ind {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, const index_t* indices, const DType* val,
                                  int req, DType* out) {
    KERNEL_ASSIGN(out[indices[i]], req, val[i]);
  }
};

template<typename DType>
MSHADOW_FORCE_INLINE void TopKSort(const Tensor<cpu, 1, DType>& dat,
                                   const Tensor<cpu, 1, index_t>& ind,
                                   const Tensor<cpu, 1, char>& work,
                                   index_t K, index_t N, bool is_ascend,
                                   Stream<cpu> *s) {
  // Use full sort when K is relatively large.
  const bool full_sort(K*8 > N);
  // Batch size.
  const index_t M(work.size(0)/(sizeof(DType)*N));
  const int omp_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount());
  #pragma omp parallel for num_threads(omp_threads)
  for (index_t i = 0; i < M; ++i) {
    // Tensor `work` stores the flattened source data, while `dat` stores the sorted result.
    DType *vals = reinterpret_cast<DType*>(work.dptr_);
    DType *sorted_vals = dat.dptr_+i*N;
    index_t *indices = ind.dptr_+i*N;
    if (is_ascend) {
      if (full_sort) {
        std::sort(indices, indices+N,
                  [&](const index_t& i1, const index_t& i2){
          return vals[i1] < vals[i2]; });
      } else {
        std::partial_sort(indices, indices+K, indices+N,
                          [&](const index_t& i1, const index_t& i2){
          return vals[i1] < vals[i2]; });
      }
    } else {
      if (full_sort) {
        std::sort(indices, indices+N,
                  [&](const index_t& i1, const index_t& i2){
          return vals[i1] > vals[i2]; });
      } else {
        std::partial_sort(indices, indices+K, indices+N,
                          [&](const index_t& i1, const index_t& i2){
          return vals[i1] > vals[i2]; });
      }
    }
    for (index_t j = 0; j < K; ++j) {
      sorted_vals[j] = vals[indices[j]];
    }
  }
}

#ifdef __CUDACC__

template<typename DType>
MSHADOW_XINLINE bool TopKCompare(DType val1, index_t ind1, DType val2, index_t ind2,
                                 bool is_ascend) {
  // Negative indices denote undefined values which are considered arbitrary small resp. large.
  return (ind2 < 0) || (ind1 >= 0 && ((is_ascend && val1 < val2) || (!is_ascend && val1 > val2)));
}

template<typename DType>
MSHADOW_XINLINE void MergeTopK(index_t K, DType *val1, index_t *ind1, DType *val2, index_t *ind2,
                               bool is_ascend) {
  // In-place merge of two sorted top-K lists into val1/ind1. First determine the intervals
  // [0,..,i1], [0,..i2] of the two lists that will be part of the merged list.
  index_t i1(K-1), i2(K-1);
  for (index_t i = 0; i < K; ++i) {
    if (TopKCompare(val1[i1], ind1[i1], val2[i2], ind2[i2], is_ascend)) {
      --i2;
    } else {
      --i1;
    }
  }
  // Now merge the lists from back to front.
  for (index_t i = K; i--;) {
    if (i2 < 0 || i1 >= 0 && TopKCompare(val2[i2], ind2[i2], val1[i1], ind1[i1], is_ascend)) {
      val1[i] = val1[i1];
      ind1[i] = ind1[i1];
      --i1;
    } else {
      val1[i] = val2[i2];
      ind1[i] = ind2[i2];
      --i2;
    }
  }
}

template<typename DType>
__global__ void PartialSortSmallK(index_t K, index_t N, DType *val, index_t *ind, bool is_ascend) {
  // Buffer for pairwise reduction.
  extern __shared__ index_t buff[];
  // Start of buffer sections associated with this thread.
  const index_t offset(threadIdx.x*K);
  index_t *ind_buff = &buff[offset];
  DType *val_buff = reinterpret_cast<DType*>(&buff[blockDim.x*K])+offset;
  // Initialize top-K values for this thread.
  for (index_t i = 0; i < K; ++i) {
    ind_buff[i] = -1;
  }
  // Range of values this thread cares about. Each thread block processes
  // a different batch item (i.e. a different set of ind/val where we
  // have to select the top-K elements). All threads within the same
  // block work on the same batch item.
  const index_t first(blockIdx.x*N+threadIdx.x), last((blockIdx.x+1)*N);
  // Select top-K from this range and store it sorted in the buffer.
  // We assume a small K, so linear insertion is o.k.
  for (index_t i = first; i < last; i += blockDim.x) {
    DType cur_val(val[i]);
    index_t cur_ind(ind[i]);
    for (index_t j = K; j-- && TopKCompare(cur_val, cur_ind, val_buff[j],
                                           ind_buff[j], is_ascend); ) {
      if (j+1 < K) {
        val_buff[j+1] = val_buff[j];
        ind_buff[j+1] = ind_buff[j];
      }
      val_buff[j] = cur_val;
      ind_buff[j] = cur_ind;
    }
  }
  // Recursive merge of sorted lists for this thread block. Note that blockDim.x is not
  // necessary a power of two, therefore the additional checks for last_s.
  for (index_t s = (blockDim.x+1)/2, last_s = blockDim.x;
       last_s > 1; last_s = s, s = (s+1)/2) {
    __syncthreads();
    if (threadIdx.x < s && threadIdx.x+s < last_s) {
      MergeTopK(K, val_buff, ind_buff, val_buff+s*K, ind_buff+s*K, is_ascend);
    }
  }
  // Final updates on master thread.
  if (threadIdx.x == 0) {
    for (index_t i = 0; i < K; ++i) {
      ind[blockIdx.x*N+i] = ind_buff[i];
      val[blockIdx.x*N+i] = val_buff[i];
    }
  }
}

template<typename DType>
MSHADOW_FORCE_INLINE void TopKSort(const Tensor<gpu, 1, DType>& dat,
                                   const Tensor<gpu, 1, index_t>& ind,
                                   const Tensor<gpu, 1, char>& work,
                                   index_t K, index_t N, bool is_ascend,
                                   Stream<gpu> *s) {
  // Use full sort for all but very small K for which we
  // can do a partial sort entirely within shared memory.
  const bool full_sort(K > 5);
  // Batch size.
  const index_t M(dat.size(0)/N);
  if (full_sort) {
    // Divide workspace into two parts. The first one is needed to store batch ids.
    size_t alignment = std::max(sizeof(DType), sizeof(index_t));
    size_t id_size = PadBytes(sizeof(index_t) * ind.size(0), alignment);
    Tensor<gpu, 1, index_t> batch_id(reinterpret_cast<index_t*>(work.dptr_),
                                     Shape1(ind.size(0)), s);
    Tensor<gpu, 1, char> sort_work(work.dptr_+id_size, Shape1(work.size(0)-id_size), s);
    mxnet::op::SortByKey(dat, ind, is_ascend, &sort_work);
    if (M > 1) {
      // Back to back sorting. Note that mxnet::op::SortByKey is a stable sort.
      batch_id = ind / N;
      mxnet::op::SortByKey(batch_id, dat, true, &sort_work);
      batch_id = ind / N;
      mxnet::op::SortByKey(batch_id, ind, true, &sort_work);
    }
  } else {
    const int nthreads(mshadow::cuda::kBaseThreadNum);
    PartialSortSmallK<<<M, nthreads, nthreads*K*(sizeof(int)+sizeof(DType)),
                        mshadow::Stream<gpu>::GetStream(s)>>>
                        (K, N, dat.dptr_, ind.dptr_, is_ascend);
  }
}

#endif


/*!
   * \brief Implementation of the TopK operation
   *
   *
   * \param ctx the running context
   * \param resource temporary resource handler
   * \param src the Source blob
   * \param ret the destination blobs
   * \param k the K elements to keep
   * \param param the topk parameters
   * \tparam xpu the device type.
   * \tparam DType type of the output value/mask.
   * \tparam IDType type of the output indices.
   */
template<typename xpu, typename DType, typename IDType>
void TopKImpl(const RunContext &ctx,
              const Resource &resource,
              const std::vector<OpReqType>& req,
              const TBlob& src,
              const std::vector<TBlob>& ret,
              const TopKParam& param) {
  using namespace mshadow;
  using namespace mshadow::expr;
  // 1. Parse and initialize information
  Stream<xpu> *s = ctx.get_stream<xpu>();
  Tensor<xpu, 1, char> workspace;
  Tensor<xpu, 1, char> temp_workspace;
  Tensor<xpu, 1, DType> sorted_dat;
  Tensor<xpu, 1, index_t> indices, sel_indices;
  size_t batch_size = 0;
  index_t element_num = 0;  // number of batches + the size of each batch
  int axis = 0;
  bool do_transpose = false;
  bool is_ascend = false;
  index_t k = 0;
  size_t alignment = std::max(sizeof(DType), sizeof(int));
  mxnet::TShape target_shape;
  ParseTopKParam(src.shape_, param,
                 &target_shape, &batch_size, &element_num, &axis, &k, &do_transpose, &is_ascend);
  CHECK_LE(element_num, mxnet::common::MaxIntegerValue<index_t>())
    << "'index_t' does not have a sufficient precision to represent "
    << "the indices of the input array. The total element_num is "
    << element_num << ", but the selected index_t can only represent "
    << mxnet::common::MaxIntegerValue<index_t>() << " elements";
  Tensor<xpu, 3, DType> dat = src.FlatTo3D<xpu, DType>(axis, axis, s);
  size_t temp_size = 0;
  // Temp space needed by the gpu-based full sorts.
  temp_size = std::max(temp_size,
    mxnet::op::SortByKeyWorkspaceSize<int, int, xpu>(src.Size()));
  temp_size = std::max(temp_size,
    mxnet::op::SortByKeyWorkspaceSize<int, DType, xpu>(src.Size()));
  temp_size = std::max(temp_size,
    mxnet::op::SortByKeyWorkspaceSize<DType, int, xpu>(src.Size()));
  // Additional temp space for gpu full sorts for batch ids.
  temp_size += PadBytes(sizeof(index_t) * src.Size(), alignment);
  // Temp space for cpu sorts.
  temp_size = std::max(temp_size, static_cast<size_t>(sizeof(DType) * src.Size()));
  size_t workspace_size = temp_size + PadBytes(sizeof(DType) * src.Size(), alignment)
                                    + PadBytes(sizeof(index_t) * src.Size(), alignment);
  if (param.ret_typ == topk_enum::kReturnMask) {
    workspace_size += PadBytes(sizeof(int) * batch_size * k, alignment);
  }
  workspace = resource.get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);
  char* workspace_curr_ptr = workspace.dptr_;
  sorted_dat = Tensor<xpu, 1, DType>(reinterpret_cast<DType*>(workspace_curr_ptr),
                                      Shape1(src.Size()), s);  // contain sorted dat
  workspace_curr_ptr += PadBytes(sizeof(DType) * src.Size(), alignment);
  indices = Tensor<xpu, 1, index_t>(reinterpret_cast<index_t*>(workspace_curr_ptr),
                                Shape1(src.Size()), s);  // indices in the original matrix
  workspace_curr_ptr += PadBytes(sizeof(index_t) * src.Size(), alignment);

  if (param.ret_typ == topk_enum::kReturnMask) {
    sel_indices = Tensor<xpu, 1, index_t>(reinterpret_cast<index_t*>(workspace_curr_ptr),
                                      Shape1(batch_size * k), s);
    workspace_curr_ptr += PadBytes(sizeof(index_t) * batch_size * k, alignment);
    CHECK_EQ(sel_indices.CheckContiguous(), true);
  }

  if (std::is_same<xpu, cpu>::value) {
    Tensor<xpu, 1, DType> flattened_data;
    if (do_transpose) {
      flattened_data = Tensor<xpu, 1, DType>(reinterpret_cast<DType*>(workspace_curr_ptr),
                                              Shape1(src.Size()), s);
      workspace_curr_ptr += sizeof(DType) * src.Size();
      flattened_data = reshape(transpose(dat, Shape3(0, 2, 1)), Shape1(src.Size()));
      CHECK_EQ(flattened_data.CheckContiguous(), true);
    } else {
      flattened_data = src.FlatTo1D<xpu, DType>(s);
    }
    // `temp_workspace` stores the flattened data
    temp_workspace = Tensor<xpu, 1, char>(reinterpret_cast<char*>(flattened_data.dptr_),
                                          Shape1(sizeof(DType)*src.Size()), s);
    CHECK_EQ(temp_workspace.CheckContiguous(), true);
  } else {
    if (do_transpose) {
      sorted_dat = reshape(transpose(dat, Shape3(0, 2, 1)), Shape1(src.Size()));
    } else {
      sorted_dat = reshape(dat, Shape1(src.Size()));
    }
    CHECK_EQ(sorted_dat.CheckContiguous(), true);
    temp_workspace = Tensor<xpu, 1, char>(workspace_curr_ptr, Shape1(temp_size), s);  // temp space
    workspace_curr_ptr += temp_size;
  }

  mxnet_op::Kernel<range_fwd, xpu>::Launch(s, batch_size * element_num, 1, index_t{0}, index_t{1},
    kWriteTo, indices.dptr_);
  CHECK_EQ(indices.CheckContiguous(), true);

  // 2. Perform inplace batch sort.
  // After sorting, each batch in `sorted_dat` will be sorted in the corresponding order
  // up to the k-th element and the `indices` will contain the corresponding index in `sorted_dat`
  // `temp_workspace` is used to store the flattend source data for CPU device, and it's used as
  // a temporal buffer for GPU device.
  TopKSort(sorted_dat, indices, temp_workspace, k, element_num, is_ascend, s);

  // 3. Assign results to the ret blob
  // When returning indices, only update(modulo) required elements instead of full elements
  // to avoid redundant calculation.
  // Cast `ret_indices` from int to real_t could introduce conversion error when the element_num
  // is large enough.
  if (param.ret_typ == topk_enum::kReturnMask) {
    Tensor<xpu, 1, DType> ret_mask = ret[0].FlatTo1D<xpu, DType>(s);
    ret_mask = scalar<DType>(0);
    sel_indices = reshape(slice<1>(
                              inplace_reshape(indices,
                                              Shape2(batch_size,
                                                     element_num)), 0, k),
                              Shape1(batch_size * k));
    if (do_transpose) {
      mxnet::TShape src_shape = src.shape_.FlatTo3D(axis);
      CHECK_EQ(sel_indices.CheckContiguous(), true);
      sel_indices = transpose_indices(sel_indices, Shape3(src_shape[0], src_shape[2], src_shape[1]),
                                      Shape3(0, 2, 1));
    }
    if (req[0] == kNullOp) {
      return;
    } else if (req[0] == kWriteTo) {
      mxnet_op::Kernel<fill_ind_to_one, xpu>::Launch(s, batch_size * k,
                                                     sel_indices.dptr_, ret_mask.dptr_);
    } else {
      LOG(FATAL) << "req=" << req[0] << " is not supported yet.";
    }
  } else if (param.ret_typ == topk_enum::kReturnIndices) {
    if (do_transpose) {
      Tensor<xpu, 3, IDType> ret_indices = ret[0].FlatTo3D<xpu, IDType>(axis, axis, s);
      ASSIGN_DISPATCH(ret_indices, req[0], tcast<IDType>(F<mshadow_op::mod>(transpose(
                      slice<2>(inplace_reshape(indices,
                                               Shape3(ret_indices.shape_[0],
                                                      ret_indices.shape_[2],
                                                      element_num)),
                               0, k),
                      Shape3(0, 2, 1)), element_num)));
    } else {
      Tensor<xpu, 2, IDType> ret_indices =
        ret[0].get_with_shape<xpu, 2, IDType>(Shape2(batch_size, k), s);
      ASSIGN_DISPATCH(ret_indices, req[0], tcast<IDType>(F<mshadow_op::mod>(slice<1>(
                      inplace_reshape(indices, Shape2(batch_size, element_num)), 0, k),
                      element_num)));
    }
  } else {
    if (do_transpose) {
      Tensor<xpu, 3, DType> ret_value = ret[0].FlatTo3D<xpu, DType>(axis, axis, s);
      Tensor<xpu, 3, IDType> ret_indices = ret[1].FlatTo3D<xpu, IDType>(axis, axis, s);
      ASSIGN_DISPATCH(ret_value, req[0], transpose(
                   slice<2>(inplace_reshape(sorted_dat,
                                    Shape3(ret_value.shape_[0], ret_value.shape_[2], element_num)),
                            0, k), Shape3(0, 2, 1)));
      ASSIGN_DISPATCH(ret_indices, req[1], tcast<IDType>(F<mshadow_op::mod>(transpose(
                      slice<2>(inplace_reshape(indices,
                                               Shape3(ret_indices.shape_[0],
                                                      ret_indices.shape_[2],
                                                      element_num)),
                               0, k), Shape3(0, 2, 1)), element_num)));
    } else {
      Tensor<xpu, 2, DType> ret_value =
        ret[0].get_with_shape<xpu, 2, DType>(Shape2(batch_size, k), s);
      Tensor<xpu, 2, IDType> ret_indices =
        ret[1].get_with_shape<xpu, 2, IDType>(Shape2(batch_size, k), s);
      ASSIGN_DISPATCH(ret_value, req[0],
             slice<1>(inplace_reshape(sorted_dat, Shape2(batch_size, element_num)), 0, k));
      ASSIGN_DISPATCH(ret_indices, req[1], tcast<IDType>(F<mshadow_op::mod>(slice<1>(
                 inplace_reshape(indices, Shape2(batch_size, element_num)), 0, k), element_num)));
    }
  }
}

template<typename xpu>
void TopK(const nnvm::NodeAttrs& attrs,
          const OpContext& ctx,
          const std::vector<TBlob>& inputs,
          const std::vector<OpReqType>& req,
          const std::vector<TBlob>& outputs) {
  const TopKParam& param = nnvm::get<TopKParam>(attrs.parsed);
  if (param.ret_typ == topk_enum::kReturnIndices || param.ret_typ == topk_enum::kReturnBoth) {
    MXNET_NO_FLOAT16_TYPE_SWITCH(inputs[0].type_flag_, DType, {
      MSHADOW_TYPE_SWITCH(param.dtype, IDType, {
        TopKImpl<xpu, DType, IDType>(ctx.run_ctx, ctx.requested[0], req, inputs[0], outputs, param);
      })
    });
  } else {
    MXNET_NO_FLOAT16_TYPE_SWITCH(inputs[0].type_flag_, DType, {
      TopKImpl<xpu, DType, index_t>(ctx.run_ctx, ctx.requested[0], req, inputs[0], outputs, param);
    });
  }
}

template<typename xpu>
void Sort(const nnvm::NodeAttrs& attrs,
          const OpContext& ctx,
          const std::vector<TBlob>& inputs,
          const std::vector<OpReqType>& req,
          const std::vector<TBlob>& outputs) {
  const SortParam& param = nnvm::get<SortParam>(attrs.parsed);
  TopKParam topk_param;
  topk_param.axis = param.axis;
  topk_param.is_ascend = param.is_ascend;
  topk_param.k = 0;
  topk_param.ret_typ = topk_enum::kReturnValue;
  MXNET_NO_FLOAT16_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    TopKImpl<xpu, DType, index_t>(ctx.run_ctx, ctx.requested[0], req, inputs[0],
                                  outputs, topk_param);
  });
}

template<typename xpu>
void ArgSort(const nnvm::NodeAttrs& attrs,
             const OpContext& ctx,
             const std::vector<TBlob>& inputs,
             const std::vector<OpReqType>& req,
             const std::vector<TBlob>& outputs) {
  const ArgSortParam& param = nnvm::get<ArgSortParam>(attrs.parsed);

  if (inputs[0].shape_.ndim() == 0) {
  // Scalar tensor only accept axis of value 0, -1 or None
    CHECK(!static_cast<bool>(param.axis) || param.axis.value() == -1 || param.axis.value() == 0)
      << "Axis can only be -1 or 0 for scalor tensor";
    MSHADOW_TYPE_SWITCH(param.dtype, DType, {
      Stream<xpu> *s = ctx.get_stream<xpu>();
      Tensor<xpu, 1, DType> outdata = outputs[0].get_with_shape<xpu, 1, DType>(Shape1(1), s);
      ASSIGN_DISPATCH(outdata, OpReqType::kWriteTo, 0);
    });
  } else if (inputs[0].shape_.Size() == 0) {
    // If the input tensor is zero size, only a check on axis is needed
    if (static_cast<bool>(param.axis)) {
      int axis = param.axis.value();
      if (axis < 0) axis += inputs[0].shape_.ndim();
      CHECK(axis >= 0 && axis < inputs[0].shape_.ndim())
        << "Axis must be within the range of input tensor's dimension";
    }
  } else {
    TopKParam topk_param;
    topk_param.axis = param.axis;
    topk_param.is_ascend = param.is_ascend;
    topk_param.k = 0;
    topk_param.dtype = param.dtype;
    topk_param.ret_typ = topk_enum::kReturnIndices;
    MXNET_NO_FLOAT16_TYPE_SWITCH(inputs[0].type_flag_, DType, {
      MSHADOW_TYPE_SWITCH(param.dtype, IDType, {
        TopKImpl<xpu, DType, IDType>(ctx.run_ctx,
                                     ctx.requested[0], req, inputs[0], outputs, topk_param);
      });
    });
  }
}

template<typename xpu, typename DType, typename IDType>
void TopKBackwardImpl(const OpContext &ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs,
                      const TopKParam& param) {
  CHECK_NE(req[0], kWriteInplace);
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.run_ctx.get_stream<xpu>();
  CHECK(param.ret_typ == topk_enum::kReturnValue || param.ret_typ == topk_enum::kReturnBoth);
  size_t batch_size = 0;
  index_t element_num = 0;  // number of batches + the size of each batch
  int axis = 0;
  bool do_transpose = false;
  bool is_ascend = false;
  index_t k = 0;
  mxnet::TShape target_shape;
  ParseTopKParam(outputs[0].shape_, param,
                 &target_shape, &batch_size, &element_num, &axis, &k, &do_transpose, &is_ascend);
  CHECK_LE(element_num, mxnet::common::MaxIntegerValue<IDType>())
    << "'IDType' does not have a sufficient precision to represent "
    << "the indices of the input array. The total element_num is " << element_num
    << ", but the selected index_t can only represent "
    << mxnet::common::MaxIntegerValue<IDType>() << " elements";
  Tensor<xpu, 1, index_t> workspace =
    ctx.requested[0].get_space_typed<xpu, 1, index_t>(Shape1(batch_size * k + batch_size), s);
  Tensor<xpu, 1, index_t> sel_indices =
    Tensor<xpu, 1, index_t>(workspace.dptr_, Shape1(batch_size * k), s);
  Tensor<xpu, 1, index_t> batch_shift =
    Tensor<xpu, 1, index_t>(workspace.dptr_ + batch_size * k, Shape1(batch_size), s);

  Tensor<xpu, 2, DType> out_grad =
    inputs[0].get_with_shape<xpu, 2, DType>(Shape2(inputs[0].shape_.Size(), 1), s);
  Tensor<xpu, 2, DType> in_grad =
    outputs[0].get_with_shape<xpu, 2, DType>(Shape2(outputs[0].shape_.Size(), 1), s);
  mxnet_op::Kernel<range_fwd, xpu>::Launch(s, batch_size, 1, index_t{0}, element_num, kWriteTo,
                                           batch_shift.dptr_);
  if (do_transpose) {
    Tensor<xpu, 1, IDType> indices = inputs[2].FlatTo1D<xpu, IDType>(s);
    mxnet::TShape src_shape = outputs[0].shape_.FlatTo3D(axis);
    sel_indices = reshape(transpose(
                            broadcast_to(inplace_reshape(batch_shift,
                                                         Shape3(src_shape[0], src_shape[2], 1)),
                                         mxnet::TShape(Shape3(src_shape[0], src_shape[2], k))),
                            Shape3(0, 2, 1)),
                          Shape1(batch_size * k));
    sel_indices += tcast<index_t>(indices);
    sel_indices = transpose_indices(sel_indices, Shape3(src_shape[0], src_shape[2], src_shape[1]),
                                    Shape3(0, 2, 1));
  } else {
    Tensor<xpu, 2, IDType> indices =
      inputs[2].get_with_shape<xpu, 2, IDType>(Shape2(batch_size, k), s);
    sel_indices = reshape(tcast<index_t>(indices) +
                          broadcast_to(inplace_reshape(batch_shift, Shape2(batch_size, 1)),
                                       mxnet::TShape(Shape2(batch_size, k))),
                          Shape1(batch_size * k));
  }
  CHECK_EQ(sel_indices.CheckContiguous(), true);
  if (kWriteTo == req[0] || kAddTo == req[0]) {
    if (kWriteTo == req[0]) {
      in_grad = scalar<DType>(0);
    }
    mxnet_op::Kernel<fill_ind, xpu>::Launch(s, batch_size * k,
                                            sel_indices.dptr_,
                                            out_grad.dptr_,
                                            req[0],
                                            in_grad.dptr_);
  } else {
    LOG(FATAL) << "Not Implemented!";
  }
}

template<typename xpu>
void TopKBackward_(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
  const TopKParam& param = nnvm::get<TopKParam>(attrs.parsed);
  if (param.ret_typ == topk_enum::kReturnBoth) {
    MXNET_NO_FLOAT16_TYPE_SWITCH(inputs[0].type_flag_, DType, {
      MSHADOW_TYPE_SWITCH(param.dtype, IDType, {
        TopKBackwardImpl<xpu, DType, IDType>(ctx, inputs, req, outputs, param);
      });
    });
  } else if (param.ret_typ == topk_enum::kReturnValue) {
    MXNET_NO_FLOAT16_TYPE_SWITCH(inputs[0].type_flag_, DType, {
      TopKBackwardImpl<xpu, DType, index_t>(ctx, inputs, req, outputs, param);
    });
  } else {
    LOG(FATAL) << "Not Implemented";
  }
}

inline uint32_t TopKNumOutputs(const NodeAttrs& attrs) {
  const TopKParam& param = nnvm::get<TopKParam>(attrs.parsed);
  if (param.ret_typ == topk_enum::kReturnIndices ||
    param.ret_typ == topk_enum::kReturnMask) {
    return static_cast<uint32_t>(1);
  } else {
    return static_cast<uint32_t>(2);
  }
}

inline uint32_t TopKNumVisibleOutputs(const NodeAttrs& attrs) {
  const TopKParam& param = nnvm::get<TopKParam>(attrs.parsed);
  if (param.ret_typ == topk_enum::kReturnBoth) {
    return static_cast<uint32_t>(2);
  } else {
    return static_cast<uint32_t>(1);
  }
}

inline bool TopKType(const nnvm::NodeAttrs& attrs,
                     std::vector<int> *in_attrs,
                     std::vector<int> *out_attrs) {
  const TopKParam& param = nnvm::get<TopKParam>(attrs.parsed);
  int data_type = -1;
  size_t in_size = in_attrs->size();
  size_t out_size = out_attrs->size();
  CHECK_EQ(in_size, 1);
  CHECK(out_size == 1 || out_size == 2);
  //  out_attr[0] -> stores value
  //  out_attr[1] -> stores indices
  if (out_size > 1) {
    if (param.ret_typ == topk_enum::kReturnValue) {
#if MXNET_USE_INT64_TENSOR_SIZE == 1
      CHECK(type_assign(&(*out_attrs)[1], mshadow::kInt64))
#else
      CHECK(type_assign(&(*out_attrs)[1], mshadow::kInt32))
#endif
          << "Failed to set the type of ret_indices.";
    } else {
      CHECK(type_assign(&(*out_attrs)[1], param.dtype))
          << "Failed to set the type of ret_indices.";
    }
  }
  if (param.ret_typ == topk_enum::kReturnIndices) {
    CHECK(type_assign(&(*out_attrs)[0], param.dtype))
            << "Failed to set the type of ret_indices.";
  } else {
    CHECK(type_assign(&data_type, (*in_attrs)[0])) << "Incompatible dtype of input, in_attrs[0]="
                                                   << (*in_attrs)[0];
    CHECK(type_assign(&data_type, (*out_attrs)[0])) << "Incompatible dtype of output, out_attrs[0]="
                                                    << (*out_attrs)[0];
    CHECK(type_assign(&(*in_attrs)[0], data_type)) << "Incompatible dtype of input, in_attrs[0]="
                                                   << (*in_attrs)[0];
    CHECK(type_assign(&(*out_attrs)[0], data_type)) << "Incompatible dtype of output, out_attrs[0]="
                                                    << (*out_attrs)[0];
    if (data_type == -1) return false;
  }
  return true;
}

inline bool TopKShapeImpl(const TopKParam& param,
                          mxnet::ShapeVector *in_attrs,
                          mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  if (param.ret_typ == topk_enum::kReturnIndices ||
    param.ret_typ == topk_enum::kReturnMask) {
    CHECK_EQ(out_attrs->size(), 1U);
  } else {
    CHECK_EQ(out_attrs->size(), 2U);
  }
  mxnet::TShape& in_shape = (*in_attrs)[0];
  size_t batch_size = 0;
  index_t element_num = 0;  // number of batches + the size of each batch
  int axis = 0;
  bool do_transpose = false;
  bool is_ascend = false;
  index_t k = 0;
  mxnet::TShape target_shape;
  ParseTopKParam(in_shape, param,
    &target_shape, &batch_size, &element_num, &axis, &k, &do_transpose, &is_ascend);
  if (param.ret_typ == topk_enum::kReturnIndices ||
    param.ret_typ == topk_enum::kReturnMask) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, target_shape);
  } else {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, target_shape);
    SHAPE_ASSIGN_CHECK(*out_attrs, 1, target_shape);
  }
  return true;
}

inline bool TopKShape(const nnvm::NodeAttrs& attrs,
                      mxnet::ShapeVector *in_attrs,
                      mxnet::ShapeVector *out_attrs) {
  const TopKParam& param = nnvm::get<TopKParam>(attrs.parsed);
  return TopKShapeImpl(param, in_attrs, out_attrs);
}

inline bool SortType(const nnvm::NodeAttrs& attrs,
                     std::vector<int> *in_attrs,
                     std::vector<int> *out_attrs) {
  int data_type = -1;
  size_t in_size = in_attrs->size();
  size_t out_size = out_attrs->size();
  CHECK_EQ(in_size, 1);
  CHECK_EQ(out_size, 2);
#if MXNET_USE_INT64_TENSOR_SIZE == 1
  CHECK(type_assign(&(*out_attrs)[1], mshadow::kInt64))
#else
  CHECK(type_assign(&(*out_attrs)[1], mshadow::kInt32))
#endif
      << "Failed to set the type of ret_indices";
  CHECK(type_assign(&data_type, (*in_attrs)[0])) << "Incompatible dtype of input, in_attrs[0]="
                                                 << (*in_attrs)[0];
  CHECK(type_assign(&data_type, (*out_attrs)[0])) << "Incompatible dtype of output, out_attrs[0]="
                                                  << (*out_attrs)[0];
  CHECK(type_assign(&(*in_attrs)[0], data_type)) << "Incompatible dtype of input, in_attrs[0]="
                                                 << (*in_attrs)[0];
  CHECK(type_assign(&(*out_attrs)[0], data_type)) << "Incompatible dtype of output, out_attrs[0]="
                                                  << (*out_attrs)[0];
  if (data_type == -1) return false;
  return true;
}

inline bool SortShape(const nnvm::NodeAttrs& attrs,
                      mxnet::ShapeVector *in_attrs,
                      mxnet::ShapeVector *out_attrs) {
  const SortParam& param = nnvm::get<SortParam>(attrs.parsed);
  TopKParam topk_param;
  topk_param.axis = param.axis;
  topk_param.is_ascend = param.is_ascend;
  topk_param.k = 0;
  topk_param.ret_typ = topk_enum::kReturnValue;
  return TopKShapeImpl(topk_param, in_attrs, out_attrs);
}

inline bool ArgSortType(const nnvm::NodeAttrs& attrs,
                        std::vector<int> *in_attrs,
                        std::vector<int> *out_attrs) {
  const ArgSortParam& param = nnvm::get<ArgSortParam>(attrs.parsed);
  CHECK(type_assign(&(*out_attrs)[0], param.dtype))
      << "Failed to set the type of ret_indices.";
  return true;
}

inline bool ArgSortShape(const nnvm::NodeAttrs& attrs,
                         mxnet::ShapeVector *in_attrs,
                         mxnet::ShapeVector *out_attrs) {
  const ArgSortParam& param = nnvm::get<ArgSortParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  mxnet::TShape& in_shape = (*in_attrs)[0];

  if (in_shape.ndim() == 0) {
    mxnet::TShape target_shape({1});
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, target_shape);
  } else if (!static_cast<bool>(param.axis)) {
    mxnet::TShape target_shape(Shape1(in_shape.Size()));
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, target_shape);
  } else {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_shape);
  }

  return true;
}
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_TENSOR_ORDERING_OP_INL_H_
