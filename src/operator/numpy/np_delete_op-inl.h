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
 *  Copyright (c) 2019 by Contributors
 * \file np_delete_op-inl.h
 * \brief Function definition of delete operators
 */
#ifndef MXNET_OPERATOR_NUMPY_NP_DELETE_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_DELETE_OP_INL_H_

#include <vector>
#include <memory>
#include <algorithm>
#include "../../common/utils.h"
#include "../tensor/sort_op.h"
#include "../tensor/init_op.h"
#include "../operator_common.h"
#include "../mxnet_op.h"
#include "../tensor/broadcast_reduce_op.h"
#ifdef __CUDACC__
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#endif

namespace mxnet {
namespace op {

struct NumpyDeleteParam : public dmlc::Parameter<NumpyDeleteParam> {
  dmlc::optional<int> start;
  dmlc::optional<int> stop;
  dmlc::optional<int> step;
  dmlc::optional<int> int_ind;
  dmlc::optional<int> axis;
  DMLC_DECLARE_PARAMETER(NumpyDeleteParam) {
    DMLC_DECLARE_FIELD(start)
    .set_default(dmlc::optional<int>())
    .describe("If 'obj' is slice, 'start' is one of it's arguments.");
    DMLC_DECLARE_FIELD(stop)
    .set_default(dmlc::optional<int>())
    .describe("If 'obj' is slice, 'stop' is one of it's arguments.");
    DMLC_DECLARE_FIELD(step)
    .set_default(dmlc::optional<int>())
    .describe("If 'obj' is slice, 'step' is one of it's arguments.");
    DMLC_DECLARE_FIELD(int_ind)
    .set_default(dmlc::optional<int>())
    .describe("If 'obj' is int, 'int_ind' is the index before which"
              "'values' is inserted");
    DMLC_DECLARE_FIELD(axis)
    .set_default(dmlc::optional<int>())
    .describe("Axis along which to insert `values`.");
  }
};

namespace delete_ {

enum DeleteOpInputs {kArr, kObj};
enum DeleteOpOutputs {kOut};
}  // namespace delete_

struct SliceToIndices {
  /*!
   * \brief transfer slice to indices array
   */
  template<typename IType>
  MSHADOW_XINLINE static void Map(int i, IType* indices, int start, int step) {
    indices[i] = start + i * step;
  }
};

struct IsDeleteCal {
  /*!
   * \brief indicate which indices need to be deleted in input
   * \param N used to check indices legality 
   * \param is_delete if is_delete[i] == False, index i needn't to be deleted from output
   *                  if is_delete[i] == True, index i need to be deleted from output
   * \param indices the indices need to be deleted
   */
  template<typename IType>
  MSHADOW_XINLINE static void Map(int i, int N, bool* is_delete, const IType* indices) {
    if ((indices[i] >= 0) && (indices[i] < N)) {
      is_delete[static_cast<int>(indices[i])] = true;
    }
  }
};

struct OutPosCal {
  /*!
   * \brief map the index from input to output. e.g.
   * \example original_position 0 1 2 3 4
   *          is_delete         F T T F F
   *          out_position      0 - - 1 2
   */
  MSHADOW_XINLINE static void Map(int i, int64_t* out_pos, const bool* is_delete) {
    if (!is_delete[i]) {
      int cnt = 0;
      for (int j = 0; j < i; ++j) {
        if (!is_delete[j]) {
          cnt++;
        }
      }
      out_pos[i] = cnt;
    }
  }
};

template<int req, int ndim>
struct DeleteKernel {
  /*!
   * \brief delete a sub-array from input along an axis according to 'is_delete'.
   * \param out_data - output: a new array with sub-arrays along an axis deleted.
   * \param in_arr - input: 'arr', original array.
   * \param is_delete - mark where will be deleted or be reminded in 'arr'
   * \param out_pos - if is_delete[i] is 'false', out_pos[i] indicates its.
   * \param arrshape - the shape of 'arr'.
   * \param out_stride - the stride of 'out_data'.
   * \param axis - delete sub-array along this axis
   */
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data,
                                  const DType* in_arr,
                                  const bool* is_delete,
                                  const int64_t* out_pos,
                                  const mshadow::Shape<ndim> arrshape,
                                  const mshadow::Shape<ndim> out_stride,
                                  const int axis) {
    // i -> position in in_arr's shape
    mshadow::Shape<ndim> arr_idx = mxnet_op::unravel(i, arrshape);
    if (!is_delete[arr_idx[axis]]) {
      arr_idx[axis] = out_pos[arr_idx[axis]];
      int64_t dest_idx = mxnet_op::dot(arr_idx, out_stride);
      KERNEL_ASSIGN(out_data[dest_idx], req, in_arr[i]);
    }
  }
};

/*!
 * /brief equals to numpy's slice.indices(range)
 * /param pstart - slice.start
 * /param pstep - slice.step
 * /param pstop - slice.stop
 * /return start - slice.indices(range).start
 * /return stop - slice.indices(range).stop
 * /return step - slice.indices(range).step
 * /return tot - total number of slice.indices(range)
 */
inline void SliceIndices(const dmlc::optional<int>& pstart,
                         const dmlc::optional<int>& pstop,
                         const dmlc::optional<int>& pstep,
                         const int range,
                         int* start, int* stop, int* step,
                         size_t* tot) {
  *step = pstep.has_value() ? pstep.value() : 1;
  CHECK_NE(*step, 0) << "'step' can not equal to 0.";
  if (pstop.has_value()) {
    *stop = pstop.value();
    *stop += (*stop < 0) ? range : 0;
    *stop = (*stop < 0) ? ((*step < 0) ? -1 : 0) : *stop;
    *stop = (*stop >= range) ? ((*step < 0) ? range - 1 : range) : *stop;
  } else {
    *stop = (*step > 0) ? range : -1;
  }
  if (pstart.has_value()) {
    *start = pstart.value();
    *start += (*start < 0) ? range : 0;
    *start = (*start < 0) ? ((*step < 0) ? -1 : 0) : *start;
    *start = (*start >= range) ? ((*step < 0) ? range - 1 : range) : *start;
  } else {
    *start = (*step > 0) ? 0 : range - 1;
  }
  if (*step > 0 && *stop >= *start) {
    *tot = static_cast<size_t>((*stop - *start + *step - 1) / *step);
  } else if (*step < 0 && *stop <= *start) {
    *tot = static_cast<size_t>((*stop - *start + *step + 1) / *step);
  }
}

template<typename xpu>
void NumpyDeleteCompute(const nnvm::NodeAttrs& attrs,
                        const OpContext &ctx,
                        const std::vector<NDArray> &inputs,
                        const std::vector<OpReqType> &req,
                        const std::vector<NDArray> &outputs) {
  using namespace mshadow;
  using namespace mxnet_op;

  const NumpyDeleteParam& param = nnvm::get<NumpyDeleteParam>(attrs.parsed);
  CHECK_EQ(inputs.size(),
          (param.step.has_value() || param.int_ind.has_value()) ? 1U : 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();

  int ndim = inputs[delete_::kArr].shape().ndim();
  int axis = param.axis.has_value() ? param.axis.value() : -1;
  NDArray arr;  // original array

  if (!param.axis.has_value()) {
    arr = inputs[delete_::kArr].Reshape(Shape1(inputs[delete_::kArr].shape().Size()));
    ndim = 1;
    axis = -1;
  } else {
    arr = inputs[delete_::kArr];
  }

  if (ndim == 0) {
    const_cast<NDArray &>(outputs[delete_::kOut]).Init(arr.shape());
    mxnet_op::copy(s, outputs[delete_::kOut].data(), inputs[delete_::kArr].data());
    return;
  }

  axis = CheckAxis(axis, ndim);
  int N = (arr.shape())[axis];
  mxnet::TShape outshape(arr.shape());
  // if obj is slice, they're obj's arguments
  int start = 0, stop = 0, step = 0;
  // total number to be deleted
  size_t numtodel = 0;
  // if obj is scaler, index is it's value
  int index = 0;

  if (param.step.has_value()) {  // obj is slice
    SliceIndices(param.start, param.stop, param.step,
                 N, &start, &stop, &step, &numtodel);
    if (numtodel == 0) {
      const_cast<NDArray &>(outputs[delete_::kOut]).Init(arr.shape());
      mxnet_op::copy(s, outputs[delete_::kOut].data(), inputs[delete_::kArr].data());
      return;
    }
    outshape[axis] -= numtodel;
    const_cast<NDArray &>(outputs[delete_::kOut]).Init(outshape);
  } else if (param.int_ind.has_value()) {  // obj is scaler
    index = param.int_ind.value();
    CHECK((index >= -1 * N) && (index < N))
      << "index " << index
      << " is out of bounds for axis " << axis
      << " with size " << N << "\n";
    index += ((index < 0) ? N : 0);
    numtodel = static_cast<size_t>(1);
    outshape[axis] -= 1;
    const_cast<NDArray &>(outputs[delete_::kOut]).Init(outshape);
  } else {  // obj is tensor
    numtodel = inputs[delete_::kObj].shape().Size();
  }

  char* out_pos_ptr = nullptr;
  char* indices_ptr = nullptr;
  char* is_delete_ptr = nullptr;
  MSHADOW_TYPE_SWITCH(((inputs.size() == 2U) ?  // obj is tensor
                      inputs[delete_::kObj].dtype() :
                      mshadow::DataType<int64_t>::kFlag), IType, {
    size_t temp_mem_size = sizeof(int64_t) * arr.shape()[axis] +
                           sizeof(IType) * numtodel +
                           sizeof(bool) * arr.shape()[axis];
    Tensor<xpu, 1, char> temp_mem =
      ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(temp_mem_size), s);
    out_pos_ptr = temp_mem.dptr_;
    indices_ptr = out_pos_ptr + sizeof(int64_t) * arr.shape()[axis];
    is_delete_ptr = indices_ptr + sizeof(IType) * numtodel;
    if (param.step.has_value()) {  // obj is slice, transfer slice to tensor
      Kernel<SliceToIndices, xpu>::Launch(
        s, numtodel, reinterpret_cast<IType*>(indices_ptr), start, step);
    } else if (param.int_ind.has_value()) {  // obj is scaler, copy it to tensor
      Fill(s, TBlob(reinterpret_cast<IType*>(indices_ptr),
           Shape1(numtodel), xpu::kDevMask), kWriteTo, index);
    } else {  // obj is tensor, copy it to a unified tensor
      mxnet_op::copy(s,
        TBlob(reinterpret_cast<IType*>(indices_ptr), inputs[delete_::kObj].shape(),
              inputs[delete_::kObj].data().dev_mask()),
        inputs[delete_::kObj].data());
    }
    mxnet_op::Kernel<mxnet_op::set_zero, xpu>::Launch(
      s, arr.shape()[axis], reinterpret_cast<bool*>(is_delete_ptr));
    // mark which position need to be deleted from input arr
    Kernel<IsDeleteCal, xpu>::Launch(
      s, numtodel, N, reinterpret_cast<bool*>(is_delete_ptr),
      reinterpret_cast<IType*>(indices_ptr));
    // calculate output data's original position in input arr
    Kernel<OutPosCal, xpu>::Launch(
      s, arr.shape()[axis], reinterpret_cast<int64_t*>(out_pos_ptr),
      reinterpret_cast<bool*>(is_delete_ptr));
  });

  if (inputs.size() == 2U) {  // obj is tensor
    // get total number of nonredundant indices
    #ifdef __CUDACC__
      thrust::device_ptr<bool>is_delete_dev(reinterpret_cast<bool*>(is_delete_ptr));
      thrust::device_vector<bool>vec_is_delete(is_delete_dev, is_delete_dev + arr.shape()[axis]);
    #else
      std::vector<bool>vec_is_delete(reinterpret_cast<bool*>(is_delete_ptr),
                                     reinterpret_cast<bool*>(is_delete_ptr) + arr.shape()[axis]);
    #endif
    numtodel = 0;
    for (int i = 0; i < arr.shape()[axis]; ++i) {
      if (vec_is_delete[i]) {
        numtodel++;
      }
    }
    outshape[axis] -= numtodel;
    const_cast<NDArray &>(outputs[delete_::kOut]).Init(outshape);
  }

  MSHADOW_TYPE_SWITCH(((inputs.size() == 2U) ?  // obj is tensor
                      inputs[delete_::kObj].dtype() :
                      mshadow::DataType<int64_t>::kFlag), IType, {
    MXNET_NDIM_SWITCH(outshape.ndim(), ndim, {
      mshadow::Shape<ndim> out_strides = mxnet_op::calc_stride(outshape.get<ndim>());
      MSHADOW_TYPE_SWITCH(outputs[delete_::kOut].dtype(), DType, {
        MXNET_ASSIGN_REQ_SWITCH(req[delete_::kOut], req_type, {
          Kernel<DeleteKernel<req_type, ndim>, xpu>::Launch(
            s, arr.shape().Size(),
            outputs[delete_::kOut].data().dptr<DType>(),
            arr.data().dptr<DType>(),
            reinterpret_cast<bool*>(is_delete_ptr),
            reinterpret_cast<int64_t*>(out_pos_ptr),
            arr.shape().get<ndim>(),
            out_strides, axis);
        });
      });
    });
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_DELETE_OP_INL_H_
