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
 * \file np_insert_op-inl.h
 * \brief Function definition of insert operators
 */
#ifndef MXNET_OPERATOR_NUMPY_NP_INSERT_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_INSERT_OP_INL_H_

#include <vector>
#include <memory>
#include <algorithm>
#include "../../common/utils.h"
#include "../tensor/sort_op.h"
#include "../tensor/init_op.h"
#include "../operator_common.h"
#include "../mxnet_op.h"
#include "./np_delete_op-inl.h"
namespace mxnet {
namespace op {

struct NumpyInsertParam : public dmlc::Parameter<NumpyInsertParam> {
  dmlc::optional<double> val;
  dmlc::optional<int> start;
  dmlc::optional<int> stop;
  dmlc::optional<int> step;
  dmlc::optional<int> int_ind;
  dmlc::optional<int> axis;
  DMLC_DECLARE_PARAMETER(NumpyInsertParam) {
    DMLC_DECLARE_FIELD(val)
    .set_default(dmlc::optional<double>())
    .describe("A scaler to be inserted into 'array'");
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
    .describe("Axis along which to insert 'values'.");
  }
};

/*!
 * \brief insert when obj is 'scaler' or a 'slice' with only one element.
 * \tparam ndim - both 'in_arr', 'in_val' and 'out_data' have same ndim before call this.
 * \param out_data - output: insert 'value' to 'arr' according to 'index'.
 * \param in_arr - input: 'arr', original array.
 * \param index - input(only for first Map): it's the only element in 'obj' indicats insert position.
 * \param in_obj - input(only for second Map): It indicats insert position, it's ndim may equals to 0.
 * \param in_val - input: 'value', insert to 'arr' according to 'index'.
 * \param N - (only for first Map) arr.shape_[axis]
 * \param numnew - extra dim size in 'out_data' compared with 'arr' in 'axis'.
 * \param axis - insert 'value' to 'arr' in 'axis'.
 * \param moveaxis - If 'obj' is a scaler, moveaxis is true;
                     If 'obj' is a slice with one element, moveaxis is false.
 * \note Different between the two Map:
         The first one use a scaler index;
         The second one use a sequence of indecies which only has one index.
 */
template<int ndim>
struct InsertSingleIndexKernel {
  template<typename DType, typename VType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data,
                                  const VType* in_val, const DType* in_arr,
                                  const mshadow::Shape<ndim> outshape,
                                  const mshadow::Shape<ndim> valshape,
                                  const int index, const int numnew,
                                  const mshadow::Shape<ndim> val_stride,
                                  const mshadow::Shape<ndim> old_val_stride,
                                  const mshadow::Shape<ndim> arr_stride,
                                  const mshadow::Shape<ndim> out_stride,
                                  const int axis, bool moveaxis, const int req) {
    // i is the global flattened index in the output
    // out_idx: i -> position in output's shape
    mshadow::Shape<ndim> out_idx = mxnet_op::unravel(i, outshape);
    int64_t dest_idx;
    if (out_idx[axis] >= index && out_idx[axis] < index + numnew) {  // from 'value'
      int idx_val = out_idx[axis] - index;
      // val_idx: i -> position in values's shape
      mshadow::Shape<ndim> val_idx(out_idx);
      val_idx[axis] = idx_val;
      for (int j = ndim - 1; j >= 0; --j) {
        if (valshape[j] == 1) {  // broadcast
          val_idx[j] = 0;
        }
      }
      dest_idx = 0;
      if (moveaxis) {  // moveaxis(values, 0, axis)
        for (int j = 0; j < axis; ++j) {
          dest_idx += old_val_stride[j + 1] * val_idx[j];
        }
        dest_idx += old_val_stride[0] * val_idx[axis];
        for (int j = axis + 1; j < ndim ; ++j) {
          dest_idx += old_val_stride[j] * val_idx[j];
        }
      } else {
        dest_idx = mxnet_op::dot(val_stride, val_idx);
      }
      KERNEL_ASSIGN(out_data[i], req, static_cast<DType>(in_val[dest_idx]));
    } else {  // from 'arr'
      int idx_arr = (out_idx[axis] < index) ?
                     out_idx[axis] : out_idx[axis] - numnew;
      // arr_idx: i -> position in arr's shape
      mshadow::Shape<ndim> arr_idx(out_idx);
      arr_idx[axis] = idx_arr;
      dest_idx = mxnet_op::dot(arr_stride, arr_idx);
      KERNEL_ASSIGN(out_data[i], req, in_arr[dest_idx]);
    }
  }

  template<typename DType, typename VType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data,
                                  const VType* in_val, const DType* in_arr,
                                  const mshadow::Shape<ndim> outshape,
                                  const mshadow::Shape<ndim> valshape,
                                  const int N, const int64_t* in_obj, const int numnew,
                                  const mshadow::Shape<ndim> val_stride,
                                  const mshadow::Shape<ndim> old_val_stride,
                                  const mshadow::Shape<ndim> arr_stride,
                                  const mshadow::Shape<ndim> out_stride,
                                  const int axis, bool moveaxis, const int req) {
    // i is the global flattened index in the output
    // out_idx: i -> position in output's shape
    mshadow::Shape<ndim> out_idx = mxnet_op::unravel(i, outshape);
    int64_t dest_idx;
    int64_t index = in_obj[0];
    if (static_cast<int64_t>(index) < 0) {
      index += static_cast<int64_t>(N);
    }
    if (out_idx[axis] >= index && out_idx[axis] < index + numnew) {  // from 'value'
      int idx_val = out_idx[axis] - index;
      // val_idx: i -> position in values's shape
      mshadow::Shape<ndim> val_idx(out_idx);
      val_idx[axis] = idx_val;
      for (int j = ndim - 1; j >= 0; --j) {
        if (valshape[j] == 1) {  // broadcast
          val_idx[j] = 0;
        }
      }
      dest_idx = 0;
      if (moveaxis) {  // moveaxis(values, 0, axis)
        for (int j = 0; j < axis; ++j) {
          dest_idx += old_val_stride[j + 1] * val_idx[j];
        }
        dest_idx += old_val_stride[0] * val_idx[axis];
        for (int j = axis + 1; j < ndim ; ++j) {
          dest_idx += old_val_stride[j] *val_idx[j];
        }
      } else {
        dest_idx = mxnet_op::dot(val_stride, val_idx);
      }
      KERNEL_ASSIGN(out_data[i], req, static_cast<DType>(in_val[dest_idx]));
    } else {  // from 'arr'
      int idx_arr = (out_idx[axis] < index) ? out_idx[axis] : out_idx[axis] - numnew;
      // arr_idx: i -> position in arr's shape
      mshadow::Shape<ndim> arr_idx(out_idx);
      arr_idx[axis] = idx_arr;
      dest_idx = mxnet_op::dot(arr_stride, arr_idx);
      KERNEL_ASSIGN(out_data[i], req, in_arr[dest_idx]);
    }
  }
};

/*!
 * \brief insert when obj is 'tensor' or 'slice' with more than one element.
 * \tparam ndim - both 'in_arr', 'in_val' and 'out_data' have same ndim before call this.
 * \param out_data - output: insert 'value' to 'arr' according to 'index'.
 * \param in_arr - input: 'arr', original array.
 * \param in_obj - input: It indicats insert position, ndim may equals to 0.
 * \param in_val - input: 'value', insert to 'arr' according to 'index'.
 * \param is_insert - if is_insert[out_idx[axis]] is true, it's from 'values', else from 'arr'.
 * \param origin_idx - indicate the original position in 'arr' or 'values' in 'axis'. 
 * \param axis - insert 'value' to 'arr' in 'axis'.
 * \note Different between the two Map:
         The first one insert a block of data, param 'in_val' is a tensor;
         The second one insert only a single data, param 'in_val' is a scaler.
 */
template<int ndim>
struct InsertSeqIndicesKernel {
  template<typename DType, typename VType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data,
                                  const VType* in_val, const DType* in_arr,
                                  const mshadow::Shape<ndim> outshape,
                                  const mshadow::Shape<ndim> valshape,
                                  const int* is_insert,
                                  const int* origin_idx,
                                  const mshadow::Shape<ndim> val_stride,
                                  const mshadow::Shape<ndim> arr_stride,
                                  const mshadow::Shape<ndim> out_stride,
                                  const int axis, const int req) {
    // i is the global flattened index in the output
    // out_idx: i -> position in output's shape
    mshadow::Shape<ndim> out_idx = mxnet_op::unravel(i, outshape);
    int64_t dest_idx;
    if (is_insert[out_idx[axis]]) {
      // the data of output[i] is from 'values'
      int idx_val = origin_idx[out_idx[axis]];
      // insert_idx: i -> position in insert's shape
      mshadow::Shape<ndim> insert_idx(out_idx);
      insert_idx[axis] = idx_val;
      // val_idx: i -> position in values's shape
      mshadow::Shape<ndim> val_idx(insert_idx);
      for (int j = ndim - 1; j >= 0; --j) {  // broadcast
        if (valshape[j] == 1) {
          val_idx[j] = 0;
        }
      }
      dest_idx = mxnet_op::dot(val_idx, val_stride);
      KERNEL_ASSIGN(out_data[i], req, static_cast<DType>(in_val[dest_idx]));
    } else {
      // the data of output[i] is from 'arr'
      int idx_arr = origin_idx[out_idx[axis]];
      // arr_idx: i -> position in arr's shape
      mshadow::Shape<ndim> arr_idx(out_idx);
      arr_idx[axis] = idx_arr;
      dest_idx = mxnet_op::dot(arr_idx, arr_stride);
      KERNEL_ASSIGN(out_data[i], req, in_arr[dest_idx]);
    }
  }
};

struct ObjToIndices {
  MSHADOW_XINLINE static void Map(int i, int64_t* indices,
                                  int N, const int64_t* obj) {
    indices[i] = obj[i];
    if (indices[i] < 0) {
      indices[i] += static_cast<int64_t>(N);
    }
  }
};

struct IndicesModify {
  MSHADOW_XINLINE static void Map(int i, int64_t* indices, const int* order) {
    indices[order[i]] += i;
  }
};

struct SetIsInsert {
  MSHADOW_XINLINE static void Map(int i, int64_t* indices, int* is_insert) {
    is_insert[static_cast<int>(indices[i])] = 1;
  }
};

struct SetOriginValuesIdx {
  MSHADOW_XINLINE static void Map(int i, const int64_t* indices, int* origin_idx) {
    origin_idx[static_cast<int>(indices[i])] = i;
  }
};

struct SetOriginArrIdx {
  MSHADOW_XINLINE static void Map(int i, const int* is_insert,
                         int* origin_idx) {
    if (!is_insert[i]) {
      int cnt = 0;
      for (int j = 0; j < i; ++j) {
        if (is_insert[j] == 0) {
          cnt++;
        }
      }
      origin_idx[i] = cnt;
    }
  }
};

template<typename xpu, int ndim>
void InsertScalerImpl(mshadow::Stream<xpu> *s, const TBlob& output,
                      const TBlob& arr, const TBlob& values,
                      const mshadow::Shape<ndim>& arr_strides,
                      const mshadow::Shape<ndim>& val_strides,
                      const mshadow::Shape<ndim>& old_val_strides,
                      const mshadow::Shape<ndim>& out_strides,
                      const mshadow::Shape<ndim>& k_outshape,
                      const mshadow::Shape<ndim>& k_valshape,
                      const int dtype, const int vtype, const int req,
                      const int axis, const int index, const int numnew,
                      const size_t len, const bool moveaxis) {
  using namespace mshadow;
  using namespace mxnet_op;
  MSHADOW_TYPE_SWITCH(dtype, DType, {
    MSHADOW_TYPE_SWITCH(vtype, VType, {
      Kernel<InsertSingleIndexKernel<ndim>, xpu>::Launch(
        s, len, output.dptr<DType>(),
        values.dptr<VType>(), arr.dptr<DType>(),
        k_outshape, k_valshape, index, numnew,
        val_strides, old_val_strides, arr_strides, out_strides,
        axis, moveaxis, req);
    });
  });
}

template<typename xpu, int ndim>
void InsertSizeOneTensorImpl(mshadow::Stream<xpu> *s, const TBlob& output,
                             const TBlob& arr, const TBlob& values,
                             const mshadow::Shape<ndim>& arr_strides,
                             const mshadow::Shape<ndim>& val_strides,
                             const mshadow::Shape<ndim>& old_val_strides,
                             const mshadow::Shape<ndim>& out_strides,
                             const mshadow::Shape<ndim>& k_outshape,
                             const mshadow::Shape<ndim>& k_valshape,
                             const int dtype, const int vtype, const int req,
                             const int axis, const TBlob& index, const int numnew,
                             const int N, const size_t len, const bool moveaxis) {
  using namespace mshadow;
  using namespace mxnet_op;
  MSHADOW_TYPE_SWITCH(dtype, DType, {
    MSHADOW_TYPE_SWITCH(vtype, VType, {
      Kernel<InsertSingleIndexKernel<ndim>, xpu>::Launch(
        s, len, output.dptr<DType>(),
        values.dptr<VType>(), arr.dptr<DType>(),
        k_outshape, k_valshape, N, index.dptr<int64_t>(), numnew,
        val_strides, old_val_strides, arr_strides, out_strides,
        axis, moveaxis, req);
    });
  });
}

template<typename xpu, int ndim>
void InsertSequenceImpl(mshadow::Stream<xpu> *s, const TBlob& output,
                        const TBlob& arr, const TBlob& values,
                        const mshadow::Shape<ndim>& arr_strides,
                        const mshadow::Shape<ndim>& val_strides,
                        const mshadow::Shape<ndim>& out_strides,
                        const mshadow::Shape<ndim>& k_outshape,
                        const mshadow::Shape<ndim>& k_valshape,
                        const int* is_insert, const int* origin_idx,
                        const int dtype, const int vtype, const int req,
                        const int axis, const size_t len) {
  using namespace mshadow;
  using namespace mxnet_op;
  MSHADOW_TYPE_SWITCH(dtype, DType, {
    MSHADOW_TYPE_SWITCH(vtype, VType, {
      Kernel<InsertSeqIndicesKernel<ndim>, xpu>::Launch(
        s, len, output.dptr<DType>(),
        values.dptr<VType>(), arr.dptr<DType>(),
        k_outshape, k_valshape, is_insert, origin_idx,
        val_strides, arr_strides, out_strides, axis, req);
    });
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_INSERT_OP_INL_H_
