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
#include "../../common/utils.h"
#include "../tensor/sort_op.h"
#include "../tensor/init_op.h"
#include "../operator_common.h"
#include "../mxnet_op.h"

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
    .describe("Axis along which to insert `values`.");
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
struct InsertSingleIndexForward {
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

template<int ndim>
inline mshadow::Shape<ndim> GetStride(const mxnet::TShape& shape) {
  mshadow::Shape<ndim>stride;
  size_t tmp = 1;
  for (int i = shape.ndim() - 1; i >= 0; --i) {
    stride[i] = tmp;
    tmp *= shape[i];
  }
  return stride;
}

template<int ndim>
inline mshadow::Shape<ndim> GetKernelShape(const mxnet::TShape& shape) {
  mshadow::Shape<ndim>k_shape;
  for (int i = 0 ; i < shape.ndim() ; ++i) {
    k_shape[i] = shape[i];
  }
  return k_shape;
}

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
struct InsertSeqIndicesForward {
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

  template<typename DType, typename VType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data,
                                  const VType in_val, const DType* in_arr,
                                  const mshadow::Shape<ndim> outshape,
                                  const int* is_insert,
                                  const int* origin_idx,
                                  const mshadow::Shape<ndim> arr_stride,
                                  const mshadow::Shape<ndim> out_stride,
                                  const int axis, const int req) {
    // i is the global flattened index in the output
    // out_idx: i -> position in output's shape
    mshadow::Shape<ndim> out_idx = mxnet_op::unravel(i, outshape);
    int64_t dest_idx;
    if (is_insert[out_idx[axis]]) {
      KERNEL_ASSIGN(out_data[i], req, static_cast<DType>(in_val));
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

struct SliceToIndices {
  MSHADOW_XINLINE static void Map(int i, int64_t* indices, int N,
                                  int start, int step) {
    indices[i] = start + i * step;
    if (indices[i] < 0) {
      indices[i] += static_cast<int64_t>(N);
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
void NumpyInsertCompute(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;

  const NumpyInsertParam& param = nnvm::get<NumpyInsertParam>(attrs.parsed);
  int input_count = param.val.has_value() ? 1 : 2;
  int insize = (param.step.has_value() || param.int_ind.has_value()) ?
               input_count : input_count + 1;
  bool obj_is_tensor = (param.val.has_value() && insize == 2) ||
                       (!param.val.has_value() && insize == 3);
  CHECK_EQ(inputs.size(), insize);
  CHECK_EQ(outputs.size(), 1);
  CHECK_EQ(req.size(), 1);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const int arr_pos = 0;
  const int val_pos = param.val.has_value() ? 0 : 1;
  const int obj_pos = val_pos + 1;
  const int out_pos = 0;
  int ndim = inputs[arr_pos].shape_.ndim();
  int axis = param.axis.has_value() ? param.axis.value() : 0;
  TBlob arr;
  TBlob values = param.val.has_value() ?
                 TBlob(nullptr, mxnet::TShape(0, 1), xpu::kDevMask, outputs[out_pos].type_flag_) :
                 inputs[val_pos];
  if (!param.axis.has_value()) {
    arr = inputs[arr_pos].reshape(Shape1(inputs[arr_pos].shape_.Size()));
    ndim = 1;
  } else if (ndim == 0) {
    if (param.val.has_value()) {
      CHECK_EQ(inputs[val_pos].shape_.ndim(), 0)
        << "'arr' is a 0-d array, 'values' can not assign to it. "
        << "alueError: assignment to 0-d array.";
      mxnet_op::copy(s, outputs[out_pos], inputs[val_pos]);
    } else {
      MSHADOW_TYPE_SWITCH(outputs[out_pos].type_flag_, DType, {
        Fill(s, outputs[out_pos], req[0], static_cast<DType>(param.val.value()));
      });
    }
    return;
  } else {
    arr = inputs[arr_pos];
    CHECK(axis >= -1 * arr.shape_.ndim() && axis < arr.shape_.ndim())
      << "Axis should be in the range of [-r, r-1] where r is the rank of input tensor";
    axis += (axis < 0) ? arr.shape_.ndim() : 0;
  }

  int N = arr.shape_[axis];
  size_t indices_len = 0;  // indices amount
  int start = 0, stop = 0, step = 0;  // arguments from 'obj' when it's 'slice'

  // get and check indices from slice or sequence of ints
  if (obj_is_tensor) {  // indices from 'tensor'
    indices_len = inputs[obj_pos].shape_.Size();
  } else if (param.step.has_value()) {  // indices from 'slice'
    SliceIndices(param.start, param.stop, param.step,
                 N, &start, &stop, &step, &indices_len);
  }

  int numnew = 0;  // numnew = output.shape[axis] - arr.shape[axis]
  int index = 0;  // save modified index, because index may be negative integer
  mxnet::TShape val_newshape(arr.shape_.ndim(), -1);
  // modify values's ndim to arr's ndim, for broadcast easily later
  // e.g. value shape: (2,) arr shape: (3, 2) => value shape: (1, 2)
  for (int i = values.shape_.ndim() - 1, j = arr.shape_.ndim() - 1;
       i >= 0 || j >= 0;
       --i, --j) {
    if (i >= 0 && j >= 0) {
      val_newshape[j] = values.shape_[i];
    } else if (i >= 0) {
      CHECK_EQ(values.shape_[i], 1) << "index exceed limits.";
    } else {
      val_newshape[j] = 1;
    }
  }
  values.shape_.assign(val_newshape.begin(), val_newshape.end());

  // get numnew
  mxnet::TShape old_valshape(values.shape_);
  if (param.int_ind.has_value() ||
    (obj_is_tensor && inputs[obj_pos].shape_.ndim() == 0)) {  // scaler
    if (param.int_ind.has_value()) {
      index = param.int_ind.value();
      CHECK(index >= -1 * N && index <= N)
        << "Index should be in the range of [-r, r-1] where r is the dim size in 'axis'";
      if (index < 0) {
        index += N;
      }
    }

    // values = moveaxis(values, 0, axis), will change values's shape
    numnew = values.shape_[0];
    mxnet::TShape axes(values.ndim(), -1);  // moved axes
    mxnet::TShape val_newshape(values.ndim(), -1);
    int axes_id = 0;
    for (int i = 1; i <= axis; ++i) {
      axes[axes_id++] = i;
    }
    axes[axes_id++] = 0;
    for (int i = axis + 1; i < values.ndim(); ++i) {
      axes[axes_id++] = i;
    }
    for (int i = 0; i < values.ndim(); ++i) {
      val_newshape[i] = values.shape_[axes[i]];
    }
    values.shape_.assign(val_newshape.begin(), val_newshape.end());
  } else if (indices_len == 1) {  // tensor with only one element
    numnew = values.shape_[axis];
    if (param.step.has_value()) {
      index = start;
      CHECK(index >= -1 * N && index <= N)
        << "Index should be in the range of [-r, r-1] where r is the dim size in 'axis'";
      if (index < 0) {
        index += N;
      }
    }
  } else {
    numnew = static_cast<int>(indices_len);
  }

  const mxnet::TShape& outshape = outputs[out_pos].shape_;
  MXNET_NDIM_SWITCH(outshape.ndim(), ndim, {
    mshadow::Shape<ndim> arr_strides = mxnet_op::calc_stride(arr.shape_.get<ndim>());
    mshadow::Shape<ndim> val_strides = mxnet_op::calc_stride(values.shape_.get<ndim>());
    mshadow::Shape<ndim> old_val_strides = mxnet_op::calc_stride(old_valshape.get<ndim>());
    mshadow::Shape<ndim> out_strides = mxnet_op::calc_stride(outshape.get<ndim>());
    mshadow::Shape<ndim> k_outshape = outshape.get<ndim>();
    mshadow::Shape<ndim> k_valshape = values.shape_.get<ndim>();
    int vtype = param.val.has_value() ?
                mshadow::DataType<double>::kFlag :
                inputs[val_pos].type_flag_;
    MSHADOW_TYPE_SWITCH(outputs[out_pos].type_flag_, DType, {
      MSHADOW_TYPE_SWITCH(vtype, VType, {
        if ((param.int_ind.has_value() ||
            (obj_is_tensor && inputs[obj_pos].shape_.ndim() == 0) ||
            (indices_len == 1)) &&
            param.val.has_value()) {
          // If insert use single index and 'value' is inputed as numerical parameter
          values = TBlob(ctx.requested[0].get_space_typed<xpu, 1, VType>(Shape1(1), s));
          Fill(s, values, kWriteTo, param.val.value());
        }
        if (param.int_ind.has_value()) {
          // 'obj' is integer, need to moveaxis
          Kernel<InsertSingleIndexForward<ndim>, xpu>::Launch(
            s, outshape.Size(), outputs[out_pos].dptr<DType>(),
            values.dptr<VType>(), arr.dptr<DType>(),
            k_outshape, k_valshape, index, numnew,
            val_strides, old_val_strides, arr_strides, out_strides,
            axis, true, req[out_pos]);
        } else if (obj_is_tensor && inputs[obj_pos].shape_.ndim() == 0) {
          // 'obj' is tensor and the tensor's ndim is 0, also need to moveaxis
            Kernel<InsertSingleIndexForward<ndim>, xpu>::Launch(
              s, outshape.Size(), outputs[out_pos].dptr<DType>(),
              values.dptr<VType>(), arr.dptr<DType>(),
              k_outshape, k_valshape, N, inputs[obj_pos].dptr<int64_t>(), numnew,
              val_strides, old_val_strides, arr_strides, out_strides,
              axis, true, req[out_pos]);
        } else if (indices_len == 1) {
          if (param.step.has_value()) {
            Kernel<InsertSingleIndexForward<ndim>, xpu>::Launch(
              s, outshape.Size(), outputs[out_pos].dptr<DType>(),
              values.dptr<VType>(), arr.dptr<DType>(),
              k_outshape, k_valshape, start, numnew,
              val_strides, old_val_strides, arr_strides, out_strides,
              axis, false, req[out_pos]);
          } else {
            Kernel<InsertSingleIndexForward<ndim>, xpu>::Launch(
              s, outshape.Size(), outputs[out_pos].dptr<DType>(),
              values.dptr<VType>(), arr.dptr<DType>(),
              k_outshape, k_valshape, N, inputs[obj_pos].dptr<int64_t>(), numnew,
              val_strides, old_val_strides, arr_strides, out_strides,
              axis, false, req[out_pos]);
          }
        } else {
          // broadcast check
          for (int i = outshape.ndim() - 1; i >= 0; --i) {
            int sz = outshape[i];
            if (i == axis) {
              sz = numnew;
            }
            CHECK((values.shape_[i] == 1) || (values.shape_[i] == sz));
          }
          size_t temp_storage_bytes, temp_mem_size;
          temp_storage_bytes = SortByKeyWorkspaceSize<int64_t, int, xpu>(indices_len, false, true);
          temp_mem_size = indices_len * sizeof(int64_t) * 2 +
                          indices_len * sizeof(int) +
                          outshape[axis] * sizeof(int) * 2 +
                          temp_storage_bytes;
          Tensor<xpu, 1, char> temp_mem =
            ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(temp_mem_size), s);
          int64_t* indices_ptr = reinterpret_cast<int64_t*>(temp_mem.dptr_);
          int64_t* sorted_indices_ptr = reinterpret_cast<int64_t*>(indices_ptr + indices_len);
          int* order_ptr = reinterpret_cast<int*>(sorted_indices_ptr + indices_len);
          int* is_insert = reinterpret_cast<int*>(order_ptr + indices_len);
          int* origin_idx = reinterpret_cast<int*>(is_insert + outshape[axis]);
          Tensor<xpu, 1, char> temp_storage(reinterpret_cast<char*>(origin_idx + outshape[axis]),
                                            Shape1(temp_storage_bytes), s);
          Tensor<xpu, 1, int64_t> indices(indices_ptr, Shape1(indices_len), s);
          Tensor<xpu, 1, int64_t> sorted_indices(sorted_indices_ptr, Shape1(indices_len), s);
          Tensor<xpu, 1, int> order(order_ptr, Shape1(indices_len), s);
          int num_bits = common::ilog2ui(static_cast<unsigned int>(indices_len) - 1);
          if (param.step.has_value()) {
            Kernel<SliceToIndices, xpu>::Launch(s, indices_len, indices_ptr, N, start, step);
          } else {
            Kernel<ObjToIndices, xpu>::Launch(s, indices_len, indices_ptr, N,
                                              inputs[obj_pos].dptr<int64_t>());
          }
          Kernel<range_fwd, xpu>::Launch(s, indices_len, 1, 0, 1, kWriteTo, order_ptr);
          mxnet::op::SortByKey(indices, order, true, &temp_storage, 0, num_bits, &sorted_indices);
          Kernel<IndicesModify, xpu>::Launch(s, indices_len, indices_ptr, order_ptr);

          mxnet_op::Kernel<mxnet_op::set_zero, xpu>::Launch(s, outshape[axis], is_insert);
          Kernel<SetIsInsert, xpu>::Launch(s, indices_len, indices_ptr, is_insert);

          Kernel<SetOriginValuesIdx, xpu>::Launch(s, indices_len, indices_ptr, origin_idx);
          Kernel<SetOriginArrIdx, xpu>::Launch(s, outshape[axis], is_insert, origin_idx);
          if (param.val.has_value()) {
            Kernel<InsertSeqIndicesForward<ndim>, xpu>::Launch(
              s, outshape.Size(),
              outputs[out_pos].dptr<DType>(),
              param.val.value(), arr.dptr<DType>(),
              k_outshape, is_insert, origin_idx,
              arr_strides, out_strides, axis, req[out_pos]);
          } else {
            Kernel<InsertSeqIndicesForward<ndim>, xpu>::Launch(
              s, outshape.Size(),
              outputs[out_pos].dptr<DType>(),
              values.dptr<VType>(), arr.dptr<DType>(),
              k_outshape, k_valshape, is_insert, origin_idx,
              val_strides, arr_strides, out_strides, axis, req[out_pos]);
          }
        }
      });
    });
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_INSERT_OP_INL_H_
