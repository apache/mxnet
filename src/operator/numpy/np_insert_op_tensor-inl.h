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
#ifndef MXNET_OPERATOR_NUMPY_NP_INSERT_OP_TENSOR_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_INSERT_OP_TENSOR_INL_H_

#include <vector>
#include <memory>
#include <algorithm>
#include "../../common/utils.h"
#include "../tensor/sort_op.h"
#include "../tensor/init_op.h"
#include "../operator_common.h"
#include "../mxnet_op.h"
#include "./np_delete_op-inl.h"
#include "./np_insert_op-inl.h"

namespace mxnet {
namespace op {

/*
 * Only support tensor indices (the type of param 'obj' is tensor).
 */
template<typename xpu>
void NumpyInsertTensorCompute(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<TBlob>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;

  const NumpyInsertParam& param = nnvm::get<NumpyInsertParam>(attrs.parsed);
  int input_count = param.val.has_value() ? 1 : 2;
  int insize = input_count + 1;
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
  size_t indices_len = inputs[obj_pos].shape_.Size();  // indices amount

  // get and check indices from tensor
  int numnew = 0;  // numnew = output.shape[axis] - arr.shape[axis]
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
  if (inputs[obj_pos].shape_.ndim() == 0) {  // scaler
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
  } else {
    numnew = static_cast<int>(indices_len);
  }

  const mxnet::TShape& outshape = outputs[out_pos].shape_;
  int dtype = outputs[out_pos].type_flag_;
  int vtype = param.val.has_value() ?
              mshadow::DataType<double>::kFlag :
              inputs[val_pos].type_flag_;
  if ((inputs[obj_pos].shape_.ndim() == 0 || indices_len == 1)
      && param.val.has_value()) {
    MSHADOW_TYPE_SWITCH(vtype, VType, {
      // If insert use single index and 'value' is inputed as numerical parameter
      values = TBlob(ctx.requested[0].get_space_typed<xpu, 1, VType>(Shape1(1), s));
      Fill(s, values, kWriteTo, param.val.value());
    });
  }
  if (inputs[obj_pos].shape_.ndim() == 0) {
    // 'obj' is tensor and the tensor's ndim is 0, also need to moveaxis
    MXNET_NDIM_SWITCH(outshape.ndim(), ndim, {
      InsertSizeOneTensorImpl<xpu, ndim>(s, outputs[out_pos], arr, values,
                                         mxnet_op::calc_stride(arr.shape_.get<ndim>()),
                                         mxnet_op::calc_stride(values.shape_.get<ndim>()),
                                         mxnet_op::calc_stride(old_valshape.get<ndim>()),
                                         mxnet_op::calc_stride(outshape.get<ndim>()),
                                         outshape.get<ndim>(), values.shape_.get<ndim>(),
                                         dtype, vtype, req[out_pos], axis, inputs[obj_pos],
                                         numnew, N, outshape.Size(), true);
    });
  } else if (indices_len == 1) {
    MXNET_NDIM_SWITCH(outshape.ndim(), ndim, {
      InsertSizeOneTensorImpl<xpu, ndim>(s, outputs[out_pos], arr, values,
                                         mxnet_op::calc_stride(arr.shape_.get<ndim>()),
                                         mxnet_op::calc_stride(values.shape_.get<ndim>()),
                                         mxnet_op::calc_stride(old_valshape.get<ndim>()),
                                         mxnet_op::calc_stride(outshape.get<ndim>()),
                                         outshape.get<ndim>(), values.shape_.get<ndim>(),
                                         dtype, vtype, req[out_pos], axis, inputs[obj_pos],
                                         numnew, N, outshape.Size(), false);
    });
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
    int num_bits = 8 * sizeof(int64_t);
    Kernel<ObjToIndices, xpu>::Launch(s, indices_len, indices_ptr, N,
                                      inputs[obj_pos].dptr<int64_t>());
    Kernel<range_fwd, xpu>::Launch(s, indices_len, 1, 0, 1, kWriteTo, order_ptr);
    mxnet::op::SortByKey(indices, order, true, &temp_storage, 0, num_bits, &sorted_indices);
    Kernel<IndicesModify, xpu>::Launch(s, indices_len, indices_ptr, order_ptr);

    mxnet_op::Kernel<mxnet_op::set_zero, xpu>::Launch(s, outshape[axis], is_insert);
    Kernel<SetIsInsert, xpu>::Launch(s, indices_len, indices_ptr, is_insert);

    Kernel<SetOriginValuesIdx, xpu>::Launch(s, indices_len, indices_ptr, origin_idx);
    Kernel<SetOriginArrIdx, xpu>::Launch(s, outshape[axis], is_insert, origin_idx);
    MXNET_NDIM_SWITCH(outshape.ndim(), ndim, {
      InsertSequenceImpl<xpu, ndim>(s, outputs[out_pos], arr, values,
                                    mxnet_op::calc_stride(arr.shape_.get<ndim>()),
                                    mxnet_op::calc_stride(values.shape_.get<ndim>()),
                                    mxnet_op::calc_stride(outshape.get<ndim>()),
                                    outshape.get<ndim>(), values.shape_.get<ndim>(),
                                    is_insert, origin_idx, dtype, vtype, req[out_pos],
                                    axis, outshape.Size());
    });
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_INSERT_OP_TENSOR_INL_H_
