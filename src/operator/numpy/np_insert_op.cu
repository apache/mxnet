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
 * \file np_insert_op.cu
 * \brief GPU Implementation of numpy insert operations
 */

#include "./np_insert_op-inl.h"

namespace mxnet {
namespace op {

template<>
void InsertOneIndicesImpl<gpu>(const OpContext &ctx,
                                const TShape& outshape, const TShape& old_valshape,
                                const NumpyInsertParam& param,
                                const std::vector<TBlob>& inputs,
                                const std::vector<TBlob>& outputs,
                                const TBlob& arr, const TBlob& values,
                                const int& dtype, const int& vtype,
                                const std::vector<OpReqType>& req,
                                const int& axis, const int& start,
                                const int& out_pos, const int& obj_pos,
                                const int& numnew, const int& N){
  using namespace mshadow;
  using namespace mxnet_op;
  mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
  MXNET_NDIM_SWITCH(outshape.ndim(), ndim, {
    if (param.step.has_value()) {
      InsertScalerObj<gpu, ndim>(s, outputs[out_pos], arr, values,
                                  mxnet_op::calc_stride(arr.shape_.get<ndim>()),
                                  mxnet_op::calc_stride(values.shape_.get<ndim>()),
                                  mxnet_op::calc_stride(old_valshape.get<ndim>()),
                                  mxnet_op::calc_stride(outshape.get<ndim>()),
                                  outshape.get<ndim>(), values.shape_.get<ndim>(),
                                  dtype, vtype, req[out_pos], axis, start, numnew,
                                  outshape.Size(), false);
    } else {
      InsertSizeOneTensorObj<gpu, ndim>(s, outputs[out_pos], arr, values,
                                        mxnet_op::calc_stride(arr.shape_.get<ndim>()),
                                        mxnet_op::calc_stride(values.shape_.get<ndim>()),
                                        mxnet_op::calc_stride(old_valshape.get<ndim>()),
                                        mxnet_op::calc_stride(outshape.get<ndim>()),
                                        outshape.get<ndim>(), values.shape_.get<ndim>(),
                                        dtype, vtype, req[out_pos], axis, inputs[obj_pos],
                                        numnew, N, outshape.Size(), false);
    }
  });
}

template<>
void InsertTensorIndicesImpl<gpu>(const OpContext &ctx,
                                  const TShape& outshape,
                                  const NumpyInsertParam& param,
                                  const std::vector<TBlob>& inputs,
                                  const std::vector<TBlob>& outputs,
                                  const TBlob& arr, const TBlob& values,
                                  const int& dtype, const int& vtype,
                                  const std::vector<OpReqType>& req,
                                  const int& axis, const int& start,
                                  const int& step, const int&indices_len,
                                  const int& out_pos, const int& obj_pos,
                                  const int& numnew, const int& N){
  using namespace mshadow;
  using namespace mxnet_op;
  mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();

  for (int i = outshape.ndim() - 1; i >= 0; --i) {
    int sz = outshape[i];
    if (i == axis) {
      sz = numnew;
    }
    CHECK((values.shape_[i] == 1) || (values.shape_[i] == sz));
  }
  size_t temp_storage_bytes, temp_mem_size;
  temp_storage_bytes = SortByKeyWorkspaceSize<int64_t, int, gpu>(indices_len, false, true);
  temp_mem_size = indices_len * sizeof(int64_t) * 2 +
                  indices_len * sizeof(int) +
                  outshape[axis] * sizeof(int) * 2 +
                  temp_storage_bytes;
  Tensor<gpu, 1, char> temp_mem =
    ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(temp_mem_size), s);
  int64_t* indices_ptr = reinterpret_cast<int64_t*>(temp_mem.dptr_);
  int64_t* sorted_indices_ptr = reinterpret_cast<int64_t*>(indices_ptr + indices_len);
  int* order_ptr = reinterpret_cast<int*>(sorted_indices_ptr + indices_len);
  int* is_insert = reinterpret_cast<int*>(order_ptr + indices_len);
  int* origin_idx = reinterpret_cast<int*>(is_insert + outshape[axis]);
  Tensor<gpu, 1, char> temp_storage(reinterpret_cast<char*>(origin_idx + outshape[axis]),
                                    Shape1(temp_storage_bytes), s);
  Tensor<gpu, 1, int64_t> indices(indices_ptr, Shape1(indices_len), s);
  Tensor<gpu, 1, int64_t> sorted_indices(sorted_indices_ptr, Shape1(indices_len), s);
  Tensor<gpu, 1, int> order(order_ptr, Shape1(indices_len), s);
  int num_bits = common::ilog2ui(static_cast<unsigned int>(indices_len) - 1);
  if (param.step.has_value()) {
    Kernel<SliceToIndices, gpu>::Launch(s, indices_len, indices_ptr, start, step);
  } else {
    Kernel<ObjToIndices, gpu>::Launch(s, indices_len, indices_ptr, N,
                                      inputs[obj_pos].dptr<int64_t>());
  }
  Kernel<range_fwd, gpu>::Launch(s, indices_len, 1, 0, 1, kWriteTo, order_ptr);
  mxnet::op::SortByKey(indices, order, true, &temp_storage, 0, num_bits, &sorted_indices);
  Kernel<IndicesModify, gpu>::Launch(s, indices_len, indices_ptr, order_ptr);

  mxnet_op::Kernel<mxnet_op::set_zero, gpu>::Launch(s, outshape[axis], is_insert);
  Kernel<SetIsInsert, gpu>::Launch(s, indices_len, indices_ptr, is_insert);

  Kernel<SetOriginValuesIdx, gpu>::Launch(s, indices_len, indices_ptr, origin_idx);
  Kernel<SetOriginArrIdx, gpu>::Launch(s, outshape[axis], is_insert, origin_idx);
  MXNET_NDIM_SWITCH(outshape.ndim(), ndim, {
    InsertSequenceObj<gpu, ndim>(s, outputs[out_pos], arr, values,
                                  mxnet_op::calc_stride(arr.shape_.get<ndim>()),
                                  mxnet_op::calc_stride(values.shape_.get<ndim>()),
                                  mxnet_op::calc_stride(outshape.get<ndim>()),
                                  outshape.get<ndim>(), values.shape_.get<ndim>(),
                                  is_insert, origin_idx, dtype, vtype, req[out_pos],
                                  axis, outshape.Size());
    
  });
}

NNVM_REGISTER_OP(_npi_insert)
.set_attr<FCompute>("FCompute<gpu>", NumpyInsertCompute<gpu>);

}
}
