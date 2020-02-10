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
 * \file np_insert_op.cc
 * \brief CPU Implementation of numpy insert operations
 */

#include <vector>
#include "./np_insert_op-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(NumpyInsertParam);

bool NumpyInsertType(const nnvm::NodeAttrs& attrs,
                     std::vector<int> *in_type,
                     std::vector<int> *out_type) {
  const NumpyInsertParam& param = nnvm::get<NumpyInsertParam>(attrs.parsed);
  int input_count = param.val.has_value() ? 1 : 2;
  int insize = (param.step.has_value() || param.int_ind.has_value()) ?
               input_count : input_count + 1;
  bool obj_is_tensor = !param.step.has_value() && !param.int_ind.has_value();
  CHECK_EQ(in_type->size(), insize);
  CHECK_EQ(out_type->size(), 1U);
  if (obj_is_tensor) {
    int obj_pos = input_count;
    CHECK_NE((*in_type)[obj_pos], -1) << "Index type must be set for insert operator\n";
    CHECK_EQ((*in_type)[obj_pos], mshadow::DataType<int64_t>::kFlag)
      << "Index type only support int64.\n";
  }
  TYPE_ASSIGN_CHECK(*out_type, 0, (*in_type)[0]);  // output type equals to input arr's
  TYPE_ASSIGN_CHECK(*in_type, 0, (*out_type)[0]);
  return (*in_type)[0] != -1;
}

bool NumpyInsertShape(const nnvm::NodeAttrs& attrs,
                      mxnet::ShapeVector *in_shape,
                      mxnet::ShapeVector *out_shape) {
  using namespace mshadow;
  const NumpyInsertParam& param = nnvm::get<NumpyInsertParam>(attrs.parsed);
  int input_count = param.val.has_value() ? 1 : 2;
  int insize = (param.step.has_value() || param.int_ind.has_value()) ?
               input_count : input_count + 1;
  bool obj_is_tensor = !param.step.has_value() && !param.int_ind.has_value();
  const int arr_pos = 0;
  const int val_pos = param.val.has_value() ? 0 : 1;
  const int obj_pos = val_pos + 1;
  CHECK_EQ(in_shape->size(), insize);
  mxnet::TShape scale_shape(0, 1);
  mxnet::TShape &arrshape = (*in_shape)[arr_pos];
  mxnet::TShape &valshape = param.val.has_value() ? scale_shape : (*in_shape)[val_pos];
  mxnet::TShape &objShape = obj_is_tensor ? (*in_shape)[obj_pos] : scale_shape;
  if (obj_is_tensor) {
    CHECK_LE(objShape.ndim(), 1)
      << "index array argument obj to insert must be one dimensional or scale.\n";
  }

  out_shape->clear();

  int ndim = arrshape.ndim();
  int axis = param.axis.has_value() ? param.axis.value() : 0;
  if (!(param.axis.has_value())) {
    arrshape = Shape1(arrshape.Size());
    ndim = 1;
  } else if (ndim == 0) {
    if (param.val.has_value()) {
      out_shape->push_back(scale_shape);
    } else {
      CHECK_EQ(valshape.ndim(), 0)
        << "'arr' is a 0-d array, 'values' can not assign to it. "
        << "alueError: assignment to 0-d array.";
      out_shape->push_back(valshape);
    }
    return shape_is_known(out_shape[0]);
  } else {
    CHECK(axis >= -1 * arrshape.ndim() && axis < arrshape.ndim())
      << "Axis should be in the range of [-r, r-1] where r is the rank of input tensor";
    axis += (axis < 0) ? arrshape.ndim() : 0;
  }

  int seq_cnt = -1;
  int N = arrshape[axis];
  if (obj_is_tensor) {
    seq_cnt = objShape.Size();
  } else if (param.step.has_value()) {
    int step = param.step.value();
    int stop, start;
    if (param.stop.has_value()) {
      stop = param.stop.value();
      stop += (stop < 0) ? N : 0;
      stop = (stop < 0) ? ((step < 0) ? -1 : 0) : stop;
      stop = (stop >= N) ? ((step < 0) ? N - 1 : N) : stop;
    } else {
      stop = (step > 0) ? N : -1;
    }
    if (param.start.has_value()) {
      start = param.start.value();
      start += (start < 0) ? N : 0;
      start = (start < 0) ? ((step < 0) ? -1 : 0) : start;
      start = (start >= N) ? ((step < 0) ? N - 1 : N) : start;
    } else {
      start = (step > 0) ? 0 : N - 1;
    }
    seq_cnt = 0;
    if (step > 0 && stop >= start) {
      seq_cnt = (stop - start + step - 1) / step;
    } else if (step < 0 && stop <= start) {
      seq_cnt = (stop - start + step + 1) / step;
    }
  }

  mxnet::TShape newshape(arrshape);
  mxnet::TShape val_newshape(arrshape.ndim(), -1);
  int numnew = 0;  // amount of new column insert to 'arr' in 'axis'
  // modify values's ndim to arr's ndim, for broadcast easily later
  // e.g. value shape: (2,) arr shape: (3, 2) => value shape: (1, 2)
  for (int i = valshape.ndim() - 1, j = arrshape.ndim() - 1; i >= 0 || j >= 0; --i, --j) {
    if (i >= 0 && j >= 0) {
      val_newshape[j] = valshape[i];
    } else if (i >= 0) {
      CHECK_EQ(valshape[i], 1) << "index exceed limits.";
    } else {
      val_newshape[j] = 1;
    }
  }
  valshape.assign(val_newshape.begin(), val_newshape.end());

  if (param.int_ind.has_value() ||
      (obj_is_tensor && objShape.ndim() == 0)) {
    // because of moveaxis(values, 0, axis)
    numnew = valshape[0];
  } else if (seq_cnt == 1) {
    numnew = valshape[axis];
  } else {
    numnew = seq_cnt;
  }

  newshape[axis] += numnew;
  out_shape->push_back(newshape);
  return shape_is_known(newshape);
}

template<>
void InsertOneIndicesImpl<cpu>(const OpContext &ctx,
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
  mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
  MXNET_NDIM_SWITCH(outshape.ndim(), ndim, {
    if (param.step.has_value()) {
      InsertScalerObj<cpu, ndim>(s, outputs[out_pos], arr, values,
                                  mxnet_op::calc_stride(arr.shape_.get<ndim>()),
                                  mxnet_op::calc_stride(values.shape_.get<ndim>()),
                                  mxnet_op::calc_stride(old_valshape.get<ndim>()),
                                  mxnet_op::calc_stride(outshape.get<ndim>()),
                                  outshape.get<ndim>(), values.shape_.get<ndim>(),
                                  dtype, vtype, req[out_pos], axis, start, numnew,
                                  outshape.Size(), false);
    } else {
      InsertSizeOneTensorObj<cpu, ndim>(s, outputs[out_pos], arr, values,
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
void InsertTensorIndicesImpl<cpu>(const OpContext &ctx,
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
  mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();

  for (int i = outshape.ndim() - 1; i >= 0; --i) {
    int sz = outshape[i];
    if (i == axis) {
      sz = numnew;
    }
    CHECK((values.shape_[i] == 1) || (values.shape_[i] == sz));
  }
  size_t temp_storage_bytes, temp_mem_size;
  temp_storage_bytes = SortByKeyWorkspaceSize<int64_t, int, cpu>(indices_len, false, true);
  temp_mem_size = indices_len * sizeof(int64_t) * 2 +
                  indices_len * sizeof(int) +
                  outshape[axis] * sizeof(int) * 2 +
                  temp_storage_bytes;
  Tensor<cpu, 1, char> temp_mem =
    ctx.requested[0].get_space_typed<cpu, 1, char>(Shape1(temp_mem_size), s);
  int64_t* indices_ptr = reinterpret_cast<int64_t*>(temp_mem.dptr_);
  int64_t* sorted_indices_ptr = reinterpret_cast<int64_t*>(indices_ptr + indices_len);
  int* order_ptr = reinterpret_cast<int*>(sorted_indices_ptr + indices_len);
  int* is_insert = reinterpret_cast<int*>(order_ptr + indices_len);
  int* origin_idx = reinterpret_cast<int*>(is_insert + outshape[axis]);
  Tensor<cpu, 1, char> temp_storage(reinterpret_cast<char*>(origin_idx + outshape[axis]),
                                    Shape1(temp_storage_bytes), s);
  Tensor<cpu, 1, int64_t> indices(indices_ptr, Shape1(indices_len), s);
  Tensor<cpu, 1, int64_t> sorted_indices(sorted_indices_ptr, Shape1(indices_len), s);
  Tensor<cpu, 1, int> order(order_ptr, Shape1(indices_len), s);
  int num_bits = common::ilog2ui(static_cast<unsigned int>(indices_len) - 1);
  if (param.step.has_value()) {
    Kernel<SliceToIndices, cpu>::Launch(s, indices_len, indices_ptr, start, step);
  } else {
    Kernel<ObjToIndices, cpu>::Launch(s, indices_len, indices_ptr, N,
                                      inputs[obj_pos].dptr<int64_t>());
  }
  Kernel<range_fwd, cpu>::Launch(s, indices_len, 1, 0, 1, kWriteTo, order_ptr);
  mxnet::op::SortByKey(indices, order, true, &temp_storage, 0, num_bits, &sorted_indices);
  Kernel<IndicesModify, cpu>::Launch(s, indices_len, indices_ptr, order_ptr);

  mxnet_op::Kernel<mxnet_op::set_zero, cpu>::Launch(s, outshape[axis], is_insert);
  Kernel<SetIsInsert, cpu>::Launch(s, indices_len, indices_ptr, is_insert);

  Kernel<SetOriginValuesIdx, cpu>::Launch(s, indices_len, indices_ptr, origin_idx);
  Kernel<SetOriginArrIdx, cpu>::Launch(s, outshape[axis], is_insert, origin_idx);
  MXNET_NDIM_SWITCH(outshape.ndim(), ndim, {
    InsertSequenceObj<cpu, ndim>(s, outputs[out_pos], arr, values,
                                  mxnet_op::calc_stride(arr.shape_.get<ndim>()),
                                  mxnet_op::calc_stride(values.shape_.get<ndim>()),
                                  mxnet_op::calc_stride(outshape.get<ndim>()),
                                  outshape.get<ndim>(), values.shape_.get<ndim>(),
                                  is_insert, origin_idx, dtype, vtype, req[out_pos],
                                  axis, outshape.Size());
    
  });
}

NNVM_REGISTER_OP(_npi_insert)
.describe(R"code(Insert values along the given axis before the given indices.)code" ADD_FILELINE)
.set_attr_parser(ParamParser<NumpyInsertParam>)
.set_num_inputs([](const NodeAttrs& attrs) {
    const NumpyInsertParam& params = nnvm::get<NumpyInsertParam>(attrs.parsed);
    int input_count = params.val.has_value() ? 1 : 2;
    return (params.step.has_value() || params.int_ind.has_value()) ? input_count : input_count + 1;
})
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    const NumpyInsertParam& params = nnvm::get<NumpyInsertParam>(attrs.parsed);
    if (params.val.has_value()) {
      return (params.step.has_value() || params.int_ind.has_value()) ?
             std::vector<std::string>{"arr"} :
             std::vector<std::string>{"arr", "obj"};
    } else {
      return (params.step.has_value() || params.int_ind.has_value()) ?
             std::vector<std::string>{"arr", "values"} :
             std::vector<std::string>{"arr", "values", "obj"};
    }
})
.set_attr<mxnet::FInferShape>("FInferShape", NumpyInsertShape)
.set_attr<nnvm::FInferType>("FInferType", NumpyInsertType)
.set_attr<mxnet::FCompute>("FCompute<cpu>", NumpyInsertCompute<cpu>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.add_argument("arr", "NDArray-or-Symbol", "Input ndarray")
.add_argument("values", "NDArray-or-Symbol", "Input ndarray")
.add_argument("obj", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(NumpyInsertParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
