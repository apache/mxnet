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
 * \file index_add-inl.h
 * \brief Function definition of index_add operator
*/
#ifndef MXNET_OPERATOR_TENSOR_INDEX_ADD_INL_H_
#define MXNET_OPERATOR_TENSOR_INDEX_ADD_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <algorithm>
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

inline bool IndexModifyOpShape(const nnvm::NodeAttrs& attrs,
                               mxnet::ShapeVector* in_attrs,
                               mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 1U);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, (*in_attrs)[0]);
  return true;
}

inline bool IndexModifyOpType(const nnvm::NodeAttrs& attrs,
                              std::vector<int>* in_attrs,
                              std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 1U);
  CHECK_NE((*in_attrs)[0], -1);
  CHECK_NE((*in_attrs)[1], -1);
  CHECK_NE((*in_attrs)[2], -1);
  CHECK_EQ((*in_attrs)[0], (*in_attrs)[2])
    << "index_add(a, ind, val) only support a.dtype == val.dtype";
  CHECK((*in_attrs)[1] == mshadow::kInt64 ||
        (*in_attrs)[1] == mshadow::kInt32)
    << "'ind' only support int dtype.";
  TYPE_ASSIGN_CHECK(*out_attrs, 0, (*in_attrs)[0]);
  return (*out_attrs)[0] != -1;
}

template<typename xpu, typename DType>
void IndexAddForwardCalc(mshadow::Stream<xpu> *s,
                         const int ind_num, DType* out,
                         const DType* val,
                         const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM>& a_tail_shape,
                         const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM>& a_pre_stride,
                         const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM>& val_stride,
                         const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM>& val_shape,
                         const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM>& a_shape,
                         const int a_tail_size,
                         const int ind_ndim, const int* ind,
                         const int a_ndim);

template<typename xpu>
void IndexAddOpForward(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 1U);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob a = inputs[0];
  TBlob ind = inputs[1];
  TBlob val = inputs[2];
  TBlob out = outputs[0];
  CHECK_GT(a.shape_.ndim(), 0) << "The first input is saclar, please use '+' instead.";
  int a_ndim = a.shape_.ndim();
  CHECK_LE(a_ndim, MXNET_SPECIAL_MAX_NDIM)
    << "ndim should less than "<< MXNET_SPECIAL_MAX_NDIM
    << "but get " << a_ndim <<"\n";
  int val_ndim = val.shape_.ndim();
  if (val_ndim == 0) {
    val.shape_ = Shape1(1);
    val_ndim = 1;
  }
  // ind=np.array([]), ind.shape_.ndim() = 1
  // ind=np.array(1), ind.shape_.ndim() = 0
  // ind=np.array([[0,0],[0,1]]), ind.shape_.ndim() = 2
  CHECK_NE(ind.shape_.Size(), 0) << "Param 'ind' is []. Please just use op 'add' instead.\n";
  CHECK_LE(ind.shape_.ndim(), 2) << "'ind' array allow 2 dimension at most.";
  if (ind.shape_.ndim() == 0) {
    ind.shape_ = Shape2(1, 1);
  } else if (ind.shape_.ndim() == 1) {
    ind.shape_ = Shape2(1, ind.shape_[0]);
  }
  int ind_ndim = ind.shape_[0];
  int ind_num = ind.shape_[1];
  CHECK_LE(ind_ndim, a_ndim) << "IndexError: too many indices for array.";

  // check 'val' broadcast legality
  CHECK_LE(val_ndim, a_ndim - ind_ndim + 1)
    << "The ndim of param 'val' is " << val_ndim
    << ", but it should less than or equal to " << a_ndim - ind_ndim + 1;
  for (int i = a_ndim - 1, j = val_ndim - 1; j >= 0 ; --i, --j) {
    if ((j == 0) && (val_ndim == a_ndim - ind_ndim + 1)) {
      // val_ndim == a_ndim - ind_ndim + 1, check the first dim of input 'val'
      CHECK(val.shape_[j] == ind_num || val.shape_[j] == 1)
        << "can not broadcast from " << val.shape_[j] << " to " << ind_num;
    } else {
      CHECK(val.shape_[j] == a.shape_[i] || val.shape_[j] == 1)
        << "can not broadcast from " << val.shape_[j] << " to " << a.shape_[i]
        << " in axis " << i;
    }
  }
  int a_tail_size = static_cast<int>(a.shape_.ProdShape(ind_ndim, a_ndim));
  mshadow::Shape<MXNET_SPECIAL_MAX_NDIM>a_shape, val_shape;
  for (int i = MXNET_SPECIAL_MAX_NDIM - 1, j = a_ndim - 1; i >= 0; --i, --j) {
    a_shape[i] = (j >= 0) ? a.shape_[j] : 1;
  }
  mshadow::Shape<MXNET_SPECIAL_MAX_NDIM>a_pre_shape(a_shape);
  mshadow::Shape<MXNET_SPECIAL_MAX_NDIM>a_tail_shape(a_shape);

  int seg = MXNET_SPECIAL_MAX_NDIM - a_ndim;
  for (int i = seg; i < ind_ndim + seg; ++i) {
    a_tail_shape[i] = 1;
  }
  for (int i = ind_ndim + seg; i < a_ndim + seg; ++i) {
    a_pre_shape[i] = 1;
  }
  for (int i = MXNET_SPECIAL_MAX_NDIM - 1, j = val_ndim - 1; i >= 0; --i, --j) {
    val_shape[i] = (j >= 0) ? val.shape_[j] : 1;
  }
  mshadow::Shape<MXNET_SPECIAL_MAX_NDIM>a_pre_stride = calc_stride(a_pre_shape);
  mshadow::Shape<MXNET_SPECIAL_MAX_NDIM>val_stride = calc_stride(val_shape);
  mxnet_op::copy(s, out, a);
  Tensor<xpu, 1, char> temp_mem = ctx.requested[0].get_space_typed<xpu, 1, char>(
                                  Shape1(sizeof(int) * ind_ndim * ind_num), s);
  TBlob t_ind = TBlob(reinterpret_cast<int*>(temp_mem.dptr_),
                      Shape1(ind_ndim * ind_num), xpu::kDevMask);
  mxnet_op::copy(s, t_ind, ind);
  MSHADOW_TYPE_SWITCH(a.type_flag_, DType, {
    IndexAddForwardCalc<xpu, DType>(s, ind_num,
                                    out.dptr<DType>(), val.dptr<DType>(),
                                    a_tail_shape, a_pre_stride,
                                    val_stride, val_shape, a_shape,
                                    a_tail_size, ind_ndim,
                                    t_ind.dptr<int>(), a_ndim);
  });
}

template<typename DType>
struct IndexAddBackwardAKernel {
  MSHADOW_XINLINE static void Map(size_t i, DType* grad_a,
                                  const DType* ograd,
                                  const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM>& stride,
                                  const int tail_size, const int ind_num, const int ind_ndim,
                                  const int32_t* ind_vec, const int req, const int out_ndim) {
    index_t id = 0;
    int seg = MXNET_SPECIAL_MAX_NDIM - out_ndim;
    for (int dim = 0; dim < ind_ndim; ++dim) {
      id += stride[seg + dim] * ind_vec[dim * ind_num + i];
    }
    for (int _i = 0; _i < tail_size; ++_i) {
      KERNEL_ASSIGN(grad_a[id + _i], req, ograd[id + _i]);
    }
  }
};

template<typename xpu, typename DType>
void IndexAddOpBackwardACalc(mshadow::Stream<xpu> *s,
                             DType* grad_a, const DType* ograd,
                             const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM>& stride,
                             const int tail_size, const int ind_num,
                             const int ind_ndim, const int32_t* ind_vec,
                             const int req, const int out_ndim);

template<typename xpu, typename DType>
void IndexAddOpBackwardValCalc(mshadow::Stream<xpu> *s,
                               DType* grad_val, const DType* ograd,
                               const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM>& ograd_tail_shape,
                               const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM>& ograd_pre_stride,
                               const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM>& val_stride,
                               const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM>& val_shape,
                               const int tail_size, const int ind_num,
                               const int ind_ndim, const int32_t* ind_vec,
                               const int out_ndim);

template<typename xpu>
void IndexAddOpBackward(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  if (req[0] == kNullOp && req[2] == kNullOp) {
    return;
  }
  CHECK_EQ(inputs.size(), 4);
  CHECK_EQ(outputs.size(), 3);
  const TBlob& ograd = inputs[0];
  const TBlob& a = inputs[1];
  TBlob ind = inputs[2];
  const TBlob& val = inputs[3];
  const TBlob& grad_a = outputs[0];
  const TBlob& grad_val = outputs[2];
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  // get the number of 'ind' index
  if (ind.shape_.ndim() == 0) {
    ind.shape_ = Shape2(1, 1);
  } else if (ind.shape_.ndim() == 1) {
    ind.shape_ = Shape2(1, ind.shape_[0]);
  }
  int ind_ndim = ind.shape_[0];
  int ind_num = ind.shape_[1];
  // broadcast 'ind'
  int ndim = ograd.shape_.ndim();
  int tail_size = static_cast<int>(ograd.shape_.ProdShape(ind_ndim, ndim));
  mshadow::Shape<MXNET_SPECIAL_MAX_NDIM>ograd_shape, val_shape;
  for (int i = MXNET_SPECIAL_MAX_NDIM - 1, j = ndim - 1; i >= 0; --i, --j) {
    ograd_shape[i] = (j >= 0) ? ograd.shape_[j] : 1;
  }
  mshadow::Shape<MXNET_SPECIAL_MAX_NDIM>ograd_stride = mxnet_op::calc_stride(ograd_shape);
  mshadow::Shape<MXNET_SPECIAL_MAX_NDIM>ograd_pre_shape(ograd_shape);
  mshadow::Shape<MXNET_SPECIAL_MAX_NDIM>ograd_tail_shape(ograd_shape);
  Tensor<xpu, 1, char> temp_mem = ctx.requested[0].get_space_typed<xpu, 1, char>(
                                  Shape1(sizeof(int32_t) * ind_ndim * ind_num), s);
  TBlob t_ind = TBlob(reinterpret_cast<int32_t*>(temp_mem.dptr_), ind.shape_, xpu::kDevMask);
  mxnet_op::copy(s, t_ind, ind);
  MSHADOW_TYPE_SWITCH(grad_a.type_flag_, DType, {
    IndexAddOpBackwardACalc<xpu, DType>(s, grad_a.dptr<DType>(),
      ograd.dptr<DType>(), ograd_stride, tail_size, ind_num, ind_ndim,
      t_ind.dptr<int32_t>(), req[0], ndim);
  });

  int seg = MXNET_SPECIAL_MAX_NDIM - ndim;
  for (int i = seg; i < seg + ind_ndim; ++i) {
    ograd_tail_shape[i] = 1;
  }
  for (int i = seg + ind_ndim; i < seg + ndim; ++i) {
    ograd_pre_shape[i] = 1;
  }
  for (int i = seg + ndim - 1, j = grad_val.shape_.ndim() - 1; i >= seg; --i, --j) {
    val_shape[i] = (j >= 0) ? val.shape_[j] : 1;
  }
  mshadow::Shape<MXNET_SPECIAL_MAX_NDIM>ograd_pre_stride = mxnet_op::calc_stride(ograd_pre_shape);
  mshadow::Shape<MXNET_SPECIAL_MAX_NDIM>val_stride = mxnet_op::calc_stride(val_shape);

  MSHADOW_TYPE_SWITCH(grad_val.type_flag_, DType, {
    IndexAddOpBackwardValCalc<xpu, DType>(
      s, grad_val.dptr<DType>(), ograd.dptr<DType>(),
      ograd_tail_shape, ograd_pre_stride, val_stride,
      val_shape, tail_size, ind_num, ind_ndim, t_ind.dptr<int32_t>(), ndim);
  });
}

}   // namespace op
}   // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_INDEX_ADD_INL_H_
