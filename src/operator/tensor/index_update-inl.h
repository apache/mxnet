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
 * \file index_update-inl.h
 * \brief Function definition of index_update operator
*/
#ifndef MXNET_OPERATOR_TENSOR_INDEX_UPDATE_INL_H_
#define MXNET_OPERATOR_TENSOR_INDEX_UPDATE_INL_H_

#include <vector>
#include <algorithm>
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "index_add-inl.h"

namespace mxnet {
namespace op {

template<typename DType, typename VType, int NDim>
struct IndexUpdateForwardKernel {
  MSHADOW_XINLINE static void Map(size_t i, DType* out,
                                  const VType* val,
                                  const mshadow::Shape<NDim> a_tail_shape,
                                  const mshadow::Shape<NDim> a_pre_stride,
                                  const mshadow::Shape<NDim> val_stride,
                                  const mshadow::Shape<NDim> val_shape,
                                  const size_t a_tail_size, const int ind_num,
                                  const int ind_ndim, const int* ind_vec,
                                  const int req, int64_t* pre) {
    size_t id = 0;
    for (int dim = 0; dim < ind_ndim; ++dim) {
      id += a_pre_stride[dim] * ind_vec[dim * ind_num + i];
    }
    if (i >= pre[id]) {
      printf("i:%d id:%d pre[id]:%d\n",i,id,pre[id]);
      pre[id] = i;
      id *= a_tail_size;
      for (int _i = 0; _i < a_tail_size; ++_i) {
        mshadow::Shape<NDim> a_tail_id = mxnet_op::unravel(_i, a_tail_shape);
        mshadow::Shape<NDim> val_id;
        for (int _j = 0; _j < NDim; ++_j) {
          val_id[_j] = (val_shape[_j] == 1) ? 0 : a_tail_id[_j];
        }
        val_id[ind_ndim - 1] = (val_shape[ind_ndim - 1] == 1) ? 0 : i;
        size_t val_dest = mxnet_op::dot(val_id, val_stride);
        KERNEL_ASSIGN(out[id + _i], req, static_cast<DType>(val[val_dest]));
      }
    }
  }
};

template<typename xpu, typename DType, typename VType, int NDim>
void IndexUpdateForwardImpl(mshadow::Stream<xpu> *s,
                            const int ind_num, DType* out,
                            const VType* val,
                            const mshadow::Shape<NDim>& a_tail_shape,
                            const mshadow::Shape<NDim>& a_pre_stride,
                            const mshadow::Shape<NDim>& val_stride,
                            const mshadow::Shape<NDim>& val_shape,
                            const size_t a_tail_size,
                            const int ind_ndim, const int* ind_vec,
                            const int req, int64_t* pre);

template<typename xpu>
void IndexUpdateOpForward(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const IndexModifyParam& param = nnvm::get<IndexModifyParam>(attrs.parsed);
  const TBlob a = inputs[0];
  TBlob val = inputs[1];
  TBlob out = outputs[0];
  CHECK_NE(a.shape_.ndim(), 0) << "Please use '=' instead.";
  int a_ndim = a.shape_.ndim();
  int val_ndim = inputs[1].shape_.ndim();
  if (val_ndim == 0) {
    val.shape_ = Shape1(1);
    val_ndim = 1;
  }
  int ind_ndim = param.ind.ndim();
  CHECK_LE(ind_ndim, a_ndim) << "IndexError: too many indices for array.";
  // ind=(), dim:0, ind[0] is invalid
  // ind=(1), dim:1, ind[0].ndim():1
  // ind=((0,0),(0,1)), dim:2, ind[0].ndim():2
  CHECK_NE(ind_ndim, 0) << "Param 'ind' is (). Please use ‘=’ directly.\n";

  // get the number of 'ind' index
  int ind_num = 0;
  for (int p_dim = 0; p_dim < ind_ndim; ++p_dim) {
    ind_num = (param.ind[p_dim].ndim() > ind_num) ? param.ind[p_dim].ndim() : ind_num;
  }
  // check 'ind' data legality
  for (int p_dim = 0; p_dim < ind_ndim; ++p_dim) {
    // broadcast check
    CHECK((param.ind[p_dim].ndim() == ind_num) ||
          (param.ind[p_dim].ndim() == 1) ||
          (param.ind[p_dim].ndim() == 0))
      << "IndexError: shape mismatch: indexing arrays could not be broadcast together"
      << " with shapes (" << ind_num << ",) (" << param.ind[p_dim].ndim() << ",)";
    if (param.ind[p_dim].ndim() == 0) {
      // nothing changed
      return;
    }
    // bounds check
    for (int p_num = 0; p_num < param.ind[p_dim].ndim(); ++p_num) {
      CHECK_LE(param.ind[p_dim][p_num], a.shape_[p_dim])
        << "IndexError: index " << param.ind[p_dim][p_num]
        << " is out of bounds for axis " << p_dim
        << " with size " << a.shape_[p_dim];
    }
  }
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

  // broadcast 'ind'
  size_t vec_size = ind_ndim * ind_num;
  std::vector<int>vec_ind(vec_size);
  for (int p_dim = 0; p_dim < ind_ndim; ++p_dim) {
    for (int p_num = 0; p_num < ind_num; ++p_num) {
      vec_ind[p_dim * ind_num + p_num] = param.ind[p_dim].ndim() == 1 ?
                                         param.ind[p_dim][0] :
                                         param.ind[p_dim][p_num];
    }
  }
  size_t a_tail_size = a.shape_.ProdShape(ind_ndim, a_ndim);
  MSHADOW_TYPE_SWITCH(a.type_flag_, DType, {
    MXNET_NDIM_SWITCH(a_ndim, NDim, {
      mxnet_op::copy(s, out, a);
      mshadow::Shape<NDim>a_shape = a.shape_.get<NDim>();
      mshadow::Shape<NDim>a_pre_shape(a_shape);
      mshadow::Shape<NDim>a_tail_shape(a_shape);
      mshadow::Shape<NDim>val_shape;
      for (int i = 0; i < ind_ndim; ++i) {
        a_tail_shape[i] = 1;
      }
      for (int i = ind_ndim; i < NDim; ++i) {
        a_pre_shape[i] = 1;
      }
      for (int i = NDim - 1, j = val_ndim - 1; i >= 0; --i, --j) {
        val_shape[i] = (j >= 0) ? val.shape_[j] : 1;
      }
      mshadow::Shape<NDim>a_pre_stride = mxnet_op::calc_stride(a_pre_shape);
      mshadow::Shape<NDim>val_stride = mxnet_op::calc_stride(val_shape);
      MSHADOW_TYPE_SWITCH(val.type_flag_, VType, {
        size_t pre_size = a.shape_.ProdShape(0, ind_ndim);
        Tensor<xpu, 1, int64_t> pre = ctx.requested[0].get_space_typed<xpu, 1, int64_t>(
                                      Shape1(pre_size), s);  // record the index of updats value
        // If different indexes point to the same position, the last value will be updated.
        // example:
        // before: a = [[0, 0], [0, 0]], ind = ((0, 0), (0, 0)) val = [1, 2]
        // after index_update(a, val, ind) : a = [[0, 2], [0, 0]]
        
        Kernel<set_zero, xpu>::Launch(s, pre_size, pre.dptr_);
        IndexUpdateForwardImpl<xpu, DType, VType, NDim>(s, ind_num,
                                                        out.dptr<DType>(), val.dptr<VType>(),
                                                        a_tail_shape, a_pre_stride, val_stride,
                                                        val_shape, a_tail_size, ind_ndim,
                                                        vec_ind.data(), req[0], pre.dptr_);
      });
    });
  });   
}

}   // namespace op
}   // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_INDEX_ADD_INL_H_
