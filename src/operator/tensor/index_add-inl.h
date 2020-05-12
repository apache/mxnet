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

#include <vector>
#include <algorithm>
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

struct IndexModifyParam : public dmlc::Parameter<IndexModifyParam> {
    mxnet::Tuple<mxnet::Tuple<int>> ind;
    DMLC_DECLARE_PARAMETER(IndexModifyParam) {
      DMLC_DECLARE_FIELD(ind)
        .describe("Index indicating where the input added values.");
    }
};

inline bool IndexModifyOpShape(const nnvm::NodeAttrs& attrs,
                               mxnet::ShapeVector* in_attrs,
                               mxnet::ShapeVector* out_attrs) {
  IndexModifyParam param = nnvm::get<IndexModifyParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, (*in_attrs)[0]);
  return true;
}

inline bool IndexModifyOpType(const nnvm::NodeAttrs& attrs,
                              std::vector<int>* in_attrs,
                              std::vector<int>* out_attrs) {
  const IndexModifyParam param = nnvm::get<IndexModifyParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  CHECK_NE((*in_attrs)[0], -1);
  CHECK_NE((*in_attrs)[1], -1);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, (*in_attrs)[0]);
  return (*out_attrs)[0] != -1;
}

template<typename xpu, typename DType, typename VType, int NDim>
void IndexAddForwardCalc(mshadow::Stream<xpu> *s,
                         const int ind_num, DType* out,
                         const VType* val,
                         const mshadow::Shape<NDim>& a_tail_shape,
                         const mshadow::Shape<NDim>& a_pre_stride,
                         const mshadow::Shape<NDim>& val_stride,
                         const mshadow::Shape<NDim>& val_shape,
                         const size_t a_tail_size,
                         const int ind_ndim, const int* ind_vec);

template<typename xpu>
void IndexAddOpForward(const nnvm::NodeAttrs& attrs,
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
  CHECK_NE(a.shape_.ndim(), 0) << "Please use '+' instead.";
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
  CHECK_NE(ind_ndim, 0) << "Param 'ind' is (). Please use op 'add' instead.\n";

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
  // MSHADOW_REAL_TYPE_SWITCH(a.type_flag_, DType, {
  //   MXNET_NDIM_SWITCH(a_ndim, NDim, {
  //     mxnet_op::copy(s, out, a);
  //     mshadow::Shape<NDim>a_shape = a.shape_.get<NDim>();
  //     mshadow::Shape<NDim>a_pre_shape(a_shape);
  //     mshadow::Shape<NDim>a_tail_shape(a_shape);
  //     mshadow::Shape<NDim>val_shape;
  //     for (int i = 0; i < ind_ndim; ++i) {
  //       a_tail_shape[i] = 1;
  //     }
  //     for (int i = ind_ndim; i < NDim; ++i) {
  //       a_pre_shape[i] = 1;
  //     }
  //     for (int i = NDim - 1, j = val_ndim - 1; i >= 0; --i, --j) {
  //       val_shape[i] = (j >= 0) ? val.shape_[j] : 1;
  //     }
  //     mshadow::Shape<NDim>a_pre_stride = mxnet_op::calc_stride(a_pre_shape);
  //     mshadow::Shape<NDim>val_stride = mxnet_op::calc_stride(val_shape);
  //     MSHADOW_REAL_TYPE_SWITCH(val.type_flag_, VType, {
  //       IndexAddForwardCalc<xpu, DType, VType, NDim>(s, ind_num,
  //                                                    out.dptr<DType>(), val.dptr<VType>(),
  //                                                    a_tail_shape, a_pre_stride, val_stride,
  //                                                    val_shape, a_tail_size, ind_ndim,
  //                                                    vec_ind.data());
  //     });
  //   });
  // });
}

template<typename DType, typename OType, int NDim>
struct IndexAddBackwardAKernel {
  MSHADOW_XINLINE static void Map(size_t i, DType* grad_a,
                                  const OType* ograd,
                                  const mshadow::Shape<NDim> stride,
                                  const size_t tail_size,
                                  const int ind_num, const int ind_ndim,
                                  const int* ind_vec, const int req) {
    size_t id = 0;
    for (int dim = 0; dim < ind_ndim; ++dim) {
      id += stride[dim] * ind_vec[dim * ind_num + i];
    }
    for (int _i = 0; _i < tail_size; ++_i) {
      KERNEL_ASSIGN(grad_a[id + _i], req, static_cast<DType>(ograd[id + _i]));
    }
  }
};

template<typename xpu, typename DType, typename OType, int NDim>
void IndexAddOpBackwardACalc(mshadow::Stream<xpu> *s,
                             DType* grad_a, const OType* ograd,
                             const mshadow::Shape<NDim>& stride,
                             const size_t tail_size, const int ind_num,
                             const int ind_ndim, const int* ind_vec,
                             const int req);

template<typename xpu, typename DType, typename OType, int NDim>
void IndexAddOpBackwardValCalc(mshadow::Stream<xpu> *s,
                               DType* grad_val, const OType* ograd,
                               const mshadow::Shape<NDim>& ograd_tail_shape,
                               const mshadow::Shape<NDim>& ograd_pre_stride,
                               const mshadow::Shape<NDim>& val_stride,
                               const mshadow::Shape<NDim>& val_shape,
                               const size_t tail_size, const int ind_num,
                               const int ind_ndim, const int* ind_vec);

template<typename xpu>
void IndexAddOpBackward(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  if (req[0] == kNullOp && req[1] == kNullOp) {
    return;
  }
  CHECK_EQ(inputs.size(), 3);
  CHECK_EQ(outputs.size(), 2);
  const TBlob& ograd = inputs[0];
  const TBlob& a = inputs[1];
  const TBlob& val = inputs[2];
  const TBlob& grad_a = outputs[0];
  const TBlob& grad_val = outputs[1];
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const IndexModifyParam& param = nnvm::get<IndexModifyParam>(attrs.parsed);
  // get the number of 'ind' index
  int ind_num = 0;
  int ind_ndim = param.ind.ndim();
  for (int p_dim = 0; p_dim < ind_ndim; ++p_dim) {
    ind_num = (param.ind[p_dim].ndim() > ind_num) ? param.ind[p_dim].ndim() : ind_num;
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
  int ndim = ograd.shape_.ndim();
  size_t tail_size = ograd.shape_.ProdShape(ind_ndim, ndim);
  // MSHADOW_REAL_TYPE_SWITCH(grad_a.type_flag_, DType, {
  //   MSHADOW_REAL_TYPE_SWITCH(ograd.type_flag_, OType, {
  //     MXNET_NDIM_SWITCH(ndim, NDim, {
  //       mshadow::Shape<NDim> stride = mxnet_op::calc_stride(ograd.shape_.get<NDim>());
  //       IndexAddOpBackwardACalc<xpu, DType, OType, NDim>(s, grad_a.dptr<DType>(),
  //         ograd.dptr<OType>(), stride, tail_size, ind_num, ind_ndim,
  //         vec_ind.data(), req[0]);
  //     });
  //   });
  // });
  // MSHADOW_REAL_TYPE_SWITCH(grad_val.type_flag_, DType, {
  //   MXNET_NDIM_SWITCH(ndim, NDim, {
  //     mshadow::Shape<NDim>ograd_shape = ograd.shape_.get<NDim>();
  //     mshadow::Shape<NDim>ograd_pre_shape(ograd_shape);
  //     mshadow::Shape<NDim>ograd_tail_shape(ograd_shape);
  //     mshadow::Shape<NDim>val_shape;
  //     for (int i = 0; i < ind_ndim; ++i) {
  //       ograd_tail_shape[i] = 1;
  //     }
  //     for (int i = ind_ndim; i < NDim; ++i) {
  //       ograd_pre_shape[i] = 1;
  //     }
  //     for (int i = NDim - 1, j = grad_val.shape_.ndim() - 1; i >= 0; --i, --j) {
  //       val_shape[i] = (j >= 0) ? val.shape_[j] : 1;
  //     }
  //     mshadow::Shape<NDim>ograd_pre_stride = mxnet_op::calc_stride(ograd_pre_shape);
  //     mshadow::Shape<NDim>val_stride = mxnet_op::calc_stride(val_shape);
  //     MSHADOW_REAL_TYPE_SWITCH(ograd.type_flag_, OType, {
  //       IndexAddOpBackwardValCalc<xpu, DType, OType>(
  //         s, grad_val.dptr<DType>(), ograd.dptr<OType>(),
  //         ograd_tail_shape, ograd_pre_stride, val_stride, val_shape,
  //         tail_size, ind_num, ind_ndim, vec_ind.data());
  //     });
  //   });
  // });
}

}   // namespace op
}   // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_INDEX_ADD_INL_H_
