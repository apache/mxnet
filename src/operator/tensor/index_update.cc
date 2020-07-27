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
 * \file index_update-inl.cc
 * \brief implementation of index_update operator
*/
#include <vector>
#include "./index_update-inl.h"

namespace mxnet {
namespace op {

template<typename DType>
struct IndexUpdateForwardCPUKernel {
  MSHADOW_XINLINE static void Map(size_t i, DType* out,
                                  const DType* val,
                                  const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> a_tail_shape,
                                  const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> a_pre_stride,
                                  const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> val_stride,
                                  const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> val_shape,
                                  const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> a_shape,
                                  const int a_tail_size, const int ind_num,
                                  const int ind_ndim, const int* ind,
                                  const int a_ndim, const int seg) {
    index_t id = 0;
    for (int dim = 0; dim < ind_ndim; ++dim) {
      CHECK_LT(ind[dim * ind_num + i], a_shape[seg + dim])
        << "IndexError: index " << ind[dim * ind_num + i]
        << " is out of bounds for axis " << dim
        << " with size " << a_shape[seg + dim];
      CHECK_GE(ind[dim * ind_num + i], 0)
        << "IndexError: index " << ind[dim * ind_num + i]
        << " should be greater or equal to 0.";
      id += a_pre_stride[seg + dim] * ind[dim * ind_num + i];
    }
    id *= a_tail_size;
    for (int _i = 0; _i < a_tail_size; ++_i) {
      mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> a_tail_id = mxnet_op::unravel(_i, a_tail_shape);
      mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> val_id;
      for (int _j = 0; _j < seg; ++_j) {
        val_id[_j] = 0;
      }
      for (int _j = seg; _j < seg + a_ndim; ++_j) {
        val_id[_j] = (val_shape[_j] == 1) ? 0 : a_tail_id[_j];
      }
      val_id[seg + ind_ndim - 1] = (val_shape[seg + ind_ndim - 1] == 1) ? 0 : i;
      index_t val_dest = mxnet_op::dot(val_id, val_stride);
      out[id + _i] = val[val_dest];
    }
  }
};

template<typename xpu, typename DType>
void IndexUpdateForwardCalc(mshadow::Stream<xpu> *s,
                            const int ind_num, DType* out,
                            const DType* val,
                            const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> a_tail_shape,
                            const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> a_pre_stride,
                            const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> val_stride,
                            const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> val_shape,
                            const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> a_shape,
                            const int a_tail_size,
                            const int ind_ndim, const int* ind,
                            const int a_ndim) {
  using namespace mxnet_op;
  using namespace mshadow;
  int seg = MXNET_SPECIAL_MAX_NDIM - a_ndim;
  Kernel<IndexUpdateForwardCPUKernel<DType>, xpu>::Launch(
    s, ind_num, out, val, a_tail_shape, a_pre_stride,
    val_stride, val_shape, a_shape, a_tail_size, ind_num,
    ind_ndim, ind, a_ndim, seg);
}


template<typename DType>
void IndexUpdateBackwardValCPUCompute(DType* grad_val,
                                      const DType* ograd,
                                      const int* ind_vec,
                                      const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> ograd_tail_shape,
                                      const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> ograd_pre_stride,
                                      const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> val_stride,
                                      const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> val_shape,
                                      const int ograd_tail_size, const int ind_num,
                                      const int ind_ndim, const int out_ndim,
                                      const int seg) {
  #pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (index_t i = 0; i < static_cast<index_t>(ind_num); ++i) {
    index_t id = 0;
    for (int dim = 0; dim < ind_ndim; ++dim) {
      id += ograd_pre_stride[seg + dim] * ind_vec[dim * ind_num + i];
    }
    id *= ograd_tail_size;
    for (int _i = 0; _i < ograd_tail_size; ++_i) {
      mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> ograd_tail_id =
        mxnet_op::unravel(_i, ograd_tail_shape);
      mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> val_id;
      for (int _j = 0; _j < seg; ++_j) {
        val_id[_j] = 0;
      }
      for (int _j = seg; _j < seg + out_ndim; ++_j) {
        val_id[_j] = (val_shape[_j] == 1) ? 0 : ograd_tail_id[_j];
      }
      val_id[seg + ind_ndim - 1] = (val_shape[seg + ind_ndim - 1] == 1) ? 0 : i;
      index_t val_dest = mxnet_op::dot(val_id, val_stride);
      #pragma omp critical
      {
        grad_val[val_dest] += ograd[id + _i];
      }
    }
  }
}

template<>
void IndexUpdateOpBackwardValImpl<cpu>(const OpContext& ctx,
                                const TBlob& grad_val,
                                const TBlob& ograd,
                                const TBlob& t_ind,
                                const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> ograd_tail_shape,
                                const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> ograd_pre_stride,
                                const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> val_stride,
                                const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> val_shape,
                                const int tail_size, const int ind_num, const int ind_ndim,
                                const int ndim) {
  using namespace mshadow;
  using namespace mxnet_op;
  int seg = MXNET_SPECIAL_MAX_NDIM - ndim;
  MSHADOW_TYPE_SWITCH(grad_val.type_flag_, DType, {
    IndexUpdateBackwardValCPUCompute<DType>(
      grad_val.dptr<DType>(), ograd.dptr<DType>(), t_ind.dptr<int>(),
      ograd_tail_shape, ograd_pre_stride, val_stride, val_shape, tail_size,
      ind_num, ind_ndim, ndim, seg);
  });
}

template<typename DType>
void IndexUpdateBackwardACPUCompute(DType* out_grad,
                                    const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> grada_pre_stride,
                                    const int tail_size, const int ind_num, const int ind_ndim,
                                    const int32_t* ind, const int seg,
                                    const int req) {
  #pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (index_t i = 0; i < static_cast<index_t>(ind_num); ++i) {
    index_t id = 0;
    for (int dim = 0; dim < ind_ndim; ++dim) {
      id += grada_pre_stride[seg + dim] * ind[dim * ind_num + i];
    }
    id *= tail_size;
    for (int _i = 0; _i < tail_size; ++_i) {
      out_grad[id + _i] = 0;
    }
  }
}

template<>
void IndexUpdateOpBackwardAImpl<cpu>(const OpContext& ctx,
                                     const TBlob& grad_a,
                                     const TBlob& ograd,
                                     const TBlob& ind,
                                     const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> grada_pre_stride,
                                     const int tail_size, const int ind_num, const int ind_ndim,
                                     const int seg, const int req) {
  using namespace mxnet_op;
  using namespace mshadow;
  mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
  MSHADOW_TYPE_SWITCH(grad_a.type_flag_, DType, {
    size_t temp_mem_size = ind.shape_.Size() * sizeof(int) +
                           ograd.shape_.Size() * sizeof(DType);
    Tensor<cpu, 1, char> temp_mem =
      ctx.requested[0].get_space_typed<cpu, 1, char>(Shape1(temp_mem_size), s);
    TBlob t_ograd = TBlob(temp_mem.dptr_, ograd.shape_, ograd.dev_mask(),
                          ograd.type_flag_, ograd.dev_id());
    TBlob t_ind = TBlob(temp_mem.dptr_ + ograd.Size() * sizeof(DType), ind.shape_, ind.dev_mask(),
                        mshadow::kInt32, ind.dev_id());
    mxnet_op::copy(s, t_ograd, ograd);
    mxnet_op::copy(s, t_ind, ind);
    IndexUpdateBackwardACPUCompute<DType>(t_ograd.dptr<DType>(),
                                          grada_pre_stride, tail_size,
                                          ind_num, ind_ndim,
                                          t_ind.dptr<int32_t>(), seg, req);
    Kernel<ReqCopy<DType>, cpu>::Launch(s, grad_a.shape_.Size(), grad_a.dptr<DType>(),
                                        t_ograd.dptr<DType>(), req);
  });
}

NNVM_REGISTER_OP(_npx_index_update)
.describe(R"code(This operators implements the "=" mimic function.
)code" ADD_FILELINE)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a", "ind", "val"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", IndexModifyOpShape)
.set_attr<nnvm::FInferType>("FInferType", IndexModifyOpType)
.set_attr<FCompute>("FCompute<cpu>", IndexUpdateOpForward<cpu>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::ObjectPtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
      auto a_grad = MakeNode("_backward_index_update_a", n->attrs.name + "_backward_a",
                              {ograds[0], n->inputs[1]}, nullptr, &n);
      auto idx_grad = MakeNode("zeros_like", n->attrs.name + "_backward_indices",
                              {n->inputs[1]}, nullptr, &n);
      auto val_grad = MakeNode("_backward_index_update_val", n->attrs.name + "_backward_val",
                              {ograds[0], n->inputs[1]}, nullptr, &n);
      // auto val_grad = MakeNode("zeros_like", n->attrs.name + "_backward_val",
      //                         {n->inputs[2]}, nullptr, &n);
      std::vector<nnvm::NodeEntry> ret;
      ret.emplace_back(a_grad);
      ret.emplace_back(idx_grad);
      ret.emplace_back(val_grad);
      return ret;
  })
.add_argument("a", "NDArray-or-Symbol", "Input ndarray")
.add_argument("ind", "NDArray-or-Symbol", "Index ndarray")
.add_argument("val", "NDArray-or-Symbol", "Update ndarray");


NNVM_REGISTER_OP(_backward_index_update_a)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", IndexUpdateOpBackwardA<cpu>);


NNVM_REGISTER_OP(_backward_index_update_val)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", IndexUpdateOpBackwardVal<cpu>);

}  // namespace op
}  // namespace mxnet
