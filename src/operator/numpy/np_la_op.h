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
 * Copyright (c) 2019 by Contributors
 * \file np_la_op.h
 * \brief Function definition of Operators for advanced linear algebra.
 */
#ifndef MXNET_OPERATOR_NUMPY_NP_LA_OP_H_
#define MXNET_OPERATOR_NUMPY_NP_LA_OP_H_

#include <tuple>
#include <vector>
#include <algorithm>
#include "../tensor/broadcast_reduce_op.h"
#include "../tensor/broadcast_reduce-inl.h"


namespace mxnet {
namespace op {


using namespace broadcast;

struct NumpyLaNormParam : public dmlc::Parameter<NumpyLaNormParam> {
  double ord;
  dmlc::optional<mxnet::TShape> axis;
  bool keepdims;
  int flag;
  DMLC_DECLARE_PARAMETER(NumpyLaNormParam) {
    DMLC_DECLARE_FIELD(ord).set_default(2)
    .describe("Order of the norm. inf means numpy’s inf object.");
    DMLC_DECLARE_FIELD(axis).set_default(dmlc::optional<mxnet::TShape>())
    .describe(R"code(If axis is an integer, it specifies the axis of x along
     which to compute the vector norms. If axis is a 2-tuple,
     it specifies the axes that hold 2-D matrices, and the matrix norms of
     these matrices are computed. If axis is None then either a vector norm (when x is 1-D)
     or a matrix norm (when x is 2-D) is returned.If axis is an integer,
     it specifies the axis of x along which to compute the vector norms.
     If axis is a 2-tuple, it specifies the axes that hold 2-D matrices,
     and the matrix norms of these matrices are computed. If axis is None then either a
     vector norm (when x is 1-D) or a matrix norm (when x is 2-D) is returned.)code");
    DMLC_DECLARE_FIELD(keepdims).set_default(false)
    .describe("If this is set to `True`, the reduced axis is left "
    "in the result as dimension with size one.");
    DMLC_DECLARE_FIELD(flag).set_default(-1)
    .describe("Mapping relations between ord and flag."
    "ord:  None,  'fro', 'nuc', 'inf'  '-inf'."
    "flag:  0 ,    1,      2,    3,      4. ");
  }
};

/*! \brief compute ln norm */
struct nrmn {
  /*! \brief do reduction into dst */
  template<typename AType, typename DType>
  MSHADOW_XINLINE static void Reduce(volatile AType& sum_of_power, volatile DType src, double scale) { // NOLINT(*)
    AType c;
    c = math::pow(src, scale);
    sum_of_power += c;
  }
  /*! \brief finalize reduction result */
  template<typename DType>
  MSHADOW_XINLINE static void Finalize(volatile DType& sum_of_power, double scale) { // NOLINT(*)
    sum_of_power = math::pow(sum_of_power, 1/scale);
  }
  /*! \brief combine the results of two reducers */
  template<typename DType>
  MSHADOW_XINLINE static void Merge(volatile DType& dst_val, volatile DType& dst_residual, volatile DType& src_val, volatile DType& scale) { // NOLINT(*)
    dst_val += src_val;
  }
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType &sum_of_power) { // NOLINT(*)
    sum_of_power = 0;
  }
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType &sum_of_power, double scale) { // NOLINT(*)
    SetInitValue(sum_of_power);
  }
};

struct nrm_zero {
  /*! \brief do reduction into dst */
  template<typename DType>
  MSHADOW_XINLINE static void Reduce(volatile DType& dst,  volatile DType src) { // NOLINT(*)
    DType res = src == 0 ? 0 : 1;
    dst +=  res;
  }
  /*! \brief do reduction into dst */
  template<typename DType>
  MSHADOW_XINLINE static void Reduce(volatile DType& dst,  volatile DType src, volatile DType &none) { // NOLINT(*)
    Reduce(dst, src);
  }
  template<typename DType>
  MSHADOW_XINLINE static void Merge(volatile DType& dst_val, volatile DType& src_val) { // NOLINT(*)
    Reduce(dst_val, src_val);
  }
  /*! \brief combine the results of two reducers */
  template<typename DType>
  MSHADOW_XINLINE static void Merge(volatile DType& dst_val, volatile DType& dst_residual, volatile DType& src_val, volatile DType& src_residual) { // NOLINT(*)
    Reduce(dst_val, src_val);
  }
  /*! \brief finalize reduction */
  template<typename DType>
  MSHADOW_XINLINE static void Finalize(volatile DType& dst) {} // NOLINT(*)
  /*! \brief finalize reduction */
  template<typename DType>
  MSHADOW_XINLINE static void Finalize(volatile DType& dst, volatile DType& residual) {} // NOLINT(*
  /*!
   *\brief set the initial value during reduction
   */
  /*!
*\brief calculate gradient of redres with respect to redsrc,
* redres: reduced result, redsrc: one of reduction element
*/
  template<typename DType>
  MSHADOW_XINLINE static DType PartialGrad(DType redres, DType redsrc) {
    return 1;
  }
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType &initv) { // NOLINT(*)
    initv = 0;
  }
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType &initv, DType &none) { // NOLINT(*)
    SetInitValue(initv);
  }
};

template<typename Reducer, int ndim, typename AType, typename DType, typename OType, typename OP>
void norm_seq_reduce_compute(const size_t N, const size_t M, const bool addto,
                             const DType *big, OType *small, const Shape<ndim> bshape,
                             const Shape<ndim> sshape, const Shape<ndim> rshape,
                             const Shape<ndim> rstride, const double ord) {
#pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (index_t idx = 0; idx < static_cast<index_t>(N); ++idx) {
    Shape<ndim> coord = unravel(idx, sshape);
    index_t j = ravel(coord, bshape);
    AType val;
    double residual = ord;
    Reducer::SetInitValue(val, residual);
    for (size_t k = 0; k < M; ++k) {
      coord = unravel(k, rshape);
      Reducer::Reduce(val, AType(OP::Map(big[j + dot(coord, rstride)])), residual);
    }
    Reducer::Finalize(val, residual);
    assign(&small[idx], addto, OType(val));
  }
}
#ifdef __CUDACC__
#include "np_la_op-inl.cuh"

#else
template <typename Reducer, int ndim, typename DType, typename OP, bool safe_acc = false>
void PowerReduce(Stream<cpu>* s, const TBlob& small, const OpReqType req,
            const Tensor<cpu, 1, char>& workspace, const TBlob& big, const double ord) {
  if (req == kNullOp) return;
  Shape<ndim> rshape, rstride;
  diff(small.shape_.get<ndim>(), big.shape_.get<ndim>(), &rshape, &rstride);
  size_t N = small.shape_.Size(), M = rshape.Size();
  if (!safe_acc) {
    norm_seq_reduce_compute<Reducer, ndim, DType, DType, DType, OP>(
        N, M, req == kAddTo, big.dptr<DType>(), small.dptr<DType>(),
        big.shape_.get<ndim>(), small.shape_.get<ndim>(), rshape, rstride, ord);
  } else {
    // Use real-only type swtich for windows temporarily due to CI issues.
#ifndef _WIN32
    MXNET_ACC_TYPE_SWITCH(mshadow::DataType<DType>::kFlag, DataType, AType, {
      typedef typename std::conditional<safe_acc, AType, DataType>::type AccType;
      MSHADOW_TYPE_SWITCH(small.type_flag_, OType, {
          typedef typename std::conditional<safe_acc, OType, DataType>::type OutType;
          norm_seq_reduce_compute<Reducer, ndim, AccType, DataType, OutType, OP>(
          N, M, req == kAddTo, big.dptr<DataType>(), small.dptr<OutType>(),
          big.shape_.get<ndim>(), small.shape_.get<ndim>(), rshape, rstride, ord);
      });
    });
#else
    MXNET_REAL_ACC_TYPE_SWITCH(mshadow::DataType<DType>::kFlag, DataType, AType, {
      typedef typename std::conditional<safe_acc, AType, DataType>::type AccType;
      MSHADOW_TYPE_SWITCH(small.type_flag_, OType, {
        typedef typename std::conditional<safe_acc, OType, DataType>::type OutType;
        norm_seq_reduce_compute<Reducer, ndim, AccType, DataType, OutType, OP>(
          N, M, req == kAddTo, big.dptr<DataType>(), small.dptr<OutType>(),
          big.shape_.get<ndim>(), small.shape_.get<ndim>(), rshape, rstride, ord);
      });
    });
#endif
  }
}
#endif

template<typename xpu, typename reducer, bool safe_acc = false, bool normalize = false,
    typename OP = op::mshadow_op::identity>
void NumpyNormPowerComputeImpl(const OpContext& ctx,
                               const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs,
                               const mxnet::TShape& small,
                               const double& ord) {
  using namespace mshadow;
  using namespace mshadow::expr;

  mxnet::TShape src_shape, dst_shape;
  BroadcastReduceShapeCompact(inputs[0].shape_, small, &src_shape, &dst_shape);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, OType, {
      const TBlob in_data = inputs[0].reshape(src_shape);
      const TBlob out_data = outputs[0].reshape(dst_shape);
      BROADCAST_NDIM_SWITCH(dst_shape.ndim(), NDim, {
        size_t workspace_size = broadcast::ReduceWorkspaceSize<NDim, DType>(
            s, out_data.shape_, req[0], in_data.shape_);
        Tensor<xpu, 1, char> workspace =
            ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);
        PowerReduce<reducer, NDim, DType, OP, safe_acc>(
            s, out_data, req[0], workspace, in_data, ord);
        if (normalize) {
          auto out = out_data.FlatTo2D<xpu, OType>(s);
          out /= scalar<OType>(src_shape.Size()/dst_shape.Size());
        }
      });
    });
  });
}

template<int req, typename OP, bool normzero>
struct numpy_norm_reduce_axes_backward_broadcast{
  template<typename DType, typename OType>
  MSHADOW_XINLINE static void Map(index_t i,
                                  DType *data,
                                  OType *out,
                                  DType *igrad,
                                  OType *ograd,
                                  mshadow::Shape<5> in_shape,
                                  mshadow::Shape<5> out_shape,
                                  const uint32_t ndim) {
    size_t in_stride = 1;
    size_t out_stride = 1;
    index_t idx = i;
    index_t out_idx = i;
    for (int iter = ndim - 1; iter >= 0; --iter) {
      size_t dim_idx = idx % in_shape[iter];
      out_idx -= dim_idx * in_stride;
      if (out_shape[iter] != 1) {
        out_idx += dim_idx * out_stride;
      }
      idx /= in_shape[iter];
      in_stride *= in_shape[iter];
      out_stride *= out_shape[iter];
    }
    int flag_data = mshadow_op::sign::Map(data[i]);
    if (normzero) {
      KERNEL_ASSIGN(igrad[i], req, DType(ograd[out_idx]) *
                    DType(OP::Map(DType(flag_data*data[i]), DType(0))*flag_data));
    } else {
      KERNEL_ASSIGN(igrad[i], req, DType(ograd[out_idx]) * DType(OP::Map(DType(flag_data*data[i]),
                    DType(out[out_idx]))*flag_data));
    }
  }
};

template<typename xpu, typename OP, bool normalize = false, bool normzero = false>
void NumpyNormReduceAxesBackwardUseInOutImpl(const OpContext& ctx,
                                             const mxnet::TShape &small,
                                             const std::vector<TBlob>& inputs,
                                             const std::vector<OpReqType>& req,
                                             const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;

  mxnet::TShape src_shape, dst_shape;
  BroadcastReduceShapeCompact(outputs[0].shape_, small, &src_shape, &dst_shape);
  Stream<xpu> *s = ctx.get_stream<xpu>();

  MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, OType, {
      mshadow::Shape<5> in_shape;
      mshadow::Shape<5> out_shape;
      for (int i = 0; i < 5; ++i) {
        if (i < dst_shape.ndim()) {
          in_shape[i] = src_shape[i];
          out_shape[i] = dst_shape[i];
        } else {
          in_shape[i] = 1;
          out_shape[i] = 1;
        }
      }
      if (dst_shape.ndim() == 2) {
        Tensor<xpu, 2, OType> igrad =
            outputs[0].get_with_shape<xpu, 2, OType>(src_shape.get<2>(), s);
        Tensor<xpu, 2, DType> ograd =
            inputs[0].get_with_shape<xpu, 2, DType>(dst_shape.get<2>(), s);
        Tensor<xpu, 2, OType> data =
            inputs[1].get_with_shape<xpu, 2, OType>(src_shape.get<2>(), s);
        Tensor<xpu, 2, DType> out =
            inputs[2].get_with_shape<xpu, 2, DType>(dst_shape.get<2>(), s);
        MXNET_REQ_TYPE_SWITCH(req[0], Req, {
          Kernel<numpy_norm_reduce_axes_backward_broadcast<Req, OP, normzero>, xpu>::Launch(
              s, outputs[0].shape_.Size(), data.dptr_, out.dptr_, igrad.dptr_, ograd.dptr_,
              in_shape, out_shape, src_shape.ndim());
        });
        if (normalize) igrad /= scalar<OType>(src_shape.Size()/dst_shape.Size());
      } else {
        const int ndim = MXNET_SPECIAL_MAX_NDIM;
        Tensor<xpu, ndim, OType> igrad =
            outputs[0].get_with_shape<xpu, ndim, OType>(src_shape.get<ndim>(), s);
        Tensor<xpu, ndim, DType> ograd =
            inputs[0].get_with_shape<xpu, ndim, DType>(dst_shape.get<ndim>(), s);
        Tensor<xpu, ndim, OType> data =
            inputs[1].get_with_shape<xpu, ndim, OType>(src_shape.get<ndim>(), s);
        Tensor<xpu, ndim, DType> out =
            inputs[2].get_with_shape<xpu, ndim, DType>(dst_shape.get<ndim>(), s);
        MXNET_REQ_TYPE_SWITCH(req[0], Req, {
          Kernel<numpy_norm_reduce_axes_backward_broadcast<Req, OP, normzero>, xpu>::Launch(
              s, outputs[0].shape_.Size(), data.dptr_, out.dptr_, igrad.dptr_, ograd.dptr_,
              in_shape, out_shape, src_shape.ndim());
        });
        if (normalize) igrad /= scalar<OType>(src_shape.Size()/dst_shape.Size());
      }
    });
  });
}
template <int req, typename OP>
struct numpy_norm_power_backward_broadcast {
  template<typename DType, typename OType>
  MSHADOW_XINLINE static void Map(index_t i,
                                  DType *data,
                                  OType *out,
                                  DType *igrad,
                                  OType *ograd,
                                  mshadow::Shape<5> in_shape,
                                  mshadow::Shape<5> out_shape,
                                  const uint32_t ndim,
                                  const double ord) {
    size_t in_stride = 1;
    size_t out_stride = 1;
    index_t idx = i;
    index_t out_idx = i;
    for (int iter = ndim - 1; iter >= 0; --iter) {
      size_t dim_idx = idx % in_shape[iter];
      out_idx -= dim_idx * in_stride;
      if (out_shape[iter] != 1) {
        out_idx += dim_idx * out_stride;
      }
      idx /= in_shape[iter];
      in_stride *= in_shape[iter];
      out_stride *= out_shape[iter];
    }
    int flag_data = mshadow_op::sign::Map(data[i]);
    KERNEL_ASSIGN(igrad[i], req, DType(ograd[out_idx]) * DType(out[out_idx])*
        DType(math::pow(DType(OP::Map(data[i])), ord - 2)) * DType(flag_data));
  }
};

template<typename xpu, typename OP, bool normalize = false>
void NumpyNormPowerBackwardUseInOutImpl(const OpContext& ctx,
                                        const mxnet::TShape &small,
                                        const std::vector<TBlob>& inputs,
                                        const std::vector<OpReqType>& req,
                                        const std::vector<TBlob>& outputs,
                                        const double ord) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;

  mxnet::TShape src_shape, dst_shape;
  BroadcastReduceShapeCompact(outputs[0].shape_, small, &src_shape, &dst_shape);
  Stream<xpu> *s = ctx.get_stream<xpu>();

  MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, OType, {
      mshadow::Shape<5> in_shape;
      mshadow::Shape<5> out_shape;
      for (int i = 0; i < 5; ++i) {
        if (i < dst_shape.ndim()) {
          in_shape[i] = src_shape[i];
          out_shape[i] = dst_shape[i];
        } else {
          in_shape[i] = 1;
          out_shape[i] = 1;
        }
      }
      if (dst_shape.ndim() == 2) {
        Tensor<xpu, 2, OType> igrad =
            outputs[0].get_with_shape<xpu, 2, OType>(src_shape.get<2>(), s);
        Tensor<xpu, 2, DType> ograd =
            inputs[0].get_with_shape<xpu, 2, DType>(dst_shape.get<2>(), s);
        Tensor<xpu, 2, OType> data =
            inputs[1].get_with_shape<xpu, 2, OType>(src_shape.get<2>(), s);
        Tensor<xpu, 2, DType> out =
            inputs[2].get_with_shape<xpu, 2, DType>(dst_shape.get<2>(), s);
        MXNET_REQ_TYPE_SWITCH(req[0], Req, {
          Kernel<numpy_norm_power_backward_broadcast<Req, OP>, xpu>::Launch(
              s, outputs[0].shape_.Size(), data.dptr_, out.dptr_, igrad.dptr_, ograd.dptr_,
              in_shape, out_shape, src_shape.ndim(), ord);
        });
        if (normalize) igrad /= scalar<OType>(src_shape.Size()/dst_shape.Size());
      } else {
        const int ndim = MXNET_SPECIAL_MAX_NDIM;
        Tensor<xpu, ndim, OType> igrad =
            outputs[0].get_with_shape<xpu, ndim, OType>(src_shape.get<ndim>(), s);
        Tensor<xpu, ndim, DType> ograd =
            inputs[0].get_with_shape<xpu, ndim, DType>(dst_shape.get<ndim>(), s);
        Tensor<xpu, ndim, OType> data =
            inputs[1].get_with_shape<xpu, ndim, OType>(src_shape.get<ndim>(), s);
        Tensor<xpu, ndim, DType> out =
            inputs[2].get_with_shape<xpu, ndim, DType>(dst_shape.get<ndim>(), s);
        MXNET_REQ_TYPE_SWITCH(req[0], Req, {
          Kernel<numpy_norm_power_backward_broadcast<Req, OP>, xpu>::Launch(
              s, outputs[0].shape_.Size(), data.dptr_, out.dptr_, igrad.dptr_, ograd.dptr_,
              in_shape, out_shape, src_shape.ndim(), ord);
        });
        if (normalize) igrad /= scalar<OType>(src_shape.Size()/dst_shape.Size());
      }
    });
  });
}

template<typename xpu>
void NumpyLaNormCompute(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  using namespace mshadow::expr;

  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  if (req[0] == kNullOp) return;
  if (outputs[0].Size() == 0U) return;
  const NumpyLaNormParam& param = nnvm::get<NumpyLaNormParam>(attrs.parsed);
  const int ndim = inputs[0].ndim();

  mxnet::TShape small;
  bool flag_axis;
  if (!param.axis.has_value()) {
    flag_axis = false;
  } else {
    flag_axis = true;
  }
  if (param.keepdims) {
    small = outputs[0].shape_;
  } else {
    small = ReduceAxesShapeImpl(inputs[0].shape_, param.axis, true, false);
  }
  // Immediately handle some default, simple, fast, and common cases.
  if (!flag_axis && (param.flag == 0 || (param.flag == 1 && ndim == 2) ||
     (param.flag == -1 && param.ord == 2 && ndim == 1))) {
    ReduceAxesComputeImpl<xpu, mshadow_op::nrm2, false, false, mshadow_op::identity>(
        ctx, inputs, req, outputs, small);
    return;
  }
  if ((flag_axis && param.axis.value().ndim() == 1) || (!flag_axis && ndim == 1)) {
    // if ord is inf
    if (param.flag == 3) {
      ReduceAxesComputeImpl<xpu, mshadow::red::maximum, false, false, mshadow_op::abs>(
          ctx, inputs, req, outputs, small);
    } else if (param.flag == 4) {
      ReduceAxesComputeImpl<xpu, mshadow::red::minimum, false, false, mshadow_op::abs>(
          ctx, inputs, req, outputs, small);
    } else if (param.ord == 1) {
      ReduceAxesComputeImpl<xpu, mshadow_op::sum, false, false, mshadow_op::abs>(
          ctx, inputs, req, outputs, small);
    } else if (param.flag == 0 || (param.flag == -1 && param.ord == 2)) {
      ReduceAxesComputeImpl<xpu, mshadow_op::nrm2, false, false, mshadow_op::identity>(
          ctx, inputs, req, outputs, small);
    } else if (param.ord == 0) {
      ReduceAxesComputeImpl<xpu, nrm_zero, false, false, mshadow_op::identity>(
          ctx, inputs, req, outputs, small);
    } else {
      NumpyNormPowerComputeImpl<xpu, nrmn, false, false, mshadow_op::abs>(
          ctx, inputs, req, outputs, small, param.ord);
      return;
    }
  } else if ((flag_axis && param.axis.value().ndim() == 2) || (!flag_axis && ndim == 2)) {
    if (param.flag == 2 || (param.flag == -1 && (param.ord == 2 || param.ord == -2))) {
      LOG(FATAL) << "Do not implement SVD for norm.";
    } else if (param.flag == 0 || param.flag == 1) {
      ReduceAxesComputeImpl<xpu, mshadow_op::nrm2, false, false, mshadow_op::identity>(
          ctx, inputs, req, outputs, small);
    } else if (param.flag == 3 || param.flag == 4 || param.ord == 1 || param.ord == -1) {
      LOG(FATAL) << "You must give axis.";
    } else {
      LOG(FATAL) << "Invalid norm order for matrices.";
    }
  } else {
    LOG(FATAL) << "Improper number of dimensions to norm.";
  }
}

template<typename xpu>
void NumpyLpNormGradCompute(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;

  if (req[0] == kNullOp) return;
  if (inputs[0].shape_.Size() == 0U) return;  // zero-size tensor
  const NumpyLaNormParam& param = nnvm::get<NumpyLaNormParam>(attrs.parsed);
  mxnet::TShape small;
  const int ndim = outputs[0].ndim();
  bool flag_axis;
  if (!param.axis.has_value()) {
    flag_axis = false;
  } else {
    flag_axis = true;
  }
  if (param.keepdims) {
    small = inputs[0].shape_;
  } else {
    small = ReduceAxesShapeImpl(outputs[0].shape_, param.axis, true, false);
  }
  // Immediately handle some default, simple, fast, and common cases.
  if (!flag_axis && (param.flag == 0 || (param.flag == 1 && ndim == 2) ||
      (param.flag == -1 && param.ord == 2 && ndim == 1))) {
    ReduceAxesBackwardUseInOutImpl<xpu, mshadow_op::div, false>(ctx, small, inputs, req, outputs);
    return;
  }
  if ((flag_axis && param.axis.value().ndim() == 1) || (!flag_axis && ndim == 1)) {
    // if numpy ord is inf
    if (param.flag == 3 || param.flag == 4 || param.ord == 1) {
      NumpyNormReduceAxesBackwardUseInOutImpl<xpu, mshadow_op::eq, false, false>(
          ctx, small, inputs, req, outputs);
    } else if (param.flag == 0 || (param.flag == -1 && param.ord == 2)) {
      ReduceAxesBackwardUseInOutImpl<xpu, mshadow_op::div, false>(
          ctx, small, inputs, req, outputs);
    } else if (param.ord == 0) {
      NumpyNormReduceAxesBackwardUseInOutImpl<xpu, mshadow_op::ne, false, true>(
          ctx, small, inputs, req, outputs);
    } else {
      NumpyNormPowerBackwardUseInOutImpl<xpu, mshadow_op::abs, false>(
          ctx, small, inputs, req, outputs, param.ord);
      return;
    }
  } else if ((flag_axis && param.axis.value().ndim() == 2) || (!flag_axis && ndim == 2)) {
    if (param.flag == 2 || (param.flag == -1 && (param.ord == 2 || param.ord == -2))) {
      LOG(FATAL) << "Do not implement SVD for norm.";
    } else if (param.flag == 0 || param.flag == 1) {
      ReduceAxesBackwardUseInOutImpl<xpu, mshadow_op::div, false>(ctx, small, inputs,
                                                                  req, outputs);
    } else if (param.flag == 3 || param.flag == 4 ||param.ord == 1 || param.ord == -1) {
      LOG(FATAL) << "You must give axis.";
    } else {
      LOG(FATAL) << "Invalid norm order for matrices.";
    }
  } else {
    LOG(FATAL) << "Improper number of dimensions to norm.";
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_LA_OP_H_
