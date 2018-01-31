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
 * Copyright (c) 2017 by Contributors
 * \file softmax-inl.h
 * \brief
*/
#ifndef MXNET_OPERATOR_NN_SOFTMAX_INL_H_
#define MXNET_OPERATOR_NN_SOFTMAX_INL_H_

#include <vector>

#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../tensor/broadcast_reduce_op.h"

namespace mxnet {
namespace op {
namespace mxnet_op {

struct softmax_fwd {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(expf(a)/b);
  }
};


struct log_softmax_fwd {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(a - logf(b));
  }
};


template<typename OP, typename DType, int ndim>
inline void Softmax(Stream<cpu> *s, DType *in, DType *out,
                    Shape<ndim> shape, int axis) {
  index_t M = shape[axis];
  index_t N = shape.Size()/M;
  Shape<ndim> stride = calc_stride(shape);
  Shape<ndim> sshape = shape;
  sshape[axis] = 1;
  index_t sa = stride[axis];

  #pragma omp parallel for
  for (int i = 0; i < static_cast<int>(N); ++i) {
    index_t base = unravel_dot(i, sshape, stride);

    DType mmax = in[base];
    for (index_t j = 1; j < M; ++j) {
      if (mmax < in[base + j*sa]) mmax = in[base + j*sa];
    }

    DType sum = DType(0);
    for (index_t j = 0; j < M; ++j) {
      sum += std::exp(in[base + j*sa] - mmax);
    }

    for (index_t j = 0; j < M; ++j) {
      out[base + j*sa] = OP::Map(in[base + j*sa] - mmax, sum);
    }
  }
}


struct softmax_bwd {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType ograd, DType out, DType sum) {
    return DType(out * (ograd - sum));
  }
};


struct log_softmax_bwd {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType ograd, DType out, DType sum) {
    return DType(ograd - expf(out)*sum);
  }
};


template<typename OP1, typename OP2, typename DType, int ndim>
inline void SoftmaxGrad(Stream<cpu> *s, DType *out, DType *ograd,
                        DType *igrad, Shape<ndim> shape, int axis) {
  index_t M = shape[axis];
  index_t N = shape.Size()/M;
  Shape<ndim> stride = calc_stride(shape);
  Shape<ndim> sshape = shape;
  sshape[axis] = 1;
  index_t sa = stride[axis];

  #pragma omp parallel for
  for (int i = 0; i < static_cast<int>(N); ++i) {
    index_t base = unravel_dot(i, sshape, stride);

    DType sum = DType(0);
    for (index_t j = 0; j < M; ++j) {
      sum += OP1::Map(ograd[base + j*sa], out[base + j*sa]);
    }

    for (index_t j = 0; j < M; ++j) {
      igrad[base + j*sa] = OP2::Map(ograd[base + j*sa], out[base + j*sa], sum);
    }
  }
}


#ifdef __CUDACC__
template<int x_bits, typename OP, typename DType, int ndim>
__global__ void softmax_compute_kernel(DType *in, DType *out, index_t M, int axis,
                                       Shape<ndim> sshape, Shape<ndim> stride) {
  const unsigned x_size = 1 << x_bits;
  __shared__ DType smem[x_size];
  index_t sa = stride[axis];
  index_t base = unravel_dot(blockIdx.x, sshape, stride);
  index_t x = threadIdx.x;

  red::maximum::SetInitValue(smem[x]);
  for (index_t i = x; i < M; i += x_size) {
    red::maximum::Reduce(smem[x], in[base + i*sa]);
  }
  __syncthreads();
  cuda::Reduce1D<red::maximum, x_bits>(smem);
  __syncthreads();
  DType smax = smem[0];
  __syncthreads();

  red::sum::SetInitValue(smem[x]);
  for (index_t i = x; i < M; i += x_size) {
    red::sum::Reduce(smem[x], static_cast<DType>(expf(in[base + i*sa] - smax)));
  }
  __syncthreads();
  cuda::Reduce1D<red::sum, x_bits>(smem);
  __syncthreads();
  DType ssum = smem[0];
  __syncthreads();

  for (index_t i = x; i < M; i += x_size) {
    out[base + i*sa] = OP::Map(in[base + i*sa] - smax, ssum);
  }
}

template<typename OP, typename DType, int ndim>
inline void Softmax(Stream<gpu> *s, DType *in, DType *out,
                    Shape<ndim> shape, int axis) {
  const int x_bits = 7;
  const int x_size = 1 << x_bits;
  index_t M = shape[axis];
  index_t N = shape.Size()/M;
  Shape<ndim> stride = calc_stride(shape);
  Shape<ndim> sshape = shape;
  sshape[axis] = 1;

  softmax_compute_kernel<x_bits, OP, DType, ndim>
    <<<N, x_size, 0, mshadow::Stream<gpu>::GetStream(s)>>>(
      in, out, M, axis, sshape, stride);
  MSHADOW_CUDA_POST_KERNEL_CHECK(softmax_compute_kernel);
}


template<int x_bits, typename OP1, typename OP2, typename DType, int ndim>
__global__ void softmax_gradient_kernel(DType *out, DType *ograd, DType *igrad,
                                        index_t M, int axis, Shape<ndim> sshape,
                                        Shape<ndim> stride) {
  const unsigned x_size = 1 << x_bits;
  __shared__ DType smem[x_size];
  index_t sa = stride[axis];
  index_t base = unravel_dot(blockIdx.x, sshape, stride);
  index_t x = threadIdx.x;

  red::sum::SetInitValue(smem[x]);
  for (index_t i = x; i < M; i += x_size) {
    red::sum::Reduce(smem[x], OP1::Map(ograd[base + i*sa], out[base + i*sa]));
  }
  __syncthreads();
  cuda::Reduce1D<red::sum, x_bits>(smem);
  __syncthreads();
  DType ssum = smem[0];
  __syncthreads();

  for (index_t i = x; i < M; i += x_size) {
    igrad[base + i*sa] = OP2::Map(ograd[base + i*sa], out[base + i*sa], ssum);
  }
}


template<typename OP1, typename OP2, typename DType, int ndim>
inline void SoftmaxGrad(Stream<gpu> *s, DType *out, DType *ograd,
                        DType *igrad, Shape<ndim> shape, int axis) {
  const int x_bits = 7;
  const int x_size = 1 << x_bits;
  index_t M = shape[axis];
  index_t N = shape.Size()/M;
  Shape<ndim> stride = calc_stride(shape);
  Shape<ndim> sshape = shape;
  sshape[axis] = 1;

  softmax_gradient_kernel<x_bits, OP1, OP2, DType, ndim>
    <<<N, x_size, 0, mshadow::Stream<gpu>::GetStream(s)>>>(
      out, ograd, igrad, M, axis, sshape, stride);
  MSHADOW_CUDA_POST_KERNEL_CHECK(softmax_gradient_kernel);
}
#endif

}  // namespace mxnet_op


struct SoftmaxParam : public dmlc::Parameter<SoftmaxParam> {
  int axis;
  DMLC_DECLARE_PARAMETER(SoftmaxParam) {
    DMLC_DECLARE_FIELD(axis).set_default(-1)
      .describe("The axis along which to compute softmax.");
  }
};

template<typename xpu, typename OP>
void SoftmaxCompute(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  if (req[0] == kNullOp) return;
  CHECK_NE(req[0], kAddTo);
  const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
  int axis = CheckAxis(param.axis, inputs[0].ndim());
  TShape shape = AxisShapeCompact(inputs[0].shape_, &axis, true);
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    if (shape.ndim() == 2) {
      Softmax<OP>(ctx.get_stream<xpu>(), inputs[0].dptr<DType>(),
              outputs[0].dptr<DType>(), shape.get<2>(), axis);
    } else {
      Softmax<OP>(ctx.get_stream<xpu>(), inputs[0].dptr<DType>(),
              outputs[0].dptr<DType>(), shape.get<3>(), axis);
    }
  });
}


template<typename xpu, typename OP1, typename OP2>
void SoftmaxGradCompute(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  if (req[0] == kNullOp) return;
  CHECK_NE(req[0], kAddTo);
  const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
  int axis = CheckAxis(param.axis, inputs[0].ndim());
  TShape shape = AxisShapeCompact(inputs[0].shape_, &axis, true);
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    if (shape.ndim() == 2) {
      SoftmaxGrad<OP1, OP2>(ctx.get_stream<xpu>(), inputs[1].dptr<DType>(),
                            inputs[0].dptr<DType>(), outputs[0].dptr<DType>(),
                            shape.get<2>(), axis);
    } else {
      SoftmaxGrad<OP1, OP2>(ctx.get_stream<xpu>(), inputs[1].dptr<DType>(),
                            inputs[0].dptr<DType>(), outputs[0].dptr<DType>(),
                            shape.get<3>(), axis);
    }
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NN_SOFTMAX_INL_H_
