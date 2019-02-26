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

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../tensor/broadcast_reduce_op.h"

namespace mxnet {
namespace op {
namespace mxnet_op {

struct softmax_fwd {
  template<typename AType>
  MSHADOW_XINLINE static AType Map(float a, AType b) {
    return AType(expf(a)/b);
  }

  template<typename AType>
  MSHADOW_XINLINE static AType Map(double a, AType b) {
    return AType(exp(a)/b);
  }
};


struct log_softmax_fwd {
  template<typename DType>
  MSHADOW_XINLINE static float Map(DType a, float b) {
    return a - logf(b);
  }

  template<typename DType>
  MSHADOW_XINLINE static double Map(DType a, double b) {
    return a - log(b);
  }
};


template<typename OP, bool negate, typename AType, typename DType, typename OType, int ndim>
inline void Softmax(Stream<cpu> *s, DType *in, OType *out,
                    Shape<ndim> shape, int axis, const DType temperature) {
  index_t M = shape[axis];
  index_t N = shape.Size()/M;
  Shape<ndim> stride = calc_stride(shape);
  Shape<ndim> sshape = shape;
  sshape[axis] = 1;
  index_t sa = stride[axis];

  #pragma omp parallel for
  for (int i = 0; i < static_cast<int>(N); ++i) {
    index_t base = unravel_dot(i, sshape, stride);

    DType mmax = negate ? -in[base] : in[base];
    DType val;
    for (index_t j = 1; j < M; ++j) {
      val = negate ? -in[base + j*sa] : in[base + j*sa];
      if (mmax < val) mmax = val;
    }

    AType sum = AType(0);
    DType in_val;
    // By default temperature is 1.0.
    // Adding a branch here to save the CPU 'divide-by-1' computation at runtime
    if (temperature == 1.0) {
      for (index_t j = 0; j < M; ++j) {
        in_val = negate ? -in[base + j*sa] : in[base + j*sa];
        sum += std::exp(in_val - mmax);
      }

      for (index_t j = 0; j < M; ++j) {
        in_val = negate ? -in[base + j*sa] : in[base + j*sa];
        out[base + j*sa] = OP::Map(in_val - mmax, sum);
      }
    } else {
      for (index_t j = 0; j < M; ++j) {
        in_val = negate ? -in[base + j*sa] : in[base + j*sa];
        sum += std::exp((in_val - mmax)/temperature);
      }

      for (index_t j = 0; j < M; ++j) {
        in_val = negate ? -in[base + j*sa] : in[base + j*sa];
        out[base + j*sa] = OP::Map((in_val - mmax)/temperature, sum);
      }
    }
  }
}


struct softmax_bwd {
  template<typename DType, typename AType>
  MSHADOW_XINLINE static AType Map(DType ograd, DType out, AType sum) {
    return AType(out * (ograd - sum));
  }
};


struct log_softmax_bwd {
  template<typename AType>
  MSHADOW_XINLINE static AType Map(float ograd, float out, AType sum) {
    return AType(ograd - expf(out)*sum);
  }

  template<typename AType>
  MSHADOW_XINLINE static AType Map(double ograd, double out, AType sum) {
    return AType(ograd - exp(out)*sum);
  }
};


template<typename OP1, typename OP2, int Req, bool negate,
  typename AType, typename DType, typename OType, int ndim>
inline void SoftmaxGrad(Stream<cpu> *s, OType *out, OType *ograd,
                        DType *igrad, Shape<ndim> shape, int axis,
                        const DType temperature) {
  index_t M = shape[axis];
  index_t N = shape.Size()/M;
  Shape<ndim> stride = calc_stride(shape);
  Shape<ndim> sshape = shape;
  sshape[axis] = 1;
  index_t sa = stride[axis];

  #pragma omp parallel for
  for (int i = 0; i < static_cast<int>(N); ++i) {
    index_t base = unravel_dot(i, sshape, stride);

    AType sum = AType(0);
    for (index_t j = 0; j < M; ++j) {
      sum += OP1::Map(ograd[base + j*sa], out[base + j*sa]);
    }

    // By default temperature is 1.0.
    // Adding a branch here to save the CPU 'divide-by-1' computation at runtime
    DType final_result;
    if (temperature == 1.0) {
      for (index_t j = 0; j < M; ++j) {
        final_result = negate ?
                       -OP2::Map(ograd[base + j*sa], out[base + j*sa], sum) :
                       OP2::Map(ograd[base + j*sa], out[base + j*sa], sum);
        KERNEL_ASSIGN(igrad[base + j*sa], Req, final_result);
      }
    } else {
      for (index_t j = 0; j < M; ++j) {
        final_result = negate ?
                       -OP2::Map(ograd[base + j*sa], out[base + j*sa], sum) / temperature :
                       OP2::Map(ograd[base + j*sa], out[base + j*sa], sum) / temperature;
        KERNEL_ASSIGN(igrad[base + j*sa], Req, final_result);
      }
    }
  }
}


#ifdef __CUDACC__
template<int x_bits, typename OP, bool negate, typename AType, int ndim,
  typename DType, typename OType>
__global__ void softmax_compute_kernel(DType *in, OType *out, index_t M, int axis,
                                       Shape<ndim> sshape, Shape<ndim> stride,
                                       const double temperature) {
  const unsigned x_size = 1 << x_bits;
  __shared__ AType smem[x_size];
  index_t sa = stride[axis];
  index_t base = unravel_dot(blockIdx.x, sshape, stride);
  index_t x = threadIdx.x;

  red::maximum::SetInitValue(smem[x]);
  for (index_t i = x; i < M; i += x_size) {
    smem[x] = ::max(smem[x], negate ? -in[base + i*sa] : in[base + i*sa]);
  }
  __syncthreads();
  cuda::Reduce1D<red::maximum, x_bits>(smem);
  __syncthreads();
  DType smax = smem[0];
  __syncthreads();

  red::sum::SetInitValue(smem[x]);
  DType val;
  for (index_t i = x; i < M; i += x_size) {
    val = negate ? -in[base + i*sa]:in[base + i*sa];
    smem[x] += static_cast<AType>(expf((val - smax) / static_cast<AType>(temperature)));
  }
  __syncthreads();
  cuda::Reduce1D<red::sum, x_bits>(smem);
  __syncthreads();
  AType ssum = smem[0];
  __syncthreads();

  for (index_t i = x; i < M; i += x_size) {
    val = negate ? -in[base + i*sa] : in[base + i*sa];
    out[base + i*sa] = OP::Map((val - smax)/static_cast<DType>(temperature), ssum);
  }
}

template<typename OP, bool negate, typename AType, typename DType, typename OType, int ndim>
inline void Softmax(Stream<gpu> *s, DType *in, OType *out,
                    Shape<ndim> shape, int axis, const double temperature) {
  const int x_bits = 7;
  const int x_size = 1 << x_bits;
  index_t M = shape[axis];
  index_t N = shape.Size()/M;
  Shape<ndim> stride = calc_stride(shape);
  Shape<ndim> sshape = shape;
  sshape[axis] = 1;

  softmax_compute_kernel<x_bits, OP, negate, AType, ndim>
    <<<N, x_size, 0, mshadow::Stream<gpu>::GetStream(s)>>>(
      in, out, M, axis, sshape, stride, temperature);
  MSHADOW_CUDA_POST_KERNEL_CHECK(softmax_compute_kernel);
}


template<int x_bits, typename OP1, typename OP2, int Req, bool negate, typename AType, int ndim,
  typename DType, typename OType>
__global__ void softmax_gradient_kernel(OType *out, OType *ograd, DType *igrad,
                                        index_t M, int axis, Shape<ndim> sshape,
                                        Shape<ndim> stride, const double temperature) {
  const unsigned x_size = 1 << x_bits;
  __shared__ AType smem[x_size];
  index_t sa = stride[axis];
  index_t base = unravel_dot(blockIdx.x, sshape, stride);
  index_t x = threadIdx.x;

  red::sum::SetInitValue(smem[x]);
  for (index_t i = x; i < M; i += x_size) {
    smem[x] += OP1::Map(ograd[base + i*sa], out[base + i*sa]);
  }
  __syncthreads();
  cuda::Reduce1D<red::sum, x_bits>(smem);
  __syncthreads();
  AType ssum = smem[0];
  __syncthreads();

  DType final_result;
  for (index_t i = x; i < M; i += x_size) {
    final_result =
      negate ?
      -OP2::Map(ograd[base + i*sa], out[base + i*sa], ssum) :
      OP2::Map(ograd[base + i*sa], out[base + i*sa], ssum);
    KERNEL_ASSIGN(igrad[base + i*sa], Req, final_result / static_cast<DType>(temperature));
  }
}


template<typename OP1, typename OP2, int Req, bool negate, typename AType, int ndim,
  typename DType, typename OType>
inline void SoftmaxGrad(Stream<gpu> *s, OType *out, OType *ograd,
                        DType *igrad, Shape<ndim> shape, int axis,
                        const double temperature) {
  const int x_bits = 7;
  const int x_size = 1 << x_bits;
  index_t M = shape[axis];
  index_t N = shape.Size()/M;
  Shape<ndim> stride = calc_stride(shape);
  Shape<ndim> sshape = shape;
  sshape[axis] = 1;

  softmax_gradient_kernel<x_bits, OP1, OP2, Req, negate, AType, ndim>
    <<<N, x_size, 0, mshadow::Stream<gpu>::GetStream(s)>>>(
      out, ograd, igrad, M, axis, sshape, stride, temperature);
  MSHADOW_CUDA_POST_KERNEL_CHECK(softmax_gradient_kernel);
}
#endif

}  // namespace mxnet_op


struct SoftmaxParam : public dmlc::Parameter<SoftmaxParam> {
  int axis;
  dmlc::optional<double> temperature;
  dmlc::optional<int> dtype;
  DMLC_DECLARE_PARAMETER(SoftmaxParam) {
    DMLC_DECLARE_FIELD(axis).set_default(-1)
    .describe("The axis along which to compute softmax.");
    DMLC_DECLARE_FIELD(temperature).set_default(dmlc::optional<double>())
    .describe("Temperature parameter in softmax");
    DMLC_DECLARE_FIELD(dtype)
    .add_enum("float16", mshadow::kFloat16)
    .add_enum("float32", mshadow::kFloat32)
    .add_enum("float64", mshadow::kFloat64)
    .set_default(dmlc::optional<int>())
    .describe("DType of the output in case this can't be inferred. "
              "Defaults to the same as input's dtype if not defined (dtype=None).");
  }
};

static inline bool softmax_has_dtype_override(const nnvm::NodeAttrs& attrs) {
  const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
  return param.dtype.has_value() && param.dtype.value() != -1;
}

static inline bool SoftmaxOpType(const nnvm::NodeAttrs& attrs,
                                 std::vector<int>* in_attrs,
                                 std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 1);
  const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);

  if (softmax_has_dtype_override(attrs)) {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, param.dtype.value());
    type_assign(&(*in_attrs)[0], (*out_attrs)[0]);
    return true;
  } else {
    return ElemwiseType<1, 1>(attrs, in_attrs, out_attrs);
  }
}

static inline bool SoftmaxGradOpShape(const nnvm::NodeAttrs& attrs,
                                      std::vector<TShape> *in_attrs,
                                      std::vector<TShape> *out_attrs) {
  if (softmax_has_dtype_override(attrs)) {
    return ElemwiseShape<3, 1>(attrs, in_attrs, out_attrs);
  } else {
    return ElemwiseShape<2, 1>(attrs, in_attrs, out_attrs);
  }
}

static inline bool SoftmaxGradOpType(const nnvm::NodeAttrs& attrs,
                                     std::vector<int>* in_attrs,
                                     std::vector<int>* out_attrs) {
  CHECK_EQ(out_attrs->size(), 1);
  if (softmax_has_dtype_override(attrs)) {
    CHECK_EQ(in_attrs->size(), 3);
    int in_dtype = (*in_attrs)[1];
    int out_dtype = (*in_attrs)[2];
    TYPE_ASSIGN_CHECK(*in_attrs, 0, out_dtype);
    TYPE_ASSIGN_CHECK(*out_attrs, 0, in_dtype);

    return (*out_attrs)[0] != -1 && (*in_attrs)[0] != -1;
  } else {
    CHECK_EQ(in_attrs->size(), 2);
    int out_dtype = (*in_attrs)[1];
    TYPE_ASSIGN_CHECK(*out_attrs, 0, out_dtype);
    TYPE_ASSIGN_CHECK(*in_attrs, 0, out_dtype);

    return (*out_attrs)[0] != -1 && (*in_attrs)[0] != -1;
  }
}

static inline std::vector<std::pair<int, int> >
SoftmaxGradOpInplaceOption(const nnvm::NodeAttrs& attrs) {
  if (softmax_has_dtype_override(attrs)) {
    return std::vector<std::pair<int, int> >{{0, 0}, {1, 0}, {2, 0}};
  } else {
    return std::vector<std::pair<int, int> >{{0, 0}, {1, 0}};
  }
}

static inline uint32_t SoftmaxGradOpNumInputs(const nnvm::NodeAttrs& attrs) {
  return softmax_has_dtype_override(attrs) ? 3 : 2;
}

static inline std::vector<std::string> SoftmaxGradOpInputNames(const nnvm::NodeAttrs& attrs) {
  if (softmax_has_dtype_override(attrs)) {
    return std::vector<std::string>{"ograd", "data", "output"};
  } else {
    return std::vector<std::string>{"ograd", "output"};
  }
}

struct SoftmaxFGradient {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    if (softmax_has_dtype_override(n->attrs)) {
      return ElemwiseGradUseInOut {op_name}(n, ograds);
    } else {
      return ElemwiseGradUseOut {op_name}(n, ograds);
    }
  }
};

template<typename xpu, typename OP, bool negate = false>
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
  const double temperature = param.temperature.has_value() ?
    param.temperature.value() : 1.0;
  TShape shape = AxisShapeCompact(inputs[0].shape_, &axis, true);
  MXNET_REAL_ACC_TYPE_SWITCH(inputs[0].type_flag_, DType, AType, {
    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, OType, {
      if (shape.ndim() == 2) {
        Softmax<OP, negate, AType>(
            ctx.get_stream<xpu>(), inputs[0].dptr<DType>(),
            outputs[0].dptr<OType>(), shape.get<2>(), axis,
            static_cast<DType>(temperature));
      } else {
        Softmax<OP, negate, AType>(
            ctx.get_stream<xpu>(), inputs[0].dptr<DType>(),
            outputs[0].dptr<OType>(), shape.get<3>(), axis,
            static_cast<DType>(temperature));
      }
    });
  });
}


template<typename xpu, typename OP1, typename OP2, bool negate = false>
void SoftmaxGradCompute(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  if (req[0] == kNullOp) return;
  const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
  int axis = CheckAxis(param.axis, inputs[0].ndim());
  const double temperature = param.temperature.has_value() ?
    param.temperature.value() : 1.0;
  TShape shape = AxisShapeCompact(inputs[0].shape_, &axis, true);

  int out_idx = softmax_has_dtype_override(attrs) ? 2 : 1;

  MXNET_REAL_ACC_TYPE_SWITCH(inputs[0].type_flag_, OType, AType, {
    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        if (shape.ndim() == 2) {
          SoftmaxGrad<OP1, OP2, Req, negate, AType>(
              ctx.get_stream<xpu>(), inputs[out_idx].dptr<OType>(),
              inputs[0].dptr<OType>(), outputs[0].dptr<DType>(),
              shape.get<2>(), axis, static_cast<DType>(temperature));
        } else {
          SoftmaxGrad<OP1, OP2, Req, negate, AType>(
              ctx.get_stream<xpu>(), inputs[out_idx].dptr<OType>(),
              inputs[0].dptr<OType>(), outputs[0].dptr<DType>(),
              shape.get<3>(), axis, static_cast<DType>(temperature));
        }
      });
    });
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NN_SOFTMAX_INL_H_
