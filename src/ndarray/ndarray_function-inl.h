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
 *  Copyright (c) 2015 by Contributors
 * \file ndarray_function-inl.h
 * \brief The real implementation of NDArray functions.
 */
#ifndef MXNET_NDARRAY_NDARRAY_FUNCTION_INL_H_
#define MXNET_NDARRAY_NDARRAY_FUNCTION_INL_H_

#include <vector>
#include "./ndarray_function.h"
// this file will be included twice by CPU and GPU
// macro to help specialize evaluation function

#ifndef DECL_TERNARY
#define DECL_TERNARY(XPU, OP, FUN)                                          \
  template<>                                                                \
  void Eval<XPU, OP>(const TBlob &lhs, const TBlob &mhs,                    \
                     const TBlob &rhs, TBlob *ret, RunContext ctx) {        \
    FUN<XPU, OP>(lhs, mhs, rhs, ret, ctx);                                  \
  }
#endif

#ifndef DECL_BINARY
#define DECL_BINARY(XPU, OP, FUN)                                                      \
  template<>                                                                           \
  void Eval<XPU, OP>(const TBlob &lhs, const TBlob &rhs, TBlob *ret, RunContext ctx) { \
    FUN<XPU, OP>(lhs, rhs, ret, ctx);                                                  \
  }
#endif

#ifndef DECL_SCALAR
#define DECL_SCALAR(XPU, OP, FUN, REVERSE)                           \
  template<>                                                         \
  void Eval<XPU, OP, REVERSE>(const TBlob &lhs, const real_t &rhs,   \
                                     TBlob *ret, RunContext ctx) {   \
    FUN<XPU, OP, REVERSE>(lhs, rhs, ret, ctx);                       \
  }
#endif

#if defined(__CUDACC__)
#define DEVICE gpu
#else
#define DEVICE cpu
#endif

namespace mxnet {
namespace ndarray {

// true implementation
template<typename xpu, typename OP>
void EvalBinary_(const TBlob &lhs, const TBlob &rhs,
                 TBlob *ret, RunContext ctx) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(ret->type_flag_, lhs.type_flag_)
    << "Only support input/output with the same data type";
  CHECK_EQ(ret->type_flag_, rhs.type_flag_)
    << "Only support input/output with the same data type";
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
    ret->FlatTo2D<xpu, DType>(s)
      = F<typename OP::mshadow_op>(lhs.FlatTo2D<xpu, DType>(s),
                                   rhs.FlatTo2D<xpu, DType>(s));
  });
}

template<typename xpu, typename OP>
void EvalOneHot_(const TBlob &index, const TBlob &rhs,
                 TBlob *ret, RunContext ctx) {
  LOG(INFO) << "The operator onehot_encode is deprecated; use one_hot instead.";
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  // TODO(eric): support mixed type encoding, i.e. int index and float rhs.
  CHECK_EQ(ret->type_flag_, mshadow::default_type_flag)
    << "one_hot_encode only support float32 as input/output";
  CHECK_EQ(rhs.type_flag_, mshadow::default_type_flag)
    << "one_hot_encode only support float32 as input/output";
  CHECK_EQ(index.type_flag_, mshadow::default_type_flag)
    << "one_hot_encode only support float32 as input/output";
  ret->get<xpu, 2, real_t>(s) =
    one_hot_encode(index.get<xpu, 1, real_t>(s),
                   rhs.shape_[1]);
}

template<typename xpu, typename OP>
void EvalMatChooseRowElem_(const TBlob &lhs, const TBlob &rhs,
                           TBlob *ret, RunContext ctx) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  // TODO(eric): support mixed type choose, i.e. int index and float rhs.
  CHECK_EQ(ret->type_flag_, mshadow::default_type_flag)
    << "mat_choose_row_element only support float32 as input/output";
  CHECK_EQ(rhs.type_flag_, mshadow::default_type_flag)
    << "mat_choose_row_element only support float32 as input/output";
  CHECK_EQ(lhs.type_flag_, mshadow::default_type_flag)
    << "mat_choose_row_element only support float32 as input/output";
  ret->get<xpu, 1, real_t>(s)
      = mat_choose_row_element(lhs.get<xpu, 2, real_t>(s),
                               rhs.get<xpu, 1, real_t>(s));
}

template<typename xpu, typename OP>
void EvalMatFillRowElem_(const TBlob &lhs, const TBlob &mhs, const TBlob &rhs,
                         TBlob *ret, RunContext ctx) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  ret->get<xpu, 2, real_t>(s)
          = mat_fill_row_element(lhs.get<xpu, 2, real_t>(s),
                                 mhs.get<xpu, 1, real_t>(s),
                                 rhs.get<xpu, 1, real_t>(s));
}

template<typename xpu, typename OP, bool reverse>
void EvalScalar_(const TBlob &lhs, const real_t &rhs,
                 TBlob *ret, RunContext ctx) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(ret->type_flag_, lhs.type_flag_)
    << "Only support input/output with the same data type";
  if (reverse) {
    MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
      ret->FlatTo2D<xpu, DType>(s)
        = F<typename OP::mshadow_op>(scalar(DType(rhs)), lhs.FlatTo2D<xpu, DType>(s));
    });
  } else {
    MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
      ret->FlatTo2D<xpu, DType>(s)
        = F<typename OP::mshadow_op>(lhs.FlatTo2D<xpu, DType>(s), scalar(DType(rhs)));
    });
  }
}

template<>
void EvalClip<DEVICE>(const TBlob &src, const real_t &a_min, const real_t &a_max,
                             TBlob *ret, RunContext ctx) {
  typedef DEVICE xpu;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(ret->type_flag_, src.type_flag_)
    << "Only support input/output with the same data type";
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
    ret->FlatTo2D<xpu, DType>(s)
      = F<ClipMax::mshadow_op>(
          F<ClipMin::mshadow_op>(src.FlatTo2D<xpu, DType>(s), scalar(DType(a_min))),
          scalar(DType(a_max)));
  });
}

template<>
void EvalRandom<DEVICE, UniformDistribution>(const real_t &a,
                                             const real_t &b,
                                             const Resource &resource,
                                             TBlob *ret,
                                             RunContext ctx) {
  typedef DEVICE xpu;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  switch (ret->type_flag_) {
  case mshadow::kFloat32:
    {
      mshadow::Random<xpu, float> *prnd = resource.get_random<xpu, float>(s);
      mshadow::Tensor<xpu, 2, float> tmp = ret->FlatTo2D<xpu, float>(s);
      prnd->SampleUniform(&tmp, float(a), float(b));  // NOLINT(*)
      break;
    }
  case mshadow::kFloat64:
    {
      mshadow::Random<xpu, double> *prnd = resource.get_random<xpu, double>(s);
      mshadow::Tensor<xpu, 2, double> tmp = ret->FlatTo2D<xpu, double>(s);
      prnd->SampleUniform(&tmp, double(a), double(b));  // NOLINT(*)
      break;
    }
  default:
    LOG(FATAL) << "Random only support float32 and float64";
  }
}

template<>
void EvalRandom<DEVICE, GaussianDistribution>(
    const real_t &mu,
    const real_t &sigma,
    const Resource &resource,
    TBlob *ret,
    RunContext ctx) {
  typedef DEVICE xpu;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  switch (ret->type_flag_) {
  case mshadow::kFloat32:
    {
      mshadow::Random<xpu, float> *prnd = resource.get_random<xpu, float>(s);
      mshadow::Tensor<xpu, 2, float> tmp = ret->FlatTo2D<xpu, float>(s);
      prnd->SampleGaussian(&tmp, float(mu), float(sigma));  // NOLINT(*)
      break;
    }
  case mshadow::kFloat64:
    {
      mshadow::Random<xpu, double> *prnd = resource.get_random<xpu, double>(s);
      mshadow::Tensor<xpu, 2, double> tmp = ret->FlatTo2D<xpu, double>(s);
      prnd->SampleGaussian(&tmp, double(mu), double(sigma));  // NOLINT(*)
      break;
    }
  default:
    LOG(FATAL) << "Random only support float32 and float64";
  }
}

template<>
void EvalRandom<DEVICE, GammaDistribution>(
    const real_t &alpha,
    const real_t &beta,
    const Resource &resource,
    TBlob *ret,
    RunContext ctx) {
  typedef cpu xpu;  // No support for gpu for this distribution.
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  switch (ret->type_flag_) {
  case mshadow::kFloat32:
    {
      mshadow::Random<xpu, float> *prnd = resource.get_random<xpu, float>(s);
      mshadow::Tensor<xpu, 2, float> tmp = ret->FlatTo2D<xpu, float>(s);
      prnd->SampleGamma(&tmp, float(alpha), float(beta));  // NOLINT(*)
      break;
    }
  case mshadow::kFloat64:
    {
      mshadow::Random<xpu, double> *prnd = resource.get_random<xpu, double>(s);
      mshadow::Tensor<xpu, 2, double> tmp = ret->FlatTo2D<xpu, double>(s);
      prnd->SampleGamma(&tmp, double(alpha), double(beta));  // NOLINT(*)
      break;
    }
  default:
    LOG(FATAL) << "Random only support float32 and float64";
  }
}


template<>
void EvalRandom<DEVICE, ExponentialDistribution>(
    const real_t &lambda,
    const real_t &dummy,  // this is to satisfy the SampleOp lambda signature
    const Resource &resource,
    TBlob *ret,
    RunContext ctx) {
  typedef cpu xpu;  // No support for gpu for this distribution.
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  switch (ret->type_flag_) {
  case mshadow::kFloat32:
    {
      mshadow::Random<xpu, float> *prnd = resource.get_random<xpu, float>(s);
      mshadow::Tensor<xpu, 2, float> tmp = ret->FlatTo2D<xpu, float>(s);
      prnd->SampleExponential(&tmp, float(lambda));  // NOLINT(*)
      break;
    }
  case mshadow::kFloat64:
    {
      mshadow::Random<xpu, double> *prnd = resource.get_random<xpu, double>(s);
      mshadow::Tensor<xpu, 2, double> tmp = ret->FlatTo2D<xpu, double>(s);
      prnd->SampleExponential(&tmp, double(lambda));  // NOLINT(*)
      break;
    }
  default:
    LOG(FATAL) << "Random only support float32 and float64";
  }
}

template<>
void EvalRandom<DEVICE, PoissonDistribution>(
    const real_t &lambda,
    const real_t &dummy,  // this is to satisfy the SampleOp lambda signature
    const Resource &resource,
    TBlob *ret,
    RunContext ctx) {
  typedef cpu xpu;  // No support for gpu for this distribution.
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  switch (ret->type_flag_) {
  case mshadow::kFloat32:
    {
      mshadow::Random<xpu, float> *prnd = resource.get_random<xpu, float>(s);
      mshadow::Tensor<xpu, 2, float> tmp = ret->FlatTo2D<xpu, float>(s);
      prnd->SamplePoisson(&tmp, float(lambda));  // NOLINT(*)
      break;
    }
  case mshadow::kFloat64:
    {
      mshadow::Random<xpu, double> *prnd = resource.get_random<xpu, double>(s);
      mshadow::Tensor<xpu, 2, double> tmp = ret->FlatTo2D<xpu, double>(s);
      prnd->SamplePoisson(&tmp, double(lambda));  // NOLINT(*)
      break;
    }
  default:
    LOG(FATAL) << "Random only support float32 and float64";
  }
}

template<>
void EvalRandom<DEVICE, NegBinomialDistribution>(
    const real_t &k,
    const real_t &p,
    const Resource &resource,
    TBlob *ret,
    RunContext ctx) {
  typedef cpu xpu;  // No support for gpu for this distribution.
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  switch (ret->type_flag_) {
  case mshadow::kFloat32:
    {
      mshadow::Random<xpu, float> *prnd = resource.get_random<xpu, float>(s);
      mshadow::Tensor<xpu, 2, float> tmp = ret->FlatTo2D<xpu, float>(s);
      prnd->SampleNegativeBinomial(&tmp, float(k), float(p));  // NOLINT(*)
      break;
    }
  case mshadow::kFloat64:
    {
      mshadow::Random<xpu, double> *prnd = resource.get_random<xpu, double>(s);
      mshadow::Tensor<xpu, 2, double> tmp = ret->FlatTo2D<xpu, double>(s);
      prnd->SampleNegativeBinomial(&tmp, double(k), double(p));  // NOLINT(*)
      break;
    }
  default:
    LOG(FATAL) << "Random only support float32 and float64";
  }
}

template<>
void EvalRandom<DEVICE, GenNegBinomialDistribution>(
    const real_t &mu,
    const real_t &alpha,
    const Resource &resource,
    TBlob *ret,
    RunContext ctx) {
  typedef cpu xpu;  // No support for gpu for this distribution.
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  switch (ret->type_flag_) {
  case mshadow::kFloat32:
    {
      mshadow::Random<xpu, float> *prnd = resource.get_random<xpu, float>(s);
      mshadow::Tensor<xpu, 2, float> tmp = ret->FlatTo2D<xpu, float>(s);
      prnd->SampleGeneralizedNegativeBinomial(&tmp, float(mu), float(alpha));  // NOLINT(*)
      break;
    }
  case mshadow::kFloat64:
    {
      mshadow::Random<xpu, double> *prnd = resource.get_random<xpu, double>(s);
      mshadow::Tensor<xpu, 2, double> tmp = ret->FlatTo2D<xpu, double>(s);
      prnd->SampleGeneralizedNegativeBinomial(&tmp, double(mu), double(alpha));  // NOLINT(*)
      break;
    }
  default:
    LOG(FATAL) << "Random only support float32 and float64";
  }
}

template<>
void Eval<DEVICE>(const real_t &rhs, TBlob *ret, RunContext ctx) {
  mshadow::Stream<DEVICE> *s = ctx.get_stream<DEVICE>();
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
    ret->FlatTo2D<DEVICE, DType>(s) = DType(rhs);
  });
}

template<>
void ElementwiseSum<DEVICE>(const std::vector<TBlob> source,
                            TBlob *dst,
                            RunContext ctx) {
  typedef DEVICE xpu;
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  for (size_t i = 1; i < source.size(); ++i) {
    CHECK_EQ(source[i].type_flag_, dst->type_flag_)
      << "Only support input/output with the same data type";
  }
  MSHADOW_TYPE_SWITCH(dst->type_flag_, DType, {
    Tensor<xpu, 2, DType> out = dst->FlatTo2D<xpu, DType>(s);

    switch (source.size()) {
      case 2: {
        Tensor<xpu, 2, DType> in_0 = source[0].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> in_1 = source[1].FlatTo2D<xpu, DType>(s);
        out = in_0 + in_1;
        break;
      }
      case 3: {
        Tensor<xpu, 2, DType> in_0 = source[0].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> in_1 = source[1].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> in_2 = source[2].FlatTo2D<xpu, DType>(s);
        out = in_0 + in_1 + in_2;
        break;
      }
      case 4: {
        Tensor<xpu, 2, DType> in_0 = source[0].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> in_1 = source[1].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> in_2 = source[2].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> in_3 = source[3].FlatTo2D<xpu, DType>(s);
        out = in_0 + in_1 + in_2 + in_3;
        break;
      }
      default: {
        Tensor<xpu, 2, DType> in_0 = source[0].FlatTo2D<xpu, DType>(s);
        out = F<op::mshadow_op::identity>(in_0);
        for (size_t i = 1; i < source.size(); ++i) {
          out += source[i].FlatTo2D<xpu, DType>(s);
        }
        break;
      }
    }
  });
}

template <>
void EvalBroadcast<DEVICE>(TBlob const& src, TBlob* ret, int size, RunContext ctx) {
  typedef DEVICE xpu;
  mshadow::Stream<xpu>* s = ctx.get_stream<xpu>();
  mshadow::Tensor<xpu, 3> out = ret->get<xpu, 3, real_t>(s);
  mshadow::Tensor<xpu, 2> in = src.get<xpu, 2, real_t>(s);
  out = mshadow::expr::broadcast_with_axis(in, 0, size);
}

// declarations
DECL_BINARY(DEVICE, MatChooseRowElem, EvalMatChooseRowElem_)
DECL_TERNARY(DEVICE, MatFillRowElem, EvalMatFillRowElem_)
DECL_BINARY(DEVICE, OneHotEncode, EvalOneHot_)
DECL_BINARY(DEVICE, Plus, EvalBinary_)
DECL_BINARY(DEVICE, Minus, EvalBinary_)
DECL_BINARY(DEVICE, Mul, EvalBinary_)
DECL_BINARY(DEVICE, Div, EvalBinary_)
DECL_SCALAR(DEVICE, Plus, EvalScalar_, true)
DECL_SCALAR(DEVICE, Minus, EvalScalar_, true)
DECL_SCALAR(DEVICE, Mul, EvalScalar_, true)
DECL_SCALAR(DEVICE, Div, EvalScalar_, true)

// for reverse seq
DECL_SCALAR(DEVICE, Plus, EvalScalar_, false)
DECL_SCALAR(DEVICE, Minus, EvalScalar_, false)
DECL_SCALAR(DEVICE, Mul, EvalScalar_, false)
DECL_SCALAR(DEVICE, Div, EvalScalar_, false)
}  // namespace ndarray
}  // namespace mxnet

#endif  // MXNET_NDARRAY_NDARRAY_FUNCTION_INL_H_
