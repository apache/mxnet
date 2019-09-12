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
 * \file np_gamma_op.h
 * \brief Operator for random sampling from gamma distribution
 */

#ifndef MXNET_OPERATOR_NUMPY_RANDOM_NP_GAMMA_OP_H_
#define MXNET_OPERATOR_NUMPY_RANDOM_NP_GAMMA_OP_H_

#include <mxnet/operator_util.h>
#include <mshadow/base.h>
#include <vector>
#include <string>
#include <algorithm>
#include "./dist_common.h"
#include "../../elemwise_op_common.h"
#include "../../tensor/elemwise_binary_broadcast_op.h"
#include "../../mshadow_op.h"
#include "../../mxnet_op.h"
#include "../../operator_common.h"

#define M 2

namespace mxnet {
namespace op {

struct NumpyGammaParam : public dmlc::Parameter<NumpyGammaParam> {
  dmlc::optional<float> shape;
  dmlc::optional<float> scale;
  std::string ctx;
  int dtype;
  dmlc::optional<mxnet::Tuple<int>> size;
  DMLC_DECLARE_PARAMETER(NumpyGammaParam) {
    DMLC_DECLARE_FIELD(shape);
    DMLC_DECLARE_FIELD(scale);
    DMLC_DECLARE_FIELD(size)
        .set_default(dmlc::optional<mxnet::Tuple<int>>())
        .describe("Output shape. If the given shape is, "
                  "e.g., (m, n, k), then m * n * k samples are drawn. "
                  "Default is None, in which case a single value is returned.");
    DMLC_DECLARE_FIELD(ctx)
    .set_default("xpu")
    .describe("Context of output, in format [xpu|xpu|xpu_pinned](n)."
              " Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype)
    .add_enum("float32", mshadow::kFloat32)
    .add_enum("float64", mshadow::kFloat64)
    .add_enum("float16", mshadow::kFloat16)
    .set_default(mshadow::kFloat32)
    .describe("DType of the output in case this can't be inferred. "
              "Defaults to float32 if not defined (dtype=None).");
  }
};


inline bool NumpyGammaOpType(const nnvm::NodeAttrs &attrs,
                                   std::vector<int> *in_attrs,
                                   std::vector<int> *out_attrs) {
  const NumpyGammaParam &param = nnvm::get<NumpyGammaParam>(attrs.parsed);
  int otype = param.dtype;
  if (otype != -1) {
    (*out_attrs)[0] = otype;
  } else {
    (*out_attrs)[0] = mshadow::kFloat32;
  }
  return true;
}

// template <typename FType>
// void _copy(FType *dst, FType *src) {
// #if USE_CUDA == 1
//   CUDA_CALL(cudaMemcpy(dst, src, sizeof(FType, cudaMemcpyDeviceToHost)))
// #else
//   *dst = *src;
// #endif
// }

// #if USE_CUDA == 1
// template <typename FType>
// void _copy(context::xpu device, FType *dst, FType *src);
// #endif

template <typename xpu>
void _copy(float *dst, float*src);

template <typename xpu>
void _copy(double *dst, double*src);

namespace mxnet_op {

template <typename IType, typename FType>
MSHADOW_XINLINE void GammaTransform(IType a, IType b,
                                    FType* uniforms, FType* normals) {
  // start
  FType d = a < 1 ? a + 2.0 / 3.0 : a - 1.0 / 3.0;
  FType k = sqrt(9.0 * d);
  FType c = 1.0 / k;
  // printf("c1 %f\n", uniforms[M - 1]);
  for (size_t i = 0; i < M - 1; i++) {
    FType u = uniforms[i];
    FType n = normals[i];
    uniforms[i] = FType(-1);
    FType ocn = 1+c*n;
    FType v = ocn*ocn*ocn;
    if (v > 0) {
      if (u <= (1 - 0.0331 * (n * n) * (n * n))) {
        // rejection sample. The second operation should be
        // performed with low probability. This is the "squeeze"
        uniforms[i] = FType(d * v * b);
      }
      if (logf(u) < 0.5 * (n * n) + d * (1 - v + logf(v))) {
        // rejection sample. The second operation should be
        // performed with low probability. This is the "squeeze"
        uniforms[i] = FType(d * v * b);
      }
    }
  }
  // printf("c2 %f\n", uniforms[M - 1]);
}


template <typename IType, typename FType>
MSHADOW_XINLINE FType GammaReduce(IType a, FType* uniforms) {
  FType u2 = uniforms[M - 1];
  for (size_t i = 0; i < M - 1; i++) {
    FType sample = uniforms[i];
    if (sample > 0) {
      // printf("a %f\n", a);
      // printf("b %f\n",sample);
      // printf("c %f\n", u2);
      return a < 1 ? sample * powf(u2, FType(1.0 / a)) : sample;
    }
  }
  return -1;
}

template<typename OType, typename FType>
struct CheckSuccessKernel {
  MSHADOW_XINLINE static void Map(int i, OType* out, FType* flag) {
    if (out[i] < 0) {
      flag[0] = -1.0;
    }
  }
};

template <int ndim, typename IType, typename OType, typename FType>
struct gamma_kernel {
  MSHADOW_XINLINE static void Map(index_t i,
                                  const Shape <ndim> &lstride, const Shape <ndim> &hstride,
                                  const Shape <ndim> &oshape,
                                  IType *shape, IType *scale,
                                  FType *uniforms, FType *normals,
                                  OType *out) {
  Shape<ndim> coord = unravel(i, oshape);
  auto lidx = static_cast<index_t>(dot(coord, lstride));
  auto hidx = static_cast<index_t>(dot(coord, hstride));
  IType shape_value = shape[lidx];
  IType scale_value = scale[hidx];
  // map phase
  GammaTransform<IType, FType>(shape_value, scale_value,
                                uniforms + i * M, normals + i * M);
  // reduce phase
  OType sample = (OType)GammaReduce<IType, FType>(shape_value, uniforms + i * M);
  out[i] = sample;
  }
};

template <int ndim, typename IType, typename OType, typename FType>
struct gamma_kernel_r {
  MSHADOW_XINLINE static void Map(index_t i,
                                  const Shape <ndim> &lstride, const Shape <ndim> &hstride,
                                  const Shape <ndim> &oshape,
                                  IType *shape, IType *scale,
                                  FType *uniforms, FType *normals,
                                  OType *out, FType* flag) {
  flag[0] = 1;
  Shape<ndim> coord = unravel(i, oshape);
  auto lidx = static_cast<index_t>(dot(coord, lstride));
  auto hidx = static_cast<index_t>(dot(coord, hstride));
  IType shape_value = shape[lidx];
  IType scale_value = scale[hidx];
  if (out[i] < 0) {
    // map phase
    GammaTransform<IType, FType>(shape_value, scale_value,
                                  uniforms + i * M, normals + i * M);
    // reduce phase
    OType sample = (OType)GammaReduce<IType, FType>(shape_value, uniforms + i * M);
    out[i] = sample;
    }
  }
};

template <int ndim, typename IType, typename OType, typename FType>
struct gamma_one_scalar_kernel {
  MSHADOW_XINLINE static void Map(index_t i, int scalar_pos,
                                  const Shape <ndim> &stride,
                                  const Shape <ndim> &oshape,
                                  IType *array, float scalar,
                                  FType *uniforms, FType *normals,
                                  OType *out) {
  Shape<ndim> coord = unravel(i, oshape);
  auto idx = static_cast<index_t>(dot(coord, stride));
  IType shape_value;
  IType scale_value;
  if (scalar_pos == 0) {
    shape_value = scalar;
    scale_value = array[idx];
  } else {
    shape_value = array[idx];
    scale_value = scalar;
  }
  // map phase
  GammaTransform<IType, FType>(shape_value, scale_value,
                                uniforms + i * M, normals + i * M);
  // reduce phase
  OType sample = (OType)GammaReduce<IType, FType>(shape_value, uniforms + i * M);
  out[i] = sample;
  }
};

template <int ndim, typename IType, typename OType, typename FType>
struct gamma_one_scalar_kernel_r {
  MSHADOW_XINLINE static void Map(index_t i, int scalar_pos,
                                  const Shape <ndim> &stride,
                                  const Shape <ndim> &oshape,
                                  IType *array, float scalar,
                                  FType *uniforms, FType *normals,
                                  OType *out, FType *flag) {
  flag[0] = 1;
  Shape<ndim> coord = unravel(i, oshape);
  auto idx = static_cast<index_t>(dot(coord, stride));
  IType shape_value;
  IType scale_value;
  if (scalar_pos == 0) {
    shape_value = scalar;
    scale_value = array[idx];
  } else {
    shape_value = array[idx];
    scale_value = scalar;
  }
  if (out[i] < 0) {
    // map phase
    GammaTransform<IType, FType>(shape_value, scale_value,
                                  uniforms + i * M, normals + i * M);
    // reduce phase
    OType sample = (OType)GammaReduce<IType, FType>(shape_value, uniforms + i * M);
    out[i] = sample;
    }
  }
};

template <typename OType, typename FType>
struct gamma_two_scalar_kernel {
  MSHADOW_XINLINE static void Map(index_t i, float shape_value,
                                  float scale_value, FType *uniforms_origin,
                                  FType *normals_origin, OType *out) {
    // map phase
    FType *uniforms = uniforms_origin + i * M;
    FType *normals = normals_origin + i * M;
    GammaTransform<float, FType>(shape_value, scale_value, uniforms,
                                  normals);
    // reduce phase
    OType sample =
        (OType)GammaReduce<float, FType>(shape_value, uniforms);
    out[i] = sample;
    }
};

template <typename OType, typename FType>
struct gamma_two_scalar_kernel_r {
  MSHADOW_XINLINE static void Map(index_t i, float shape_value,
                                  float scale_value,
                                  FType *uniforms_origin,
                                  FType *normals_origin, OType *out, FType* flag) {
    flag[0] = 1;
    // map phase
    FType *uniforms = uniforms_origin + i * M;
    FType *normals = normals_origin + i * M;
    if (out[i] < 0) {
      GammaTransform<float, FType>(shape_value, scale_value, uniforms,
                                  normals);
    // reduce phase
      OType sample =
        (OType)GammaReduce<float, FType>(shape_value, uniforms);
      out[i] = sample;
      }
    }
};
}  // namespace mxnet_op

template <typename xpu, typename FType>
void NumpyGammaForward(const nnvm::NodeAttrs &attrs, const OpContext &ctx,
                         const std::vector<TBlob> &inputs,
                         const std::vector<OpReqType> &req,
                         const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  const NumpyGammaParam &param = nnvm::get<NumpyGammaParam>(attrs.parsed);
  CHECK_EQ(outputs.size(), 1);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  // Generate base random number.
  Random<xpu, FType> *prnd = ctx.requested[0].get_random<xpu, FType>(s);
  index_t output_len = outputs[0].Size();
  Tensor<xpu, 1, FType> random_tensor =
      ctx.requested[1].get_space_typed<xpu, 1, FType>(Shape1(output_len * 2 * M + 1), s);
  Tensor<xpu, 1, FType> uniform_tensor = random_tensor.Slice(0, output_len * M);
  Tensor<xpu, 1, FType> normal_tensor = random_tensor.Slice(output_len * M, output_len * 2 * M);
  prnd->SampleUniform(&uniform_tensor, 0, 1);
  prnd->SampleGaussian(&normal_tensor, 0, 1);
  mxnet::TShape new_lshape, new_hshape, new_oshape;
  FType failure_indicator = 1.0;
  Tensor<xpu, 1, FType> failure_indic_workspace =
      random_tensor.Slice(output_len * 2 * M, output_len * 2 * M + 1);
  FType *failure_indicator_device = failure_indic_workspace.dptr_;
  // [scalar scalar] case
  if (inputs.size() == 0U) {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, OType, {
      Kernel<gamma_two_scalar_kernel<OType, FType>, xpu>::Launch(
          s, outputs[0].Size(), param.shape.value(), param.scale.value(),
          uniform_tensor.dptr_, normal_tensor.dptr_, outputs[0].dptr<OType>());
      Kernel<CheckSuccessKernel<OType, FType>, xpu>::Launch(
          s, outputs[0].Size(), outputs[0].dptr<OType>(),
          failure_indicator_device);
      _copy<xpu>(&failure_indicator, failure_indicator_device);
      // cout<<failure_indicator<<endl;
      while (1) {
        if (failure_indicator >= 0) {
          break;
        } else {
          prnd->SampleUniform(&uniform_tensor, 0, 1);
          prnd->SampleGaussian(&normal_tensor, 0, 1);
          Kernel<gamma_two_scalar_kernel_r<OType, FType>, xpu>::Launch(
              s, outputs[0].Size(), param.shape.value(), param.scale.value(),
              uniform_tensor.dptr_, normal_tensor.dptr_,
              outputs[0].dptr<OType>(), failure_indicator_device);
          failure_indicator = 1.0;
          Kernel<CheckSuccessKernel<OType, FType>, xpu>::Launch(
              s, outputs[0].Size(), outputs[0].dptr<OType>(),
              failure_indicator_device);
          _copy<xpu>(&failure_indicator, failure_indicator_device);
        }
      }
    });
  } else if (inputs.size() == 1U) {
    // [scalar tensor], [tensor scalar] case
    int ndim = FillShape(inputs[0].shape_, inputs[0].shape_, outputs[0].shape_,
                         &new_lshape, &new_lshape, &new_oshape);
    int scalar_pos;
    float scalar_value;
    // int type_flag = param.t;
    if (param.shape.has_value()) {
      scalar_pos = 0;
      scalar_value = param.shape.value();
    } else {
      scalar_pos = 1;
      scalar_value = param.scale.value();
    }
    MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, IType, {
      MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, OType, {
        BROADCAST_NDIM_SWITCH(ndim, NDim, {
        mshadow::Shape<NDim> oshape = new_oshape.get<NDim>();
        mshadow::Shape<NDim> stride =
            mxnet_op::calc_stride(new_lshape.get<NDim>());
        mxnet_op::Kernel<gamma_one_scalar_kernel<NDim, IType, OType, FType>, xpu>::Launch(
            s, outputs[0].Size(), scalar_pos, stride, oshape,
            inputs[0].dptr<IType>(), scalar_value,
            uniform_tensor.dptr_, normal_tensor.dptr_,
            outputs[0].dptr<OType>());
        Kernel<CheckSuccessKernel<OType, FType>, xpu>::Launch(
          s, outputs[0].Size(), outputs[0].dptr<OType>(),
          failure_indicator_device);
        _copy<xpu>(&failure_indicator, failure_indicator_device);
        while (1) {
        if (failure_indicator >= 0) {
          break;
        } else {
          prnd->SampleUniform(&uniform_tensor, 0, 1);
          prnd->SampleGaussian(&normal_tensor, 0, 1);
          mxnet_op::Kernel<gamma_one_scalar_kernel_r<NDim, IType, OType, FType>, xpu>::Launch(
            s, outputs[0].Size(), scalar_pos, stride, oshape,
            inputs[0].dptr<IType>(), scalar_value,
            uniform_tensor.dptr_, normal_tensor.dptr_,
            outputs[0].dptr<OType>(), failure_indicator_device);
          failure_indicator = 1.0;
          Kernel<CheckSuccessKernel<OType, FType>, xpu>::Launch(
              s, outputs[0].Size(), outputs[0].dptr<OType>(),
              failure_indicator_device);
          _copy<xpu>(&failure_indicator, failure_indicator_device);
        }
      }
        });
      });
    });
  } else if (inputs.size() == 2U) {
    // [tensor tensor] case
    int ndim = FillShape(inputs[0].shape_, inputs[1].shape_, outputs[0].shape_,
                         &new_lshape, &new_hshape, &new_oshape);
    MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, IType, {
      MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, OType, {
        BROADCAST_NDIM_SWITCH(ndim, NDim, {
        mshadow::Shape<NDim> oshape = new_oshape.get<NDim>();
        mshadow::Shape<NDim> lstride =
            mxnet_op::calc_stride(new_lshape.get<NDim>());
        mshadow::Shape<NDim> hstride =
            mxnet_op::calc_stride(new_hshape.get<NDim>());
        mxnet_op::Kernel<gamma_kernel<NDim, IType, OType, FType>, xpu>::Launch(
            s, outputs[0].Size(), lstride, hstride, oshape,
            inputs[0].dptr<IType>(), inputs[1].dptr<IType>(),
            uniform_tensor.dptr_, normal_tensor.dptr_,
            outputs[0].dptr<OType>());
        Kernel<CheckSuccessKernel<OType, FType>, xpu>::Launch(
          s, outputs[0].Size(), outputs[0].dptr<OType>(),
          failure_indicator_device);
        _copy<xpu>(&failure_indicator, failure_indicator_device);
        while (1) {
        if (failure_indicator >= 0) {
          break;
        } else {
          prnd->SampleUniform(&uniform_tensor, 0, 1);
          prnd->SampleGaussian(&normal_tensor, 0, 1);
          mxnet_op::Kernel<gamma_kernel_r<NDim, IType, OType, FType>, xpu>::Launch(
            s, outputs[0].Size(), lstride, hstride, oshape,
            inputs[0].dptr<IType>(), inputs[1].dptr<IType>(),
            uniform_tensor.dptr_, normal_tensor.dptr_,
            outputs[0].dptr<OType>(), failure_indicator_device);
          failure_indicator = 1.0;
          Kernel<CheckSuccessKernel<OType, FType>, xpu>::Launch(
              s, outputs[0].Size(), outputs[0].dptr<OType>(),
              failure_indicator_device);
          _copy<xpu>(&failure_indicator, failure_indicator_device);
          }
        }
        });
      });
    });
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_RANDOM_NP_GAMMA_OP_H_
