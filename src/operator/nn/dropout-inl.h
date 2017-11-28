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
 * Copyright (c) 2015 by Contributors
 * \file dropout-inl.h
 * \brief
 * \author Bing Xu, Da Zheng
*/

#ifndef MXNET_OPERATOR_NN_DROPOUT_INL_H_
#define MXNET_OPERATOR_NN_DROPOUT_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include "../../engine/openmp.h"
#include "../operator_common.h"
#include "../mshadow_op.h"

#if defined(USE_MKL) && defined(_OPENMP)
#include <omp.h>

#include <mkl_vml_functions.h>
#include <mkl_vsl.h>
#endif  // USE_MKL && _OPENMP

namespace dropout {
enum DropoutOpInputs {kData};
enum DropoutOpOutputs {kOut, kMask};
enum DropoutOpForwardResource {kRandom};
enum DropoutOpMode {kTraining, kAlways};
}  // namespace dropout

namespace mxnet {
namespace op {

#if defined(USE_MKL) && defined(_OPENMP)
static void bernoulli_generate(int n, double p, int* r) {
  const int seed = 17 + rand() % 4096;  // NOLINT(runtime/threadsafe_fn)
  const int nthr = engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
# pragma omp parallel num_threads(nthr)
  {
    const int ithr = omp_get_thread_num();
    const int avg_amount = (n + nthr - 1) / nthr;
    const int my_offset = ithr * avg_amount;
    const int my_amount = std::min(my_offset + avg_amount, n) - my_offset;
    if (my_amount > 0) {
      VSLStreamStatePtr stream;
      vslNewStream(&stream, VSL_BRNG_MCG31, seed);
      vslSkipAheadStream(stream, my_offset);
      viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF, stream, my_amount,
        r + my_offset, p);
      vslDeleteStream(&stream);
    }
  }
}
#endif  // USE_MKL && _OPENMP

struct DropoutParam : public dmlc::Parameter<DropoutParam> {
  float p;
  int mode;
  DMLC_DECLARE_PARAMETER(DropoutParam) {
    DMLC_DECLARE_FIELD(p).set_default(0.5)
    .set_range(0, 1)
    .describe("Fraction of the input that gets dropped out during training time.");
    DMLC_DECLARE_FIELD(mode)
    .add_enum("training", dropout::kTraining)
    .add_enum("always", dropout::kAlways)
    .set_default(dropout::kTraining)
    .describe("Whether to only turn on dropout during training or to also turn on for inference.");
  }
};  // struct DropoutParam

template<typename xpu, typename DType>
void DropoutForward(const OpContext &ctx, const DropoutParam &param,
                    const std::vector<TBlob> &in_data, const std::vector<OpReqType> &req,
                    const std::vector<TBlob> &out_data) {
  using namespace mshadow;
  using namespace mshadow::expr;
  real_t pkeep_ = 1.0f - param.p;
  int mode_ = param.mode;
  CHECK_EQ(in_data.size(), 1U);
  if (ctx.is_train) {
    CHECK_EQ(out_data.size(), 2U);
  }
  Stream<xpu> *s = ctx.get_stream<xpu>();
  Tensor<xpu, 2, DType> data = in_data[dropout::kData].FlatTo2D<xpu, DType>(s);
  Tensor<xpu, 2, DType> out = out_data[dropout::kOut].FlatTo2D<xpu, DType>(s);
  if (ctx.is_train || mode_ == dropout::kAlways) {
    Tensor<xpu, 2, DType> mask = out_data[dropout::kMask].FlatTo2D<xpu, DType>(s);
#if !defined(__CUDACC__) && defined(USE_MKL) && defined(_OPENMP)
    DType* outptr = out.dptr_;
    DType* dataptr = data.dptr_;
    auto maskptr = reinterpret_cast<int*>(mask.dptr_);
    int count = mask.shape_[0]*mask.shape_[1];
    bernoulli_generate(count, pkeep_, maskptr);
    const float pk_1 = 1.0f / pkeep_;
#pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
    for (int i = 0; i < count; ++i) {
      outptr[i] = dataptr[i] * maskptr[i] * pk_1;
    }
#else
    Random<xpu> *prnd = ctx.requested[dropout::kRandom].get_random<xpu, real_t>(s);
    mask = tcast<DType>(F<mshadow_op::threshold>(
            prnd->uniform(mask.shape_), pkeep_) * (1.0f / pkeep_));
    Assign(out, req[dropout::kOut], data * mask);
#endif  // USE_MKL && _OPENMP
  } else {
    Assign(out, req[dropout::kOut], F<mshadow_op::identity>(data));
  }
}

template<typename xpu, typename DType>
void DropoutBackward(const OpContext &ctx, const DropoutParam &param,
                     const TBlob &out_grad, const TBlob &out_data_mask,
                     const OpReqType &req, const TBlob &in_grad) {
  using namespace mshadow;
  using namespace mshadow::expr;
  int mode_ = param.mode;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  Tensor<xpu, 2, DType> grad = out_grad.FlatTo2D<xpu, DType>(s);
  Tensor<xpu, 2, DType> mask = out_data_mask.FlatTo2D<xpu, DType>(s);
  Tensor<xpu, 2, DType> gdata = in_grad.FlatTo2D<xpu, DType>(s);
  if (ctx.is_train || mode_ == dropout::kAlways) {
#if !defined(__CUDACC__) && defined(USE_MKL) && defined(_OPENMP)
    real_t pkeep_ = 1.0f - param.p;
    DType* ingradptr = gdata.dptr_;
    DType* outgradptr = grad.dptr_;
    auto maskptr = reinterpret_cast<int*>(mask.dptr_);
    int count = mask.shape_[0]*mask.shape_[1];
    const float pk_1 = 1.0f / pkeep_;
#pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
    for (int i = 0; i < count; ++i) {
      ingradptr[i] = outgradptr[i] * maskptr[i] * pk_1;
    }
#else  // USE_MKL && _OPENMP
    Assign(gdata, req, grad * mask);
#endif  // USE_MKL && _OPENMP
  } else {
    Assign(gdata, req, F<mshadow_op::identity>(grad));
  }
}

template<typename xpu>
void DropoutCompute(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  const DropoutParam& param = nnvm::get<DropoutParam>(attrs.parsed);
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    DropoutForward<xpu, DType>(ctx, param, inputs, req, outputs);
  });
}

template<typename xpu>
void DropoutGradCompute(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  const DropoutParam& param = nnvm::get<DropoutParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1);
  CHECK_EQ(req.size(), 1);

  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    DropoutBackward<xpu, DType>(ctx, param, inputs[0], inputs[1], req[0],
                                outputs[0]);
  });
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NN_DROPOUT_INL_H_
