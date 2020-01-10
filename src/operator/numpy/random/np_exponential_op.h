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
 * \file np_exponential_op.h
 * \brief Operator for numpy sampling from exponential distribution.
 */

#ifndef MXNET_OPERATOR_NUMPY_RANDOM_NP_EXPONENTIAL_OP_H_
#define MXNET_OPERATOR_NUMPY_RANDOM_NP_EXPONENTIAL_OP_H_

#include <mxnet/operator_util.h>
#include <algorithm>
#include <string>
#include <vector>
#include <cmath>
#include "../../elemwise_op_common.h"
#include "../../mshadow_op.h"
#include "../../mxnet_op.h"
#include "../../operator_common.h"
#include "../../tensor/elemwise_binary_broadcast_op.h"
#include "./dist_common.h"

namespace mxnet {
namespace op {

struct NumpyExponentialParam : public dmlc::Parameter<NumpyExponentialParam> {
    dmlc::optional<float> scale;
    dmlc::optional<mxnet::Tuple<int>> size;
    DMLC_DECLARE_PARAMETER(NumpyExponentialParam) {
        DMLC_DECLARE_FIELD(scale)
        .set_default(dmlc::optional<float> (1.0));
        DMLC_DECLARE_FIELD(size)
        .set_default(dmlc::optional<mxnet::Tuple<int>>())
        .describe("Output shape. If the given shape is, "
            "e.g., (m, n, k), then m * n * k samples are drawn. "
            "Default is None, in which case a single value is returned.");
    }
};

template <typename DType>
struct scalar_exponential_kernel {
  MSHADOW_XINLINE static void Map(index_t i, float scale, float *threshold,
                                  DType *out) {
    out[i] = -scale * log(threshold[i]);
  }
};

template <typename xpu>
void NumpyExponentialForward(const nnvm::NodeAttrs &attrs,
                         const OpContext &ctx,
                         const std::vector<TBlob> &inputs,
                         const std::vector<OpReqType> &req,
                         const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  const NumpyExponentialParam &param = nnvm::get<NumpyExponentialParam>(attrs.parsed);
  CHECK_GE(param.scale.value(), 0.0) << "ValueError: expect scale >= 0";
  Stream<xpu> *s = ctx.get_stream<xpu>();
  index_t output_len = outputs[0].Size();
  Random<xpu, float> *prnd = ctx.requested[0].get_random<xpu, float>(s);
  Tensor<xpu, 1, float> workspace =
      ctx.requested[1].get_space_typed<xpu, 1, float>(Shape1(output_len), s);
  prnd->SampleUniform(&workspace, 0.0, 1.0);
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Kernel<scalar_exponential_kernel<DType>, xpu>::Launch(
                                        s, outputs[0].Size(), param.scale.value(),
                                        workspace.dptr_, outputs[0].dptr<DType>());
  });
  
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_RANDOM_NP_BERNOULLI_OP_H_