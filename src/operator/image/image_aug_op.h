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

#ifndef MXNET_OPERATOR_IMAGE_IMAGE_AUG_OP_H_
#define MXNET_OPERATOR_IMAGE_IMAGE_AUG_OP_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <utility>
#include <algorithm>
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "../mxnet_op.h"

namespace mxnet {
namespace op {

struct NormalizeParam : public dmlc::Parameter<NormalizeParam> {
  nnvm::Tuple<float> mean, std;
  DMLC_DECLARE_PARAMETER(NormalizeParam) {
    DMLC_DECLARE_FIELD(mean).set_default(nnvm::Tuple<float>({0.f}))
      .describe("");
    DMLC_DECLARE_FIELD(std).set_default(nnvm::Tuple<float>({1.f}))
      .describe("");
  }
};


void NormalizeCompute(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<NDArray>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<NDArray>& outputs) {
  using namespace mxnet_op;
  const auto& params = dmlc::get<NormalizeParam>(attrs.parsed);
  CHECK_NE(req[0], kAddTo);
  MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    auto num_channel = inputs[0].shape_[0];
    auto size = inputs[0].Size(1, inputs[0].ndim());
    nnvm::Tuple<DType> mean(params.mean.begin(), params.mean.end());
    nnvm::Tuple<DType> std(params.std.begin(), params.std.end());
    DType* src = inputs[0].dptr<DType>();
    DType* dst = outputs[0].dptr<DType>();
    for (int i = 0; i < num_channel; ++i) {
      for (int j = 0; j < size; ++j, ++out, ++src) {
        *out = (*src - mean[i]) / std[i];
      }
    }
  });
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_IMAGE_IMAGE_AUG_OP_H_
