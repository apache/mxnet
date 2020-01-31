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
 *  Copyright (c) 2019 by Contributors
 * \file multi_lamb.cc
 * \brief multi-tensor LAMB optimizer
 * \author Moises Hernandez
 */

#include "./multi_lamb-inl.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

template<typename MPDType, bool has_mixed_precision>
struct MultiLAMBKernelStep1 {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i,
                                  const MultiLAMBKernelParam<DType, MPDType>& kernel_params,
                                  const float beta1, const float beta2,
                                  const float epsilon,
                                  const float clip_gradient,
                                  const bool bias_correction,
                                  const float rescale_grad,
                                  float* temp_g) {
    using namespace mshadow_op;
    for (size_t index = 0; index < kernel_params.ntensors; ++index) {
      if ((size_t)i < kernel_params.sizes[index]) {
        MPDType w = has_mixed_precision ? kernel_params.weights32[index][i]:
                                          MPDType(kernel_params.weights[index][i]);
        MPDType scaled_grad = static_cast<MPDType>(kernel_params.grads[index][i])*rescale_grad;
        if (clip_gradient >= 0.0f)
            scaled_grad = mshadow_op::clip::Map(scaled_grad, static_cast<MPDType>(clip_gradient));
        MPDType mean = static_cast<MPDType>(beta1) * kernel_params.mean[index][i] +
          (static_cast<MPDType>(1.0f) - static_cast<MPDType>(beta1)) * scaled_grad;
        MPDType var = static_cast<MPDType>(beta2) * kernel_params.var[index][i] +
          (static_cast<MPDType>(1.0f) - static_cast<MPDType>(beta2)) * scaled_grad * scaled_grad;
        kernel_params.mean[index][i] = mean;
        kernel_params.var[index][i] = var;

        MPDType g;
        if (bias_correction) {
          MPDType mean_hat = mean / (static_cast<MPDType>(1.0f) -
                                    power::Map(static_cast<MPDType>(beta1),
                                    static_cast<MPDType>(kernel_params.step_count[index])));
          MPDType var_hat = var / (static_cast<MPDType>(1.0f) -
                                  power::Map(static_cast<MPDType>(beta2),
                                  static_cast<MPDType>(kernel_params.step_count[index])));
          g = mean_hat / (sqrt(var_hat) + static_cast<MPDType>(epsilon)) +
                         kernel_params.wds[index] * w;
        } else {
          g = mean / (sqrt(var) + static_cast<MPDType>(epsilon)) +
                      kernel_params.wds[index] * w;
        }
        temp_g[kernel_params.tensor2temp_g[index]+i] = g;
      }
    }
  }
};

template<typename MPDType, bool has_mixed_precision>
struct MultiLAMBKernelStep2 {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i,
                                  const MultiLAMBKernelParam<DType, MPDType>& kernel_params,
                                  const float* sum_sq_weigths,
                                  const float* sum_sq_temp_g,
                                  const float* temp_g,
                                  const float lower_bound,
                                  const float upper_bound,
                                  const OpReqType req) {
    for (size_t index = 0; index < kernel_params.ntensors; ++index) {
      if ((size_t)i < kernel_params.sizes[index]) {
        MPDType w = has_mixed_precision ? kernel_params.weights32[index][i]:
                                          MPDType(kernel_params.weights[index][i]);
        float r1 = sqrt(sum_sq_weigths[index]);
        float r2 = sqrt(sum_sq_temp_g[index]);
        if (lower_bound >= 0)
          r1 = std::max(r1, lower_bound);
        if (upper_bound >= 0)
          r1 = std::min(r1, upper_bound);

        // calculate lamb_trust_ratio
        MPDType r;
        if (r1 == 0.0f || r2 == 0.0f)
          r = 1.0f;
        else
          r = r1/r2;

        MPDType lr_adjusted = kernel_params.learning_rates[index] * r;
        w -= lr_adjusted * temp_g[kernel_params.tensor2temp_g[index]+i];

        // update weights
        if (has_mixed_precision)
          kernel_params.weights32[index][i] = w;
        KERNEL_ASSIGN(kernel_params.out_data[index][i], req, w);
      }
    }
  }
};

template<typename MPDType, typename DType>
void CallKernel1(Stream<cpu>* s,
                  const MultiLAMBKernelParam<DType, MPDType>& kernel_params,
                  const MultiLAMBParam &param,
                  float* temp_g,
                  int* block_to_tensor,
                  int* block_to_chunk) {
  Kernel<MultiLAMBKernelStep1<MPDType, !std::is_same<DType, MPDType>::value>, cpu>::
                               Launch(s, kernel_params.max_size,
                               kernel_params,
                               param.beta1, param.beta2,
                               param.epsilon,
                               param.clip_gradient,
                               param.bias_correction,
                               param.rescale_grad,
                               temp_g);
}

template<typename MPDType, typename DType>
void CallKernel2(Stream<cpu>* s,
                  const MultiLAMBKernelParam<DType, MPDType>& kernel_params,
                  const MultiLAMBParam &param,
                  float* r1, float* r2,
                  float* temp_g,
                  int* block_to_tensor,
                  int* block_to_chunk,
                  const OpReqType req) {
  Kernel<MultiLAMBKernelStep2<MPDType, !std::is_same<DType, MPDType>::value>, cpu>::
                               Launch(s, kernel_params.max_size,
                               kernel_params,
                               r1, r2,
                               temp_g,
                               param.lower_bound, param.upper_bound,
                               req);
}

DMLC_REGISTER_PARAMETER(MultiLAMBParam);

std::vector<std::string> LAMBParamToVector(uint32_t num_tensors,
                                           const char *p_names[],
                                           size_t n_params) {
  std::vector<std::string> ret;
  for (uint32_t i = 0; i < num_tensors; ++i) {
    const auto idx = std::to_string(i);
    for (size_t j = 0; j < n_params; ++j)
      ret.push_back(std::string(p_names[i]) + idx);
  }
  return ret;
}

inline uint32_t NumTensors(const nnvm::NodeAttrs& attrs) {
  return static_cast<uint32_t>(dmlc::get<MultiLAMBParam>(attrs.parsed).num_tensors);
}

NNVM_REGISTER_OP(_multi_lamb_update)
.describe(R"code(Compute the LAMB coefficients of multiple weights and grads"
)code" ADD_FILELINE)
.set_num_inputs([](const nnvm::NodeAttrs& attrs) {
    return NumTensors(attrs) * 4;
  })
.set_num_outputs([](const nnvm::NodeAttrs& attrs) {
    return NumTensors(attrs);
  })
.set_attr_parser(ParamParser<MultiLAMBParam>)
.set_attr<mxnet::FInferShape>("FInferShape", MultiLAMBInferShape<MultiLAMBParam, 4>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<-1, -1>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    const char *param_names[] = {"weight_", "grad_", "mean_", "var_"};
    return LAMBParamToVector(NumTensors(attrs), param_names,
                             sizeof(param_names)/sizeof(param_names[0]));
  })
// mutable: mean, var
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    std::vector<uint32_t> ret;
    const auto i_max = NumTensors(attrs);
    for (size_t i = 0; i < i_max; ++i) {
      ret.push_back(i * 4 + 2);
      ret.push_back(i * 4 + 3);
    }
    return ret;
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", MultiLAMBUpdate<cpu, false>)
.add_argument("data", "NDArray-or-Symbol[]", "data")
.add_arguments(MultiLAMBParam::__FIELDS__());


NNVM_REGISTER_OP(_multi_mp_lamb_update)
.describe(R"code(Compute the LAMB coefficients of multiple weights and grads with Mix Precision"
)code" ADD_FILELINE)
.set_num_inputs([](const nnvm::NodeAttrs& attrs) {
    return NumTensors(attrs) * 5;
  })
.set_num_outputs([](const nnvm::NodeAttrs& attrs) {
    return NumTensors(attrs);
  })
.set_attr_parser(ParamParser<MultiLAMBParam>)
.set_attr<mxnet::FInferShape>("FInferShape", MultiLAMBInferShape<MultiLAMBParam, 5>)
.set_attr<nnvm::FInferType>("FInferType", MPMultiLAMBInferType<MultiLAMBParam, 5>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    const char *param_names[] = {"weight_", "grad_", "mean_", "var_", "weight32_"};
    return LAMBParamToVector(NumTensors(attrs), param_names,
                             sizeof(param_names)/sizeof(param_names[0]));
  })
// mutable: mean, var, weights32
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    std::vector<uint32_t> ret;
    const auto i_max = NumTensors(attrs);
    for (size_t i = 0; i < i_max; ++i) {
      ret.push_back(i * 5 + 2);
      ret.push_back(i * 5 + 3);
      ret.push_back(i * 5 + 4);
    }
    return ret;
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", MultiLAMBUpdate<cpu, true>)
.add_argument("data", "NDArray-or-Symbol[]", "data")
.add_arguments(MultiLAMBParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
