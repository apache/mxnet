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
 * \file dnnl_masked_softmax-inl.h
 */
#ifndef MXNET_OPERATOR_NN_DNNL_DNNL_MASKED_SOFTMAX_INL_H_
#define MXNET_OPERATOR_NN_DNNL_DNNL_MASKED_SOFTMAX_INL_H_
#include <numeric>
#include <utility>
#include <vector>

#include "dnnl_base-inl.h"
#include "operator/nn/softmax-inl.h"
#include "dnnl_softmax-inl.h"

#if MXNET_USE_ONEDNN == 1

namespace mxnet {
namespace op {

struct Primitives {
 public:
  dnnl::eltwise_forward minusone;
  dnnl::eltwise_forward::primitive_desc minusone_pd;

  dnnl::binary mask_input;
  dnnl::binary::primitive_desc mask_input_pd;

  dnnl::binary mask_output;
  dnnl::binary::primitive_desc mask_output_pd;
};

class DNNLMaskedSoftmaxFwd {
 public:
  struct Tensors {
    Tensors(const std::vector<NDArray>& inputs, const std::vector<NDArray>& outputs);

    const NDArray& input;
    const NDArray& mask;
    const NDArray& output;
  };

  DNNLMaskedSoftmaxFwd(const MaskedSoftmaxParam& param, const Tensors& tensors)
      : primitives(CreatePrimitives(param, tensors)) {}

  static DNNLMaskedSoftmaxFwd& GetCached(const MaskedSoftmaxParam& param, const Tensors& tensors);
  static std::shared_ptr<Primitives> CreatePrimitives(const MaskedSoftmaxParam& param,
                                                      const Tensors& tensors);

  void Execute(const Tensors& tensors,
               const std::vector<OpReqType>& req,
               const MaskedSoftmaxParam& param,
               bool is_train) const;

 private:
  std::shared_ptr<Primitives> primitives;
};

DNNLMaskedSoftmaxFwd::Tensors::Tensors(const std::vector<NDArray>& inputs,
                                       const std::vector<NDArray>& outputs)
    : input(inputs[0]), mask(inputs[1]), output(outputs[0]) {}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_NN_DNNL_DNNL_MASKED_SOFTMAX_INL_H_
