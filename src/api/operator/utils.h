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
 * \file utils.h
 * \brief Utility functions for operator invoke
 */
#ifndef MXNET_API_OPERATOR_UTILS_H_
#define MXNET_API_OPERATOR_UTILS_H_

#include <mxnet/runtime/ffi_helper.h>
#include <mxnet/runtime/container.h>
#include <mxnet/runtime/packed_func.h>
#include <mxnet/api_registry.h>
#include <mxnet/base.h>
#include <nnvm/c_api.h>
#include <vector>
#include "../../imperative/imperative_utils.h"

namespace mxnet {

void SetInOut(std::vector<NDArray*>* ndinputs,
              std::vector<NDArray*>* ndoutputs,
              int num_inputs,
              NDArray** inputs,
              int *num_outputs,
              int infered_num_outputs,
              int num_visible_outputs,
              NDArray** out_array);

template<typename T>
std::vector<NDArray*> Invoke(const nnvm::Op* op,
                             nnvm::NodeAttrs* attrs,
                             int num_inputs,
                             NDArray** inputs,
                             int* num_outputs,
                             NDArray** outputs) {
  int infered_num_outputs;
  int num_visible_outputs;
  imperative::SetNumOutputs(op, *attrs, num_inputs, &infered_num_outputs, &num_visible_outputs);

  std::vector<NDArray*> ndinputs, ndoutputs;
  SetInOut(&ndinputs, &ndoutputs, num_inputs, inputs,
      num_outputs, infered_num_outputs, num_visible_outputs, outputs);

  auto state = Imperative::Get()->Invoke(Context::CPU(), *attrs, ndinputs, ndoutputs);
  if (Imperative::Get()->is_recording()) {
    ::dmlc::get<T>(attrs->parsed).SetAttrDict(&(attrs->dict));
    Imperative::Get()->RecordOp(std::move(*attrs), ndinputs, ndoutputs, state);
  }
  for (int i = *num_outputs; i < infered_num_outputs; ++i) delete ndoutputs[i];
  return ndoutputs;
}

}  // namespace mxnet

#endif  // MXNET_API_OPERATOR_UTILS_H_
