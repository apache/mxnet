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

#include <mxnet/base.h>
#include <nnvm/c_api.h>
#include <vector>
#include <string>
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

std::vector<NDArray*> Invoke(const nnvm::Op* op,
                             nnvm::NodeAttrs* attrs,
                             int num_inputs,
                             NDArray** inputs,
                             int* num_outputs,
                             NDArray** outputs);

template<typename T>
void SetAttrDict(nnvm::NodeAttrs* attrs) {
  if (Imperative::Get()->is_recording()) {
    ::dmlc::get<T>(attrs->parsed).SetAttrDict(&(attrs->dict));
  }
}

}  // namespace mxnet

#endif  // MXNET_API_OPERATOR_UTILS_H_
