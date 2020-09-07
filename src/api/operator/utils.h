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

bool is_recording();
bool is_deferred_compute();

template<typename T>
void SetAttrDict(nnvm::NodeAttrs* attrs) {
  if (is_recording() || is_deferred_compute()) {
    ::dmlc::get<T>(attrs->parsed).SetAttrDict(&(attrs->dict));
  }
}

template<typename ValueType, typename T>
Tuple<ValueType> Obj2Tuple(const runtime::ObjectRef& src) {
  runtime::ADT adt = Downcast<runtime::ADT, runtime::ObjectRef>(src);
  Tuple<ValueType> ret(adt.size(), 0);
  for (size_t i = 0; i < adt.size(); ++i) {
    ret[i] = Downcast<T, runtime::ObjectRef>(adt[i])->value;
  }
  return ret;
}

}  // namespace mxnet

#endif  // MXNET_API_OPERATOR_UTILS_H_
