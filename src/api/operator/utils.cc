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
 * \file utils.cc
 * \brief Utility functions for operator invoke
 */
#include "utils.h"
#include "../../imperative/imperative_utils.h"

namespace mxnet {

bool is_recording() {
  return Imperative::Get()->is_recording();
}

void SetInOut(std::vector<NDArray*>* ndinputs,
              std::vector<NDArray*>* ndoutputs,
              int num_inputs,
              NDArray** inputs,
              int *num_outputs,
              int infered_num_outputs,
              int num_visible_outputs,
              NDArray** out_array) {
  ndinputs->clear();
  ndinputs->reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    NDArray* inp = reinterpret_cast<NDArray*>(inputs[i]);
    if (!features::is_enabled(features::INT64_TENSOR_SIZE)) {
      if (shape_is_known(inp->shape())) {  // Shape may be unknown after dynamic shape operators
        CHECK_LT(inp->shape().Size(), (int64_t{1} << 31) - 1)
          << "[SetInOut] Size of tensor you are trying to allocate is larger than "
               "2^31 elements. Please build with flag USE_INT64_TENSOR_SIZE=1";
      }
    }
    ndinputs->emplace_back(inp);
  }

  ndoutputs->clear();
  ndoutputs->reserve(infered_num_outputs);
  if (out_array == nullptr) {
    for (int i = 0; i < infered_num_outputs; ++i) {
      ndoutputs->emplace_back(new NDArray());
    }
    *num_outputs = num_visible_outputs;
  } else {
    CHECK(*num_outputs == infered_num_outputs || *num_outputs == num_visible_outputs)
      << "Operator expects " << infered_num_outputs << " (all) or "
      << num_visible_outputs << " (visible only) outputs, but got "
      << *num_outputs << " instead.";
    for (int i = 0; i < *num_outputs; ++i) {
      ndoutputs->emplace_back(out_array[i]);
    }
    for (int i = *num_outputs; i < infered_num_outputs; ++i) {
      ndoutputs->emplace_back(new NDArray());
    }
  }
}

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

  if (Imperative::Get()->is_deferred_compute()) {
    Imperative::Get()->RecordDeferredCompute(std::move(*attrs), ndinputs, ndoutputs);
  } else {
    for (NDArray *input : ndinputs) {
      Imperative::DCInfo::Compute(*input);
    }
    auto state = Imperative::Get()->Invoke(Context::CPU(), *attrs, ndinputs, ndoutputs);
    if (Imperative::Get()->is_recording()) {
      Imperative::Get()->RecordOp(std::move(*attrs), ndinputs, ndoutputs, state);
    }
  }
  for (int i = *num_outputs; i < infered_num_outputs; ++i) delete ndoutputs[i];
  return ndoutputs;
}

}  // namespace mxnet
