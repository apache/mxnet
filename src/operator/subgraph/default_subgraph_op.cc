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

#include <mxnet/ndarray.h>
#include "./common.h"
#include "../../imperative/imperative_utils.h"
#include "../../imperative/cached_op.h"

namespace mxnet {
namespace op {

#define DEBUG_SUBGRAPH 0

class DefaultSubgraphOperator {
 public:
  explicit DefaultSubgraphOperator(const Symbol& sym) : subgraph_sym_(sym) {
    subgraph_exec_.reset(new CachedOp(sym, {{"static_alloc", "true"},
                                            {"static_shape", "true"}}));
  }

  void Forward(const OpContext& ctx,
               const std::vector<NDArray>& inputs,
               const std::vector<OpReqType>& req,
               const std::vector<NDArray>& outputs);
  void Backward(const OpContext& ctx,
                const std::vector<NDArray>& inputs,
                const std::vector<OpReqType>& req,
                const std::vector<NDArray>& outputs) {
    LOG(FATAL) << "Not implemented";
  }

 private:
  nnvm::Symbol subgraph_sym_;
  CachedOpPtr subgraph_exec_;
};

void DefaultSubgraphOperator::Forward(const OpContext& ctx,
                                      const std::vector<NDArray>& inputs,
                                      const std::vector<OpReqType>& req,
                                      const std::vector<NDArray>& outputs) {
  std::vector<NDArray> tmp_inputs = inputs;
  std::vector<NDArray*> input_ptrs;
  input_ptrs.reserve(inputs.size());
  for (auto& nd : tmp_inputs) {
    input_ptrs.push_back(&nd);
  }
  std::vector<NDArray> tmp_outputs = outputs;
  std::vector<NDArray*> output_ptrs;
  for (auto& nd : tmp_outputs) {
    output_ptrs.push_back(&nd);
  }
#if DEBUG_SUBGRAPH
  for (size_t i = 0; i < inputs.size(); ++i) {
    LOG(INFO) << "inputs[" << i << "].version = " << inputs[i].version();
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    LOG(INFO) << "outputs[" << i << "].version = " << outputs[i].version();
  }
#endif
  subgraph_exec_->Forward(subgraph_exec_, input_ptrs, output_ptrs);
}

OpStatePtr CreateDefaultSubgraphOpState(const NodeAttrs& attrs,
                                        Context ctx,
                                        const std::vector<TShape>& in_shapes,
                                        const std::vector<int>& in_types) {
  return OpStatePtr::Create<DefaultSubgraphOperator>(*attrs.subgraphs[0]);
}

void DefaultSubgraphOpForward(const OpStatePtr& state_ptr,
                              const OpContext& ctx,
                              const std::vector<NDArray>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<NDArray>& outputs) {
  DefaultSubgraphOperator& op = state_ptr.get_state<DefaultSubgraphOperator>();
  op.Forward(ctx, inputs, req, outputs);
}

NNVM_REGISTER_OP(_default_subgraph_op)
.describe(R"code(_default_subgraph_op)code" ADD_FILELINE)
.set_num_inputs(DefaultSubgraphOpNumInputs)
.set_num_outputs(DefaultSubgraphOpNumOutputs)
.set_attr<nnvm::FListInputNames>("FListInputNames", DefaultSubgraphOpListInputs)
.set_attr<nnvm::FListOutputNames>("FListOutputNames", DefaultSubgraphOpListOutputs)
.set_attr<FCreateOpState>("FCreateOpState", CreateDefaultSubgraphOpState)
.set_attr<nnvm::FInferShape>("FInferShape", DefaultSubgraphOpShape)
.set_attr<nnvm::FInferType>("FInferType", DefaultSubgraphOpType)
.set_attr<FInferStorageType>("FInferStorageType", DefaultSubgraphOpStorageType)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", DefaultSubgraphOpForward)
.set_attr<nnvm::FMutateInputs>("FMutateInputs", DefaultSubgraphOpMutableInputs)
.set_attr<std::string>("key_var_num_args", "num_args")
.set_attr<FExecType>("FExecType", DefaultSubgraphOpExecType)
.add_argument("data", "NDArray-or-Symbol[]", "input data list");

}  // namespace op
}  // namespace mxnet
