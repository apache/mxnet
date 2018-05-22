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

#ifndef MXNET_OPERATOR_SUBGRAPH_OP_COMMON_H_
#define MXNET_OPERATOR_SUBGRAPH_OP_COMMON_H_

#include <mxnet/io.h>
#include <mxnet/base.h>
#include <mxnet/op_attr_types.h>
#include <vector>
#include "../imperative/imperative_utils.h"

namespace mxnet {
namespace op {

/*
 * Infer the data types of inputs and outputs of an operator that contains a
 * subgraph.
 */
bool InferSubgraphDataType(const nnvm::Symbol &subgraph, std::vector<int> *in_type,
                           std::vector<int> *out_type);

/*
 * Infer the storage types of inputs and outputs of an operator that contains a
 * subgraph.
 */
bool InferSubgraphStorage(const nnvm::Symbol &subgraph,
                          const int dev_mask,
                          DispatchMode* dispatch_mode,
                          std::vector<int> *in_attrs,
                          std::vector<int> *out_attrs);

/*
 * Infer the storage types of inputs and outputs of the backward computation of
 * an operator that contains a subgraph.
 */
bool InferSubgraphBackwardStorage(const nnvm::Symbol &subgraph,
                                  const int dev_mask,
                                  DispatchMode* dispatch_mode,
                                  std::vector<int> *in_attrs,
                                  std::vector<int> *out_attrs);

/*
 * This contains the states for running a loop and provides methods
 * of running the subgraph computation for an iteration.
 */
class LoopState {
  // These are output arrays from all iterations.
  // They also contain the Op state for each CachedOp.
  std::vector<std::vector<NDArray> > all_outputs;
  std::vector<std::vector<NDArray> > all_inputs;
  std::vector<std::vector<NDArray> > all_gradients;
  std::vector<CachedOpPtr> iter_ops;
  Symbol subgraph_sym;
  nnvm::Graph subgraph;

 public:
  LoopState(const Symbol &g) {
    this->subgraph_sym = g;
    this->subgraph.outputs = g.outputs;
  }

  void Forward(std::vector<NDArray> cinputs,
               const std::vector<OpReqType>& req,
               std::vector<NDArray> coutputs,
               bool is_recording);
  void Backward(int iter_no,
                std::vector<NDArray> ograds,
                const std::vector<OpReqType> &req,
                std::vector<NDArray> igrads);
  void Cleanup() {
    all_outputs.clear();
    all_inputs.clear();
    all_gradients.clear();
    iter_ops.clear();
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_SUBGRAPH_OP_COMMON_H_
