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

// Threadsafe and minimal functionality cached op version for Inference
// lot of code reused from cached_op.h
#ifndef MXNET_IMPERATIVE_NAIVE_CACHED_OP_H_
#define MXNET_IMPERATIVE_NAIVE_CACHED_OP_H_

#include <mxnet/imperative.h>
#include <vector>
#include <atomic>
#include <utility>
#include <string>
#include <unordered_map>
#include "./cached_op.h"



namespace mxnet {
/*! \brief NaiveCachedOp which does not involve engine which is useful when executed in parallel.
    It does not support advanced features of CachedOp, including backward/recording, etc...
 */
class NaiveCachedOp : public CachedOp {
 public:
  NaiveCachedOp(
      const nnvm::Symbol &sym,
      const std::vector<std::pair<std::string, std::string>> &flags) : CachedOp(sym, flags) {}
  virtual ~NaiveCachedOp() {}
  OpStatePtr Forward(
      const std::shared_ptr<CachedOp>& op_ptr,
      const std::vector<NDArray*>& inputs,
      const std::vector<NDArray*>& outputs,
      const Context& default_ctx) override;
  void Backward(
      const bool retain_graph,
      const OpStatePtr& state,
      const std::vector<NDArray*>& inputs,
      const std::vector<OpReqType>& reqs,
      const std::vector<NDArray*>& outputs) override {
          LOG(FATAL) << "Backward is not supported in NaiveCachedOp.";
      }
  // backward storage type inference
  bool BackwardStorageType(
      const nnvm::NodeAttrs& attrs,
      const int dev_mask,
      DispatchMode* dispatch_mode,
      std::vector<int> *in_attrs,
      std::vector<int> *out_attrs) override {
          LOG(FATAL) << "Backward is not supported in NaiveCachedOp.";
          return false;
      }
};  // NaiveCachedOp

using NaiveCachedOpPtr = std::shared_ptr<NaiveCachedOp>;

}  // namespace mxnet
#endif  // MXNET_IMPERATIVE_NAIVE_CACHED_OP_H_
