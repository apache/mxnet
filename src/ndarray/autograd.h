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
 * \file autograd.h
 * \brief AutogradRuntime can automatically compute gradients
 */
#ifndef MXNET_NDARRAY_AUTOGRAD_H_
#define MXNET_NDARRAY_AUTOGRAD_H_

#include <dmlc/logging.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/op_attr_types.h>
#include <mxnet/c_api.h>
#include <nnvm/symbolic.h>
#include <nnvm/op.h>
#include <nnvm/graph.h>
#include <vector>
#include <atomic>
#include <unordered_map>

namespace mxnet {
namespace autograd {

class AGNode {
 public:
  OpReqType grad_req;
  nnvm::NodePtr nn_node;
  OpStatePtr state;
  std::vector<AGNodeEntry> inputs;
  std::vector<NDArray> outputs;
  std::vector<NDArray> out_grads;
  bool fresh_out_grad;

  explicit AGNode(const nnvm::NodePtr& nn_node_) :
    grad_req(kNullOp), nn_node(nn_node_), fresh_out_grad(false) {}

  static AGNodePtr Create(const nnvm::NodePtr& nn_node_) {
    return std::make_shared<AGNode>(nn_node_);
  }

  void clear_history() {
    if (out_grads.size()) return;
    state.reset();
    outputs.clear();
    nn_node.reset();
    for (auto& i : inputs) i.ag_node->clear_history();
    inputs.clear();
  }
};

/*!
 * \brief AutogradRuntime Interface
 */
class AutogradRuntime {
 public:
  /*! \brief turn on or turn off operator recording for autograd. */
  bool SetIsTraining(bool is_train) {
      bool old = is_train_;
      is_train_ = is_train;
      return old;
  }
  /*! \brief whether operator recording is on. */
  bool IsTraining() const {
    return is_train_;
  }
  /*! \brief turn on or turn off operator recording for autograd. */
  bool SetIsRecording(bool is_recording) {
      bool old = is_recording_;
      is_recording_ = is_recording;
      return old;
  }
  /*! \brief whether operator recording is on. */
  bool IsRecording() const {
    return is_recording_;
  }
  /*! \brief mark variables for computing gradients. */
  void MarkVariables(const std::vector<NDArray*>& variables,
                     const std::vector<mx_uint>& grad_reqs,
                     const std::vector<NDArray*>& gradients);
  /*! \brief find the input/output ndarrays that are needed for backward */
  void GetBackwardDependency(
      const nnvm::NodePtr& node,
      uint32_t num_inputs, uint32_t num_outputs,
      std::vector<bool> *p_save_inputs,
      std::vector<bool> *p_save_outputs);
  /*! \brief to record operator, return corresponding node. */
  void RecordOp(nnvm::NodeAttrs&& attrs,
                std::vector<NDArray>* p_inputs,
                std::vector<NDArray>* p_outputs,
                const OpStatePtr& state = OpStatePtr(),
                std::vector<bool>* p_save_inputs = nullptr,
                std::vector<bool>* p_save_outputs = nullptr);
  /*! \brief compute the gradient of outputs w.r.t variables. */
  void ComputeGradient(const std::vector<NDArray>& outputs,
                       const std::vector<NDArray>& ograds,
                       bool retain_graph, bool is_train);
  /*! \return AutogradRuntime singleton */
  static AutogradRuntime* Get();
  /*! \brief Get shared pointer reference to AutogradRuntime singleton.
   *   Most user should not call this function.
   *   This function is called by another singleton X who requires
   *   AutogradRuntime to be destructed after X.
   *
   *  \return A shared pointer to AutogradRuntime singleton.
   */
  static std::shared_ptr<AutogradRuntime> _GetSharedRef();

 protected:
  /*! \brief make constructor protected. */
  AutogradRuntime();

 private:
  /*! \brief AutogradRuntime singleton. */
  static AutogradRuntime* instance_;
  /*! \brief indicate whether is training. */
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local bool is_train_;
  static thread_local bool is_recording_;
#else
  static MX_THREAD_LOCAL bool is_train_;
  static MX_THREAD_LOCAL bool is_recording_;
#endif
  /*! \brief node count used for naming */
  std::atomic<uint64_t> node_count_{0};
  /*! \brief variable count used for naming */
  std::atomic<uint64_t> variable_count_{0};
};

}  // namespace autograd
}  // namespace mxnet
#endif  // MXNET_NDARRAY_AUTOGRAD_H_
