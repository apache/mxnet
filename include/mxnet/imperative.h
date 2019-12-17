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

#ifndef MXNET_IMPERATIVE_H_
#define MXNET_IMPERATIVE_H_

#include <mxnet/op_attr_types.h>
#include <mxnet/graph_attr_types.h>
#include <mxnet/c_api.h>
#include <nnvm/symbolic.h>
#include <nnvm/op.h>
#include <nnvm/graph.h>
#include <vector>
#include <atomic>
#include <utility>
#include <string>
#include <unordered_map>

#include "./ndarray.h"

namespace mxnet {
  /*! \brief there are three numpy shape flags based on priority.
   * GlobalOn
   *   turn on numpy shape flag globally, it includes thread local.
   *   The flag can be seen in any thread.
   * ThreadLocalOn
   *   only turn on thread local numpy shape flag, it cannot be seen
   *   in other threads.
   * Off
   *   turn off numpy shape flag globally.
   * */
  enum NumpyShape{Off, ThreadLocalOn, GlobalOn};
/*! \brief runtime functions for NDArray */
class Imperative {
 public:
  /*! \brief */
  class AGInfo {
   public:
    Context ctx;
    OpReqType grad_req;
    OpStatePtr state;
    std::vector<NDArray> outputs;
    std::vector<NDArray> out_grads;
    bool fresh_out_grad;

    AGInfo() :
      grad_req(kNullOp), fresh_out_grad(false) {}

    static void Clear(const nnvm::NodePtr& node) {
      if (node == nullptr || node->info.empty()) return;
      AGInfo& info = Get(node);
      if (info.grad_req != kNullOp) return;
      node->info.clear();
    }

    static AGInfo& Get(const nnvm::NodePtr& node) {
      return dmlc::get<AGInfo>(node->info);
    }

    static AGInfo& Create(const nnvm::NodePtr& node) {
      node->info.construct<AGInfo>();
      return Get(node);
    }

    static bool IsNone(const NDArray& arr) {
      return arr.entry_.node == nullptr || arr.entry_.node->info.empty();
    }

    static bool IsVariable(const nnvm::NodePtr& node) {
      AGInfo& info = Get(node);
      return info.grad_req != kNullOp && info.outputs.size() == 1
             && info.out_grads.size() == 1;
    }
  };
  /*! \brief whether operator recording is on. */
  bool is_training() const {
    return is_train_;
  }
  /*! \brief turn on or turn off operator recording for autograd. */
  bool set_is_training(bool is_train) {
      bool old = is_train_;
      is_train_ = is_train;
      return old;
  }
  /*! \brief whether operator recording is on. */
  bool is_recording() const {
    return is_recording_;
  }
  /*! \brief turn on or turn off operator recording for autograd. */
  bool set_is_recording(bool is_recording) {
      bool old = is_recording_;
      is_recording_ = is_recording;
      return old;
  }
  /*! \brief return current numpy compatibility status,
   *  GlobalOn(2), ThreadLocalOn(1), Off(0).
   * */
  int is_np_shape() const {
    if (is_np_shape_global_) {
      return 2;
    }
    return is_np_shape_thread_local_ ? 1 : 0;
  }
  /*! \brief specify numpy compatibility off, thread local on or global on. */
  bool set_is_np_shape(int is_np_shape) {
    NumpyShape flag = static_cast<NumpyShape>(is_np_shape);
    bool old = this->is_np_shape();
    switch (flag) {
      case GlobalOn:
        is_np_shape_global_ = true;
        is_np_shape_thread_local_ = true;
        break;
      case ThreadLocalOn:
        is_np_shape_thread_local_ = true;
        break;
      case Off:
        is_np_shape_global_ = false;
        is_np_shape_thread_local_ = false;
        break;
    }
    return old;
  }
  /*! \brief to record operator, return corresponding node. */
  void RecordOp(nnvm::NodeAttrs&& attrs,
                const std::vector<NDArray*>& inputs,
                const std::vector<NDArray*>& outputs,
                const OpStatePtr& state = OpStatePtr(),
                std::vector<bool>* p_save_inputs = nullptr,
                std::vector<bool>* p_save_outputs = nullptr);
  /*! \brief */
  OpStatePtr Invoke(const Context& default_ctx,
                    const nnvm::NodeAttrs& attrs,
                    const std::vector<NDArray*>& inputs,
                    const std::vector<NDArray*>& outputs);
  /*! \brief */
  OpStatePtr InvokeOp(const Context& ctx,
                      const nnvm::NodeAttrs& attrs,
                      const std::vector<NDArray*>& inputs,
                      const std::vector<NDArray*>& outputs,
                      const std::vector<OpReqType>& req,
                      const DispatchMode dispatch_mode,
                      OpStatePtr state = OpStatePtr());
  /*! \brief mark variables for computing gradients. */
  void MarkVariables(const std::vector<NDArray*>& variables,
                     const std::vector<uint32_t>& grad_reqs,
                     const std::vector<NDArray*>& gradients);
  /*! \brief compute the gradient of outputs w.r.t variables. */
  std::vector<NDArray*> Backward(const std::vector<NDArray*>& outputs,
                                 const std::vector<NDArray*>& ograds,
                                 const std::vector<NDArray*>& variables,
                                 bool is_train, bool retain_graph,
                                 bool create_graph);
  /*! \return AutogradRuntime singleton */
  static Imperative* Get();
  /*! \brief Should op execution bulking be employed during inference. */
  static bool PreferBulkExecInference() {
    return dmlc::GetEnv("MXNET_EXEC_BULK_EXEC_INFERENCE", true);
  }
  /*! \brief Should op execution bulking be employed during training. */
  static bool PreferBulkExecTrain() {
    return dmlc::GetEnv("MXNET_EXEC_BULK_EXEC_TRAIN", true);
  }
  /*! \brief The max number of op nodes in a bulk during forward pass of training. */
  static int BulkExecMaxNodeTrainFwd() {
    return dmlc::GetEnv("MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD",
                        dmlc::GetEnv("MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN", 15));
  }
  /*! \brief The max number of op nodes in a bulk during backward pass of training. */
  static int BulkExecMaxNodeTrainBwd() {
    return dmlc::GetEnv("MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD",
                        dmlc::GetEnv("MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN", 15));
  }

 private:
  friend class NDArray;
  /*! \brief make constructor protected. */
  Imperative() {
    if (PreferBulkExecTrain())
      backward_bulk_size_ = BulkExecMaxNodeTrainBwd();
  }
  /*! \brief find the input/output ndarrays that are needed for backward */
  void GetBackwardDependency(
      const nnvm::NodePtr& node,
      uint32_t num_inputs, uint32_t num_outputs,
      std::vector<bool> *p_save_inputs,
      std::vector<bool> *p_save_outputs);
  /*! \brief indicate whether is training. */
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local bool is_train_;
  static thread_local bool is_recording_;
  // TOOD(junwu): Added numpy compatibility switch for backward compatibility.
  // Delete it in the next major release.
  static thread_local bool is_np_shape_thread_local_;
#else
  static MX_THREAD_LOCAL bool is_train_;
  static MX_THREAD_LOCAL bool is_recording_;
  // TOOD(junwu): Added numpy compatibility switch for backward compatibility.
  // Delete it in the next major release.
  static MX_THREAD_LOCAL bool is_np_shape_thread_local_;
#endif
  bool is_np_shape_global_{false};
  /*! \brief node count used for naming */
  std::atomic<uint64_t> node_count_{0};
  /*! \brief variable count used for naming */
  std::atomic<uint64_t> variable_count_{0};
  /*! \brief default backward bulk size */
  int backward_bulk_size_{0};
};

}  // namespace mxnet
#endif  // MXNET_IMPERATIVE_H_
