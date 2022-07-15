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

constexpr char OPT_CONSTRAINT_ATTR[] = "__opt_constraint__";
enum class OptConstraint : unsigned int {
  None       = 0,
  DisableAMP = 1 << 0
  // DisableQuantization = 1 << 1
};
using OptConstraint_int_t = std::underlying_type_t<OptConstraint>;

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
enum NumpyShape { Off, ThreadLocalOn, GlobalOn };
typedef NumpyShape NumpyDefaultDtype;

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
    std::vector<NDArray> out_grads;  // used to hold gradient arrays the user is
                                     // interested in (marked variables)
    bool fresh_out_grad;

    AGInfo() : grad_req(kNullOp), fresh_out_grad(false) {}

    static void Clear(const nnvm::ObjectPtr& node) {
      if (node == nullptr || node->info.empty())
        return;
      AGInfo& info = Get(node);
      if (info.grad_req != kNullOp)
        return;
      node->info.clear();
    }

    static AGInfo& Get(const nnvm::ObjectPtr& node) {
      return dmlc::get<AGInfo>(node->info);
    }

    static AGInfo& Create(const nnvm::ObjectPtr& node) {
      node->info.construct<AGInfo>();
      return Get(node);
    }

    static bool IsNone(const NDArray& arr) {
      return arr.autograd_entry_.node == nullptr || arr.autograd_entry_.node->info.empty();
    }

    static bool IsVariable(const nnvm::ObjectPtr& node) {
      AGInfo& info = Get(node);
      return info.grad_req != kNullOp && info.outputs.size() == 1 && info.out_grads.size() == 1;
    }
  };

  /*! \brief DCInfo datastructure to enable deferred computation */
  class DCInfo {
   public:
    explicit DCInfo(const std::vector<NDArray*>& inputs, const std::vector<NDArray*>& outputs);

    /*! \brief Compute the outputs of the associated operator. */
    static void Compute(const NDArray& arr);

    static DCInfo& Get(const nnvm::ObjectPtr& node) {
      return dmlc::get<DCInfo>(node->info);
    }

    static bool IsNone(const NDArray& arr) {
      return arr.deferredcompute_entry_.node == nullptr ||
             arr.deferredcompute_entry_.node->info.empty();
    }

    static bool IsComputed(const NDArray& arr) {
      return IsNone(arr) || dmlc::get<DCInfo>(arr.deferredcompute_entry_.node->info).is_computed_;
    }

    static DCInfo& Create(const nnvm::ObjectPtr& node,
                          const std::vector<NDArray*>& inputs,
                          const std::vector<NDArray*>& outputs);

    static void Clear(const nnvm::ObjectPtr& node) {
      if (node == nullptr || node->info.empty())
        return;
      node->info.clear();
    }

   private:
    friend class Imperative;

    /*! \brief Copies of input NDArrays
     *
     * If respective input NDArray is deallocated on the frontend, we still need
     * to keep a copy around to facilitate deferred computation of this array.
     * The copies share the chunk.
     *
     * They are automatically deallocated after computation finished.
     */
    std::vector<NDArray> inputs_;

    /*! \brief Handles of input NDArrays used by frontend
     *
     * Frontend may request conversion to Symbol, specifying a list of NDArray
     * handles corresponding to inputs and outputs of the Symbol. We store the
     * handles used by frontend to facilitate matching in
     * GetDeferredComputeSymbol.
     *
     * Note that the frontend may have deallocated the NDArray* and the
     * input_handles stored here may point to invalid memory.
     */
    std::vector<const NDArray*> input_handles_;

    /*! \brief Copies of output NDArrays
     *
     * If respective output NDArray is deallocated on the frontend, we still
     * need to keep a copy around to facilitate deferred computation of arrays
     * relying on the output array. The copies share the chunk.
     *
     * They are automatically deallocated after computation finished.
     */
    std::vector<NDArray> outputs_;

    /*! \brief Remember if the outputs associated with this DCInfo have been computed already */
    bool is_computed_ = false;
  };

  /*! \brief whether operator recording is on. */
  bool is_training() const {
    return is_train_;
  }
  /*! \brief turn on or turn off operator recording for autograd. */
  bool set_is_training(bool is_train) {
    bool old  = is_train_;
    is_train_ = is_train;
    return old;
  }
  /*! \brief whether operator recording is on. */
  bool is_recording() const {
    return is_recording_;
  }
  /*! \brief turn on or turn off operator recording for autograd. */
  bool set_is_recording(bool is_recording) {
    bool old      = is_recording_;
    is_recording_ = is_recording;
    return old;
  }
  /*! \brief whether deferred compute mode is on. */
  bool is_deferred_compute() const {
    return is_deferred_compute_;
  }
  /*! \brief turn on or turn off operator recording for autograd. */
  bool set_is_deferred_compute(bool is_deferred_compute) {
    bool old             = is_deferred_compute_;
    is_deferred_compute_ = is_deferred_compute;
    return old;
  }
  /*! \brief return current numpy compatibility status,
   *  GlobalOn(2), ThreadLocalOn(1), Off(0).
   * */
  int is_np_shape() const {
    if (is_np_shape_global_) {
      return NumpyShape::GlobalOn;
    }
    return is_np_shape_thread_local_ ? NumpyShape::ThreadLocalOn : NumpyShape::Off;
  }
  /*! \brief specify numpy compatibility off, thread local on or global on. */
  bool set_is_np_shape(int is_np_shape) {
    NumpyShape flag = static_cast<NumpyShape>(is_np_shape);
    bool old        = this->is_np_shape();
    switch (flag) {
      case GlobalOn:
        is_np_shape_global_       = true;
        is_np_shape_thread_local_ = true;
        break;
      case ThreadLocalOn:
        is_np_shape_thread_local_ = true;
        break;
      case Off:
        is_np_shape_global_       = false;
        is_np_shape_thread_local_ = false;
        break;
    }
    return old;
  }
  /*! \brief return current numpy default dtype compatibility status.
   * */
  bool is_np_default_dtype() const {
    if (is_np_default_dtype_global_) {
      return true;
    }
    return false;
  }
  /*! \brief specify numpy default dtype off or global on. */
  bool set_is_np_default_dtype(bool is_np_default_dtype) {
    bool old = this->is_np_default_dtype();
    if (is_np_default_dtype) {
      is_np_default_dtype_global_ = true;
    } else {
      is_np_default_dtype_global_ = false;
    }
    return old;
  }
  /*! \brief return current optimization constraints. */
  OptConstraint get_opt_constraints() const {
    return opt_constraints_;
  }
  /*! \brief set optimization constraints. */
  OptConstraint set_opt_constraints(OptConstraint constraints) {
    OptConstraint old = opt_constraints_;
    opt_constraints_  = constraints;
    return old;
  }
  /*! \brief to record operator, return corresponding node. */
  void RecordOp(nnvm::NodeAttrs&& attrs,
                const std::vector<NDArray*>& inputs,
                const std::vector<NDArray*>& outputs,
                const OpStatePtr& state           = OpStatePtr(),
                std::vector<bool>* p_save_inputs  = nullptr,
                std::vector<bool>* p_save_outputs = nullptr);
  /*! \brief to record operator, return corresponding node. */
  void RecordDeferredCompute(nnvm::NodeAttrs&& attrs,
                             const std::vector<NDArray*>& inputs,
                             const std::vector<NDArray*>& outputs);
  /*! \brief obtain symbol representation of deferred compute session. */
  nnvm::Symbol GetDeferredComputeSymbol(const std::vector<NDArray*>& outputs);
  /*! \brief associate arrays with variables for deferred compute */
  void SetDeferredComputeVariable(NDArrayHandle* arrays, SymbolHandle* variables, const int num);
  /*! \brief clear info node associated with array */
  void DeferredComputeClear(NDArrayHandle* arrays, const int num);
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
  /*! \brief unmark nonleaf variables to free the memory. */
  void DropGrads(const std::vector<NDArray*>& variables);
  /*! \brief compute the gradient of outputs w.r.t variables. */
  std::vector<NDArray*> Backward(const std::vector<NDArray*>& outputs,
                                 const std::vector<NDArray*>& ograds,
                                 const std::vector<NDArray*>& variables,
                                 bool is_train,
                                 bool retain_graph,
                                 bool create_graph);
  /*! \brief Return the marked nonleaf nodes. */
  std::vector<nnvm::ObjectPtr> ListNonleafVariables(const nnvm::Symbol& sym) const;
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
  void GetBackwardDependency(const nnvm::ObjectPtr& node,
                             uint32_t num_inputs,
                             uint32_t num_outputs,
                             std::vector<bool>* p_save_inputs,
                             std::vector<bool>* p_save_outputs);
  /*! \brief indicate whether is training. */
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local bool is_train_;
  static thread_local bool is_recording_;
  static thread_local bool is_deferred_compute_;
  static thread_local OptConstraint opt_constraints_;
  // TOOD(junwu): Added numpy compatibility switch for backward compatibility.
  // Delete it in the next major release.
  static thread_local bool is_np_shape_thread_local_;
#else
  static MX_THREAD_LOCAL bool is_train_;
  static MX_THREAD_LOCAL bool is_recording_;
  static MX_THREAD_LOCAL bool is_deferred_compute_;
  static MX_THREAD_LOCAL OptConstraint opt_constraints_;
  // TOOD(junwu): Added numpy compatibility switch for backward compatibility.
  // Delete it in the next major release.
  static MX_THREAD_LOCAL bool is_np_shape_thread_local_;
#endif
  bool is_np_shape_global_{false};
  bool is_np_default_dtype_global_{false};
  /*! \brief node count used for naming */
  std::atomic<uint64_t> node_count_{0};
  /*! \brief variable count used for naming */
  std::atomic<uint64_t> variable_count_{0};
  /*! \brief default backward bulk size */
  int backward_bulk_size_{0};
};

}  // namespace mxnet
#endif  // MXNET_IMPERATIVE_H_
