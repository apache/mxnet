/*!
 *  Copyright (c) 2017 by Contributors
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
  std::shared_ptr<Operator> opr;
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
    opr.reset();
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
  /*! \brief mark variables for computing gradients. */
  void MarkVariables(const std::vector<NDArray*>& variables,
                     const std::vector<mx_uint>& grad_reqs,
                     const std::vector<NDArray*>& gradients);
  /*! \brief record imperative operator which is executed by fcompute. */
  void RecordImperativeFCompute(const nnvm::Op* op,
                                const nnvm::NodeAttrs& attrs,
                                std::vector<NDArray>* p_inputs,
                                std::vector<NDArray>* p_outputs);
  /*! \brief record imperative operator which is executed by operator. */
  void RecordImperativeOperator(const std::shared_ptr<Operator>& opr,
                                const nnvm::Op* op,
                                const nnvm::NodeAttrs& attrs,
                                std::vector<NDArray>* p_inputs,
                                std::vector<NDArray>* p_outputs);
  /*! \brief compute the gradient of outputs w.r.t variables. */
  void ComputeGradient(const std::vector<NDArray>& outputs,
                       const std::vector<NDArray>& ograds,
                       bool retain_graph);
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
  /*! \brief to record operator, return corresponding node. */
  AGNodePtr RecordOp(const nnvm::Op* op,
                     const nnvm::NodeAttrs& attrs,
                     std::vector<NDArray>* p_inputs,
                     std::vector<NDArray>* p_outputs,
                     const std::shared_ptr<Operator>& opr);
  /*! \brief AutogradRuntime singleton. */
  static AutogradRuntime* instance_;
  /*! \brief indicate whether is training. */
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local bool is_train_;
#else
  static MX_THREAD_LOCAL bool is_train_;
#endif
  /*! \brief node count used for naming */
  std::atomic<uint64_t> node_count_{0};
  /*! \brief variable count used for naming */
  std::atomic<uint64_t> variable_count_{0};
};

}  // namespace autograd
}  // namespace mxnet
#endif  // MXNET_NDARRAY_AUTOGRAD_H_
