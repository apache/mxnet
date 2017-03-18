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
#include <nnvm/symbolic.h>
#include <nnvm/op.h>
#include <nnvm/graph.h>
#include <vector>
#include <unordered_map>

namespace mxnet {
namespace autograd {

/*!
 * \brief AutogradRuntime Interface
 */
class AutogradRuntime {
 public:
  /*! \brief turn on or turn off operator recording for autograd. */
  void SetRecording(bool recording);
  /*! \brief whether operator recording is on. */
  bool IsRecording() const;
  /*! \brief mark variables for computing gradients. */
  void MarkVariables(std::vector<NDArray*>* p_variables);
  /*! \brief record imperative operator which is executed by fcompute. */
  void RecordImperativeFCompute(FCompute fn,
                                const nnvm::Op* op,
                                const nnvm::NodeAttrs& attrs,
                                std::vector<NDArray>* p_inputs,
                                std::vector<NDArray>* p_outputs);
  /*! \brief record imperative operator which is executed by operator. */
  void RecordImperativeOperator(std::shared_ptr<Operator> opr,
                                const nnvm::Op* op,
                                const nnvm::NodeAttrs& attrs,
                                std::vector<NDArray>* p_inputs,
                                std::vector<NDArray>* p_outputs);
  /*! \brief compute the gradient of outputs w.r.t variables. */
  std::vector<NDArray> ComputeGradient(const std::vector<NDArray>& outputs);
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
  nnvm::NodePtr RecordOp(const nnvm::Op* op,
                         const nnvm::NodeAttrs& attrs,
                         std::vector<NDArray>* p_inputs,
                         std::vector<NDArray>* p_outputs);
  /*! \brief clear the record data. */
  void ClearRecords();
  /*! \brief AutogradRuntime singleton. */
  static AutogradRuntime* instance_;
  /*! \brief indicate whether operator recording is on. */
  bool is_recording_{false};
  /*! \brief node count used for naming */
  int node_count_{0};
  /*! \brief variable count used for naming */
  int variable_count_{0};
  /*! \brief mapping from node entry to saved ndarray. */
  nnvm::NodeEntryMap<NDArray> saved_ndarray_;
  /*! \brief mapping from node to saved operator. */
  std::unordered_map<const nnvm::Node*, std::shared_ptr<Operator>> saved_opr_;
};

}  // namespace autograd
}  // namespace mxnet
#endif  // MXNET_NDARRAY_AUTOGRAD_H_
