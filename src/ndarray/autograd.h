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

class AutogradRuntime {
 public:
  void SetRecording(bool recording);
  bool IsRecording() const;
  void SetMarkForRecord(const std::vector<NDArray*>& arrays, bool mark);
  void RecordImperativeFCompute(FCompute fn,
                                const nnvm::Op* op,
                                const nnvm::NodeAttrs& attrs,
                                std::vector<NDArray>& inputs,
                                std::vector<NDArray>& outputs);
  void RecordImperativeOperator(std::shared_ptr<Operator> opr,
                                const nnvm::Op* op,
                                const nnvm::NodeAttrs& attrs,
                                std::vector<NDArray>& inputs,
                                std::vector<NDArray>& outputs);
  std::vector<NDArray> ComputeGradient(std::vector<NDArray>& inputs,
                                       std::vector<NDArray>& grad_outputs);
  static AutogradRuntime* Get();
  static std::shared_ptr<AutogradRuntime> _GetSharedRef();

 protected:
  AutogradRuntime();

 private:
  void ClearRecords();
  nnvm::NodePtr RecordOp(const nnvm::Op* op,
                         const nnvm::NodeAttrs& attrs,
                         std::vector<NDArray>& inputs,
                         std::vector<NDArray>& outputs);
  static AutogradRuntime* instance_;
  bool is_recording_{false};
  nnvm::NodeEntryMap<NDArray> saved_ndarray_;
  std::unordered_map<const nnvm::Node*, std::shared_ptr<Operator>> saved_opr_;
};

}  // namespace autograd
}  // namespace mxnet
#endif  // MXNET_NDARRAY_AUTOGRAD_H_
