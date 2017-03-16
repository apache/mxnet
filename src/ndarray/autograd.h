/*!
 *  Copyright (c) 2017 by Contributors
 * \file autograd.h
 * \brief (TODO)
 */
#ifndef MXNET_NDARRAY_AUTOGRAD_H_
#define MXNET_NDARRAY_AUTOGRAD_H_

#include <dmlc/logging.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <nnvm/symbolic.h>
#include <nnvm/op.h>
#include <nnvm/graph.h>
#include <vector>
#include <unordered_map>

namespace mxnet {
namespace ndarray {

class AutogradRuntime {
 public:
  void SetRecording(bool recording);
  bool IsRecording();
  void SetMarkForRecord(const std::vector<NDArray*>& arrays, bool mark);
  void RecordImperative(const nnvm::Op* op,
                const nnvm::NodeAttrs& attrs,
                std::vector<NDArray>& inputs,
                std::vector<NDArray>& outputs);
  std::vector<NDArray> ComputeGradient(std::vector<NDArray>& inputs,
                       std::vector<NDArray>& grad_outputs);
  static AutogradRuntime* Get();

 protected:
  AutogradRuntime();

 private:
  std::vector<NDArray> Execute(nnvm::Symbol sym,
      nnvm::NodeEntryMap<NDArray> feed_dict);

  static AutogradRuntime* instance_;
  std::unordered_map<const NDArray*, bool> bp_flags;
  bool is_recording_{false};
  nnvm::NodeEntryMap<NDArray> entry_ndarray_map_;
  nnvm::NodeEntryMap<nnvm::NodePtr> new_input_variables_;
  nnvm::NodeEntryMap<TShape> entry_shape_map_;
  std::unordered_map<const nnvm::Node*, nnvm::NodeEntry> var_entry_map_;
  // Symbol graph;
};

nnvm::Graph GetBackwardGraph(nnvm::Graph g);

// I will delete this after finish
inline void PrintSymbol(nnvm::Symbol s, std::string name = "") {
  std::cout << "\n\n";
  LOG(INFO) << name;
  s.Print(std::cout);
  std::cout << "\n\n";
}

inline void PrintSymbol(nnvm::Graph g, std::string name = "") {
  nnvm::Symbol s;
  s.outputs = g.outputs;
  PrintSymbol(s, name);
}

}  // namespace ndarray
}  // namespace mxnet
#endif  // MXNET_NDARRAY_AUTOGRAD_H_
