/*!
 *  Copyright (c) 2017 by Contributors
 * \file autograd.cc
 * \brief (TODO)
 */

#include <mxnet/operator.h>
#include <mxnet/executor.h>
#include <nnvm/pass_functions.h>
#include <unordered_set>
#include <iostream>
#include "../executor/graph_executor.h"
#include "./autograd.h"

namespace mxnet {

// forward declaration
namespace exec {
nnvm::NodeEntry AttrHint(nnvm::NodeEntry src, nnvm::NodeEntry like);
nnvm::NodeEntry AggregateGradient(std::vector<nnvm::NodeEntry>&& v);
}  // namespace exec

namespace ndarray {

using nnvm::Node;
using nnvm::NodePtr;
using nnvm::NodeEntry;
using nnvm::NodeEntryMap;

AutogradRuntime* AutogradRuntime::instance_ = new AutogradRuntime();

AutogradRuntime* AutogradRuntime::Get() {
  return instance_;
}

AutogradRuntime::AutogradRuntime() {}

void AutogradRuntime::SetRecording(bool recording) {
  is_recording_ = recording;
}

bool AutogradRuntime::IsRecording() {
  return is_recording_;
}

void AutogradRuntime::SetMarkForRecord(const std::vector<NDArray*>& arrays, bool mark) {
  for (const NDArray* arr : arrays) {
    bp_flags[arr] = mark;
  }
}

void AutogradRuntime::RecordImperative(const nnvm::Op* op,
                                       const nnvm::NodeAttrs& attrs,
                                       std::vector<NDArray>& inputs,
                                       std::vector<NDArray>& outputs) {
  NodePtr node = Node::Create();
  // (TODO) name of operator
  node->attrs = attrs;
  node->attrs.name = op->name + "_temp";
  for (size_t i = 0; i < inputs.size(); ++i) {
    NodeEntry &e = inputs[i].entry_;
    if (e.node->is_variable() &&
        e.node->attrs.name.empty()) {
      e.node->attrs.name = "variable_" + std::to_string(i);
      entry_ndarray_map_[e] = inputs[i];
    }
    node->inputs.push_back(e);
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    NodeEntry &e = outputs[i].entry_;
    e.node = node;
    e.index = i;
    entry_ndarray_map_[e] = outputs[i];
  }
}

std::vector<NDArray> AutogradRuntime::ComputeGradient(std::vector<NDArray>& inputs,
                                      std::vector<NDArray>& outputs) {

  nnvm::Symbol ff_sym;
  for (size_t i = 0; i < outputs.size(); ++i) {
    ff_sym.outputs.push_back(outputs[i].entry_);
  }
  PrintSymbol(ff_sym, "Forward Graph");

  // nnvm::Graph full_graph = GetBackwardGraph(ff_sym);
  // PrintSymbol(full_graph, "Full Graph");
  // nnvm::Symbol full_sym;
  // full_sym.outputs = full_graph.outputs;
  // (TODO) should pass full_sym
  // std::vector<NDArray> result = Execute(ff_sym, entry_ndarray_map_);
  std::vector<NDArray> result = Execute(ff_sym, entry_ndarray_map_);
  return result;
}

template<typename ValueType>
inline ValueType get_node_attr(
    const nnvm::Node& node,
    const std::string& key, ValueType default_value) {
  auto it = node.attrs.dict.find(key);
  if (it == node.attrs.dict.end()) {
    return default_value;
  } else {
    ValueType ret;
    dmlc::parameter::FieldEntry<ValueType> e;
    e.Init(key, &ret, ret);
    e.Set(&ret, it->second);
    return ret;
  }
}

nnvm::Graph GetBackwardGraph(nnvm::Graph g) {
  Symbol symbol;
  symbol.outputs = g.outputs;

  std::vector<NodeEntry> head_grad_entry;
  static int head_grad_count = 0;
  for (size_t i = 0; i < g.outputs.size(); ++i) {
    NodeEntry ngrad{nnvm::Node::Create(), 0, 0};
    ngrad.node->attrs.name = "head_grad" + std::to_string(head_grad_count++);
    head_grad_entry.emplace_back(exec::AttrHint(ngrad, g.outputs[i]));
  }
  std::vector<NodePtr> args = symbol.ListInputs(nnvm::Symbol::kReadOnlyArgs);
  std::vector<NodeEntry> xs;
  for (size_t i = 0; i < args.size(); ++i) {
    xs.emplace_back(NodeEntry{args[i], 0, 0});
  }

  int do_mirror = dmlc::GetEnv("MXNET_BACKWARD_DO_MIRROR", 0);
  auto need_mirror = [do_mirror](const nnvm::Node& node) -> int {
    if (node.is_variable()) return 0;
    const std::string& type = node.attrs.op->name;
    if (type == "Dropout") return false;
    if (get_node_attr(node, "__force_mirroring__", false)) return true;
    if (do_mirror == 0) return false;
    if (type == "Convolution") return false;
    if (type == "FullyConnected") return false;
    if (type == "Concat") return false;
    if (type == "SoftmaxOutput") return false;
    if (type == "CuDNNBatchNorm") return false;
    return true;
  };
  // take gradient
  nnvm::Graph g_grad = nnvm::pass::Gradient(
      g, symbol.outputs, xs, head_grad_entry,
      exec::AggregateGradient, need_mirror);
  CHECK_EQ(g_grad.outputs.size(), xs.size());

  return g_grad;
}

std::vector<NDArray> AutogradRuntime::Execute(nnvm::Symbol sym,
    nnvm::NodeEntryMap<NDArray> feed_dict) {

  // (TODO) only forward part now
  // const Op* id = nnvm::Op::Get("_identity_with_attr_like_rhs");
  // DFSVisit(sym.outputs, [&](const NodePtr& node) {
  //     if (node->op() == id) {
  //       NDArray output = feed_dict.at(node->inputs[1]);
  //       NDArray grad(output.shape(), output.ctx());
  //       grad = static_cast<real_t>(1.0);
  //       feed_dict.insert({node->inputs[0], grad});
  //     }
  //   });

  std::cout << std::endl;
  LOG(INFO) << "bind";
  Executor *exec = Executor::NewBind(sym, feed_dict);
  LOG(INFO) << "Enter Run";
  exec->Run(feed_dict);
  LOG(INFO) << "Exit Run";
  return exec->grads();
}

}  // namespace ndarray
}  // namespace mxnet
