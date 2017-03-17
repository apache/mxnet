/*!
 *  Copyright (c) 2017 by Contributors
 * \file autograd.cc
 * \brief Implementation of AutogradRuntime module.
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


namespace autograd {

using nnvm::Symbol;
using nnvm::Node;
using nnvm::NodePtr;
using nnvm::NodeEntry;
using nnvm::NodeEntryMap;
using exec::GraphExecutor;

AutogradRuntime::AutogradRuntime() {}

void AutogradRuntime::SetRecording(bool recording) {
  if (recording) ClearRecords();
  is_recording_ = recording;
}

bool AutogradRuntime::IsRecording() const {
  return is_recording_;
}

void AutogradRuntime::SetMarkForRecord(const std::vector<NDArray*>& arrays, bool mark) {
  // (TODO)
}

void AutogradRuntime::RecordImperativeFCompute(FCompute fn,
                                               const nnvm::Op* op,
                                               const nnvm::NodeAttrs& attrs,
                                               std::vector<NDArray>& inputs,
                                               std::vector<NDArray>& outputs) {
  RecordOp(op, attrs, inputs, outputs);
}

void AutogradRuntime::RecordImperativeOperator(std::shared_ptr<Operator> opr,
                                               const nnvm::Op* op,
                                               const nnvm::NodeAttrs& attrs,
                                               std::vector<NDArray>& inputs,
                                               std::vector<NDArray>& outputs) {
  NodePtr node = RecordOp(op, attrs, inputs, outputs);
  saved_opr_.insert({node.get(), opr});
}

std::vector<NDArray> Execute(Symbol sym,
                             const NodeEntryMap<NDArray>& feed_dict,
                             const NodeOperatorMap& saved_opr);

std::vector<NDArray> AutogradRuntime::ComputeGradient(std::vector<NDArray>& inputs,
                                                      std::vector<NDArray>& outputs) {

  Symbol ff_sym;
  for (size_t i = 0; i < outputs.size(); ++i) {
    ff_sym.outputs.push_back(outputs[i].entry_);
  }
  // TODO(ziheng) should pass full_sym in the future
  // std::vector<NDArray> result = Execute(ff_sym, saved_ndarray_);
  std::vector<NDArray> result = Execute(ff_sym, saved_ndarray_, saved_opr_);
  return result;
}

void AutogradRuntime::ClearRecords() {
  saved_ndarray_.clear();
  saved_opr_.clear();
}

std::shared_ptr<AutogradRuntime> AutogradRuntime::_GetSharedRef() {
  static std::shared_ptr<AutogradRuntime> inst(new AutogradRuntime());
  return inst;
}

AutogradRuntime* AutogradRuntime::Get() {
  static AutogradRuntime *ptr = _GetSharedRef().get();
  return ptr;
}

NodePtr AutogradRuntime::RecordOp(const nnvm::Op* op,
                                  const nnvm::NodeAttrs& attrs,
                                  std::vector<NDArray>& inputs,
                                  std::vector<NDArray>& outputs) {
  NodePtr node = Node::Create();
  // (TODO) name of operator
  node->attrs = attrs;
  node->attrs.name = op->name + "_temp";
  for (size_t i = 0; i < inputs.size(); ++i) {
    NodeEntry &e = inputs[i].entry_;
    if (e.node->is_variable() && e.node->attrs.name.empty()) {
      e.node->attrs.name = "variable_" + std::to_string(i);
      saved_ndarray_[e] = inputs[i];
    }
    node->inputs.push_back(e);
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    NodeEntry &e = outputs[i].entry_;
    e.node = node;
    e.index = i;
    saved_ndarray_[e] = outputs[i];
  }
  return node;
}

// (TODO) should remove saved_opr from arguments
GraphExecutor *NewBind(Symbol symbol,
                       const NodeEntryMap<TShape>& shapes,
                       const NodeOperatorMap& saved_opr) {
  std::vector<NodePtr> input_nodes =
    symbol.ListInputs(Symbol::ListInputOption::kAll);
  std::vector<NDArray> inputs;

  // default context (TODO) fixme
  Context ctx = Context::CPU();
  std::map<std::string, Context> ctx_map;

  // prepare inputs
  inputs.reserve(input_nodes.size());
  for (const NodePtr& n : input_nodes) {
    NodeEntry e = NodeEntry{n, 0, 0};
    if (shapes.count(e)) {
      NDArray nd(shapes.at(e), ctx);
      inputs.push_back(nd);
    } else {
      LOG(FATAL) << "no corresponding ndarray: " << n->attrs.name;
    }
  }

  // grads for every inputs
  std::vector<NDArray> grads;
  std::vector<OpReqType> grad_reqs;
  for (size_t i = 0; i < inputs.size(); ++i) {
    NDArray grad(inputs[i].shape(), inputs[i].ctx());
    grad = static_cast<real_t>(1.0);
    grads.push_back(grad);
    grad_reqs.push_back(OpReqType::kWriteTo);
  }

  // empty aux_states
  std::vector<NDArray> aux_states;

  auto exec = new exec::GraphExecutor();
  // (TODO) too hack
  exec->shape_hints_ = shapes;
  exec->saved_opr_ = saved_opr;
  exec->Init(symbol, ctx, ctx_map,
             inputs, grads, grad_reqs, aux_states);

  return exec;
}

std::vector<NDArray> Run(GraphExecutor* exec,
                         const NodeEntryMap<NDArray>& feed_dict) {
  const nnvm::IndexedGraph& idx = exec->graph_.indexed_graph();

  for (const auto& kv : feed_dict) {
    if (idx.exist(kv.first.node.get())) {
      uint32_t entry_id = idx.entry_id(kv.first);
      CopyFromTo(kv.second, &(exec->data_entry_[entry_id]));
    }
  }

  std::vector<NDArray> head_grads;
  head_grads.reserve(exec->head_grad_array_.size());

  for (size_t i = 0; i < exec->output_arrays_.size(); ++i) {
    NDArray grad(exec->output_arrays_[i].shape(), exec->output_arrays_[i].ctx());
    grad = static_cast<real_t>(1.0);
    head_grads.push_back(grad);
  }

  exec->Backward(head_grads);

  std::vector<NDArray> results;
  results.reserve(exec->grad_store_.size());
  for (const auto& kv : exec->grad_store_) {
    results.emplace_back(kv.second);
  }
  return results;
}

std::vector<NDArray> Execute(Symbol sym,
                             const NodeEntryMap<NDArray>& feed_dict,
                             const NodeOperatorMap& saved_opr) {
  NodeEntryMap<TShape> shapes;
  for (const auto& kv : feed_dict) {
    shapes.insert({kv.first, kv.second.shape()});
  }
  exec::GraphExecutor *exec = NewBind(sym, shapes, saved_opr);
  return Run(exec, feed_dict);
}

}  // namespace autograd
}  // namespace mxnet
