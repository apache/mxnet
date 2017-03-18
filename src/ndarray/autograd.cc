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
namespace autograd {

using nnvm::Symbol;
using nnvm::Node;
using nnvm::NodePtr;
using nnvm::NodeEntry;
using nnvm::NodeEntryMap;
using exec::GraphExecutor;

AutogradRuntime::AutogradRuntime() {}

void AutogradRuntime::SetRecording(bool recording) {
  is_recording_ = recording;
}

bool AutogradRuntime::IsRecording() const {
  return is_recording_;
}

void AutogradRuntime::MarkVariables(std::vector<NDArray*> *p_variables) {
  std::vector<NDArray*>& variables = *p_variables;
  for (NDArray* var : variables) {
    NodeEntry& e = var->entry_;
    e.node = Node::Create();
    e.node->attrs.name = "ag_variables_" + std::to_string(variable_count_++);
  }
}

void AutogradRuntime::RecordImperativeFCompute(FCompute fn,
                                               const nnvm::Op* op,
                                               const nnvm::NodeAttrs& attrs,
                                               std::vector<NDArray> *p_inputs,
                                               std::vector<NDArray> *p_outputs) {
  RecordOp(op, attrs, p_inputs, p_outputs);
}

void AutogradRuntime::RecordImperativeOperator(std::shared_ptr<Operator> opr,
                                               const nnvm::Op* op,
                                               const nnvm::NodeAttrs& attrs,
                                               std::vector<NDArray> *p_inputs,
                                               std::vector<NDArray> *p_outputs) {
  NodePtr node = RecordOp(op, attrs, p_inputs, p_outputs);
  saved_opr_.insert({node.get(), opr});
}

std::vector<NDArray> Execute(Symbol sym,
                             const NodeEntryMap<NDArray>& feed_dict,
                             const NodeOperatorMap& saved_opr);

std::vector<NDArray> AutogradRuntime::ComputeGradient(const std::vector<NDArray>& outputs) {
  Symbol ff_sym;
  for (size_t i = 0; i < outputs.size(); ++i) {
    ff_sym.outputs.push_back(outputs[i].entry_);
  }
  std::vector<NDArray> result = Execute(ff_sym, saved_ndarray_, saved_opr_);
  ClearRecords();
  return result;
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
                                  std::vector<NDArray> *p_inputs,
                                  std::vector<NDArray> *p_outputs) {
  std::vector<NDArray>& inputs  = *p_inputs;
  std::vector<NDArray>& outputs = *p_outputs;

  NodePtr node = Node::Create();
  node->attrs = attrs;
  node->attrs.name = "ag_" + op->name + "_" + std::to_string(node_count_++);

  for (size_t i = 0; i < outputs.size(); ++i) {
    NodeEntry &e = outputs[i].entry_;
    e.node = node;
    e.index = i;
    saved_ndarray_[e] = outputs[i];
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
    NodeEntry &e = inputs[i].entry_;
    CHECK(e.node.get() != nullptr)
      << "not support partial gradient yet, all the "
      << "inputs of autograd should be marked as variable";
    if (!saved_ndarray_.count(e)) {
      saved_ndarray_[e] = inputs[i];
    }
    node->inputs.push_back(e);
  }
  return node;
}

void AutogradRuntime::ClearRecords() {
  node_count_ = 0;
  variable_count_ = 0;
  saved_ndarray_.clear();
  saved_opr_.clear();
}

GraphExecutor *Bind(Symbol symbol,
                    const NodeEntryMap<TShape>&  shapes,
                    const NodeEntryMap<Context>& ctxs,
                    const NodeOperatorMap& saved_opr) {
  std::vector<NodePtr> input_nodes =
    symbol.ListInputs(Symbol::ListInputOption::kAll);

  size_t input_size = input_nodes.size();
  std::vector<NDArray> inputs;
  inputs.reserve(input_size);
  std::vector<NDArray> grads;
  grads.reserve(input_size);
  std::vector<OpReqType> grad_reqs;
  grad_reqs.reserve(input_size);

  // prepare inputs and set grad for every input
  for (size_t i = 0; i < input_size; ++i) {
    NodeEntry e = NodeEntry{input_nodes[i], 0, 0};
    if (shapes.count(e) && ctxs.count(e)) {
      TShape  shape = shapes.at(e);
      Context ctx   = ctxs.at(e);
      inputs.emplace_back(shape, ctx);
      NDArray grad(shape, ctx);
      grad = static_cast<real_t>(1.0);
      grads.emplace_back(grad);
      grad_reqs.emplace_back(OpReqType::kWriteTo);
    } else {
      LOG(FATAL) << "no corresponding ndarray: "
                 << input_nodes[i]->attrs.name << "(0)";
    }
  }

  // default context, assuming use the same context
  CHECK_GT(ctxs.size(), 0)
    << "The size of context mapping should be greater than zero";
  Context ctx = ctxs.begin()->second;

  std::map<std::string, Context> ctx_map;
  std::vector<NDArray> aux_states;

  auto exec = new exec::GraphExecutor();
  // (TODO) too hack here
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
  NodeEntryMap<Context> ctxs;
  for (const auto& kv : feed_dict) {
    const NodeEntry& e = kv.first;
    shapes.insert({kv.first, kv.second.shape()});
    ctxs.insert({kv.first, kv.second.ctx()});
  }
  exec::GraphExecutor *exec = Bind(sym, shapes, ctxs, saved_opr);
  std::vector<NDArray> res = Run(exec, feed_dict);
  return res;
}

}  // namespace autograd
}  // namespace mxnet
