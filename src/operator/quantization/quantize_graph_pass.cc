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
 * \file quantization.cc
 * \brief
 */

#include <mxnet/op_attr_types.h>
#include <nnvm/graph.h>
#include <nnvm/pass.h>
#include <queue>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "quantize_v2-inl.h"
#include "../../common/utils.h"

namespace mxnet {
namespace op {

using nnvm::Graph;
using nnvm::Node;
using nnvm::NodeEntry;
using nnvm::ObjectPtr;
using nnvm::Symbol;

static inline std::string GetOutputName(const nnvm::Node* node, int index) {
  // Map where Key is Op and Value is function registered by FListOutputNames
  static const auto& flist_outputs = nnvm::Op::GetAttr<nnvm::FListOutputNames>("FListOutputNames");
  std::vector<std::string> output_names;

  // If operator has registered FListOutputNames function
  // get names by calling it with passed node attributes as argument
  // else return output index as a string
  if (flist_outputs.count(node->op())) {
    output_names = flist_outputs[node->op()](node->attrs);
    CHECK_GT(output_names.size(), index);
    return output_names[index];
  }

  CHECK_GT(node->num_outputs(), index);
  return std::to_string(index);
}

static inline size_t GetNumOutputs(ObjectPtr node) {
  // Get NumOutputs, check if current node has NumVisibleOutputs function, if yes, return
  // num_visible_outputs
  size_t num_outputs = node->num_outputs();
  static const auto& num_visible_outputs_attr =
      nnvm::Op::GetAttr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs");
  auto num_visible_output_func = num_visible_outputs_attr.get(node->op(), nullptr);
  if (num_visible_output_func != nullptr) {
    num_outputs = num_visible_output_func(node->attrs);
  }
  return num_outputs;
}

ObjectPtr CreateNode(std::string op_name, std::string node_name) {
  ObjectPtr node   = Node::Create();
  node->attrs.name = node_name;
  if (op_name == "nullptr") {
    node->attrs.op = nullptr;
    // ugly workaround because VariableParam is not exposed
    node->attrs.parsed =
        nnvm::Symbol::CreateVariable(node->attrs.name).outputs[0].node->attrs.parsed;
  } else {
    node->attrs.op = Op::Get(op_name);
  }
  return node;
}

/*!
 * \brief Insert a node named with node_name holding the op of op_name
 * before the node current and after the node previous.
 */
ObjectPtr InsertNode(std::string op_name,
                     std::string node_name,
                     ObjectPtr current,
                     NodeEntry previous) {
  ObjectPtr node = CreateNode(op_name, node_name);
  node->inputs.emplace_back(previous);
  current->inputs.emplace_back(node);
  return node;
}

std::vector<NodeEntry> OfflineParams(std::vector<NodeEntry>&& outputs,
                                     const std::unordered_set<std::string>& offline_params) {
  std::string node_suffixs[3] = {"", "_min", "_max"};
  std::unordered_map<Node*, ObjectPtr> mirror_map;
  nnvm::NodeEntryMap<ObjectPtr> entry_var;
  auto need_offline = [&](ObjectPtr n) {
    return (n->op() == Op::Get("_contrib_quantize_v2")) && n->inputs[0].node->is_variable() &&
           offline_params.count(n->inputs[0].node->attrs.name);
  };
  DFSVisit(outputs, [&](const ObjectPtr& node) {
    for (NodeEntry& e : node->inputs) {
      if (need_offline(e.node)) {
        std::string node_name = e.node->attrs.name;
        if (!entry_var.count(e)) {
          entry_var[e] = CreateNode("nullptr", node_name + node_suffixs[e.index]);
        }
        e.node    = entry_var[e];
        e.index   = 0;
        e.version = 0;
      }
    }
  });
  return std::move(outputs);
}

// To check if a node is registered with a computation function on a target device.
bool isRegistered(ObjectPtr node, const int& dev_type) {
  const auto& op      = node->op();
  Context ctx         = Context::Create(static_cast<Context::DeviceType>(dev_type), 0);
  FCompute fcompute   = common::GetFCompute<FCompute>(op, "FCompute", ctx);
  FComputeEx fcomp_ex = common::GetFCompute<FComputeEx>(op, "FComputeEx", ctx);
  FStatefulCompute fcomputestateful =
      common::GetFCompute<FStatefulCompute>(op, "FStatefulCompute", ctx);
  FStatefulComputeEx fcomputestateful_ex =
      common::GetFCompute<FStatefulComputeEx>(op, "FStatefulComputeEx", ctx);
  return (fcompute != nullptr || fcomp_ex != nullptr || fcomputestateful != nullptr ||
          fcomputestateful_ex != nullptr);
}

inline QuantizeType NeedQuantize(ObjectPtr node,
                                 const std::unordered_set<std::string>& excluded_nodes,
                                 const std::unordered_set<std::string>& excluded_ops,
                                 const int& dev_type,
                                 std::unordered_map<ObjectPtr, ObjectPtr>* quantized_node_map,
                                 const std::string quantize_granularity) {
  std::unordered_map<ObjectPtr, ObjectPtr> quantized_node;
  static auto& quantizable_map  = Op::GetAttr<mxnet::FQuantizable>("FQuantizable");
  static auto& quantized_op_map = Op::GetAttr<mxnet::FQuantizedOp>("FQuantizedOp");
  static auto& fexec_type       = nnvm::Op::GetAttr<FExecType>("FExecType");
  const auto& op                = node->op();
  bool need                     = false;
  if (op && quantized_op_map.count(op)) {
    need = true;
    // If the quantized node is not registered with a computation function, the node
    // will be excluded automatically.
    auto q_ptr = quantized_op_map[node->op()];
    auto qnode = q_ptr(node->attrs);
    if (!isRegistered(qnode, dev_type)) {
      LOG(INFO) << "Neither FCompute nor FComputeEx registered, " << node->op()->name
                << " is excluded automatically.";
      need = false;
    } else {
      if (excluded_nodes.count(node->attrs.name) || excluded_ops.count(node->op()->name)) {
        need = false;
      } else if (!node->attrs.subgraphs.empty()) {
        ExecType exec_type = fexec_type.count(op) ? fexec_type[op](node->attrs) : ExecType::kSync;
        if (exec_type != ExecType::kSubgraphExec) {
          // This is a fused subgraph node, try to match inner node.
          CHECK_EQ(node->attrs.subgraphs.size(), 1);
          auto subgraph_sym = node->attrs.subgraphs[0];
          DFSVisit(subgraph_sym->outputs, [&](const nnvm::ObjectPtr& n) {
            if (n->is_variable())
              return;
            if (excluded_nodes.count(n->attrs.name)) {
              need = false;
            }
          });
        }
      }
    }
    if (need) {
      auto quantized_node = quantized_op_map[op](node->attrs);
      if (!quantized_node->op())
        need = false;
      if (need) {
        if ((quantize_granularity == "channel-wise") &&
            (node->op() == Op::Get("_sg_onednn_fully_connected"))) {
          quantized_node->attrs.dict["channel_wise_quantize"] = "True";
        }
        quantized_node_map->insert(std::make_pair(node, quantized_node));
      }
      if (quantizable_map.count(op)) {
        return quantizable_map[op](node->attrs);
      } else {
        return QuantizeType::kSupport;
      }
    }
  }
  CHECK(!need);
  return QuantizeType::kNone;
}

enum quantize_bit {
  kFromInput  = 1,
  kFromOutput = 2,
};

static void MarkQuantizedNodes(const Graph& src,
                               std::unordered_map<ObjectPtr, ObjectPtr>* quantized_node_map) {
  const auto excluded_nodes       = src.GetAttr<std::unordered_set<std::string>>("excluded_nodes");
  const auto excluded_ops         = src.GetAttr<std::unordered_set<std::string>>("excluded_ops");
  const auto quantize_mode        = src.GetAttr<std::string>("quantize_mode");
  const auto dev_type             = src.GetAttr<int>("target_ctx");
  const auto quantize_granularity = src.GetAttr<std::string>("quantize_granularity");

  std::unordered_map<ObjectPtr, std::vector<ObjectPtr>> node_output_map;
  std::unordered_set<ObjectPtr> must_quantize_nodes;
  std::unordered_map<ObjectPtr, int> support_quantize_nodes;
  // Build node_output_map, must_quantize_nodes and support_quantize_nodes;
  DFSVisit(src.outputs, [&](const ObjectPtr& node) {
    auto quantize_type = NeedQuantize(
        node, excluded_nodes, excluded_ops, dev_type, quantized_node_map, quantize_granularity);
    if (quantize_type == QuantizeType::kMust) {
      must_quantize_nodes.insert(node);
    } else if (quantize_type == QuantizeType::kSupport) {
      support_quantize_nodes[node] = 0;
    }
    for (size_t i = 0; i < node->inputs.size(); ++i) {
      node_output_map[node->inputs[i].node].push_back(node);
    }
  });

  if (quantize_mode == "full") {
    return;
  } else if (quantize_mode == "smart") {
    // Mark quantized nodes from input
    std::queue<ObjectPtr> task_queue;
    for (const auto& node : must_quantize_nodes) {
      task_queue.push(node);
    }
    while (!task_queue.empty()) {
      const auto& node = task_queue.front();
      task_queue.pop();
      for (auto& i : node->inputs) {
        const auto& input = i.node;
        auto it           = support_quantize_nodes.find(input);
        if (it != support_quantize_nodes.end()) {
          it->second = it->second | kFromInput;
          task_queue.push(input);
        }
      }
    }

    // Mark quantized nodes from output
    for (const auto& node : must_quantize_nodes) {
      task_queue.push(node);
    }
    while (!task_queue.empty()) {
      const auto& node = task_queue.front();
      task_queue.pop();
      const auto& outputs = node_output_map[node];
      for (const auto& output : outputs) {
        auto it = support_quantize_nodes.find(output);
        if (it != support_quantize_nodes.end()) {
          it->second = it->second | kFromOutput;
          task_queue.push(output);
        }
      }
    }

    // Summarize the result
    for (const auto& node : support_quantize_nodes) {
      CHECK(quantized_node_map->count(node.first));
      if (node.second != (kFromInput | kFromOutput)) {
        quantized_node_map->erase(node.first);
      }
    }
  } else {
    LOG(FATAL) << "unrecognized quantize mode: " << quantize_mode;
  }
}

Graph QuantizeGraph(Graph&& src) {
  static const auto& need_requantize_map = Op::GetAttr<mxnet::FNeedRequantize>("FNeedRequantize");
  static const auto& avoid_quantize_input_map =
      Op::GetAttr<mxnet::FAvoidQuantizeInput>("FAvoidQuantizeInput");
  static const auto& flist_inputs = nnvm::Op::GetAttr<nnvm::FListOutputNames>("FListInputNames");
  static const auto& avoid_dequantize_map =
      Op::GetAttr<mxnet::FAvoidDequantizeOutput>("FAvoidDequantizeOutput");
  static const auto& need_asym_quantize_map =
      Op::GetAttr<mxnet::FNeedAsymQuantizeInput>("FNeedAsymQuantizeInput");
  const auto offline_params       = src.GetAttr<std::unordered_set<std::string>>("offline_params");
  const auto quantized_dtype      = src.GetAttr<std::string>("quantized_dtype");
  const auto quantize_granularity = src.GetAttr<std::string>("quantize_granularity");
  const auto dev_type             = src.GetAttr<int>("target_ctx");

  if (dev_type == Context::kGPU && quantize_granularity == "channel-wise") {
    LOG(FATAL) << "`channel-wise` quantization option is not supported yet by GPU,"
               << " please set quantize_granularity to `tensor-wise` when quantizing model.";
  }

  std::unordered_map<ObjectPtr, ObjectPtr> quantized_node_map;
  MarkQuantizedNodes(src, &quantized_node_map);

  // mirror_map stores the mapping from the currently visited graph to the newly created quantized
  // graph. Key is the currently visited graph's node pointer, and value is a copied node of the key
  // node. The existing key's value may be updated with the newly created quantize/dequantize op.
  std::unordered_map<Node*, ObjectPtr> mirror_map;
  std::unordered_map<ObjectPtr, ObjectPtr> reverse_mirror_map;
  nnvm::NodeEntryMap<NodeEntry> mirror_entry_map;
  static int verbose = dmlc::GetEnv("MXNET_QUANTIZATION_VERBOSE", 0);
  DFSVisit(src.outputs, [&](const ObjectPtr& node) {
    ObjectPtr new_node = Node::Create();
    // If the currently visited node needs quantization, insert a quantize op node before the
    // current node and replace the current node with the quantized version in the new graph.
    if (quantized_node_map.count(node)) {
      if (verbose) {
        LOG(INFO) << node->attrs.name << " is quantized.";
      }
      new_node = quantized_node_map[node];

      // add data into quantized op input
      for (size_t i = 0; i < node->inputs.size(); ++i) {
        const auto& e          = node->inputs[i];
        ObjectPtr mirror_node  = mirror_map.at(e.node.get());
        NodeEntry mirror_entry = NodeEntry{mirror_node, e.index, e.version};
        // If the NodeEntry e's node does not need quantization, and (the mirror_node is a variable,
        // or the mirror_node's op is not a quantize op), create quantize op, min op, and max op
        // taking mirror_entry as input to generate a quantized NDArray. Save the mapping between
        // e's source node and the newly created quantize op so that the quantize op can be
        // reused next time when the same entry is visited again.
        if (avoid_quantize_input_map.count(node->op()) &&
            avoid_quantize_input_map[node->op()](node->attrs, i, quantize_granularity)) {
          new_node->inputs.emplace_back(mirror_entry);
        } else if (!quantized_node_map.count(e.node) ||
                   (avoid_dequantize_map.count(e.node->op()) &&
                    avoid_dequantize_map[e.node->op()](e.node->attrs, e.index))) {
          // If the input of current quantized node has non-support of quantization, a quantize op
          // is supposed to insert into the position after the input node to quantize the float
          // input to int8/uint8 type. Also, a quantized operator with avoid-dequantize attribute
          // can produce float outputs directly. A quantize op is necessary to convert them into
          // int8/uint8 type as the input of current quantized node.
          if (mirror_entry_map.count(e)) {
            new_node->inputs.emplace_back(mirror_entry_map[e]);
          } else {
            // When there're multiple entrys outgoing from a single node, need to add entry
            // index (or output name) into quantize/min/max node to distinguish them.
            // Or the output name is not ending with 'output', just put the output name here
            // to better align with calibration phase. No need to change name to weights/bias.
            std::string suffix   = "";
            std::string new_name = e.node->attrs.name;

            if (mirror_node->op() != nullptr) {
              std::string name = GetOutputName(e.node.get(), e.index);
              suffix           = "_" + name;
            } else if (!offline_params.count(new_name)) {
              std::string input_name;
              if (flist_inputs.count(node->op())) {
                input_name = flist_inputs[node->op()](node->attrs)[i];
                new_name   = node->attrs.name + "_" + input_name;
              } else {
                new_name = node->attrs.name + "_" + e.node->attrs.name;
              }
            }
            ObjectPtr quantize_node;
            if (need_asym_quantize_map.count(node->op()) &&
                need_asym_quantize_map[node->op()](node->attrs, i)) {
              quantize_node = InsertNode("_contrib_quantize_asym",
                                         new_name + suffix + "_quantize",
                                         new_node,
                                         mirror_entry);
            } else {
              quantize_node = InsertNode(
                  "_contrib_quantize_v2", new_name + suffix + "_quantize", new_node, mirror_entry);
              // If current node is rnn op, the quantize op is supposed to quantize the result of
              // pre-node to uint8, as quantized rnn op requires uint8 input.
              quantize_node->attrs.dict["out_type"] = quantized_dtype;
            }
            quantize_node->op()->attr_parser(&(quantize_node->attrs));
            mirror_entry_map[e] = NodeEntry{quantize_node, 0, e.version};
          }
        } else if (mirror_node->op() == Op::Get("_contrib_dequantize")) {
          new_node->inputs.emplace_back(mirror_node->inputs[0].node, e.index, e.version);
        } else {
          // If the entry e's node needs quantization, or mirror_entry is from a quantize op,
          // simply add mirror_entry to the input of the new_node.
          new_node->inputs.emplace_back(mirror_entry);
        }
        // the input should be `quantize` or quantized version op now
      }

      // add min and max into quantized op input assume order of quantized op inputs is:
      // data1, data2, ..., min1, max1, min2, max2, ...
      for (size_t i = 0; i < node->inputs.size(); ++i) {
        const auto& e         = node->inputs[i];
        ObjectPtr mirror_node = mirror_map.at(e.node.get());
        if (mirror_node->op() == Op::Get("_contrib_dequantize")) {
          mirror_node = mirror_node->inputs[0].node;
        }
        NodeEntry mirror_entry = NodeEntry{mirror_node, e.index, e.version};
        // for quantize node
        uint32_t min_index = 1;
        uint32_t max_index = 2;
        if (avoid_quantize_input_map.count(node->op()) &&
            avoid_quantize_input_map[node->op()](node->attrs, i, quantize_granularity)) {
          // skip non-quantized input
          continue;
        }
        if (quantized_node_map.count(e.node)) {
          // here we calculate the output number (exclude min/max, in order to
          // calculate min/max index from mirror node) based on assumption that
          // there is only 1min and 1max output from mirror node (which is
          // currently true)
          size_t num_outputs = GetNumOutputs(mirror_node) - 2;
          min_index          = num_outputs + 2 * e.index;
          max_index          = num_outputs + 2 * e.index + 1;
        } else {
          CHECK(mirror_entry_map.count(e)) << "The input is not quantize or quantized_op";
        }
        if (mirror_entry_map.count(e)) {
          auto quantize_entry = mirror_entry_map[e];
          new_node->inputs.emplace_back(quantize_entry.node, min_index, 0);
          new_node->inputs.emplace_back(quantize_entry.node, max_index, 0);
        } else {
          new_node->inputs.emplace_back(mirror_node, min_index, 0);
          new_node->inputs.emplace_back(mirror_node, max_index, 0);
        }
      }

      // If the new_node op registered attr FNeedRequantize, insert requantize node after it.
      // Here it's assumed that the quantized_op node only produces three outputs:
      // out_data, min_range, and max_range.
      if (need_requantize_map.count(new_node->op()) > 0 &&
          need_requantize_map[new_node->op()](new_node->attrs)) {
        ObjectPtr requantize_node               = Node::Create();
        requantize_node->attrs.op               = Op::Get("_contrib_requantize");
        requantize_node->attrs.name             = "requantize_" + node->attrs.name;
        requantize_node->attrs.dict["out_type"] = quantized_dtype;
        if (requantize_node->op()->attr_parser != nullptr) {
          requantize_node->op()->attr_parser(&(requantize_node->attrs));
        }
        for (size_t i = 0; i < 3; ++i) {
          requantize_node->inputs.emplace_back(new_node, static_cast<uint32_t>(i), 0);
        }
        new_node = requantize_node;
      }
    } else {
      // If the currently visited node does not need quantization, copy the current node to become
      // the new_node. Meanwhile, check whether any inputs of the current node need quantization
      // (e.g., a quantized_conv2d node), and insert a dequantize op node in the new graph if there
      // are any. Otherwise, simply add a copy of the current node's entry to the inputs of
      // the new_node.
      if (verbose && !node->is_variable())
        LOG(INFO) << node->attrs.name << " is NOT quantized.";
      *new_node = *node;
      new_node->inputs.clear();
      for (const auto& e : node->inputs) {
        ObjectPtr mirror_node  = mirror_map.at(e.node.get());
        NodeEntry mirror_entry = NodeEntry{mirror_node, e.index, e.version};
        // If input node is quantized operator, add dequantize node. But if input node is a
        // quantized operator with avoid-dequantize attribute, its output may be already in float
        // type, which dosen't need a dequantize op.
        if (quantized_node_map.count(e.node) &&
            mirror_node->op() != Op::Get("_contrib_dequantize") &&
            !(avoid_dequantize_map.count(e.node->op()) &&
              avoid_dequantize_map[e.node->op()](e.node->attrs, e.index))) {
          // here we calculate the output number (exclude min/max, in order to
          // calculate min/max index from mirror node) based on assumption that
          // there is only 1 min and 1 max output from mirror node (which is
          // currently true)
          size_t num_outputs = GetNumOutputs(mirror_node) - 2;
          uint32_t min_index = num_outputs + 2 * e.index;
          uint32_t max_index = num_outputs + 2 * e.index + 1;
          ObjectPtr dequantize_node =
              CreateNode("_contrib_dequantize", e.node->attrs.name + "_dequantize");
          dequantize_node->inputs.emplace_back(mirror_entry);
          dequantize_node->inputs.emplace_back(mirror_node, min_index, 0);
          dequantize_node->inputs.emplace_back(mirror_node, max_index, 0);
          dequantize_node->op()->attr_parser(&(dequantize_node->attrs));

          new_node->inputs.emplace_back(dequantize_node, 0, 0);
          mirror_map[e.node.get()]            = dequantize_node;
          reverse_mirror_map[dequantize_node] = e.node;
        } else if (mirror_entry_map.count(e)) {
          new_node->inputs.emplace_back(
              mirror_entry_map[e].node->inputs[0].node, e.index, e.version);
        } else {
          new_node->inputs.emplace_back(mirror_node, e.index, e.version);
        }
      }
    }
    mirror_map[node.get()]       = new_node;
    reverse_mirror_map[new_node] = node;
  });

  std::vector<NodeEntry> outputs;
  for (const auto& e : src.outputs) {
    if (quantized_node_map.count(e.node) &&
        !(avoid_dequantize_map.count(e.node->op()) &&
          avoid_dequantize_map[e.node->op()](e.node->attrs, e.index))) {
      // Only insert dequantize for those Ops supports quantize and not excluded.
      ObjectPtr mirror_node  = mirror_map.at(e.node.get());
      NodeEntry mirror_entry = NodeEntry{mirror_node, e.index, e.version};
      // here we calculate the output number (exclude min/max, in order to
      // calculate min/max index from mirror node) based on assumption that
      // there is only 1 min and 1 max output from mirror node (which is
      // currently true)
      size_t num_outputs = GetNumOutputs(e.node);
      uint32_t min_index = num_outputs + 2 * e.index;
      uint32_t max_index = num_outputs + 2 * e.index + 1;

      ObjectPtr dequantize_node =
          CreateNode("_contrib_dequantize", e.node->attrs.name + "_dequantize");
      dequantize_node->inputs.emplace_back(mirror_entry);
      dequantize_node->inputs.emplace_back(mirror_node, min_index, 0);
      dequantize_node->inputs.emplace_back(mirror_node, max_index, 0);
      dequantize_node->op()->attr_parser(&(dequantize_node->attrs));
      outputs.emplace_back(dequantize_node, 0, 0);
    } else {
      outputs.emplace_back(mirror_map.at(e.node.get()), e.index, e.version);
    }
  }

  if (!offline_params.empty())
    outputs = OfflineParams(std::move(outputs), offline_params);

  Graph ret;
  ret.outputs = std::move(outputs);

  static const auto& need_calib_input_map =
      Op::GetAttr<mxnet::FNeedCalibrateInput>("FNeedCalibrateInput");
  static const auto& need_calib_output_map =
      Op::GetAttr<mxnet::FNeedCalibrateOutput>("FNeedCalibrateOutput");

  std::unordered_set<nnvm::ObjectPtr> calib_variables;
  std::vector<std::string> calib_nodes;
  DFSVisit(ret.outputs, [&](const ObjectPtr& node) {
    if (node->op() && !calib_variables.empty()) {
      // find nodes where input is variable node
      // and add proper input_name to calib_nodes
      for (int i = 0; i < node->inputs.size(); i++) {
        const auto& input_node = node->inputs[i];
        if (calib_variables.find(input_node.node) != std::end(calib_variables)) {
          auto fp32_node = std::find_if(
              std::begin(quantized_node_map),
              std::end(quantized_node_map),
              [&](const std::pair<ObjectPtr, ObjectPtr>& pair) { return pair.second == node; });
          if (fp32_node != std::end(quantized_node_map)) {
            const auto& fp32_in_node = fp32_node->first;
            std::string node_input_name;
            if (flist_inputs.count(fp32_in_node->op())) {
              std::string op_input_name = flist_inputs[fp32_in_node->op()](fp32_in_node->attrs)[i];
              node_input_name           = fp32_in_node->attrs.name + "_" + op_input_name;
            } else {
              node_input_name = fp32_in_node->attrs.name + "_" + input_node.node->attrs.name;
            }
            calib_nodes.push_back(node_input_name);
            calib_variables.erase(input_node.node);
          }
        }
      }
    }
    if (need_calib_input_map.count(node->op())) {
      const auto calib_idx = need_calib_input_map[node->op()](node->attrs);
      for (const auto& idx : calib_idx) {
        if (reverse_mirror_map.count(node)) {
          const auto& fp32_in_node = reverse_mirror_map[node];
          std::string name         = GetOutputName(fp32_in_node.get(), node->inputs[idx].index);
          calib_nodes.push_back(fp32_in_node->attrs.name + "_" + name);
        } else {
          const auto& e = node->inputs[idx];
          if (e.node->is_variable()) {
            // monitor callback join operator name and variable name as observable node name,
            // instead of using variable output we can use op node input
            //
            //                  data_output/fc_input
            // e.g. data (var.) ----------------------> FC (op)
            // remember current node and compare with inputs of next nodes
            calib_variables.insert(node);
          } else {
            if (reverse_mirror_map.count(e.node)) {
              const auto& fp32_in_node = reverse_mirror_map.at(e.node);
              std::string name         = GetOutputName(fp32_in_node.get(), e.index);
              calib_nodes.push_back(fp32_in_node->attrs.name + "_" + name);
            } else {
              LOG(FATAL) << "Can't find calibration node for " << node->attrs.name;
            }
          }
        }
      }
    } else if (need_calib_output_map.count(node->op())) {
      const auto calib_idx = need_calib_output_map[node->op()](node->attrs);
      for (const auto& idx : calib_idx) {
        if (reverse_mirror_map.count(node)) {
          const auto& fp32_in_node = reverse_mirror_map[node];
          std::string name         = GetOutputName(fp32_in_node.get(), static_cast<uint32_t>(idx));
          calib_nodes.push_back(fp32_in_node->attrs.name + "_" + name);
        } else {
          std::string name = GetOutputName(node.get(), static_cast<uint32_t>(idx));
          calib_nodes.push_back(node->attrs.name + "_" + name);
        }
      }
    }
  });
  ret.attrs["calib_nodes"] = std::make_shared<dmlc::any>(std::move(calib_nodes));
  return ret;
}

static inline void SetCalibTableForEntry(
    const NodeEntry& e,
    const ObjectPtr& node,
    const std::unordered_map<std::string, std::pair<float, float>>& calib_table) {
  std::string out_name       = GetOutputName(e.node.get(), e.index);
  std::string full_node_name = e.node->attrs.name;

  if (!e.node->is_variable()) {
    full_node_name += "_" + out_name;
  } else {
    const std::string suffix = "_quantize";
    full_node_name           = node->attrs.name;
    full_node_name = std::string(full_node_name.begin(), full_node_name.end() - suffix.size());
  }

  const std::string prefix = "quantized_";
  if (e.node->attrs.name.rfind(prefix, 0) == 0) {
    full_node_name = full_node_name.substr(prefix.size());
  }
  const auto calib_table_iter = calib_table.find(full_node_name);
  static int verbose          = dmlc::GetEnv("MXNET_QUANTIZATION_VERBOSE", 0);
  if (calib_table_iter != calib_table.end()) {
    if (verbose) {
      LOG(INFO) << "Set calibration result to " << node->attrs.name
                << " : min=" << calib_table_iter->second.first
                << " max=" << calib_table_iter->second.second;
    }
    node->attrs.dict["min_calib_range"] = std::to_string(calib_table_iter->second.first);
    node->attrs.dict["max_calib_range"] = std::to_string(calib_table_iter->second.second);
    if (node->op() && node->op()->attr_parser)
      node->op()->attr_parser(&(node->attrs));
  } else {
    if (verbose) {
      LOG(INFO) << "Can't find calibration result for " << node->attrs.name;
    }
  }
}

Graph SetCalibTableToQuantizedGraph(Graph&& g) {
  const auto& calib_table =
      g.GetAttr<std::unordered_map<std::string, std::pair<float, float>>>("calib_table");
  static const auto& need_calib_input_map =
      Op::GetAttr<mxnet::FNeedCalibrateInput>("FNeedCalibrateInput");
  static const auto& need_calib_output_map =
      Op::GetAttr<mxnet::FNeedCalibrateOutput>("FNeedCalibrateOutput");
  static int verbose = dmlc::GetEnv("MXNET_QUANTIZATION_VERBOSE", 0);
  if (verbose) {
    LOG(INFO) << "Set calibration result to quantized symbol.";
  }
  DFSVisit(g.outputs, [&](const ObjectPtr& node) {
    if (need_calib_input_map.count(node->op())) {
      const auto calib_idx = need_calib_input_map[node->op()](node->attrs);
      CHECK_EQ(calib_idx.size(), 1);
      const auto& idx = calib_idx[0];
      SetCalibTableForEntry(node->inputs[idx], node, calib_table);
    } else if (need_calib_output_map.count(node->op())) {
      const auto calib_idx = need_calib_output_map[node->op()](node->attrs);
      CHECK_EQ(calib_idx.size(), 1);
      const auto& idx = calib_idx[0];
      SetCalibTableForEntry({node, static_cast<uint32_t>(idx), 0}, node, calib_table);
    }
  });
  return std::move(g);
}

NNVM_REGISTER_PASS(QuantizeGraph)
    .describe("")
    .set_body(QuantizeGraph)
    .provide_graph_attr("calib_nodes")
    .set_change_graph(true);

NNVM_REGISTER_PASS(SetCalibTableToQuantizedGraph)
    .describe("")
    .set_body(SetCalibTableToQuantizedGraph)
    .set_change_graph(true);

}  // namespace op
}  // namespace mxnet
