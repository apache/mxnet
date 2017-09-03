/*!
 *  Copyright (c) 2016 by Contributors
 * \file quantization.cc
 * \brief
 */
#include <nnvm/graph.h>
#include <nnvm/pass.h>
#include <mxnet/op_attr_types.h>
#include <unordered_set>

namespace mxnet {
namespace op {

using nnvm::Symbol;
using nnvm::Node;
using nnvm::NodePtr;
using nnvm::NodeEntry;
using nnvm::Graph;

static const std::string   quantize_op_name = "_contrib_quantize";
static const std::string dequantize_op_name = "_contrib_dequantize";
static const std::string quantize_down_and_shrink_range_op_name =
    "quantize_down_and_shrink_range";

NodePtr CreateNode(std::string op_name, std::string node_name) {
  NodePtr node     = Node::Create();
  if (op_name != "nullptr") node->attrs.op = Op::Get(op_name);
  node->attrs.name = node_name;
  return node;
}

NodePtr InsertNode(std::string op_name,
    std::string node_name, NodePtr current, NodeEntry previous) {
  NodePtr node = CreateNode(op_name, node_name);
  node->inputs.emplace_back(previous);
  current->inputs.emplace_back(NodeEntry{node, 0, 0});
  return node;
}

std::vector<NodeEntry> OfflineParams(std::vector<NodeEntry>&& outputs,
                                     std::unordered_set<std::string>&& offline_params) {

  std::string node_suffixs[3] = {"", "_min", "_max"};
  std::unordered_map<Node*, NodePtr> mirror_map;
  nnvm::NodeEntryMap<NodePtr> entry_var;
  auto need_offline = [&](NodePtr n) {
    return n->op() &&
           (n->op()->name == quantize_op_name) &&
           n->inputs[0].node->is_variable() &&
           offline_params.count(n->inputs[0].node->attrs.name);
  };
  DFSVisit(outputs, [&](const NodePtr& node) {
    for (NodeEntry& e : node->inputs) {
      if (need_offline(e.node)) {
        std::string node_name = e.node->attrs.name;
        if (!entry_var.count(e)) {
          entry_var[e] = CreateNode("nullptr", node_name + node_suffixs[e.index]);
        }
        e.node = entry_var[e];
        e.index = 0;
        e.version = 0;
      }
    }
  });
  return outputs;
}

inline bool NeedQuantize(NodePtr node, const std::unordered_set<NodePtr> ignore_nodes) {
  static auto& quantized_op_map = Op::GetAttr<mxnet::FQuantizedOp>("FQuantizedOp");
  return quantized_op_map.count(node->op()) && !ignore_nodes.count(node);
}

Graph QuantizeGraph(Graph &&src) {
  static auto& quantized_op_map = Op::GetAttr<mxnet::FQuantizedOp>("FQuantizedOp");
  static auto& need_shrink_map = Op::GetAttr<mxnet::TQuantizationNeedShrink>("TQuantizationNeedShrink");
  auto offline_params = src.GetAttr<std::unordered_set<std::string>>("offline_params");
  auto ignore_nodes = src.GetAttr<std::unordered_set<NodePtr>>("ignore_nodes");

  std::unordered_map<Node*, NodePtr> mirror_map;
  DFSVisit(src.outputs, [&](const NodePtr& node) {

    NodePtr new_node = Node::Create();
    if (NeedQuantize(node, ignore_nodes)) {
      auto fquantized_op = quantized_op_map[node->op()];
      new_node = fquantized_op(node);

      // add data into quantized op input
      for (const auto& e : node->inputs) {
        NodePtr mirror_node = mirror_map.at(e.node.get());
        NodeEntry mirror_entry = NodeEntry{
          mirror_node, e.index, e.version};
        if (!NeedQuantize(e.node, ignore_nodes) &&
            (mirror_node->op() == nullptr ||
             mirror_node->op()->name != quantize_op_name)) {
          NodePtr quantize_node = InsertNode(quantize_op_name,
            e.node->attrs.name + "_quantize", new_node, mirror_entry);
          quantize_node->op()->attr_parser(&(quantize_node->attrs));

          NodePtr min_node = InsertNode("min",
              e.node->attrs.name + "_min", quantize_node, mirror_entry);
          min_node->op()->attr_parser(&(min_node->attrs));

          NodePtr max_node = InsertNode("max",
              e.node->attrs.name + "_max", quantize_node, mirror_entry);
          max_node->op()->attr_parser(&(max_node->attrs));

          mirror_map[e.node.get()] = std::move(quantize_node);
        } else {
          new_node->inputs.emplace_back(mirror_entry);
        }
        // the input should be `quantize` or quantized version op now
      }

      // add min and max into quantized op input
      // assume order of quantized op inputs is:
      //   data1, data2, ..., min1, max1, min2, max2, ...
      for (const auto& e : node->inputs) {
        NodePtr mirror_node = mirror_map.at(e.node.get());
        NodeEntry mirror_entry = NodeEntry{
          mirror_node, e.index, e.version};
        // for quantize node
        uint32_t min_index = 1;
        uint32_t max_index = 2;
        if (quantized_op_map.count(e.node->op())) {
          size_t  num_outputs = e.node->num_outputs();
          min_index = num_outputs + 2 * e.index;
          max_index = num_outputs + 2 * e.index + 1;
        } else {
          CHECK(mirror_node->op()->name == quantize_op_name)
            << "The input is not quantize or quantized_op";
        }
        new_node->inputs.emplace_back(NodeEntry{mirror_node, min_index, 0});
        new_node->inputs.emplace_back(NodeEntry{mirror_node, max_index, 0});
      }

      if (need_shrink_map.get(new_node->op(), false)) {
        NodePtr shrink_node = Node::Create();
        shrink_node->attrs.op = Op::Get(quantize_down_and_shrink_range_op_name);
        shrink_node->attrs.name = quantize_down_and_shrink_range_op_name +
            "_" + node->attrs.name;
        if (shrink_node->op()->attr_parser != nullptr) {
          shrink_node->op()->attr_parser(&(shrink_node->attrs));
        }
        for (size_t i = 0; i < 3; ++i) {
          shrink_node->inputs.emplace_back(NodeEntry{new_node, i, 0});
        }
        new_node = shrink_node;
      }
    } else {
      *new_node = *node;
      new_node->inputs.clear();
      for (const auto& e : node->inputs) {
        NodePtr mirror_node = mirror_map.at(e.node.get());
        NodeEntry mirror_entry = NodeEntry{
          mirror_node, e.index, e.version};
        size_t num_outputs = e.node->num_outputs();
        uint32_t min_index = num_outputs + 2 * e.index;
        uint32_t max_index = num_outputs + 2 * e.index + 1;

        // if input node is quantized operator, add dequantize node
        if (NeedQuantize(e.node, ignore_nodes)) {
          NodePtr dequantize_node = CreateNode(dequantize_op_name,
            e.node->attrs.name + "_dequantize");
          dequantize_node->inputs.emplace_back(mirror_entry);
          dequantize_node->inputs.emplace_back(NodeEntry{mirror_node, min_index, 0});
          dequantize_node->inputs.emplace_back(NodeEntry{mirror_node, max_index, 0});
          dequantize_node->op()->attr_parser(&(dequantize_node->attrs));

          new_node->inputs.emplace_back(NodeEntry{dequantize_node, 0, 0});
          mirror_map[e.node.get()] = std::move(dequantize_node);
        } else {
          new_node->inputs.emplace_back(NodeEntry{mirror_node, e.index, e.version});
        }
      }
    }
    mirror_map[node.get()] = std::move(new_node);
  });

  std::vector<NodeEntry> outputs;
  for (const auto& e: src.outputs) {
    if (quantized_op_map.count(e.node->op())) {
      NodePtr mirror_node = mirror_map.at(e.node.get());
      NodeEntry mirror_entry = NodeEntry{mirror_node, e.index, e.version};
      size_t num_inputs = e.node->num_inputs();
      uint32_t min_index = num_inputs + 2 * e.index;
      uint32_t max_index = num_inputs + 2 * e.index + 1;

      NodePtr dequantize_node = CreateNode(dequantize_op_name,
          e.node->attrs.name + "_dequantize");
      dequantize_node->inputs.emplace_back(mirror_entry);
      dequantize_node->inputs.emplace_back(NodeEntry{mirror_node, min_index, 0});
      dequantize_node->inputs.emplace_back(NodeEntry{mirror_node, max_index, 0});
      dequantize_node->op()->attr_parser(&(dequantize_node->attrs));
      outputs.emplace_back(NodeEntry{dequantize_node, 0, 0});
    } else {
      outputs.emplace_back(NodeEntry{mirror_map.at(e.node.get()), e.index, e.version});
    }
  }

  if (!offline_params.empty()) outputs =
    OfflineParams(std::move(outputs), std::move(offline_params));

  Graph ret;
  ret.outputs = std::move(outputs);
  return ret;
}

NNVM_REGISTER_PASS(QuantizeGraph)
.describe("")
.set_body(QuantizeGraph)
.set_change_graph(true);

}  // namespace op
}  // namespace mxnet
