/*!
 *  Copyright (c) 2015 by Contributors
 * \file static_graph.cc
 * \brief static graph of mxnet
 */
#include <dmlc/logging.h>
#include <mxnet/symbolic.h>
#include <vector>
#include <queue>
#include "../operator/operator_common.h"

namespace mxnet {
std::vector<uint32_t> StaticGraph::TopoSort() const {
  std::vector<int> out_degree(nodes.size(), 0);
  for (const Node& n : nodes) {
    for (const DataEntry& e : n.inputs) {
      ++out_degree[e.source_id];
    }
    if (n.is_backward()) {
      ++out_degree[n.backward_source_id];
    }
  }
  std::vector<uint32_t> ret(nodes.size());
  auto result = ret.rbegin();
  std::queue<uint32_t> queue;
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (out_degree[i] == 0) {
      queue.push(static_cast<uint32_t>(i));
    }
  }
  while (!queue.empty()) {
    uint32_t node_id = queue.front();
    queue.pop();
    *result = node_id;
    ++result;
    const Node& n = nodes[node_id];
    for (const DataEntry& e : n.inputs) {
      if (--out_degree[e.source_id] == 0) {
        queue.push(e.source_id);
      }
    }
    if (n.is_backward()) {
      if (--out_degree[n.backward_source_id] == 0) {
        queue.push(n.backward_source_id);
      }
    }
  }
  return std::move(ret);
}

bool StaticGraph::InferNodeShapes(const std::vector<uint32_t> &topo_order,
                                  std::vector<std::vector<TShape> > *node_out_shapes) const {
  for (uint32_t nid : topo_order) {
    const Node& node = nodes[nid];
    if (node.is_forward()) {
      std::vector<TShape> in_shape;
      for (const DataEntry& e : node.inputs) {
        in_shape.push_back((*node_out_shapes)[e.source_id][e.index]);
      }
      try {
        if (!node.op->InferShape(&in_shape, &(*node_out_shapes)[nid])) return false;
      } catch (const op::InferShapeError &err) {
        // error handling
        const std::string &op_name = node.name;
        std::string arg_name = node.op->ListArguments()[err.index];
        std::ostringstream os;
        os << "InferShape Error in "
           << op_name << "\'s" << ' ' << arg_name << " argument\n";
        auto &source = nodes[node.inputs[err.index].source_id];
        if (source.is_variable()) {
          os << "Corresponding keyword of symbol: " << source.name << '\n' << err.msg;
        }
        throw dmlc::Error(os.str());
      }
      for (size_t i = 0; i < node.inputs.size(); ++i) {
        const DataEntry& e = node.inputs[i];
        (*node_out_shapes)[e.source_id][e.index] = in_shape[i];
      }
    } else if (node.is_backward()) {
      // simply use shapes from forward pass to assign backward shape
      const Node& forward = nodes[node.backward_source_id];
      CHECK(forward.is_forward());
      std::vector<TShape>& in_grad_shapes = (*node_out_shapes)[nid];
      CHECK(in_grad_shapes.size() == forward.inputs.size());
      // assign the input shape to output gradients
      for (size_t i = 0; i < forward.inputs.size(); ++i) {
        const DataEntry &e = forward.inputs[i];
        try {
          SHAPE_ASSIGN_CHECK(in_grad_shapes, i, (*node_out_shapes)[e.source_id][e.index]);
        } catch (const op::InferShapeError &err) {
          const std::string &op_name = forward.name;
          std::string arg_name = forward.op->ListArguments()[e.index];
          std::ostringstream os;
          os << "InferShape Error in "
             << op_name << "\'s" << ' ' << arg_name << " gradient argument\n"
             << err.msg;
          throw dmlc::Error(os.str());
        }
      }
      // consistent check for input shapes
      auto& out_data_shapes = (*node_out_shapes)[node.backward_source_id];
      // use BackwardInputs to select entries corresponding to node.inputs
      auto in_shape = forward.op->BackwardInputs(
          out_data_shapes, in_grad_shapes, out_data_shapes);
      for (size_t i = 0; i < node.inputs.size(); ++i) {
        const DataEntry& e = node.inputs[i];
        try {
          SHAPE_ASSIGN_CHECK((*node_out_shapes)[e.source_id], e.index, in_shape[i]);
        } catch (const op::InferShapeError &err) {
          const std::string &op_name = nodes[e.source_id].name;
          std::ostringstream os;
          os << "InferShape Error in "
             << op_name << "\'s" << " gradient values\n"
             << err.msg;
          throw dmlc::Error(os.str());
        }
      }
    }
  }
  // TODO(bing) assign shape for head gradient
  return true;
}

bool StaticGraph::InferShape(std::vector<TShape> *in_shape,
                             std::vector<TShape> *out_shape) const {
  std::vector<std::vector<TShape> > node_out_shapes(nodes.size());
  for (size_t i = 0; i < nodes.size(); ++i) {
    int nout = 1;
    if (nodes[i].is_forward()) {
      nout = nodes[i].op->NumReturns();
    } else if (nodes[i].is_backward()) {
      nout = static_cast<int>(nodes[nodes[i].backward_source_id].inputs.size());
    }
    node_out_shapes[i].resize(nout);
  }
  CHECK(in_shape->size() == arg_nodes.size())
        << "Wrong number of inputs to infer shape";
  for (size_t i = 0; i < arg_nodes.size(); ++i) {
    node_out_shapes[arg_nodes[i]][0] = (*in_shape)[i];
  }
  if (!InferNodeShapes(this->TopoSort(),
                       &node_out_shapes)) return false;
  for (size_t i = 0; i < arg_nodes.size(); ++i) {
    (*in_shape)[i] = node_out_shapes[arg_nodes[i]][0];
  }
  out_shape->resize(heads.size());
  for (size_t i = 0; i < heads.size(); ++i) {
    const DataEntry &e = heads[i];
    (*out_shape)[i] = node_out_shapes[e.source_id][e.index];
  }
  return true;
}
}  // namespace mxnet
