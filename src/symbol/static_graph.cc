/*!
 *  Copyright (c) 2015 by Contributors
 * \file static_graph.cc
 * \brief static graph of mxnet
 */
#include <dmlc/logging.h>
#include <mxnet/symbolic.h>
#include <vector>
#include <queue>

std::vector<uint32_t> StaticGraph::TopoSort() const {
  std::vector<int> out_degree(nodes.size(), 0);
  for (const Node &n : nodes) {
    for (const DataEntry &e : n.inputs) {
      ++out_degree[e.source_id];
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
    for (const DataEntry &e : nodes[node_id].inputs) {
      out_degree[e.source_id] -= 1;
      if (out_degree[e.source_id] == 0) {
        queue.push(e.source_id);
      }
    }
  }
  return std::move(ret);
}

bool StaticGraph::InferShape(const std::vector<uint32_t> &topo_order,
                             std::vector<std::vector<TShape> > *node_out_shapes) const {
  bool success = true;
  for (uint32_t nid : topo_order) {
    const Node &node = nodes[nid];
    if (node.sym != nullptr) {
      std::vector<TShape> in_shape;
      for (const DataEntry &e : node.inputs) {
        in_shape.push_back(node_out_shapes[e.source_id][e.index]);
      }
      if (!node.sym->InferShape(&in_shape, &(*node_out_shapes)[nid])) return false;
      for (size_t i = 0; i < node.inputs.size(); ++i) {
        const DataEntry &e = node.inputs[i];
        node_out_shapes[e.source_id][e.index] = in_shape[i];
      }
    }
  }
  return true;
}

bool StaticGraph::InferShape(std::vector<TShape> *in_shape,
                             std::vector<TShape> *out_shape) const {
  std::vector<std::vector<TShape> > node_out_shapes(nodes.size());
  for (size_t i = 0; i < nodes.size(); ++i) {
    int nout = 1;
    if (nodes[i].sym != nullptr) {
      nout = nodes[i].sym->NumReturns();
    }
    node_out_shapes[i].resize(nout);
  }
  CHECK(in_shape->size() == arg_nodes.size())
        << "Wrong number of inputs to infer shape";
  for (size_t i = 0; i < arg_nodes.size(); ++i) {
      node_out_shapes[nid][0] = (*in_shape)[i];
  }
  if (!InferNodeShapes(this->TopoSort(),
                       &node_out_shapes)) return false;
  for (size_t i = 0; i < arg_nodes.size(); ++i) {
    (*in_shape)[i] = node_out_shapes[nid][0];
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    DataEntry e = outputs[i];
    (*out_shape)[i] = node_out_shapes[e.source_id][e.index];
  }
}
