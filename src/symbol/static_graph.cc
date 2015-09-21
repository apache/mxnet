/*!
 *  Copyright (c) 2015 by Contributors
 * \file static_graph.cc
 * \brief static graph of mxnet
 */
#include <dmlc/logging.h>
#include <mxnet/symbolic.h>
#include <vector>
#include <queue>
#include <map>
#include "./static_graph.h"
#include "../operator/operator_common.h"

namespace mxnet {
std::vector<uint32_t> StaticGraph::TopoSort() const {
  std::vector<std::pair<uint32_t, uint32_t> > stack;
  std::unordered_set<uint32_t> visited;
  std::vector<uint32_t> ret(nodes.size());
  std::vector<uint32_t> head_node;
  // out degree
  std::vector<int> out_degree(nodes.size(), 0);
  for (const Node& n : nodes) {
    for (const DataEntry& e : n.inputs) {
      ++out_degree[e.source_id];
    }
    if (n.is_backward()) {
      ++out_degree[n.backward_source_id];
    }
  }
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (out_degree[i] == 0) {
      stack.push_back(std::make_pair(static_cast<uint32_t>(i), 0));
    }
  }
  // heads
  for (auto &head : head_node) {
    stack.push_back(std::make_pair(head, 0));
  }
  int count = 0;
  while (!stack.empty()) {
    std::pair<uint32_t, uint32_t>& back = stack.back();
    const Node& n = nodes[back.first];
    if (back.second == n.inputs.size() + (n.is_backward() ? 1 : 0)) {
      ret[count++] = back.first;
      visited.insert(back.first);
      stack.pop_back();
    } else {
      uint32_t input;
      if (back.second == n.inputs.size() && n.is_backward()) {
        input = n.backward_source_id;
        back.second++;
      } else {
        input = n.inputs[back.second++].source_id;
      }
      if (visited.count(input) == 0) {
        stack.push_back(std::make_pair(input, 0));
      }
    }
  }
  return ret;
}

bool StaticGraph::InferNodeShapes(const std::vector<uint32_t> &topo_order,
                                  std::vector<std::vector<TShape> > *node_out_shapes,
                                  std::vector<std::vector<TShape> > *node_aux_shapes) const {
  for (uint32_t nid : topo_order) {
    const Node& node = nodes[nid];
    if (node.is_forward()) {
      std::vector<TShape> in_shape;
      for (const DataEntry& e : node.inputs) {
        in_shape.push_back((*node_out_shapes)[e.source_id][e.index]);
      }
      try {
        if (!node.op->InferShape(&in_shape,
                                 &(*node_out_shapes)[nid],
                                 &(*node_aux_shapes)[nid])) return false;
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
    } else if (nodes[nid].is_backward()) {
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
                             std::vector<TShape> *out_shape,
                             std::vector<TShape> *aux_shape) const {
  std::vector<std::vector<TShape> > node_out_shapes(nodes.size());
  std::vector<std::vector<TShape> > node_aux_shapes(nodes.size());
  for (size_t i = 0; i < nodes.size(); ++i) {
    int nout = 1;
    if (nodes[i].is_forward()) {
      nout = nodes[i].op->NumOutputs();
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
                       &node_out_shapes,
                       &node_aux_shapes)) return false;
  for (size_t i = 0; i < arg_nodes.size(); ++i) {
    (*in_shape)[i] = node_out_shapes[arg_nodes[i]][0];
  }
  out_shape->resize(heads.size());
  for (size_t i = 0; i < heads.size(); ++i) {
    const DataEntry &e = heads[i];
    (*out_shape)[i] = node_out_shapes[e.source_id][e.index];
  }
  aux_shape->clear();
  for (size_t i = 0; i < node_aux_shapes.size(); ++i) {
    if (node_aux_shapes[i].size() > 0) {
      for (auto const &shape : node_aux_shapes[i]) {
        aux_shape->push_back(shape);
      }
    }
  }
  return true;
}

StaticGraph::Node StaticGraph::CreateSumNode(
    const std::vector<DataEntry> &grad_source) {
  // find multiple gradients, need aggregate
  std::ostringstream os_size;
  Node agg_node;
  agg_node.op.reset(OperatorProperty::Create("ElementWiseSum"));
  os_size << grad_source.size();
  agg_node.op->Init({{"num_args", os_size.str()}});
  agg_node.inputs = grad_source;
  return agg_node;
}

void StaticGraph::MakeBackwardPass(std::vector<uint32_t> *head_grad_nodes,
                                   std::vector<DataEntry> *arg_grads) {
  arg_grads->clear();
  head_grad_nodes->clear();
  // get topo order of nodes, before new nodes are added
  std::vector<uint32_t> topo_order = TopoSort();
  // map out_data entry to out_grad
  std::map<DataEntry, std::vector<DataEntry> > grad_map;
  // allocate head gradient nodes
  for (DataEntry head : heads) {
    Node node;
    std::ostringstream os;
    os << nodes[head.source_id].name << '_' << head.index << "_grad";
    // TODO(bing): add index to name
    node.name = os.str();
    // node id
    uint32_t nid = static_cast<uint32_t>(nodes.size());
    nodes.push_back(std::move(node));
    // create a variable node for gradient input
    DataEntry igrad(nid, 0);
    head_grad_nodes->push_back(nid);
    // update gradient map
    auto it = grad_map.find(head);
    if (it == grad_map.end()) {
      grad_map[head] = {igrad};
    } else {
      it->second.push_back(igrad);
    }
  }
  // do backward pass traverse
  for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
    uint32_t nid = *it;
    // skip variables
    if (nodes[nid].is_variable()) continue;
    CHECK(nodes[nid].is_forward()) << "Do not support Backward of Backward";
    // get out_grad and out_data entry
    std::vector<DataEntry> out_grad, out_data;
    // nvisible is out_grad.size()
    int nvisible = nodes[nid].op->NumVisibleOutputs();
    // ntotal is out_data.size()
    int ntotal = nodes[nid].op->NumOutputs();
    // check all outpus
    for (int i = 0; i < ntotal; ++i) {
      DataEntry odata(nid, static_cast<uint32_t>(i));
      out_data.push_back(odata);
      if (i >= nvisible) continue;
      // get out_grad
      auto it = grad_map.find(odata);
      CHECK(it != grad_map.end()) << "bad graph";
      std::vector<DataEntry> &gnodes = it->second;
      if (gnodes.size() == 1) {
        out_grad.push_back(gnodes[0]);
      } else {
        std::ostringstream os_name;
        Node agg_node = StaticGraph::CreateSumNode(gnodes);
        os_name << nodes[nid].name << '_' << i << "_out_grad_agg";
        agg_node.name = os_name.str();
        uint32_t agg_node_id = static_cast<uint32_t>(nodes.size());
        nodes.push_back(std::move(agg_node));
        out_grad.push_back(DataEntry(agg_node_id, 0));
      }
    }
    // Create a gradient backward node
    Node grad_node;
    // Point to the corresponding source
    grad_node.backward_source_id = nid;
    // select out the dependent inputs
    grad_node.inputs = nodes[nid].op->BackwardInputs(
        out_grad, nodes[nid].inputs, out_data);
    grad_node.name = nodes[nid].name + "_backward";
    uint32_t grad_node_id = static_cast<uint32_t>(nodes.size());
    nodes.push_back(std::move(grad_node));
    // update gradient map
    for (size_t i = 0; i < nodes[nid].inputs.size(); ++i) {
      DataEntry idata = nodes[nid].inputs[i];
      DataEntry igrad(grad_node_id, static_cast<uint32_t>(i));
      auto it = grad_map.find(idata);
      if (it == grad_map.end()) {
        grad_map[idata] = {igrad};
      } else {
        it->second.push_back(igrad);
      }
    }
  }
  // create return values of arg_grads
  arg_grads->resize(arg_nodes.size());
  for (size_t i = 0; i < arg_nodes.size(); ++i) {
    DataEntry odata(arg_nodes[i], 0);
    auto it = grad_map.find(odata);
    CHECK(it != grad_map.end()) << "bad graph";
    if (it->second.size() == 1) {
      arg_grads->at(i) = it->second[0];
    } else {
      std::ostringstream os_name;
      Node agg_node = StaticGraph::CreateSumNode(it->second);
      os_name << nodes[arg_nodes[i]].name << "_grad_agg";
      agg_node.name = os_name.str();
      uint32_t agg_node_id = static_cast<uint32_t>(nodes.size());
      nodes.push_back(std::move(agg_node));
      arg_grads->at(i) = DataEntry(agg_node_id, 0);
    }
  }
}

void StaticGraph::Node::Save(dmlc::JSONWriter *writer) const {
  writer->BeginObject();
  if (op.get() != nullptr) {
    writer->WriteObjectKeyValue("op", op->TypeString());
    std::map<std::string, std::string> param = op->GetParams();
    writer->WriteObjectKeyValue("param", param);
  } else {
    std::map<std::string, std::string> empty_param;
    std::string json_null = "null";
    writer->WriteObjectKeyValue("op", json_null);
    writer->WriteObjectKeyValue("param", empty_param);
  }
  writer->WriteObjectKeyValue("name", name);
  writer->WriteObjectKeyValue("inputs", inputs);
  writer->WriteObjectKeyValue("backward_source_id", backward_source_id);
  writer->EndObject();
}

void StaticGraph::Node::Load(dmlc::JSONReader *reader) {
  dmlc::JSONObjectReadHelper helper;
  std::string op_type_str;
  std::map<std::string, std::string> param;
  helper.DeclareField("op", &op_type_str);
  helper.DeclareField("param", &param);
  helper.DeclareField("name", &name);
  helper.DeclareField("inputs", &inputs);
  helper.DeclareField("backward_source_id", &backward_source_id);
  helper.ReadAllFields(reader);

  if (op_type_str != "null") {
    op.reset(OperatorProperty::Create(op_type_str.c_str()));
    std::vector<std::pair<std::string, std::string> > vec(param.begin(), param.end());
    op->Init(vec);
  } else {
    op.reset(nullptr);
  }
}

void StaticGraph::Save(dmlc::JSONWriter *writer) const {
  writer->BeginObject();
  writer->WriteObjectKeyValue("nodes", nodes);
  writer->WriteObjectKeyValue("arg_nodes", arg_nodes);
  writer->WriteObjectKeyValue("heads", heads);
  writer->EndObject();
}

void StaticGraph::Load(dmlc::JSONReader *reader) {
  dmlc::JSONObjectReadHelper helper;
  helper.DeclareField("nodes", &nodes);
  helper.DeclareField("arg_nodes", &arg_nodes);
  helper.DeclareField("heads", &heads);
  helper.ReadAllFields(reader);
}
}  // namespace mxnet
