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

std::vector<uint32_t> StaticGraph::PostDFSOrder(const std::vector<uint32_t>& head_nodes,
                                                const std::unordered_set<uint32_t>& banned) const {
  std::vector<uint32_t> ret;
  std::unordered_set<uint32_t> visited;
  ret.reserve(nodes.size() / 2);
  std::vector<std::pair<uint32_t, uint32_t> > stack;
  // heads
  for (auto head : head_nodes) {
    if (visited.count(head) != 0) continue;
    stack.push_back(std::make_pair(head, 0));
    CHECK_EQ(banned.count(head), 0);
    // bugfix
    visited.insert(head);
    while (!stack.empty()) {
      std::pair<uint32_t, uint32_t>& back = stack.back();
      const Node& n = nodes[back.first];
      if (back.second == n.inputs.size() + (n.is_backward() ? 1 : 0)) {
        ret.push_back(back.first);
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
        if (visited.count(input) == 0 && banned.count(input) == 0) {
          stack.push_back(std::make_pair(input, 0));
        }
      }
    }
  }
  return ret;
}

std::vector<uint32_t> StaticGraph::TopoSort() const {
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
  std::vector<uint32_t> head_nodes;
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (out_degree[i] == 0) {
      head_nodes.push_back(static_cast<uint32_t>(i));
    }
  }
  return PostDFSOrder(head_nodes, std::unordered_set<uint32_t>());
}

bool StaticGraph::InferNodeShapes(const std::vector<uint32_t> &topo_order,
                                  std::vector<std::vector<TShape> > *node_out_shapes,
                                  std::vector<std::vector<TShape> > *node_aux_shapes,
                                  bool partial_infer) const {
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
                                 &(*node_aux_shapes)[nid])) {
          if (partial_infer)
            continue;
          return false;
        }
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
      } catch (const dmlc::Error &err) {
        const std::string &op_name = node.name;
        std::ostringstream os;
        os << "InferShape Error in " << op_name << ": " << err.what();
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
      for (size_t i = 0; i < node.inputs.size() - node.addto_index.size(); ++i) {
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

      // set for auxilary states shape.
      auto& source_aux_shapes = (*node_aux_shapes)[node.backward_source_id];
      for (size_t i = 0; i < source_aux_shapes.size(); ++i) {
        try {
          (*node_aux_shapes)[nid].push_back(source_aux_shapes[i]);
        } catch (const op::InferShapeError &err) {
          const std::string &op_name = nodes[nid].name;
          std::ostringstream os;
          os << "InferShape Error in "
             << op_name << "\'s" << " aux states\n"
             << err.msg;
          throw dmlc::Error(os.str());
        }
      }
    }
  }
  // TODO(bing) assign shape for head gradient
  return true;
}

bool StaticGraph::InferNodeTypes(const std::vector<uint32_t> &topo_order,
                                  std::vector<std::vector<int> > *node_out_types,
                                  std::vector<std::vector<int> > *node_aux_types) const {
  for (uint32_t nid : topo_order) {
    const Node& node = nodes[nid];
    if (node.is_forward()) {
      std::vector<int> in_type;
      for (const DataEntry& e : node.inputs) {
        in_type.push_back((*node_out_types)[e.source_id][e.index]);
      }
      try {
        if (!node.op->InferType(&in_type,
                                 &(*node_out_types)[nid],
                                 &(*node_aux_types)[nid])) return false;
      } catch (const op::InferTypeError &err) {
        // error handling
        const std::string &op_name = node.name;
        std::string arg_name = node.op->ListArguments()[err.index];
        std::ostringstream os;
        os << "InferType Error in "
           << op_name << "\'s" << ' ' << arg_name << " argument\n";
        auto &source = nodes[node.inputs[err.index].source_id];
        if (source.is_variable()) {
          os << "Corresponding keyword of symbol: " << source.name << '\n' << err.msg;
        }
        throw dmlc::Error(os.str());
      } catch (const dmlc::Error &err) {
        const std::string &op_name = node.name;
        std::ostringstream os;
        os << "InferType Error in " << op_name << ": " << err.what();
        throw dmlc::Error(os.str());
      }
      for (size_t i = 0; i < node.inputs.size(); ++i) {
        const DataEntry& e = node.inputs[i];
        (*node_out_types)[e.source_id][e.index] = in_type[i];
      }
    } else if (nodes[nid].is_backward()) {
      // simply use types from forward pass to assign backward type
      const Node& forward = nodes[node.backward_source_id];
      CHECK(forward.is_forward());
      std::vector<int>& in_grad_types = (*node_out_types)[nid];
      CHECK(in_grad_types.size() == forward.inputs.size());
      // assign the input type to output gradients
      for (size_t i = 0; i < forward.inputs.size(); ++i) {
        const DataEntry &e = forward.inputs[i];
        try {
          TYPE_ASSIGN_CHECK(in_grad_types, i, (*node_out_types)[e.source_id][e.index]);
        } catch (const op::InferTypeError &err) {
          const std::string &op_name = forward.name;
          std::string arg_name = forward.op->ListArguments()[e.index];
          std::ostringstream os;
          os << "InferType Error in "
             << op_name << "\'s" << ' ' << arg_name << " gradient argument\n"
             << err.msg;
          throw dmlc::Error(os.str());
        }
      }
      // consistent check for input types
      auto& out_data_types = (*node_out_types)[node.backward_source_id];
      // use BackwardInputs to select entries corresponding to node.inputs
      auto in_type = forward.op->BackwardInputs(
          out_data_types, in_grad_types, out_data_types);
      for (size_t i = 0; i < node.inputs.size() - node.addto_index.size(); ++i) {
        const DataEntry& e = node.inputs[i];
        try {
          TYPE_ASSIGN_CHECK((*node_out_types)[e.source_id], e.index, in_type[i]);
        } catch (const op::InferTypeError &err) {
          const std::string &op_name = nodes[e.source_id].name;
          std::ostringstream os;
          os << "InferType Error in "
             << op_name << "\'s" << " gradient values\n"
             << err.msg;
          throw dmlc::Error(os.str());
        }
      }

      // set for auxilary states type.
      auto& source_aux_types = (*node_aux_types)[node.backward_source_id];
      for (size_t i = 0; i < source_aux_types.size(); ++i) {
        try {
          (*node_aux_types)[nid].push_back(source_aux_types[i]);
        } catch (const op::InferTypeError &err) {
          const std::string &op_name = nodes[nid].name;
          std::ostringstream os;
          os << "InferType Error in "
             << op_name << "\'s" << " aux states\n"
             << err.msg;
          throw dmlc::Error(os.str());
        }
      }
    }
  }
  // TODO(bing) assign type for head gradient
  return true;
}

bool StaticGraph::InferShape(std::vector<TShape> *in_shape,
                             std::vector<TShape> *out_shape,
                             std::vector<TShape> *aux_shape,
                             bool partial_infer) const {
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
                       &node_aux_shapes,
                       partial_infer)) return false;
  for (size_t i = 0; i < arg_nodes.size(); ++i) {
    (*in_shape)[i] = node_out_shapes[arg_nodes[i]][0];
  }
  out_shape->resize(heads.size());
  for (size_t i = 0; i < heads.size(); ++i) {
    const DataEntry &e = heads[i];
    (*out_shape)[i] = node_out_shapes[e.source_id][e.index];
  }

  // set back auxiliary nodes.
  aux_shape->clear();
  std::vector<uint32_t> head_nodes;
  for (const auto& head : heads) {
    head_nodes.push_back(head.source_id);
  }
  std::vector<uint32_t> fwd_nodes = PostDFSOrder(head_nodes, std::unordered_set<uint32_t>());
  uint32_t counter = 0;
  for (uint32_t nid : fwd_nodes) {
    // backward consistentcy check.
    CHECK(nid == counter++);
    if (node_aux_shapes[nid].size() > 0) {
      for (auto const &shape : node_aux_shapes[nid]) {
        aux_shape->push_back(shape);
      }
    }
  }
  return true;
}

bool StaticGraph::InferType(std::vector<int> *in_type,
                             std::vector<int> *out_type,
                             std::vector<int> *aux_type) const {
  std::vector<std::vector<int> > node_out_types(nodes.size());
  std::vector<std::vector<int> > node_aux_types(nodes.size());
  for (size_t i = 0; i < nodes.size(); ++i) {
    int nout = 1;
    if (nodes[i].is_forward()) {
      nout = nodes[i].op->NumOutputs();
    } else if (nodes[i].is_backward()) {
      nout = static_cast<int>(nodes[nodes[i].backward_source_id].inputs.size());
    }
    node_out_types[i].resize(nout, -1);
  }
  CHECK(in_type->size() == arg_nodes.size())
        << "Wrong number of inputs to infer type";
  for (size_t i = 0; i < arg_nodes.size(); ++i) {
    node_out_types[arg_nodes[i]][0] = (*in_type)[i];
  }
  if (!InferNodeTypes(this->TopoSort(),
                       &node_out_types,
                       &node_aux_types)) return false;
  for (size_t i = 0; i < arg_nodes.size(); ++i) {
    (*in_type)[i] = node_out_types[arg_nodes[i]][0];
  }
  out_type->resize(heads.size());
  for (size_t i = 0; i < heads.size(); ++i) {
    const DataEntry &e = heads[i];
    (*out_type)[i] = node_out_types[e.source_id][e.index];
  }

  // set back auxiliary nodes.
  aux_type->clear();
  std::vector<uint32_t> head_nodes;
  for (const auto& head : heads) {
    head_nodes.push_back(head.source_id);
  }
  std::vector<uint32_t> fwd_nodes = PostDFSOrder(head_nodes, std::unordered_set<uint32_t>());
  uint32_t counter = 0;
  for (uint32_t nid : fwd_nodes) {
    // backward consistentcy check.
    CHECK(nid == counter++);
    if (node_aux_types[nid].size() > 0) {
      for (auto const &type : node_aux_types[nid]) {
        aux_type->push_back(type);
      }
    }
  }
  return true;
}

StaticGraph::Node StaticGraph::CreateGradSumNode(
    const std::vector<DataEntry> &grad_source) {
  // start to use inplace gradient sum when it is greater than cap.
  static size_t inplace_sum_cap = dmlc::GetEnv("MXNET_EXEC_INPLACE_GRAD_SUM_CAP", 8);
  // find multiple gradients, need aggregate
  std::vector<DataEntry> gsource;
  if (grad_source.size() < inplace_sum_cap) {
    gsource = grad_source;
  } else {
    for (size_t i = 1; i < grad_source.size(); ++i) {
      nodes[grad_source[i].source_id]
          .addto_index.push_back(grad_source[i].index);
      nodes[grad_source[i].source_id]
          .inputs.push_back(grad_source[i - 1]);
    }
    gsource.push_back(grad_source.back());
  }

  std::ostringstream os_size;
  Node agg_node;
  agg_node.op.reset(OperatorProperty::Create("ElementWiseSum"));
  os_size << gsource.size();
  agg_node.op->Init({{"num_args", os_size.str()}});
  agg_node.inputs = gsource;
  return agg_node;
}

StaticGraph::Node StaticGraph::CreateCopyNode(const DataEntry &source) {
  // find multiple gradients, need aggregate
  Node copy_node;
  copy_node.op.reset(OperatorProperty::Create("_CrossDeviceCopy"));
  copy_node.inputs = {source};
  return copy_node;
}

void StaticGraph::MakeBackwardPass(std::vector<uint32_t> *head_grad_nodes,
                                   std::vector<DataEntry>* arg_grads,
                                   std::map<uint32_t, uint32_t>* out_mirror_map) {
  // get topo order of nodes, before new nodes are added
  std::vector<uint32_t> topo_order = TopoSort();

  // build a mirror map, experimental
  std::map<uint32_t, uint32_t>& mirror_map = *out_mirror_map;
  mirror_map.clear();
  int do_mirror = dmlc::GetEnv("MXNET_BACKWARD_DO_MIRROR", 0);
  int mirror_step = dmlc::GetEnv("MXNET_BACKWARD_MIRROR_STEP", 100);
  int counter = 0;
  int *pcounter = &counter;

  auto need_mirror = [this, do_mirror, pcounter, mirror_step](uint32_t nid) {
    if (nodes[nid].is_variable()) return false;
    if (!nodes[nid].is_forward()) return false;
    std::string type = nodes[nid].op->TypeString();
    if (type == "Dropout") return false;
    if (nodes[nid].get_attr("force_mirroring", false)) return true;
    if (do_mirror == 0) return false;
    if (type == "Convolution") return false;
    if (type == "FullyConnected") return false;
    if (type == "Concat") return false;
    if (type == "SoftmaxOutput") return false;
    if (type == "CuDNNBatchNorm") return false;
    ++pcounter[0];
    if (pcounter[0] % mirror_step == 0) return false;
    return true;
  };

  for (uint32_t nid : topo_order) {
    if (need_mirror(nid)) {
      uint32_t dup_node_id = static_cast<uint32_t>(nodes.size());
      Node node(nodes[nid]);
      node.name += "_mirror";
      for (DataEntry& e : node.inputs) {
        e.source_id = mirror_map.at(e.source_id);
      }
      nodes.push_back(std::move(node));
      mirror_map[nid] = dup_node_id;
    } else {
      mirror_map[nid] = nid;
    }
  }

  // normal gradient
  arg_grads->clear();
  head_grad_nodes->clear();
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
    uint32_t mirror_nid = mirror_map[nid];
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
      DataEntry odata(mirror_nid, static_cast<uint32_t>(i));
      DataEntry okey(nid, static_cast<uint32_t>(i));
      out_data.push_back(odata);
      if (i >= nvisible) continue;
      // get out_grad
      auto it = grad_map.find(okey);
      CHECK(it != grad_map.end()) << "bad graph: Cannot find node "
                                  << nodes[nid].name << "'s " << i << "-th output";
      std::vector<DataEntry> &gnodes = it->second;
      if (gnodes.size() == 1) {
        out_grad.push_back(gnodes[0]);
      } else {
        std::ostringstream os_name;
        Node agg_node = this->CreateGradSumNode(gnodes);
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
    grad_node.backward_source_id = mirror_nid;

    std::vector<DataEntry> source_inputs;
    for (const DataEntry& e : nodes[nid].inputs) {
      source_inputs.push_back(DataEntry(mirror_map[e.source_id], e.index));
    }
    // select out the dependent inputs
    grad_node.inputs = nodes[mirror_nid].op->BackwardInputs(
        out_grad, source_inputs, out_data);

    grad_node.name = nodes[mirror_nid].name + "_backward";
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
      Node agg_node = this->CreateGradSumNode(it->second);
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
  if (attr.size() != 0) writer->WriteObjectKeyValue("attr", attr);
  CHECK_EQ(addto_index.size(), 0)
      << "Not support serializing addto_index for now";
  writer->EndObject();
}

void StaticGraph::Node::Load(dmlc::JSONReader *reader) {
  attr.clear();
  dmlc::JSONObjectReadHelper helper;
  std::string op_type_str;
  std::map<std::string, std::string> param;
  helper.DeclareField("op", &op_type_str);
  helper.DeclareField("param", &param);
  helper.DeclareField("name", &name);
  helper.DeclareField("inputs", &inputs);
  helper.DeclareField("backward_source_id", &backward_source_id);
  helper.DeclareOptionalField("attr", &attr);
  helper.ReadAllFields(reader);

  if (op_type_str != "null") {
    try {
      op.reset(OperatorProperty::Create(op_type_str.c_str()));
      std::vector<std::pair<std::string, std::string> > vec(param.begin(), param.end());
      op->Init(vec);
    } catch (const dmlc::Error &err) {
      std::ostringstream os;
      os << "Failed loading Op " << name << " of type " << op_type_str << ": " << err.what();
      throw dmlc::Error(os.str());
    }
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
