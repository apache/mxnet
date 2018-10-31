/*!
 *  Copyright (c) 2016 by Contributors
 * \file nnvm/pass_functions.h
 * \brief Pass functions that simply redirect the calls to ApplyPass
 *
 *  This file serves as documentation on how to use functions implemented in "src/pass".
 *  It is totally optional to add these functions when you add a new pass, since
 *  ApplyPass can be directly called.
 */
#ifndef NNVM_PASS_FUNCTIONS_H_
#define NNVM_PASS_FUNCTIONS_H_

#include <string>
#include <memory>
#include <vector>
#include "base.h"
#include "pass.h"
#include "graph_attr_types.h"

namespace nnvm {
namespace pass {

/*!
 * \brief Load a graph from JSON string, redirects to "LoadJSON" pass.
 * \param json_str The json string.
 * \return Loaded graph.
 */
inline Graph LoadJSON(const std::string& json_str) {
  Graph ret;
  ret.attrs["json"] = std::make_shared<any>(json_str);
  return ApplyPass(ret, "LoadJSON");
}

/*!
 * \brief Save a graph to json, redirects to "SaveJSON" pass.
 * \param graph The graph to be saved as json format.
 * \return The json string.
 */
inline std::string SaveJSON(Graph graph) {
  Graph ret = ApplyPass(std::move(graph), "SaveJSON");
  return ret.GetAttr<std::string>("json");
}


/*!
 * \brief Print graph ir
 * \param graph The graph to be printed
 * \return The graph ir string.
 */
inline std::string PrintGraphIR(Graph graph) {
  Graph ret = ApplyPass(std::move(graph), "PrintGraphIR");
  return ret.GetAttr<std::string>("graphir");
}

/*!
 * \brief Add control flow dependencies between nodes.
 *
 *  This function will enforce the correct order between
 *  write (mutable operators) and read (immutable operators)
 *  to sovle write-after-read and read-after-write problems.
 *
 * \param src The input graph.
 * \return A graph with proper control flow dependencies added.
 */
inline Graph OrderMutation(Graph src) {
  return ApplyPass(std::move(src), "OrderMutation");
}

/*!
 * \brief Infer shapes in the graph given the information.
 * \param graph The input graph.
 * \param shape_inputs The shapes of input symbols to the graph.
 * \param shape_attr_key The key to the node attribute that can indicate shape. This is
 *                       the place where manual hint for shapes could be injected.
 * \return A graph with new attribute "shape" containing inferred shape of each NodeEntry.
 *         The index of ShapeVector is given by graph.indexed_graph().entry_id.
 */
inline Graph InferShape(Graph graph,
                        ShapeVector shape_inputs,
                        std::string shape_attr_key = "") {
  if (shape_inputs.size() != 0) {
    graph.attrs["shape_inputs"] = std::make_shared<any>(std::move(shape_inputs));
  }
  if (shape_attr_key.length() != 0) {
    graph.attrs["shape_attr_key"] = std::make_shared<any>(std::move(shape_attr_key));
  }
  return ApplyPass(std::move(graph), "InferShape");
}

/*!
 * \brief Infer types in the graph given the information.
 * \param graph The input graph.
 * \param dtype_inputs The types of input symbols to the graph.
 * \param dtype_attr_key The key to the node attribute that can indicate types. This is
 *                       the place where manual hint for types could be injected.
 * \return A graph with new attribute "dtype" containing inferred type of each NodeEntry.
 *         The index of ShapeVector is given by graph.indexed_graph().entry_id.
 */
inline Graph InferType(Graph graph,
                       DTypeVector dtype_inputs,
                       std::string dtype_attr_key = "") {
  if (dtype_inputs.size() != 0) {
    graph.attrs["dtype_inputs"] = std::make_shared<any>(std::move(dtype_inputs));
  }
  if (dtype_attr_key.length() != 0) {
    graph.attrs["dtype_attr_key"] = std::make_shared<any>(std::move(dtype_attr_key));
  }
  return ApplyPass(std::move(graph), "InferType");
}

/*!
 * \brief Place the devices for each operator in the graph.
 *
 *  Current device placement is quite simple. Each operator is assigned to a "group" (stored
 *  in `device_group_attr_key` attribute). Each group is assigned to a device (stored in
 *  `device_assign_map` attribute). Operators will be placed to the device assigned to its
 *  group. Copy operators will be injected if cross device reference happens.
 *
 * \param graph The input graph.
 * \param device_group_attr_key The attribute name for hints of device group.
 * \param device_assign_map The assignment map of device.
 * \param device_copy_op The name of copy op to be inserted when cross device copy happened.
 * \return A graph with new attribute "device", cotaining device information of each node.
 */
inline Graph PlaceDevice(Graph graph,
                         std::string device_group_attr_key,
                         DeviceAssignMap device_assign_map,
                         std::string device_copy_op) {
  graph.attrs["device_group_attr_key"] = std::make_shared<any>(std::move(device_group_attr_key));
  graph.attrs["device_assign_map"] = std::make_shared<any>(std::move(device_assign_map));
  graph.attrs["device_copy_op"] = std::make_shared<any>(std::move(device_copy_op));
  return ApplyPass(std::move(graph), "PlaceDevice");
}

/*!
 * \brief Get the gradient graph whose outputs are gradients of xs wrt to ys.
 * \param graph The input graph.
 * \param ys The entries we want to take gradient from.
 * \param xs The input to take gradient with respect to.
 * \param ys_out_grad The symbol for additional gradient to be propagate back to y.
 * \param aggregate_fun Aggregation function applied to aggregate the inputs.
 * \param mirror_fun Optional mirror function to do mirror optimization and save memory.
 * \param attr_hint_fun Optional, hint function to output a node that like src, but its attr is same as like.
 * \param zero_ops Optional, list of operators that outputs a single zero array. The first one
 *  must be zeros_like.
 * \param copy_op_str Optional, name of the copy operation required to handle duplicates
 *  on the edge of the graph
 * \return A new graph, whose outputs correspond to inputs of xs.
 */
inline Graph Gradient(
    Graph graph,
    std::vector<NodeEntry> ys,
    std::vector<NodeEntry> xs,
    std::vector<NodeEntry> ys_out_grad,
    std::function<NodeEntry(std::vector<NodeEntry>&& inputs)> aggregate_fun = nullptr,
    std::function<int(const Node& node)> mirror_fun = nullptr,
    std::function<NodeEntry(const NodeEntry& src, const NodeEntry &like)>
    attr_hint_fun = nullptr,
    std::vector<const Op*> zero_ops = std::vector<const Op*>(),
    std::string copy_op_str = std::string()) {
  graph.attrs["grad_ys"] = std::make_shared<any>(std::move(ys));

  graph.attrs["grad_xs"] = std::make_shared<any>(std::move(xs));
  graph.attrs["grad_ys_out_grad"] = std::make_shared<any>(std::move(ys_out_grad));
  if (aggregate_fun != nullptr) {
    graph.attrs["grad_aggregate_fun"] = std::make_shared<any>(aggregate_fun);
  }

  if (mirror_fun != nullptr) {
    graph.attrs["grad_mirror_fun"] = std::make_shared<any>(mirror_fun);
  }

  if (attr_hint_fun != nullptr) {
    graph.attrs["attr_hint_fun"] = std::make_shared<any>(attr_hint_fun);
  }

  if (zero_ops.size()) {
    graph.attrs["zero_ops"] = std::make_shared<any>(std::move(zero_ops));
  }

  if (copy_op_str != std::string()) {
      graph.attrs["copy_op"] = std::make_shared<any>(std::move(copy_op_str));
  }

  return ApplyPass(std::move(graph), "Gradient");
}

}  // namespace pass
}  // namespace nnvm
#endif  // NNVM_PASS_FUNCTIONS_H_
