/*!
 *  Copyright (c) 2016 by Contributors
 * \file nnvm/pass.h
 * \brief Pass that can be applied to a graph.
 */
#ifndef NNVM_PASS_H_
#define NNVM_PASS_H_

#include <vector>
#include <functional>
#include "base.h"
#include "graph.h"

namespace nnvm {

/*!
 * \brief A PassFunction is an "Operator on Graph".
 *  It takes a source graph and return a graph that may or may
 *  not be the same as the input one.
 *
 *  A pass function can either change the graph structure (thus,
 *  generating a new Graph), or add new attributes to the graph.
 *
 * \param src The graph to be transformed.
 * \return The generated graph.
 */
typedef std::function<Graph (Graph src)> PassFunction;

/*!
 * \brief Apply a series of pass transformations on the input graph.
 * \param src The graph to be transformed.
 * \param passes A list of pass names to be applied.
 * \return The transformed graph
 */
Graph ApplyPasses(Graph src,
                  const std::vector<std::string>& passes);

/*!
 * \brief Apply one pass to the graph.
 * \param src The graph to be transformed.
 * \param pass The name of pass to be applied.
 * \return The transformed graph.
 */
inline Graph ApplyPass(Graph src, const std::string& pass) {
  return ApplyPasses(src, {pass});
}


/*!
 * \brief Registry entry for pass functions.
 */
struct PassFunctionReg
    : public dmlc::FunctionRegEntryBase<PassFunctionReg,
                                        PassFunction> {
  /*!
   * \brief Whether the pass will change graph structure
   *  If this is false, the pass will only change attributes.
   */
  bool change_graph{false};
  /*! \brief dependencies on operator attributes */
  std::vector<std::string> op_attr_dependency;
  /*! \brief dependencies on attributes in the graph */
  std::vector<std::string> graph_attr_dependency;
  /*! \brief generated targets of graph attributes */
  std::vector<std::string> graph_attr_targets;
  /*!
   * \brief Set whether this pass will change graph structure.
   * \param v If true, the pass will change graph structure.
   * \return Reference to self.
   */
  PassFunctionReg& set_change_graph(bool v) {  // NOLINT(*)
    change_graph = v;
    return *this;
  }
  /*!
   * \brief Declare that this pass will generate the given graph attribute name
   *        once it is applied on the graph.
   * \param attr_name Name of the graph attribute.
   * \return Reference to self.
   */
  PassFunctionReg& provide_graph_attr(const std::string& attr_name) {  // NOLINT(*)
    graph_attr_targets.push_back(attr_name);
    return *this;
  }
  /*!
   * \brief Declare this pass requires the given operator attribute to be
   *        available before being applied on the graph.
   * \param attr_name Name of the attribute.
   * \return Reference to self.
   */
  PassFunctionReg& depend_op_attr(const std::string& attr_name) {  // NOLINT(*)
    op_attr_dependency.push_back(attr_name);
    return *this;
  }
  /*!
   * \brief Declare this pass requires the given graph attribute to be
   *        available before being applied on the graph.
   * \param attr_name Name of the attribute.
   * \return Reference to self.
   */
  PassFunctionReg& depend_graph_attr(const std::string& attr_name) {  // NOLINT(*)
    graph_attr_dependency.push_back(attr_name);
    return *this;
  }
};

/*!
 * \def NNVM_REGISTER_PASS
 * \brief Macro to register pass fuctions.
 *
 * \code
 * // example of registering a shape inference pass
 * NNVM_REGISTER_PASS(InferShape)
 * .describe("Shape Inference function, generate graph attributes")
 * .provide_graph_attr("data_shape")
 * .depend_graph_attr("indexed_graph")
 * .depend_op_attr("infer_shape")
 * .set_body([](const Graph& g) {
 *     // shape inference logic
 *   });
 * \endcode
 */
#define NNVM_REGISTER_PASS(name)                                     \
  DMLC_REGISTRY_REGISTER(::nnvm::PassFunctionReg, PassFunctionReg, name)

}  // namespace nnvm

#endif  // NNVM_PASS_H_
