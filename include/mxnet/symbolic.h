/*!
 * Copyright (c) 2015 by Contributors
 * \file symbolic.h
 * \brief Symbolic interface of mxnet.
 * \author Min Lin, Bing Xu
 */
#ifndef MXNET_SYMBOLIC_H_
#define MXNET_SYMBOLIC_H_

#include <dmlc/base.h>
#include <dmlc/json.h>
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <utility>
#include "./base.h"
#include "./c_api.h"
#include "./ndarray.h"
#include "./operator.h"

// check c++11
#if DMLC_USE_CXX11 == 0
#error "CXX11 was required for symbolic module"
#endif

namespace mxnet {
/*!
 * \brief Internal data structure used for
 *  graph serializaion and graph algorithms.
 */
class StaticGraph;
/*!
 * \brief Symbol is used to represent dynamically generated symbolic computation graph.
 *
 *   This class is used as a tool to generate computation graphs(aka. configuration) of the network.
 *   Symbol is always composite, the head Node is the output node of the symbol.
 *   An atomic symbol can be seen as a special case of the composite symbol with only the head node.
 */
class Symbol {
 public:
  /*!
   * \brief copy the symbol
   * \return a deep copy of the graph
   */
  Symbol Copy() const;
  /*!
   * \brief print the symbol info to output stream.
   * \param os the output stream we like to print to
   */
  void Print(std::ostream &os) const; // NOLINT(*)
  /*!
   * \brief List the arguments names.
   *
   * The position of the returned list also corresponds to calling position in operator()
   * \return the arguments list of this symbol, they can be either named or unnamed (empty string).
   */
  std::vector<std::string> ListArguments() const;
  /*! \return get the descriptions of outputs for this symbol */
  std::vector<std::string> ListOutputs() const;
  /*! \return get the descriptions of auxiliary data for this symbol */
  std::vector<std::string> ListAuxiliaryStates() const;
  /*!
   * \brief get the index th element from the returned tuple.
   * \param index index of multi output
   * \return the symbol corresponds to the indexed element.
   */
  Symbol operator[] (size_t index) const;
  /*!
   * \brief Compose the symbol with arguments, this changes current symbol.
   *
   * The positional arguments passed in must be complete(contain all arguments).
   *
   * \param args positional arguments for the symbol
   * \param name name of returned symbol.
   */
  void Compose(const std::vector<Symbol>& args,
               const std::string& name);
  /*!
   * \brief Compose the symbol with arguments, this changes the current symbol.
   * The kwargs passed in can be in-complete,
   *
   * The rest of the symbols will remain the same name.
   *
   * \param kwargs keyword arguments for the symbol
   * \param name name of returned symbol.
   */
  void Compose(const std::unordered_map<std::string, Symbol>& kwargs,
               const std::string& name);
  /*!
   * \brief Get name from the symbol.
   *  This only works for symbol with outputs from single operators.
   *  For grouped sybmbol, an error will be raised.
   * \param out the output value of the name.
   */
  bool GetName(std::string* out);
  /*!
   * \brief set additional attributes of the symbol,
   *  This only works for symbol with outputs from single operators.
   *  For grouped sybmbol, an error will be raised.
   * \param key the key of the attribute
   * \param value the value of the attribute.
   */
  void SetAttr(const std::string &key, const std::string& value);
  /*!
   * \brief Get attributes from the symbol.
   *  This only works for symbol with outputs from single operators.
   *  For grouped sybmbol, an error will be raised.
   * \param key Key of the attribute.
   * \param out the output value of the attribute.
   * \return true if the attribute exists, false if the attribute do not exist.
   */
  bool GetAttr(const std::string& key, std::string* out);
  /*!
   * \brief Get attribute dictionary from the symbol and all children. Each
   *  attribute name is pre-pended with the symbol name.
   *  For grouped sybmbol, an error will be raised.
   * \return a dictionary.
   */
  std::map<std::string, std::string> ListAttr();
  /*!
   * \brief Get attribute dictionary from the symbol.
   *  This only works for symbol with outputs from single operators.
   *  For grouped sybmbol, an error will be raised.
   * \return a dictionary.
   */
  std::map<std::string, std::string> ListAttrShallow();
  /*!
   * \brief Apply the symbol as a function, compose with arguments
   * \param args positional arguments for the symbol
   * \param name name of returned symbol.
   * \return a new Symbol which is the composition of current symbol with its arguments
   */
  Symbol operator () (const std::vector<Symbol>& args, const std::string& name) const;
  /*!
   * \brief compose with named arguments
   * \param kwargs keyword arguments for the symbol
   * \param name name of returned symbol.
   * \return a new symbol which is the composition of current symbol with its arguments
   */
  Symbol operator () (const std::unordered_map<std::string, Symbol>& kwargs,
                      const std::string& name) const;
  /*
   * \brief Get all the internal nodes of the symbol.
   * \return symbol A new symbol whose output contains all the outputs of the symbols
   *  Including input variables and intermediate outputs.
   */
  Symbol GetInternals() const;
  /*!
   * \brief get the gradient graph
   * \param wrt with respect to the input
   * \return the new symbol with gradient graph
   */
  Symbol Grad(const std::vector<std::string>& wrt) const;
  /*!
   * \brief infer the shapes of outputs and unknown input arguments
   * \param arg_shapes the shape of input arguments of the operator
   *     this should be of same length as the vector returned by ListArguments
   *     in_shape allows unknown elements, which are checked by shape.ndim() == 0.
   *     For unknown shapes, InferShape will try to fill in the correct Shape in in_shape
   *     For known shapes, InferShape will check shape consistency
   *
   *     common practice: set the shape of data input, and usually weight's shape can be inferred
   *
   * \param out_shapes Use to store the inferred shapes of outputs.
   * \param aux_shapes Use to store the inferred shapes of auxiliary states
   * \param partial_infer Return partially inferred results if true.
   * \return true if the shape inference is successful, false if there is not enough information.
   * \throws dmlc::Error if the known arg_shapes are inconsistent.
   */
  bool InferShape(std::vector<TShape> *arg_shapes,
                  std::vector<TShape> *out_shapes,
                  std::vector<TShape> *aux_shapes,
                  bool partial_infer = false) const;

  /*!
   * \brief infer the shapes by providing shapes of known arguments.
   * \param known_arg_shapes map of argument name to shape of arguments with known shapes.
   * \param arg_shapes used to store inferred shapes of arguments.
   * \param out_shapes used to store inferred shapes of outputs.
   * \param aux_shapes Use to store the inferred shapes of auxiliary states
   * \param partial_infer Return partially inferred results if true.
   * \return true if the shape inference is successful, false if there is not enough information.
   * \throws dmlc::Error if the known arg_shapes are inconsistent.
   */
  bool InferShape(const std::unordered_map<std::string, TShape> &known_arg_shapes,
                  std::vector<TShape> *arg_shapes,
                  std::vector<TShape> *out_shapes,
                  std::vector<TShape> *aux_shapes,
                  bool partial_infer = false) const;

  /*!
   * \brief infer the types of outputs and unknown input arguments
   * \param arg_types the type of input arguments of the operator
   *     this should be of same length as the vector returned by ListArguments
   *     in_type allows unknown elements, which are checked by type.ndim() == 0.
   *     For unknown types, Infertype will try to fill in the correct type in in_type
   *     For known types, Infertype will check type consistency
   *
   *     common practice: set the type of data input, and usually weight's type can be inferred
   *
   * \param out_types Use to store the inferred types of outputs.
   * \param aux_types Use to store the inferred types of auxiliary states
   * \return true if the type inference is successful, false if there is not enough information.
   * \throws dmlc::Error if the known arg_types are inconsistent.
   */
  bool InferType(std::vector<int> *arg_types,
                  std::vector<int> *out_types,
                  std::vector<int> *aux_types) const;
  /*!
   * \brief infer the types by providing types of known arguments.
   * \param known_arg_types map of argument name to type of arguments with known types.
   * \param arg_types used to store inferred types of arguments.
   * \param out_types used to store inferred types of outputs.
   * \param aux_types Use to store the inferred types of auxiliary states
   * \return true if the type inference is successful, false if there is not enough information.
   * \throws dmlc::Error if the known arg_types are inconsistent.
   */
  bool InferType(const std::unordered_map<std::string, int> &known_arg_types,
                  std::vector<int> *arg_types,
                  std::vector<int> *out_types,
                  std::vector<int> *aux_types) const;
  /*!
   * \brief interface for json serialization.
   * \param writer the JSON writer write json.
   */
  void Save(dmlc::JSONWriter *writer) const;
  /*!
   * \brief interface for json serialization.
   * \param reader the JSON read to read json.
   */
  void Load(dmlc::JSONReader *reader);
  /*!
   * \brief get number of outputs of this symbol
   * \return number of outputs
   */
  inline size_t NumOutputs() const {
    return heads_.size();
  }
  /*!
   * \brief create Symbol by wrapping OperatorProperty
   * This function takes the ownership of op
   *
   * \param op the OperatorProperty of the Operator
   * \return Symbol
   * \sa OperatorProperty::Create
   */
  static Symbol Create(OperatorProperty *op);
  /*!
   * \brief create equivalence of symbol by grouping the symbols together
   * \param symbols list of symbols
   * \return the grouped symbol
   */
  static Symbol CreateGroup(const std::vector<Symbol> &symbols);
  /*!
   * \brief create variable symbol node
   * \param name name of the variable
   * \return the new variable
   */
  static Symbol CreateVariable(const std::string &name);

 protected:
  // Declare node, internal data structure.
  struct Node;
  /*! \brief an entry that represents output data from a node */
  struct DataEntry {
    /*! \brief the source node of this data */
    std::shared_ptr<Node> source;
    /*! \brief index of output from the source. */
    uint32_t index;
    /*! \brief enabled default copy constructor */
    DataEntry() {}
    /*! \brief constructor from index */
    DataEntry(std::shared_ptr<Node> source, uint32_t index)
        : source(source), index(index) {}
  };
  /*!
   * \brief the head nodes of Symbols
   * This head is only effective when
   */
  std::vector<DataEntry> heads_;

 private:
  /*! \return whwther the symbol is atomic */
  inline bool is_atomic() const;
  /*!
   * \brief Visit all the nodes in left-to-right depth first order.
   *
   *  This function will visit the graph in DFS order, call fvisit exactly once
   *  for each Node, and store the result in out_result.
   *
   * \param fvisit function applied for each visit.
   * \tparam FVisit visiting function type
   */
  template<typename FVisit>
  inline void DFSVisit(FVisit fvisit) const;
  /*!
   * \brief Find duplicate arguments in the composition
   * \param out the map of argument-name -> occurence count
   * \return maximum number of duplication factor
   */
  int FindDuplicateArgs(std::unordered_map<std::string, int> *out) const;
  /*!
   * \brief Convert symbol into internal static graph
   *
   * \param out_graph the pointer holder of the output graph
   */
  void ToStaticGraph(StaticGraph *out_graph) const;
  /*!
   * \brief create equivalence of symbol from static graphs.
   *  This operation will change the content of current symbol.
   * \param graph the static graph
   */
  void FromStaticGraph(const StaticGraph &graph);
  /*! \brief let static graph know the contents */
  friend class StaticGraph;
};

/*!
 * \brief Executor of a computation graph.
 *  Executor can be created by Binding a symbol.
 */
class Executor {
 public:
  /*! \brief destructor */
  virtual ~Executor() {}
  /*!
   * \brief Perform a Forward operation of Operator
   *  After this operation, user can get the result by using function head.
   */
  virtual void Forward(bool is_train) = 0;
  /*!
   * \brief Perform a Partial Forward operation of Operator.
   *  Only issue operation specified by step.
   *  The caller must keep calling PartialForward with increasing steps, until step_left=0.
   * \param is_train Whether this is training phase.
   * \param step current step, user can always start from 0
   * \param step_left Number of steps left to finish the forward.
   */
  virtual void PartialForward(bool is_train, int step, int *step_left) = 0;
  /*!
   * \brief Perform a Backward operation of the Operator.
   *  This must be called after Forward.
   *  After this operation, NDArrays specified by grad_in_args_store will be updated accordingly.
   *  User is allowed to pass in an empty Array if the head node is
   *  loss function and head gradeitn is not needed.
   *
   * \param head_grads the gradient of head nodes to be backproped.
   */
  virtual void Backward(const std::vector<NDArray> &head_grads) = 0;
  /*!
   * \brief print the execution plan info to output stream.
   * \param os the output stream we like to print to.
   */
  virtual void Print(std::ostream &os) const {} // NOLINT(*)
  /*!
   * \brief get array of outputs in the executor.
   * \return array of outputs in the executor.
   */
  virtual const std::vector<NDArray> &outputs() const = 0;
  /*!
   * \brief Create an operator by bind symbol with context and arguments.
   *  If user do not want to compute the gradients of i-th argument, grad_req_type[i] can be kNullOp.
   *
   * \param default_ctx the default context of binding.
   * \param group2ctx Context mapping group to context.
   * \param symbol the symbol that specifies the output of Forward pass.
   * \param in_args the NDArray that stores the input arguments to the symbol.
   * \param arg_grad_store NDArray that is used to store the gradient output of the input arguments.
   * \param grad_req_type requirment type of gradient saving. Can only be in {kNullOp, kAddTo, kWriteTo}.
   * \param aux_states NDArray that is used as internal state in op
   * \param shared_exec input executor to share memory with.
   * \return a new executor.
   */
  static Executor *Bind(Symbol symbol,
                        const Context& default_ctx,
                        const std::map<std::string, Context>& group2ctx,
                        const std::vector<NDArray> &in_args,
                        const std::vector<NDArray> &arg_grad_store,
                        const std::vector<OpReqType> &grad_req_type,
                        const std::vector<NDArray> &aux_states,
                        Executor* shared_exec = NULL);
  /*!
   * \brief the prototype of user-defined monitor callback
   */
  typedef std::function<void(const char*, void*)> MonitorCallback;
  /*!
   * \brief Install a callback to notify the completion of operation.
   */
  virtual void SetMonitorCallback(const MonitorCallback& callback) {}
};  // class operator
}  // namespace mxnet
#endif  // MXNET_SYMBOLIC_H_
