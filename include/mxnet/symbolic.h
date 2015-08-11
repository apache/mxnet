/*!
 * Copyright (c) 2015 by Contributors
 * \file symbolic.h
 * \brief
 * \author Bing Xu
*/

#ifndef MXNET_SYMBOLIC_H_
#define MXNET_SYMBOLIC_H_

#include <vector>
#include <memory>
#include <string>
#include <map>
#if DMLC_USE_CXX11 == 1
#include <unordered_map>
#include <unordered_set>
#endif
#include "./base.h"

namespace mxnet {
// forward declare StaticOperator
class StaticOperator;
/*!
 * \brief AtomicSymbol is the base class of all atomic symbols.
 *  This is not meant to be used by user, it should be wrapped in Symbol, so that the same instance
 *  of AtomicSymbol can be shared in the graphs of different Symbols
 */
class AtomicSymbol {
 public:
  /*!
   * \brief virtual destructor
   */
  virtual ~AtomicSymbol() {}
  /*! \brief get the descriptions of inputs for this symbol */
  virtual std::vector<std::string> ListArguments() const {
    // default implementation returns "data"
    return std::vector<std::string>(1, std::string("data"));
  }
  /*! \brief get the descriptions of outputs for this symbol */
  virtual std::vector<std::string> ListReturns() const {
    // default implementation returns "output"
    return std::vector<std::string>(1, std::string("output"));
  }
  /*! \brief number of outputs of the symbol */
  virtual int NumReturns() const {
    return 1;
  }
  /*!
   *  \brief set param for the symbol from string
   *  \param name parameter name
   *  \param val string for the configuration
   */
  virtual void SetParam(const char *name, const char *val) {}
  /*!
   * \brief infer the shapes of outputs and unknown input arguments
   * \param in_shape the shape of input arguments of the operator
   *     this should be of same length as the vector returned by DescribeArgs
   *     in_shape allows unknown elements, which are checked by shape.ndim() == 0.
   *     For unknown shapes, InferShape will try to fill in the correct Shape in in_shape
   *     For known shapes, InferShape will check shape consistency
   *
   *     common practice: set the shape of data input, and usually weight's shape can be infered
   *
   * \param out_shape the shape of outputs of the operator
   *     InferShape will modify the vector to fill output TShape
   * \return if the shape inference is successful, return true, else return false.
   */
  virtual bool InferShape(std::vector<TShape> *in_shape, std::vector<TShape> *out_shape) const = 0;
  /*!
   * \brief Copy this AtomicSymbol and returns a pointer to the copied object.
   *  this is a virtual function because different subclass of AtomicSymbol would copy differently.
   * \return a pointer to the copied atomic symbol
   */
  virtual AtomicSymbol* Copy() const = 0;
  /*!
   * \brief Bind this AtomicSymbol to a context and get back a static operator
   *  Bind function of AtomicSymbol does not return NArrayOperator, but static operator.
   *  Calling bind from the Symbol wrapper would generate a NArrayOperator.
   */
  template<typename xpu>
  StaticOperator* Bind(Context ctx) const;
  /*!
   * \brief return the type string of the atomic symbol
   *  subclasses override this function.
   */
  virtual std::string TypeString() const = 0;
  friend class Symbol;

  /*!
   * \brief create atomic symbol by type name
   * \param type_name the type string of the AtomicSymbol
   * \return a new constructed AtomicSymbol
   */
  static AtomicSymbol *Create(const char* type_name);
};
#if DMLC_USE_CXX11 == 1
/*!
 * \brief StaticGraph is the configuration of computation graphs.
 *  This is the "configuration file" of mxnet.
 *  It can be converted to/from Symbol, and can be used to bind to operators.
 */
class StaticGraph {
 public:
  /*! \brief represents a data in the graph */
  struct DataEntry {
    /*! \brief the source node id in the computation graph */
    uint32_t source_id;
    /*!
     * \brief index of output from the source.
     * If index == -1, it represents all the outputs.
     */
    int32_t index;
  };
  /*! \brief Operation Node in static graph */
  struct Node {
    /*! \brief wrapped atomic symbol */
    std::unique_ptr<AtomicSymbol> sym;
    /*! \brief name of the node */
    std::string name;
    /*! \brief inputs (node_id, index) for of the nodes*/
    std::vector<DataEntry> inputs;
  };
  /*! \brief all nodes in the graph */
  std::vector<Node> nodes;
  /*! \brief index is nodes that correspods to arguments */
  std::vector<uint32_t> arg_nodes;
  /*! \brief outputs(heads) of the graph */
  std::vector<DataEntry> outputs;
  // funtions to help inference in static graph
  /*!
   * \brief Perform a topological sort on the graph
   * \return a topological order of node indices.
   */
  std::vector<uint32_t> TopoSort() const;
  /*!
   * \brief infer the node shapes in the computation graph.
   *
   *  When calling this function, user can setup the shape information known into right position.
   *  Unknown shape are indicated by shape.ndim() == 0.
   *
   * \param topo_order The topological order of node index, as created by TopoSort.
   * \param node_out_shapes The shapes of the each outputs of nodes in the graph.
   * \return if the shape inference is successful, return true, else return false.
   */
  bool InferNodeShapes(const std::vector<uint32_t> &topo_order,
                       std::vector<std::vector<TShape> > *node_out_shapes) const;
  /*!
   * \brief infer the shapes of outputs and unknown input arguments
   * \param in_shape the shape of input arguments of the operator
   *     this should be of same length as the vector returned by ListArguments
   *     in_shape allows unknown elements, which are checked by shape.ndim() == 0.
   *     For unknown shapes, InferShape will try to fill in the correct Shape in in_shape
   *     For known shapes, InferShape will check shape consistency
   *
   *     common practice: set the shape of data input, and usually weight's shape can be infered
   *
   * \param out_shape the shape of outputs of the operator
   *     InferShape will modify the vector to fill output TShape
   * \return if the shape inference is successful, return true, else return false.
   */
  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape) const;
};
#endif
#if DMLC_USE_CXX11 == 1
/*!
 * \brief Symbol is used to represent dynamically generated symbolic computation graph.
 *
 *   This class is used as a tool to generate computation graphs(aka. configuration) of the network.
 *   Symbol is always composite, the head Node is the output node of the symbol.
 *   An atomic symbol can be seen as a special case of the composite symbol with only the head node.
 *
 *   The symbol can be converted from/to StaticGraph, the actual configuration used by mxnet.
 *   Symbol offers more flexible way to composite nodes than StaticGraph, which makes it good
 *   tool to generate configurations from language bindings such as python.
 * \sa StaticGraph
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
  std::vector<std::string> ListReturns() const;
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
   * \brief Convert a list of symbols into static graph
   *
   *  The user can go further to call bind function on static graph
   *
   * \param out_graph the pointer holder of the output graph
   */
  void ToStaticGraph(StaticGraph *out_graph) const;
  /*!
   * \brief Apply the symbol as a function, compose with arguments
   * \param args positional arguments for the symbol
   * \param name name of returned symbol.
   * \return a new Symbol which is the composition of current symbol with its arguments
   */
  inline Symbol operator () (const std::vector<Symbol>& args,
                             const std::string& name) const {
    Symbol s = this->Copy();
    s.Compose(args, name);
    return s;
  }
  /*!
   * \brief compose with named arguments
   * \param kwargs keyword arguments for the symbol
   * \param name name of returned symbol.
   * \return a new symbol which is the composition of current symbol with its arguments
   */
  inline Symbol operator () (const std::unordered_map<std::string, Symbol>& kwargs,
                             const std::string& name) const {
    Symbol s = this->Copy();
    s.Compose(kwargs, name);
    return s;
  }
  /*!
   * \brief infer the shapes of outputs and unknown input arguments
   * \param in_shape the shape of input arguments of the operator
   *     this should be of same length as the vector returned by ListArguments
   *     in_shape allows unknown elements, which are checked by shape.ndim() == 0.
   *     For unknown shapes, InferShape will try to fill in the correct Shape in in_shape
   *     For known shapes, InferShape will check shape consistency
   *
   *     common practice: set the shape of data input, and usually weight's shape can be infered
   *
   * \param out_shape the shape of outputs of the operator
   *     InferShape will modify the vector to fill output TShape
   * \return if the shape inference is successful, return true, else return false.
   */
  inline bool InferShape(std::vector<TShape> *in_shape,
                         std::vector<TShape> *out_shape) const {
    StaticGraph g;
    this->ToStaticGraph(&g);
    return g.InferShape(in_shape, out_shape);
  }
  /*!
   * \brief get number of outputs of this symbol
   * \return number of outputs
   */
  inline size_t NumReturns() const {
    return heads_.size();
  }
  /*!
   * \brief create Symbol by wrapping AtomicSymbol
   * This function takes the ownership of atomic_symbol.
   *
   * \param atomic_symbol the AtomicSymbol
   * \return Symbol
   * \sa AtomicSymbol::Create
   */
  static Symbol Create(AtomicSymbol *atomic_symbol);
  /*!
   * \brief create equivalence of symbol from static graphs
   * \param graph the static graph
   * \return the created symbol
   */
  static Symbol Create(const StaticGraph &graph);

  /*!
   * \brief create equivalence of symbol by grouping the symbols together
   * \param symbols list of symbols
   * \return the grouped symbol
   */
  static Symbol CreateGroup(const std::vector<Symbol> &symbols) {
    Symbol ret;
    for (const auto &s : symbols) {
      ret.heads_.insert(ret.heads_.end(), s.heads_.begin(), s.heads_.end());
    }
    return std::move(ret);
  }
  /*!
   * \brief create variable symbol node
   * \param name name of the variable
   * \return the new variable
   */
  inline static Symbol CreateVariable(const std::string &name) {
    Symbol s;
    s.heads_.push_back(DataEntry(std::make_shared<Node>(nullptr, name), 0));
    return std::move(s);
  }

 protected:
  // forward declare Node
  struct Node;
  /*! \brief an entry that represents output data from a node */
  struct DataEntry {
    /*! \brief the source node of this data */
    std::shared_ptr<Node> source;
    /*!
     * \brief index of output from the source.
     */
    uint32_t index;
    /*! \brief enabled default copy constructor */
    DataEntry() {}
    /*! \brief constructor from index */
    DataEntry(std::shared_ptr<Node> source, uint32_t index)
        : source(source), index(index) {}
  };
  /*!
   * \brief Node is represents node of an operator in the symbolic graph.
   *
   * It stores connection to the inputs to function represented by AtomicSymbol
   * NOTE on data structure: there are three types of node:
   * - Normal node: contains all the necessary elements of a graph.
   * - AtomicSymbol: the inputs_ is empty, represents an AtomicSymbol that has not been applied.
   * - Variable: the sym_ is nullptr, represents an named Variable of tensors that can be composed.
   */
  struct Node {
    /*! \brief wrapped atomic symbol */
    std::unique_ptr<AtomicSymbol> sym;
    /*! \brief name of the node */
    std::string name;
    /*! \brief inputs to this node */
    std::vector<DataEntry> inputs;
    /*!
     * \brief constructor
     * \param sym the AtomicSymbol to construct the symbol
     * \param name the name of the symbol
     */
    explicit Node(AtomicSymbol* sym = nullptr, const std::string& name = "")
        : sym(sym), name(name) {
    }
    /*! \return Whether the symbol is AtomicSymbol */
    inline bool is_atomic() const {
      return inputs.size() == 0 && sym != nullptr;
    }
    /*! \return Whetehr the symbolc is a PlaceHolder */
    inline bool is_variable() const {
      return sym == nullptr;
    }
  };
  /*!
   * \brief the head nodes of Symbols
   * This head is only effective when
   */
  std::vector<DataEntry> heads_;
  /*! \return whwther the symbol is AtomicSymbol */
  inline bool is_atomic() const {
    return heads_.size() == 1 && heads_[0].source->is_atomic();
  }
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
  inline void DFSVisit(FVisit fvisit) const {
    std::vector<Node*> stack;
    std::unordered_set<Node*> visited;
    // put the head into the graph
    for (auto &head : heads_) {
      Node *ptr = head.source.get();
      if (visited.count(ptr) == 0) {
        stack.push_back(ptr);
        visited.insert(ptr);
      }
    }
    while (!stack.empty()) {
      Node* back = stack.back();
      stack.pop_back();
      fvisit(back);
      for (auto it = back->inputs.rbegin(); it != back->inputs.rend(); ++it) {
        Node *ptr = it->source.get();
        if (visited.count(ptr) == 0) {
          stack.push_back(ptr);
          visited.insert(ptr);
        }
      }
    }
  }
  /*!
   * \brief Find duplicate arguments in the composition
   * \param out the map of argument-name -> occurence count
   * \return maximum number of duplication factor
   */
  int FindDuplicateArgs(std::unordered_map<std::string, int> *out) const;
};
#endif
}  // namespace mxnet
#endif  // MXNET_SYMBOLIC_H_
