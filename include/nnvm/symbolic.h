/*!
 *  Copyright (c) 2016 by Contributors
 * \file nnvm/symbolic.h
 * \brief Symbolic graph construction API
 *
 *  This API is optional, but useful to allow user
 *  to construct NNVM Graph easily, and quickly create
 *  front-end host languages.
 */
#ifndef NNVM_SYMBOLIC_H_
#define NNVM_SYMBOLIC_H_

#include <string>
#include <vector>
#include <tuple>
#include <utility>

#include "base.h"
#include "node.h"

namespace nnvm {
/*!
 * \brief Symbol is help class used to represent the operator node in Graph.
 *
 *  Symbol acts as an interface for building graphs from different components
 *  like Variable, Functor and Group. Symbol is also exported to python front-end
 *  (while Graph is not) to enable quick test and deployment. Conceptually,
 *  symbol is the final operation of a graph and thus including all the information
 *  required (the graph) to evaluate its output value.
 */
class NNVM_DLL Symbol {
 public:
  /*! \brief option passed to ListAttr */
  enum ListAttrOption {
    /*! \brief recursively list all attributes */
    kRecursive = 0,
    /*! \brief only list attributes in current node */
    kShallow = 1
  };
  /*! \brief option passed to ListInputNames */
  enum ListInputOption {
    /*! \brief list all the arguments */
    kAll = 0,
    /*! \brief list only read only arguments */
    kReadOnlyArgs = 1,
    /*!
     * \brief List auxiliary states that can be mutated by the graph.
     *  This excludes the ReadOnly arguments
     */
    kAuxiliaryStates = 2
  };

  /*! \brief output entries contained in the symbol */
  std::vector<NodeEntry> outputs;

  /*!
   * \brief Copy the symbol.
   * \return A deep copy of this symbol.
   */
  Symbol Copy() const;
  /*!
   * \brief Print the symbol info to output stream.
   * \param os The output stream to print to.
   */
  void Print(std::ostream &os) const; // NOLINT(*)
  /*!
   * \brief Get the index-th element from the returned tuple.
   * \param index Index of multi output.
   * \return The symbol corresponds to the indexed element.
   */
  Symbol operator[] (size_t index) const;
  /*!
   * \brief List the input variable nodes.
   *
   *  The order of the returned list is the same as the order of the input list to `operator()`.
   *
   * \param option The options to list the arguments.
   * \return The arguments list of this symbol, they can be either named or unnamed (empty string).
   * \sa ListInputOption
   */
  std::vector<NodePtr> ListInputs(ListInputOption option) const;
  /*!
   * \brief List the input names.
   *
   *  The order of the returned list is the same as the order of the input list to `operator()`.
   *
   * \param option The options to list the arguments.
   * \return The arguments list of this symbol, they can be either named or unnamed (empty string).
   * \sa ListInputOption
   */
  std::vector<std::string> ListInputNames(ListInputOption option) const;
  /*!
   * \brief List the names of outputs for this symbol.
   *
   *  For normal operators, it is usually symbol node name + "_output".
   *
   * \return get the descriptions of outputs for this symbol.
   */
  std::vector<std::string> ListOutputNames() const;
  /*!
   * \brief Compose the symbol with arguments, this changes the current symbol.
   * The kwargs passed in can be in-complete,
   *
   * The rest of the symbols will remain the same name.
   *
   * \param args Positional arguments.
   * \param kwargs Keyword arguments for the symbol.
   * \param name Name of returned symbol.
   */
  void Compose(const array_view<const Symbol*>& args,
               const std::unordered_map<std::string, const Symbol*>& kwargs,
               const std::string& name);
  /*!
   * \brief Apply the symbol as a function, compose with arguments
   *
   *  This is equivalent to Copy then Compose.
   *
   * \param args Positional arguments for the symbol.
   * \param kwargs Keyword arguments for the symbol.
   * \param name Name of returned symbol.
   * \return A new Symbol which is the composition of current symbol with its arguments.
   */
  Symbol operator () (const array_view<const Symbol*>& args,
                      const std::unordered_map<std::string, const Symbol*>& kwargs,
                      const std::string& name) const;
  /*!
   * \brief Add control flow dependencies to the operators in symbols.
   *
   *  For grouped symbol, an error will be raised. This mutates current symbolic Node.
   *
   * \param src The symbols to depend on.
   */
  void AddControlDeps(const Symbol& src);
  /*
   * \brief Get all the internal nodes of the symbol.
   * \return symbol A new symbol whose output contains all the outputs of the symbols
   *                including input variables and intermediate outputs.
   */
  Symbol GetInternals() const;
  /*
   * \brief Get the direct inputs of the head node(s) of this symbol.
   * \return symbol A new symbol whose output contains all the inputs of the head
   *                node(s).
   */
  Symbol GetChildren() const;
  /*!
   * \brief Set additional attributes to current node.
   *
   *  This only works for symbol with outputs from single operators.
   *  For grouped symbol, an error will be raised.
   *
   *  This function mutates the node's symbol and is not recommended.
   *
   * \param attrs The attributes to set.
   */
  void SetAttrs(const std::vector<std::pair<std::string, std::string> >& attrs);
  /*!
   * \brief Get attributes from the symbol.
   *
   *  This only works for symbol with outputs from single operators.
   *  For grouped symbol, an error will be raised.
   *
   * \param key Key of the attribute. When key == "name", it returns the name attirbute.
   * \param out The output value of the attribute.
   * \return true If the attribute exists, false if the attribute does not exist.
   */
  bool GetAttr(const std::string& key, std::string* out) const;
  /*!
   * \brief Get attribute dictionary from the symbol.
   *
   *  For grouped symbol, an error will be raised.
   *
   * \param option If recursive flag is set, the attributes of all children are retrieved.
   *               The name of symbol will be pre-pended to each key.
   * \return The created attribute.
   */
  std::unordered_map<std::string, std::string> ListAttrs(ListAttrOption option) const;
  /*!
   * \brief Get attribute dictionary from the symbol and all children.
   *
   *  For grouped symbol, an error will be raised.
   *
   * \return The created attribute in format <operator_name, key, value>.
   */
  std::vector<std::tuple<std::string, std::string, std::string> >
      ListAttrsRecursive() const;
  /*!
   * \brief Create symbolic functor(AtomicSymbol) by given operator and attributes.
   * \param op The operator.
   * \param attrs The additional attributes.
   * \return Symbol that can be used to call compose further.
   */
  static Symbol CreateFunctor(const Op* op,
                              std::unordered_map<std::string, std::string> attrs);
  /*!
   * \brief Create symbolic functor(AtomicSymbol) by given node attributes.
   * \param attrs pre-initialized Node attributes.
   * \return Symbol that can be used to call compose further.
   */
  static Symbol CreateFunctor(const NodeAttrs& attrs);
  /*!
   * \brief Create symbol node representing variable.
   * \param name Name of the variable.
   * \return The symbol.
   */
  static Symbol CreateVariable(const std::string& name);
  /*!
   * \brief Create equivalence of symbol by grouping the symbols together.
   * \param symbols A list of symbols to be grouped.
   * \return The grouped symbol.
   */
  static Symbol CreateGroup(const std::vector<Symbol>& symbols);
};

}  // namespace nnvm

#endif  // NNVM_SYMBOLIC_H_
