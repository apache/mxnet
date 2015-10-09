/*!
 *  Copyright (c) 2015 by Contributors
 * \file symbol.h
 * \brief Rcpp Symbolic construction interface of MXNet
 */
#ifndef MXNET_RCPP_SYMBOL_H_
#define MXNET_RCPP_SYMBOL_H_

#include <Rcpp.h>
#include <mxnet/c_api.h>
#include <string>
#include <algorithm>
#include <vector>

namespace mxnet {
namespace R {
// forward declare symbol functiono
class SymbolFunction;

// TODO(KK): switch exposed function into roxygen style

/*! \brief The Rcpp Symbol class of MXNet */
class Symbol : public MXNetClassBase<Symbol, SymbolHandle, MXSymbolFree> {
 public:
  /*!
   * \brief Apply the symbol as function on kwargs
   * \param kwargs keyword arguments to the data
   * \return A resulting symbol.
   */
  RObjectType Apply(const Rcpp::List& kwargs) const;
  /*!
   * \brief Print the debug string of symbol
   * \return the debug string.
   */
  std::string DebugStr() const;
  /*! \return the arguments in the symbol */
  std::vector<std::string> ListArguments() const;
  /*! \return the auxiliary states symbol */
  std::vector<std::string> ListAuxiliaryStates() const;
  /*! \return the outputs in the symbol */
  std::vector<std::string> ListOuputs() const;

  /*!
   * \brief Save the symbol to file
   * \param fname the file name we need to save to
   */
  void Save(const std::string& fname) const;
  /*!
   * \brief save the symbol to json string
   * \return a JSON string representation of symbol.
   */
  std::string AsJSON() const;
  /*!
   * \brief Get a new grouped symbol whose output contains all the
   *     internal outputs of this symbol.
   * \return The internal of the symbol.
   */
  RObjectType GetInternals() const;
  /*!
   * \brief Get index-th outputs of the symbol.
   * \param symbol The symbol
   * \param index the Index of the output.
   * \param out The output symbol whose outputs are the index-th symbol.
   */
  RObjectType GetOutput(mx_uint index) const;
  /*! \brief Infer the shapes of arguments, outputs, and auxiliary states */
  SEXP InferShape(const Rcpp::List& kwargs) const;
  //
  //' @title
  //' mx.symbol.Variable
  //'
  //’ Create a symbolic variable with specified name.
  //’
  //’ @param name string, Name of the variable.
  //’ @return The created variable symbol.
  //
  static RObjectType Variable(const std::string& name);
  //
  //' @title
  //' mx.symbol.load
  //'
  //’ Load a symbol variable from filename.
  //’
  //’ @param filename string, the path to the symbol file.
  //’ @return The loaded corresponding symbol.
  //
  static RObjectType Load(const std::string& filename);
  //
  //' @title
  //' mx.symbol.load.json
  //'
  //’ Load a symbol variable from json string
  //’
  //’ @param json string, json string of symbol.
  //’ @return The loaded corresponding symbol.
  //
  static RObjectType LoadJSON(const std::string& json);
  //
  //' @title
  //' Create a symbol that groups symbols together.
  //’
  //’ @param ... List of symbols to be grouped.
  //’ @return The created grouped symbol.
  //
  static RObjectType Group(const Rcpp::List& symbols);
  /*! \brief static function to initialize the Rcpp functions */
  static void InitRcppModule();

 private:
  // friend with SymbolFunction
  friend class SymbolFunction;
  friend class MXNetClassBase<Symbol, SymbolHandle, MXSymbolFree>;
  // enable trivial copy constructors etc.
  Symbol() {}
  // constructor
  explicit Symbol(SymbolHandle handle) {
    this->handle_ = handle;
  }
  /*!
   * \brief Return a clone of Symbol
   * \param obj The source to be cloned from
   * \return a Cloned Symbol
   */
  inline RObjectType Clone() const;
  /*!
   * \brief Compose the symbol with kwargs
   * \param kwargs keyword arguments to the data
   * \param name name of the symbol.
   */
  void Compose(const Rcpp::List& kwargs, const std::string &name);
};

/*! \brief The Symbol functions to be invoked */
class SymbolFunction : public ::Rcpp::CppFunction {
 public:
  virtual SEXP operator() (SEXP* args);

  virtual int nargs() {
    return 1;
  }

  virtual bool is_void() {
    return false;
  }

  virtual void signature(std::string& s, const char* name) {  // NOLINT(*)
    ::Rcpp::signature< SEXP, ::Rcpp::List >(s, name);
  }

  virtual const char* get_name() {
    return name_.c_str();
  }

  virtual SEXP get_formals() {
    return Rcpp::List::create(Rcpp::_["alist"]);
  }

  virtual DL_FUNC get_function_ptr() {
    return (DL_FUNC)NULL; // NOLINT(*)
  }
  /*! \brief static function to initialize the Rcpp functions */
  static void InitRcppModule();

 private:
  // make constructor private
  explicit SymbolFunction(AtomicSymbolCreator handle);
  /*! \brief internal creator handle. */
  AtomicSymbolCreator handle_;
  // name of the function
  std::string name_;
  // hint used to generate the names
  std::string name_hint_;
  // key to variable size arguments, if any.
  std::string key_var_num_args_;
};
}  // namespace R
}  // namespace mxnet

RCPP_EXPOSED_CLASS_NODECL(::mxnet::R::Symbol);

#endif  // MXNET_RCPP_SYMBOL_H_
