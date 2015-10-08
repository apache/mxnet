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

namespace mxnet {
namespace R {
// forward declare symbol functiono
class SymbolFunction;

// TODO(KK): switch exposed function into roxygen style

/*! \brief The Rcpp Symbol class of MXNet */
class Symbol {
 public:
  /*! \brief The type of Symbol in R's side */
  typedef Rcpp::RObject RObjectType;
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
  // internal constructor
  explicit Symbol(SymbolHandle handle)
      : handle_(handle) {}
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
  /*!
   * \brief R side finalizer, will free the handle.
   * \param sym the symbol object
   */
  static void Finalizer(Symbol *sym);
  /*!
   * \brief create a R object that correspond to the Symbol.
   * \param handle the SymbolHandle needed for output.
   */
  inline static RObjectType RObject(SymbolHandle handle);
  /*!
   * \brief return extenral pointer representation of Symbol from its R object
   * \param obj The R NDArray object
   * \return The external pointer to the object
   */
  inline static Symbol* XPtr(const Rcpp::RObject& obj);

  /*! \brief handle to the symbol */
  SymbolHandle handle_;
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
