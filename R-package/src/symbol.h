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

/*! \brief The Rcpp Symbol class of MXNet */
class Symbol {
 public:
  // typedef RObjectType
  typedef Rcpp::RObject RObjectType;
  /*! \return typename from R side. */
  inline static const char* TypeName() {
    return "MXSymbol";
  }
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
  /*!
   *  \brief Create a symbolic variable with specified name.
   *
   *  \param name string, Name of the variable.
   *  \return The created variable symbol.
   */
  static RObjectType Variable(const std::string& name);
  /*!
   *  \brief Load a symbol variable from filename.
   *
   *  \param filename string, the path to the symbol file.
   *  \return The loaded corresponding symbol.
   */
  static RObjectType Load(const std::string& filename);
  /*!
   *  \brief Load a symbol variable from json string
   *
   *  \param json string, json string of symbol.
   *  \return The loaded corresponding symbol.
   */
  static RObjectType LoadJSON(const std::string& json);
  /*!
   *  \brief Create a symbol that groups symbols together.
   *
   *  \param ... List of symbols to be grouped.
   *  \return The created grouped symbol.
   */
  static RObjectType Group(const Rcpp::List& symbols);
  /*! \brief static function to initialize the Rcpp functions */
  static void InitRcppModule();
  // destructor
  ~Symbol() {
    MX_CALL(MXSymbolFree(handle_));
  }
  // get external pointer of Symbol
  inline static Symbol* XPtr(const Rcpp::RObject& obj) {
    return Rcpp::as<Symbol*>(obj);
  }

 private:
  // friend with SymbolFunction
  friend class SymbolFunction;
  friend class Executor;
  // enable trivial copy constructors etc.
  Symbol() {}
  // constructor
  explicit Symbol(SymbolHandle handle)
      : handle_(handle) {}
  /*!
   * \brief create a R object that correspond to the Class
   * \param handle the Handle needed for output.
   */
  inline static Rcpp::RObject RObject(SymbolHandle handle) {
    return Rcpp::internal::make_new_object(new Symbol(handle));
  }
  /*!
   * \brief Return a clone of Symbol
   * Do not expose to R side
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

  /*! \brief internal executor handle */
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

namespace Rcpp {
  template<>
  inline bool is<mxnet::R::Symbol>(SEXP x) {
    return internal::is__module__object_fix<mxnet::R::Symbol>(x);
  }
}
#endif  // MXNET_RCPP_SYMBOL_H_
