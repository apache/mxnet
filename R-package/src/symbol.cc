/*!
 *  Copyright (c) 2015 by Contributors
 * \file symbol.cc
 * \brief Rcpp Symbol of MXNet.
 */
#include <Rcpp.h>
#include <string>
#include <algorithm>
#include "./base.h"
#include "./symbol.h"
#include "./name.h"

namespace mxnet {
namespace R {

NameManager* NameManager::Get() {
  static NameManager inst;
  return &inst;
}

inline Symbol::RObjectType Symbol::RObject(SymbolHandle handle) {
  Symbol* p = new Symbol(handle);
  // TODO(KK) can we avoid use internal::make_new_object?
  return Rcpp::internal::make_new_object(p);
}

inline Symbol::RObjectType Symbol::Clone() const {
  SymbolHandle ohandle;
  MX_CALL(MXSymbolCopy(handle_, &ohandle));
  return Symbol::RObject(ohandle);
}

inline Symbol* Symbol::XPtr(const Rcpp::RObject& obj) {
  Symbol* ptr = Rcpp::as<Symbol*>(obj);
  return ptr;
}

void Symbol::Finalizer(Symbol *sym) {
  MX_CALL(MXSymbolFree(sym->handle_));
}

Symbol::RObjectType Symbol::Apply(const Rcpp::List& kwargs) const {
  RObjectType ret = this->Clone();
  if (kwargs.containsElementNamed("name")) {
    int index = kwargs.findName("name");
    std::string name = kwargs[index];
    Rcpp::List kw(kwargs);
    kw.erase(index);
    Symbol::XPtr(ret)->Compose(kw, name);
  } else {
    std::string name;
    Symbol::XPtr(ret)->Compose(kwargs, name);
  }
  return ret;
}

std::string Symbol::DebugStr() const {
  const char *str;
  MX_CALL(MXSymbolPrint(handle_, &str));
  return str;
}

void Symbol::Compose(const Rcpp::List& kwargs, const std::string &name) {
  std::string target_name;
  std::vector<std::string> keys = SafeGetListNames(kwargs);
  // get names
  bool positional = keys.size() == 0 || keys[0].length() == 0;
  for (size_t i = 0; i < keys.size(); ++i) {
    RCHECK((keys[i].length() == 0) == positional)
        << "Input symbols need to be either positional or key=value style, not both\n";
  }
  if (positional) keys.resize(0);

  // string parameter keys
  std::vector<const char*> c_keys = CKeys(keys);
  // string parameter values
  std::vector<SymbolHandle> handles(kwargs.size());
  RLOG_INFO << "Compose=here\n";
  for (size_t i = 0; i < kwargs.size(); ++i) {
    handles[i] = Symbol::XPtr(kwargs[i])->handle_;
  }
  RLOG_INFO << "Compose=aaz\n";
  MX_CALL(MXSymbolCompose(
      handle_, name.c_str(),
      static_cast<mx_uint>(handles.size()),
      dmlc::BeginPtr(c_keys), dmlc::BeginPtr(handles)));
}

Symbol::RObjectType Symbol::Variable(const std::string& name) {
  SymbolHandle out;
  MX_CALL(MXSymbolCreateVariable(name.c_str(), &out));
  return Symbol::RObject(out);
}

Symbol::RObjectType Symbol::Group(const Rcpp::List& kwargs) {
  std::vector<SymbolHandle> handles(kwargs.size());
  for (size_t i = 0; i < kwargs.size(); ++i) {
    RCHECK(Rcpp::is<Symbol>(kwargs[i]))
        << "Group only accept MXSymbol as input\n";
    handles[i] = Symbol::XPtr(kwargs[i])->handle_;
  }
  SymbolHandle out;
  MX_CALL(MXSymbolCreateGroup(static_cast<mx_uint>(handles.size()),
                              dmlc::BeginPtr(handles), &out));
  return Symbol::RObject(out);
}

SymbolFunction::SymbolFunction(AtomicSymbolCreator handle)
    : handle_(handle) {
  const char* name;
  const char* description;
  mx_uint num_args;
  const char **arg_names;
  const char **arg_type_infos;
  const char **arg_descriptions;
  const char *key_var_num_args;

  MX_CALL(MXSymbolGetAtomicSymbolInfo(
      handle_, &name, &description, &num_args,
      &arg_names, &arg_type_infos, &arg_descriptions,
      &key_var_num_args));
  if (key_var_num_args != nullptr) {
    key_var_num_args_ = key_var_num_args;
  }
  name_hint_ = name;
  std::transform(name_hint_.begin(), name_hint_.end(),
                 name_hint_.begin(), ::tolower);

  if (name[0] == '_') {
    name_ = std::string("mx.symbol.fun.internal.") + (name + 1);
  } else {
    name_ = std::string("mx.symbol.fun.") + name;
  }
  std::ostringstream os;
  os << description << "\n\n"
     << "Parameters\n"
     << "----------\n"
     << MakeDocString(num_args, arg_names, arg_type_infos, arg_descriptions)
     << "Returns\n"
     << "-------\n"
     << "out : Symbol\n"
     << "    The resulting Symbol";
  this->docstring = os.str();
}

SEXP SymbolFunction::operator() (SEXP* args) {
  BEGIN_RCPP;
  Rcpp::List kwargs(args[0]);
  std::vector<std::string> keys = SafeGetListNames(kwargs);
  // string key and values
  std::vector<std::string> str_keys;
  std::vector<std::string> str_vals;
  // symbol key and values
  std::vector<std::string> sym_keys;
  std::vector<Rcpp::RObject> sym_vals;
  // name of the result
  std::string name;

  // classify keys
  for (size_t i = 0; i < kwargs.size(); ++i) {
    if (keys[i] == "name") {
      name = keys[i]; continue;
    }
    if (!IsSimpleArg(kwargs[i])) {
      sym_keys.push_back(keys[i]);
      sym_vals.push_back(kwargs[i]);
    } else {
      RCHECK(keys[i].length() != 0)
          << "Non Symbol parameters is only accepted via key=value style.";
      str_keys.push_back(keys[i]);
      str_vals.push_back(AsPyString(kwargs[i]));
    }
  }
  SymbolHandle shandle;
  std::vector<const char*> c_str_keys = CKeys(str_keys);
  std::vector<const char*> c_str_vals = CKeys(str_vals);
  MX_CALL(MXSymbolCreateAtomicSymbol(
      handle_, static_cast<mx_uint>(str_keys.size()),
      dmlc::BeginPtr(c_str_keys),
      dmlc::BeginPtr(c_str_vals),
      &shandle));
  Symbol::RObjectType ret = Symbol::RObject(shandle);
  Rcpp::List compose_args = Rcpp::wrap(sym_vals);
  compose_args.names() = sym_keys;
  name = NameManager::Get()->GetName(name, name_hint_);
  Symbol::XPtr(ret)->Compose(compose_args, name);
  return ret;
  END_RCPP;
}

void Symbol::InitRcppModule() {
  using namespace Rcpp;  // NOLINT(*)
  class_<Symbol>("MXSymbol")
      .finalizer(&Symbol::Finalizer)
      .method("debug.str", &Symbol::DebugStr,
              "Return the debug string of internals of symbol")
      .method("apply", &Symbol::Apply,
              "Return a new Symbol by applying current symbols into input");
  function("mx.symbol.Variable",
           &Symbol::Variable,
           List::create(_["name"]),
           "Create a symbolic variable with specified name.");
  function("mx.symbol.fun.Group",
           &Symbol::Group,
           List::create(_["slist"]),
           "Create a symbol that groups symbols together.");
}

void SymbolFunction::InitRcppModule() {
  Rcpp::Module* scope = ::getCurrentScope();
  RCHECK(scope != nullptr)
      << "Init Module need to be called inside scope";
  mx_uint out_size;
  AtomicSymbolCreator *arr;
  MX_CALL(MXSymbolListAtomicSymbolCreators(&out_size, &arr));
  for (int i = 0; i < out_size; ++i) {
    SymbolFunction *f = new SymbolFunction(arr[i]);
    scope->Add(f->get_name(), f);
  }
}
}  // namespace R
}  // namespace mxnet
