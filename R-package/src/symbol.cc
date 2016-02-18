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
#include "./export.h"

namespace mxnet {
namespace R {

NameManager* NameManager::Get() {
  static NameManager inst;
  return &inst;
}

inline Symbol::RObjectType Symbol::Clone() const {
  SymbolHandle ohandle;
  MX_CALL(MXSymbolCopy(handle_, &ohandle));
  return Symbol::RObject(ohandle);
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
  for (size_t i = 0; i < kwargs.size(); ++i) {
    handles[i] = Symbol::XPtr(kwargs[i])->handle_;
  }
  MX_CALL(MXSymbolCompose(
      handle_, name.c_str(),
      static_cast<mx_uint>(handles.size()),
      dmlc::BeginPtr(c_keys), dmlc::BeginPtr(handles)));
}

std::vector<std::string> Symbol::ListArguments() const {
  mx_uint size;
  const char **ret;
  MX_CALL(MXSymbolListArguments(handle_, &size, &ret));
  return std::vector<std::string>(ret, ret + size);
}

std::vector<std::string> Symbol::ListAuxiliaryStates() const {
  mx_uint size;
  const char **ret;
  MX_CALL(MXSymbolListAuxiliaryStates(handle_, &size, &ret));
  return std::vector<std::string>(ret, ret + size);
}

std::vector<std::string> Symbol::ListOuputs() const {
  mx_uint size;
  const char **ret;
  MX_CALL(MXSymbolListOutputs(handle_, &size, &ret));
  return std::vector<std::string>(ret, ret + size);
}

void Symbol::Save(const std::string& fname) const {
  MX_CALL(MXSymbolSaveToFile(handle_, fname.c_str()));
}

std::string Symbol::AsJSON() const {
  const char *json;
  MX_CALL(MXSymbolSaveToJSON(handle_, &json));
  return json;
}

Symbol::RObjectType Symbol::GetInternals() const {
  SymbolHandle out;
  MX_CALL(MXSymbolGetInternals(handle_, &out));
  return Symbol::RObject(out);
}

Symbol::RObjectType Symbol::GetOutput(mx_uint index) const {
  SymbolHandle out;
  MX_CALL(MXSymbolGetOutput(handle_, index - 1, &out));
  return Symbol::RObject(out);
}

// helper function to convert shape into Rcpp vector
inline Rcpp::List BuildShapeData(mx_uint shape_size,
                                 const mx_uint *shape_ndim,
                                 const mx_uint **shape_data,
                                 const std::vector<std::string> &names) {
  Rcpp::List ret(shape_size);
  for (mx_uint i = 0; i < shape_size; ++i) {
    Rcpp::IntegerVector dim(shape_data[i], shape_data[i] + shape_ndim[i]);
    std::reverse(dim.begin(), dim.end());
    ret[i] = dim;
  }
  ret.names() = names;
  return ret;
}

SEXP Symbol::InferShape(const Rcpp::List& kwargs) const {
  RCHECK(HasName(kwargs))
      << "Need to pass parameters in key=value style.\n";
  std::vector<std::string> keys = kwargs.names();
  std::vector<mx_uint> arg_ind_ptr(1, 0);
  std::vector<mx_uint> arg_shape_data;

  for (size_t i = 0; i < kwargs.size(); ++i) {
    RCHECK(keys[i].length() != 0)
      << "Need to pass parameters in key=value style.\n";
    std::vector<mx_uint> dim = Dim2InternalShape(kwargs[i]);
    arg_shape_data.insert(arg_shape_data.end(), dim.begin(), dim.end());
    arg_ind_ptr.push_back(static_cast<mx_uint>(arg_shape_data.size()));
  }
  std::vector<const char*> c_keys = CKeys(keys);

  mx_uint in_shape_size;
  const mx_uint *in_shape_ndim;
  const mx_uint **in_shape_data;
  mx_uint out_shape_size;
  const mx_uint *out_shape_ndim;
  const mx_uint **out_shape_data;
  mx_uint aux_shape_size;
  const mx_uint *aux_shape_ndim;
  const mx_uint **aux_shape_data;
  int complete;

  MX_CALL(MXSymbolInferShape(
      handle_, static_cast<mx_uint>(kwargs.size()), dmlc::BeginPtr(c_keys),
      dmlc::BeginPtr(arg_ind_ptr), dmlc::BeginPtr(arg_shape_data),
      &in_shape_size, &in_shape_ndim, &in_shape_data,
      &out_shape_size, &out_shape_ndim, &out_shape_data,
      &aux_shape_size, &aux_shape_ndim, &aux_shape_data,
      &complete));

  if (complete != 0) {
    return Rcpp::List::create(
        Rcpp::Named("arg.shapes") = BuildShapeData(
            in_shape_size, in_shape_ndim, in_shape_data, ListArguments()),
        Rcpp::Named("out.shapes") = BuildShapeData(
            out_shape_size, out_shape_ndim, out_shape_data, ListOuputs()),
        Rcpp::Named("aux.shapes") = BuildShapeData(
            aux_shape_size, aux_shape_ndim, aux_shape_data, ListAuxiliaryStates()));
  } else {
    return R_NilValue;
  }
}

Symbol::RObjectType Symbol::Variable(const std::string& name) {
  SymbolHandle out;
  MX_CALL(MXSymbolCreateVariable(name.c_str(), &out));
  return Symbol::RObject(out);
}

Symbol::RObjectType Symbol::Load(const std::string& filename) {
  SymbolHandle out;
  MX_CALL(MXSymbolCreateFromFile(filename.c_str(), &out));
  return Symbol::RObject(out);
}

Symbol::RObjectType Symbol::LoadJSON(const std::string& json) {
  SymbolHandle out;
  MX_CALL(MXSymbolCreateFromJSON(json.c_str(), &out));
  return Symbol::RObject(out);
}

Symbol::RObjectType Symbol::Group(const Rcpp::List& symbols) {
  // allow pass in single list
  Rcpp::List kwargs = symbols;
  if (symbols.size() == 1 && Rcpp::is<Rcpp::List>(symbols[0])) {
    kwargs = symbols[0];
  }

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
  const char *ret_type;

  MX_CALL(MXSymbolGetAtomicSymbolInfo(
      handle_, &name, &description, &num_args,
      &arg_names, &arg_type_infos, &arg_descriptions,
      &key_var_num_args, &ret_type));
  if (key_var_num_args != nullptr) {
    key_var_num_args_ = key_var_num_args;
  }
  name_hint_ = name;
  std::transform(name_hint_.begin(), name_hint_.end(),
                 name_hint_.begin(), ::tolower);
  if (name[0] == '_') {
    name_ = std::string("mx.varg.symbol.internal.") + (name + 1);
  } else {
    name_ = std::string("mx.varg.symbol.") + name;
  }
  std::ostringstream os;
  os << description << "\n\n"
     << MakeDocString(num_args, arg_names, arg_type_infos, arg_descriptions)
     << "@param name  string, optional\n"
     << "    Name of the resulting symbol.\n"
     << "@return out The result mx.symbol\n\n"
     << "@export\n";
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
      name = Rcpp::as<std::string>(kwargs[i]);
      continue;
    }
    if (!isSimple(kwargs[i]) && Rcpp::is<Symbol>(kwargs[i])) {
      sym_keys.push_back(keys[i]);
      sym_vals.push_back(kwargs[i]);
    } else {
      RCHECK(keys[i].length() != 0)
          << "Non Symbol parameters is only accepted via key=value style.";
      str_keys.push_back(FormatParamKey(keys[i]));
      str_vals.push_back(toPyString(keys[i], kwargs[i]));
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
      .method("debug.str", &Symbol::DebugStr,
              "Return the debug string of internals of symbol")
      .method("apply", &Symbol::Apply,
              "Return a new Symbol by applying current symbols into input")
      .method("as.json", &Symbol::AsJSON,
              "Return a json string representation of symbol")
      .method("save", &Symbol::Save,
              "Save symbol to file")
      .property("arguments", &Symbol::ListArguments,
              "List the arguments names of the symbol")
      .property("outputs", &Symbol::ListOuputs,
              "List the outputs names of the symbol")
      .property("auxiliary.states", &Symbol::ListAuxiliaryStates,
              "List the auxiliary state names of the symbol")
      .method("get.internals", &Symbol::GetInternals,
              "Get a symbol that contains all the internals")
      .method("get.output", &Symbol::GetOutput,
              "Get index-th output symbol of current one")
      .method("[[", &Symbol::GetOutput,
              "Get index-th output symbol of current one")
      .method("infer.shape", &Symbol::InferShape,
              "Inference the shape information given unknown ones");

  function("mx.symbol.Variable",
           &Symbol::Variable,
           List::create(_["name"]),
           "Create a symbolic variable with specified name.");
  function("mx.symbol.load",
           &Symbol::Load,
           List::create(_["file.name"]),
           "Load a symbol from file.");
  function("mx.symbol.load.json",
           &Symbol::LoadJSON,
           List::create(_["json.str"]),
           "Load a symbol from json string.");
  function("mx.varg.symbol.internal.Group",
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
