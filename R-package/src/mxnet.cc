/*!
 *  Copyright (c) 2015 by Contributors
 * \file mxnet.cc
 * \brief The registry of all module functions and objects
 */
#include <Rcpp.h>
#include <fstream>
#include "./base.h"
#include "./ndarray.h"
#include "./symbol.h"
#include "./executor.h"
#include "./io.h"
#include "./kvstore.h"

namespace mxnet {
namespace R {
void SetSeed(int seed) {
  MX_CALL(MXRandomSeed(seed));
}

void NotifyShutdown(int seed) {
  MX_CALL(MXNotifyShutdown());
}

// init rcpp module in base
void InitRcppModule() {
  using namespace Rcpp;  // NOLINT(*)
  function("mx.internal.set.seed", &SetSeed);
  function("mx.internal.notify.shudown", &NotifyShutdown);
}
}  // namespace R
}  // namespace mxnet


RCPP_MODULE(mxnet) {
  using namespace mxnet::R; // NOLINT(*)
  mxnet::R::InitRcppModule();
  Context::InitRcppModule();
  NDArray::InitRcppModule();
  NDArrayFunction::InitRcppModule();
  Symbol::InitRcppModule();
  SymbolFunction::InitRcppModule();
  Executor::InitRcppModule();
  DataIter::InitRcppModule();
  DataIterCreateFunction::InitRcppModule();
  KVStore::InitRcppModule();
}

std::string fun_generate(const std::string & fun_name) {
  std::string res = "mx.symbol.";
  res = res + fun_name.substr(15, fun_name.length() - 15) +
        "<- function(...) {\n" +
        "  " + fun_name + "(list(...))\n" +
        "}\n";
  return res;
}

void script_generate(Rcpp::RObject fun_lst, std::string path) {
  std::ofstream script;
  path = path + "/mxnet_generated.R";
  Rcpp::Rcout << path << " generated" << std::endl;
  script.open(path.c_str());
  script << "## script generated, do not modify by hand" << std::endl;
  Rcpp::CharacterVector lst(fun_lst);
  std::string varg = "mx.varg.symbol";
  for (int i = 0; i < lst.size(); i++) {
    std::string fun_name = Rcpp::as<std::string>(lst[i]);
    if (fun_name.compare(0, varg.length(), varg) == 0) {
      script << fun_generate(fun_name) << std::endl; 
    }
  }
  script.close();
}

RcppExport SEXP mxnet_generate(SEXP path) {
BEGIN_RCPP
  Rcpp::RObject fun_lst;
  Rcpp::Function ls("ls");
  fun_lst = ls("package:mxnet");
  script_generate(fun_lst, Rcpp::as<std::string>(path));
  return R_NilValue;
END_RCPP
}
