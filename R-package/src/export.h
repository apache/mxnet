/*!
 *  Copyright (c) 2015 by Contributors
 * \file export.h
 * \brief Export module that takes charge of code generation and document
 *  Generation for functions exported from R-side
 */
#ifndef MXNET_RCPP_EXPORT_H_
#define MXNET_RCPP_EXPORT_H_

#include <Rcpp.h>
#include <string>

namespace mxnet {
namespace R {
/*! \brief exporter class*/
class Exporter {
 public:
  /*!
   * \brief Export the generated file into path.
   * \param path The path to be exported.
   */
  static void Export(const std::string& path);
  // intialize the Rcpp module
  static void InitRcppModule();

 public:
  // get the singleton of exporter
  static Exporter* Get();
  /*! \brief The scope of current module to export */
  Rcpp::Module* scope_;
};

/*!
 * \brief Get human readable roxygen style function information.
 * \param name the name of function.
 * \parma num_args number of arguments.
 * \parma arg_names name of arguments
 * \parma arg_type_infos type information of arguments.
 * \param arg_descriptions descriptions of arguments.
 * \param remove_dup Whether to remove duplications
 */
std::string MakeDocString(mx_uint num_args,
                          const char **arg_names,
                          const char **arg_type_infos,
                          const char **arg_descriptions,
                          bool remove_dup = true);
}  // namespace R
}  // namespace mxnet
#endif  // MXNET_RCPP_EXPORT_H_
