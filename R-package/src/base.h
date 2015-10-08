/*!
 *  Copyright (c) 2015 by Contributors
 * \file base.h
 * \brief Rcpp interface of MXNet
 *  All the interface is done through C API,
 *  to achieve maximum portability when we need different compiler for libmxnet.
 */
#ifndef MXNET_RCPP_BASE_H_
#define MXNET_RCPP_BASE_H_

#include <Rcpp.h>
#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <mxnet/c_api.h>
#include <string>
#include <sstream>
#include <set>
#include <vector>

/*! \brief namespace of mxnet */
namespace mxnet {
/*! \brief namespace of R package */
namespace R {

/*! \brief macro to be compatible with non c++11 env */
#if DMLC_USE_CXX11 == 0
#define nullptr NULL
#endif

/*!
 * \brief Log that enables Stop and print message to R console
 */
class RLogFatal {
 public:
  inline std::ostringstream &stream() {
    return log_stream_;
  }
  ~RLogFatal()
#if DMLC_USE_CXX11
  noexcept(false)
#endif
  {
    std::string msg = log_stream_.str();
    throw Rcpp::exception(msg.c_str());
  }

 private:
  std::ostringstream log_stream_;
};

/*!
 * \brief LOG FATAL to report error to R console
 *  Need to append newline to it.
 */
#define RLOG_FATAL ::mxnet::R::RLogFatal().stream()

/*! \brief LOG INFO to report message to R console, need to append newline */
#define RLOG_INFO ::Rcpp::Rcout

/*!
 * \brief Checking macro for Rcpp code, report error ro R console
 * \code
 *  RCHECK(data.size() == 1) << "Data size must be 1";
 * \endcode
 */
#define RCHECK(x)                                           \
  if (!(x)) RLOG_FATAL << "Check failed: " #x << ' ' /* NOLINT(*) */

/*!
 * \brief protected MXNet C API call, report R error if happens.
 * \param func Expression to call.
 */
#define MX_CALL(func)                                              \
  {                                                                \
    int e = (func);                                                \
    if (e != 0) {                                                  \
      throw Rcpp::exception(MXGetLastError());                     \
    }                                                              \
  }

/*! \brief Context of device enviroment */
struct Context {
  /*! \brief The device ID of the context */
  int dev_type;
  /*! \brief The device ID of the context */
  int dev_id;
  /*! \brief The R object type of the context */
  typedef Rcpp::List RObjectType;
  /*! \brief default constructor  */
  Context() {}
  /*!
   * \brief Constructor
   * \param src source R representation.
   */
  explicit Context(const Rcpp::RObject& src) {
    Rcpp::List list(src);
    this->dev_id = list[1];
    this->dev_type = list[2];
  }
  /*! \return R object representation of the context */
  inline RObjectType RObject() const {
    const char *dev_name = "cpu";
    if (dev_type == kGPU) dev_name = "gpu";
    Rcpp::List ret = Rcpp::List::create(
        Rcpp::Named("device") = dev_name,
        Rcpp::Named("device_id") = dev_id,
        Rcpp::Named("device_typeid") = dev_type);
    ret.attr("class") = "MXContext";
    return ret;
  }
  /*!
   * Create a CPU context.
   * \param dev_id the device id.
   * \return CPU Context.
   */
  inline static RObjectType CPU(int dev_id = 0) {
    Context ctx;
    ctx.dev_type = kCPU;
    ctx.dev_id = dev_id;
    return ctx.RObject();
  }
  /*!
   * Create a GPU context.
   * \param dev_id the device id.
   * \return GPU Context.
   */
  inline static RObjectType GPU(int dev_id) {
    Context ctx;
    ctx.dev_type = kGPU;
    ctx.dev_id = dev_id;
    return ctx.RObject();
  }
  /*! \brief initialize all the Rcpp module functions */
  inline static void InitRcppModule() {
    using namespace Rcpp;  // NOLINT(*);
    function("mx.cpu", &CPU,
             List::create(_["device_id"] = 0),
             "Create a CPU context.");
    function("mx.gpu", &GPU,
             List::create(_["device_id"]),
             "Create a GPU context with specific device_id.");
  }
  /*! \brief the device type id for CPU */
  static const int kCPU = 1;
  /*! \brief the device type id for GPU */
  static const int kGPU = 2;
};

/*!
 * \brief Get a C char pointer vector representation of keys
 * The keys must stay alive when using c_keys
 * \param keys the string vector to get keys from
 * \return the C char pointer
 */
inline std::vector<const char*> CKeys(const std::vector<std::string> &keys) {
  std::vector<const char*> c_keys(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    c_keys[i] = keys[i].c_str();
  }
  return c_keys;
}

/*!
 * \brief Get human readable function information
 * \param name the name of function.
 * \parma num_args number of arguments.
 * \parma arg_names name of arguments
 * \parma arg_type_infos type information of arguments.
 * \param arg_descriptions descriptions of arguments.
 * \param remove_dup Whether to remove duplications
 */
inline std::string MakeDocString(mx_uint num_args,
                                 const char **arg_names,
                                 const char **arg_type_infos,
                                 const char **arg_descriptions,
                                 bool remove_dup = true) {
  std::set<std::string> visited;
  std::ostringstream os;
  for (mx_uint i = 0; i < num_args; ++i) {
    std::string arg = arg_names[i];
    if (visited.count(arg) != 0 && remove_dup) continue;
    visited.insert(arg);
    os << "    " << arg_names[i] << " : "  << arg_type_infos[i] << "\n"
       << "        " << arg_descriptions[i] << "\n";
  }
  return os.str();
}

/*!
 * \brief Create a API compatile string presentation of value
 * \return A c++ string representation
 */
inline std::string AsPyString(const SEXP val) {
  using namespace Rcpp;  // NOLINT(*)
  // TODO(KK) , better string handling.
  std::ostringstream os;
  if (is<IntegerVector>(val)) {
    IntegerVector vec(val);
    std::ostringstream os;
    os << "(";
    for (size_t i = 0; i < vec.size(); ++i) {
      int value = vec[i];
      os << value;
      if (i != 0) os << ", ";
    }
    if (vec.size() == 1) os << ",";
    os << ")";
    return os.str();
  }
  switch (TYPEOF(val)) {
    case STRSXP: return as<std::string>(val);
    case INTSXP: {
      os << as<int>(val);
      return os.str();
    }
    case REALSXP: {
      os << as<double>(val);
      return os.str();
    }
    default: {
      RLOG_FATAL << "Unsupported parameter type\n";
    }
  }
  return os.str();
}

/*!
 *\return whether the expression is simple arguments
 * That is not module object and can be converted to string
 */
inline bool IsSimpleArg(SEXP args) {
  // TODO(KK) maybe more improvements, support bool?
  switch (TYPEOF(args)) {
    case REALSXP:
    case VECSXP:
    case INTSXP:
    case CPLXSXP:
    case STRSXP: return true;
    default: return false;
  }
}

/*!
 * \brief Get names from list, return vector of empty strings if names do not present
 * \param src the source list
 * \retunr vector of string of same length as src.
 */
inline std::vector<std::string> SafeGetListNames(const Rcpp::List& src) {
  Rcpp::RObject obj = src.names();
  if (obj == R_NilValue) {
    return std::vector<std::string>(src.size(), std::string());
  } else {
    return src.names();
  }
}
}  // namespace R
}  // namespace mxnet
#endif  // MXNET_RCPP_BASE_H_
