/*!
 *  Copyright (c) 2015 by Contributors
 * \file name.h
 * \brief Name manager to get default names.
 */
#ifndef MXNET_RCPP_NAME_H_
#define MXNET_RCPP_NAME_H_

#include <map>
#include <string>

namespace mxnet {
namespace R {

/*!
 * \brief A name manager to attach names
 *  This is a very simple implementation.
 */
class NameManager {
 public:
  /*!
   * \brief Get a canonical name given name and hint
   * \param name The name passed in from parameter
   * \param hint The hint used to generate the name.
   */
  virtual std::string GetName(const std::string& name,
                              const std::string& hint) {
    if (name.length() != 0) return name;
    if (counter_.count(hint) == 0) {
      counter_[hint] = 0;
    }
    size_t cnt = counter_[hint]++;
    std::ostringstream os;
    os << hint << cnt;
    return os.str();
  }
  /*! \return  global singleton of the manager */
  static NameManager *Get();

 private:
  // internal counter
  std::map<std::string, size_t> counter_;
};
}  // namespace R
}  // namespace mxnet
#endif  // MXNET_RCPP_NAME_H_
