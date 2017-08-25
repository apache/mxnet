/*!
*  Copyright (c) 2017 by Contributors
* \file monitor.h
* \brief monitor definition
* \author Xin Li
*/

#ifndef CPP_PACKAGE_INCLUDE_MXNET_CPP_MONITOR_H_
#define CPP_PACKAGE_INCLUDE_MXNET_CPP_MONITOR_H_

#include <regex>
#include <tuple>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <functional>
#include "mxnet-cpp/base.h"
#include "mxnet-cpp/ndarray.h"
#include "mxnet-cpp/executor.h"

namespace mxnet {
namespace cpp {

/*!
* \brief Default function for monitor that computes statistics of the input tensor,
* which is the mean absolute |x|/size(x)
* \param x The input tensor
* \return The statistics of the input tensor
*/
NDArray _default_monitor_func(const NDArray &x);

/*!
* \brief Monitor interface
*/
class Monitor {
 public:
  typedef std::function<NDArray(const NDArray&)> StatFunc;
  typedef std::tuple<int, std::string, NDArray> Stat;

  /*!
  * \brief Monitor constructor
  * \param interval Number of batches between printing.
  * \param pattern A regular expression specifying which tensors to monitor.
  * \param stat_func A function that computes statistics of tensors. Defaults to mean
  * absolute value |x|/size(x).
  */
  Monitor(int interval, std::regex pattern = std::regex(".*"),
      StatFunc stat_func = _default_monitor_func);

  /*!
  * \brief install callback to executor. Supports installing to multiple executors.
  * \param exe The executor to install to.
  */
  void install(Executor *exe);

  /*!
  * \brief Start collecting stats for current batch. Call before calling forward.
  */
  void tic();

  /*!
  * \brief End collecting for current batch and return results. Call after computation
  * of current batch.
  */
  std::vector<Stat> toc();

  /*!
  * \brief End collecting and print results.
  */
  void toc_print();

 protected:
  int interval;
  std::regex pattern;
  StatFunc stat_func;
  std::vector<Executor*> exes;

  int step;
  bool activated;
  std::vector<Stat> stats;

  static void executor_callback(const char *name, NDArrayHandle ndarray, void *monitor_ptr);
};

}  // namespace cpp
}  // namespace mxnet
#endif  // CPP_PACKAGE_INCLUDE_MXNET_CPP_MONITOR_H_
