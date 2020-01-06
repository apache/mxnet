/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
*  Copyright (c) 2017 by Contributors
* \file monitor.h
* \brief monitor definition
* \author Xin Li
*/

#ifndef MXNET_CPP_MONITOR_H_
#define MXNET_CPP_MONITOR_H_

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
  * \param monitor_all If true, monitor both input and output, otherwise monitor output only.
  */
  void install(Executor *exe, bool monitor_all = false);

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
#endif  // MXNET_CPP_MONITOR_H_
