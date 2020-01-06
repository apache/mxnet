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
* \file monitor.hpp
* \brief monitor implementation
* \author Xin Li
*/

#ifndef MXNET_CPP_MONITOR_HPP_
#define MXNET_CPP_MONITOR_HPP_

#include <cmath>
#include <sstream>
#include <algorithm>
#include <vector>
#include <string>
#include "mxnet-cpp/monitor.h"

namespace mxnet {
namespace cpp {
inline NDArray _default_monitor_func(const NDArray &x) {
  return Operator("norm").PushInput(x).Invoke()[0] / std::sqrt(x.Size());
}

inline Monitor::Monitor(int interval, std::regex pattern, StatFunc stat_func)
  : interval(interval), pattern(pattern), stat_func(stat_func), step(0) {
}

inline void Monitor::install(Executor *exe, bool monitor_all) {
  MXExecutorSetMonitorCallbackEX(exe->handle_,
                                 static_cast<ExecutorMonitorCallback>(&Monitor::executor_callback),
                                 this, monitor_all);
  exes.push_back(exe);
}

inline void Monitor::tic() {
  if (step % interval == 0) {
    activated = true;
    stats.clear();
  }
}

inline std::vector<Monitor::Stat> Monitor::toc() {
  std::vector<Monitor::Stat> results;
  if (activated) {
    activated = false;

    for (auto* exe : exes) {
      for (auto& arg : exe->arg_arrays) {
        arg.WaitToRead();
      }
      for (auto& aux : exe->aux_arrays) {
        aux.WaitToRead();
      }

      for (auto &pair : exe->arg_dict()) {
        if (std::regex_match(pair.first, pattern)) {
          stats.emplace_back(step, pair.first, stat_func(pair.second));
        }
      }
      for (auto &pair : exe->aux_dict()) {
        if (std::regex_match(pair.first, pattern)) {
          stats.emplace_back(step, pair.first, stat_func(pair.second));
        }
      }
    }
    results.swap(stats);
  }
  ++step;
  return results;
}

inline void Monitor::toc_print() {
  auto results = toc();
  std::vector<float> data(1);
  for (auto& stat : results) {
    NDArray ndarray = std::get<2>(stat);

    std::string str;
    if (ndarray.Size() == 1) {
      if (ndarray.GetContext().GetDeviceType() != DeviceType::kGPU) {
        data[0] = ndarray.GetData()[0];
      } else {
        ndarray.SyncCopyToCPU(&data);
      }
      str = std::to_string(data[0]);
    } else {
      std::ostringstream out;
      out << ndarray;
      str = out.str();
    }

    LG << "Batch: " << std::get<0>(stat) << ' ' << std::get<1>(stat) << ' ' << str;
  }
}

inline void Monitor::executor_callback(const char *name, NDArrayHandle handle,
    void *monitor_ptr) {
  Monitor *monitor = static_cast<Monitor*>(monitor_ptr);
  if (monitor->activated && std::regex_match(name, monitor->pattern)) {
    monitor->stats.emplace_back(monitor->step, name, monitor->stat_func(NDArray(handle)));
  }
}

}  // namespace cpp
}  // namespace mxnet
#endif  // MXNET_CPP_MONITOR_HPP_
