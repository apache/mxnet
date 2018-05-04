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
 * Copyright (c) 2015 by Contributors
 * \file profiler.cc
 * \brief implements profiler
 */
#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <mxnet/base.h>
#include <fstream>
#include <thread>
#include <iomanip>
#include "./profiler.h"

namespace mxnet {
namespace profiler {

template<typename DType>
inline float MicroToMilli(const DType micro) {
  return static_cast<float>(static_cast<double>(micro) / 1000);
}

void AggregateStats::OnProfileStat(const ProfileStat& stat) {
  std::unique_lock<std::mutex> lk(m_);
  stat.SaveAggregate(&stats_[stat.categories_.c_str()][stat.name_.c_str()]);
}

void AggregateStats::Dump(std::ostream& os, bool clear) {
  std::ios state(nullptr);
  state.copyfmt(os);
  os << std::endl
     << "Profile Statistics." << std::endl
     << "\tNote that counter items are counter values and not time units."
     << std::endl;
  std::unique_lock<std::mutex> lk(m_);
  for (auto type_iter = stats_.begin(), type_e_iter = stats_.end();
       type_iter != type_e_iter; ++type_iter) {
    const std::string& type = type_iter->first;
    const std::unordered_map<std::string, StatData>& mm = type_iter->second;
    if (!mm.empty()) {
      os << type << std::endl << "=================" << std::endl;
      os << std::setw(25) << std::left  << "Name"
         << std::setw(16) << std::right << "Total Count"
         << " "
         << std::setw(16) << std::right
         << "Time (ms)"
         << " "
         << std::setw(16) << std::right
         << "Min Time (ms)"
         << " "
         << std::setw(16) << std::right
         << "Max Time (ms)"
         << " "
         << std::setw(16) << std::right
         << "Avg Time (ms)"
         << std::endl;
      os << std::setw(25) << std::left  << "----"
         << std::setw(16) << std::right << "-----------"
         << " "
         << std::setw(16) << std::right
         << "---------"
         << " "
         << std::setw(16) << std::right
         << "-------------"
         << " "
         << std::setw(16) << std::right
         << "-------------"
         << " "
         << std::setw(16) << std::right
         << "-------------"
         << std::endl;
      for (auto iter = mm.begin(), e_iter = mm.end(); iter != e_iter; ++iter) {
        const StatData &data = iter->second;
        if (data.type_ == StatData::kDuration || data.type_ == StatData::kCounter) {
          const std::string &name = iter->first;
          os << std::setw(25) << std::left << name
             << std::setw(16) << std::right << data.total_count_;
          os << " "
             << std::fixed << std::setw(16) << std::setprecision(4) << std::right
             << MicroToMilli(data.total_aggregate_)
             << " "
             << std::fixed << std::setw(16) << std::setprecision(4) << std::right
             << MicroToMilli(data.min_aggregate_)
             << " "
             << std::fixed << std::setw(16) << std::setprecision(4) << std::right
             << MicroToMilli(data.max_aggregate_);
          if (data.type_ == StatData::kCounter) {
            os << " "
               << std::fixed << std::setw(16) << std::setprecision(4) << std::right
               << (MicroToMilli(data.max_aggregate_ - data.min_aggregate_) / 2);
          } else {
            os << " "
               << std::fixed << std::setw(16) << std::setprecision(4) << std::right
               << (MicroToMilli(static_cast<double>(data.total_aggregate_)
                                / data.total_count_));
          }
          os << std::endl;
        }
      }
      os << std::endl;
    }
  }
  os << std::flush;
  os.copyfmt(state);
  if (clear) {
    stats_.clear();
  }
}

}  // namespace profiler
}  // namespace mxnet
