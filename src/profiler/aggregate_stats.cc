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
#include <queue>
#include <utility>
#include "./profiler.h"

namespace mxnet {
namespace profiler {

using pi = std::pair<double, std::string>;

template<typename DType>
inline float MicroToMilli(const DType micro) {
  return static_cast<float>(static_cast<double>(micro) / 1000);
}

template<typename DType>
inline float ByteToKilobyte(const DType byte) {
  return static_cast<float>(static_cast<double>(byte) / 1000);
}

inline std::priority_queue<pi>
  BuildHeap(const std::unordered_map<std::string, AggregateStats::StatData>& map,
            int sort_by, int ascending) {
  std::priority_queue<pi> heap;
  for (const auto& iter : map) {
    const std::string& name = iter.first;
    const AggregateStats::StatData& data = iter.second;
    double value = 0;
    switch (static_cast<AggregateStats::SortBy>(sort_by)) {
      case AggregateStats::SortBy::Total:
        value = data.total_aggregate_;
        break;
      case AggregateStats::SortBy::Avg:
        if (data.type_ == AggregateStats::StatData::kCounter)
          value = (data.max_aggregate_ - data.min_aggregate_) / 2;
        else
          value = static_cast<double>(data.total_aggregate_)
                                / data.total_count_;
        break;
      case AggregateStats::SortBy::Min:
        value = data.min_aggregate_;
        break;
      case AggregateStats::SortBy::Max:
        value = data.max_aggregate_;
        break;
      case AggregateStats::SortBy::Count:
        value = data.total_count_;
        break;
      default:
        LOG(FATAL) << "Invalid value for parameter sort_by";
        break;
    }
    if (ascending != 0)
      value = -value;
    heap.push(std::make_pair(value, name));
  }
  return heap;
}

void AggregateStats::OnProfileStat(const ProfileStat& stat) {
  std::unique_lock<std::mutex> lk(m_);
  if (stat.enable_aggregate_) {
    stat.SaveAggregate(&stats_[stat.categories_.c_str()][stat.name_.c_str()]);
  }
}

void AggregateStats::DumpTable(std::ostream& os, int sort_by, int ascending) {
  std::ios state(nullptr);
  state.copyfmt(os);
  os << std::endl
     << "Profile Statistics:" << std::endl
     << "\tNote the difference in units for different entries."
     << std::endl;
  std::unique_lock<std::mutex> lk(m_);
  for (const auto& stat : stats_) {
    const std::string& type = stat.first;
    const std::unordered_map<std::string, StatData>& mm = stat.second;
    bool is_memory = (type == "Device Storage"  || type == "Pool Memory");
    os << type << std::endl << "=================" << std::endl;
    os << std::setw(25) << std::left  << "Name"
        << std::setw(16) << std::right << "Total Count"
        << " "
        << (is_memory ? std::setw(0) : std::setw(16)) << std::right
        << (is_memory ? "" : "Time (ms)")
        << (is_memory ? "" : " ")
        << std::setw(16) << std::right
        << (is_memory ? "Min Use  (kB)" : "Min Time (ms)")
        << " "
        << std::setw(16) << std::right
        << (is_memory ? "Max Use  (kB)" : "Max Time (ms)")
        << " "
        << std::setw(16) << std::right
        << (is_memory ? "Avg Use  (kB)" : "Avg Time (ms)")
        << std::endl;
    os << std::setw(25) << std::left  << "----"
        << std::setw(16) << std::right << "-----------"
        << " "
        << (is_memory ? std::setw(0) : std::setw(16)) << std::right
        << (is_memory ? "" : "---------")
        << (is_memory ? "" : " ")
        << std::setw(16) << std::right
        << "-------------"
        << " "
        << std::setw(16) << std::right
        << "-------------"
        << " "
        << std::setw(16) << std::right
        << "-------------"
        << std::endl;
    auto heap = BuildHeap(mm, sort_by, ascending);
    while (!heap.empty()) {
      const std::string& name = heap.top().second;
      const StatData &data = mm.at(name);
      if (data.type_ == StatData::kDuration || data.type_ == StatData::kCounter) {
        os << std::setw(25) << std::left << name
           << std::setw(16) << std::right << data.total_count_ << " "
           << std::fixed << (is_memory ? std::setw(0) : std::setw(16))
           << std::setprecision(4) << std::right;
        if (!is_memory)
          os << MicroToMilli(data.total_aggregate_) << " ";
        os << std::fixed << std::setw(16) << std::setprecision(4) << std::right
           << (is_memory ? ByteToKilobyte(data.min_aggregate_) : MicroToMilli(data.min_aggregate_))
           << " "
           << std::fixed << std::setw(16) << std::setprecision(4) << std::right
           << (is_memory ? ByteToKilobyte(data.max_aggregate_) : MicroToMilli(data.max_aggregate_))
           << " "
           << std::fixed << std::setw(16) << std::setprecision(4) << std::right
           << (data.type_ == AggregateStats::StatData::kCounter ?
                    ByteToKilobyte((data.max_aggregate_ - data.min_aggregate_) / 2) :
                    MicroToMilli(static_cast<double>(data.total_aggregate_)/ data.total_count_));
        os << std::endl;
      }
      heap.pop();
    }
    os << std::endl;
  }
  os << std::flush;
  os.copyfmt(state);
}

void AggregateStats::DumpJson(std::ostream& os, int sort_by, int ascending) {
  std::ios state(nullptr);
  state.copyfmt(os);
  std::unique_lock<std::mutex> lk(m_);
  std::stringstream memory_ss;
  std::stringstream time_ss;
  std::stringstream *ss;
  for (const auto& stat : stats_) {
    const std::string& type = stat.first;
    const std::unordered_map<std::string, StatData>& mm = stat.second;
    bool is_memory = (type == "Device Storage"  || type == "Pool Memory");
    ss = is_memory ? &memory_ss : &time_ss;
    if (ss->tellp() != std::streampos(0))
      *ss << "        ," << std::endl;
    *ss << "        \"" << type << "\": {" << std::endl;
    auto heap = BuildHeap(mm, sort_by, ascending);
    bool first_pass = true;
    while (!heap.empty()) {
      const std::string& name = heap.top().second;
      const StatData &data = mm.at(name);
      if (data.type_ == AggregateStats::StatData::kDuration ||
          data.type_ == AggregateStats::StatData::kCounter) {
        if (!first_pass)
          *ss << "            ," << std::endl;
        first_pass = false;
        *ss << "            \"" << name << "\": {" << std::endl
            << "                \"Count\": "
            << data.total_count_
            << "," << std::endl;
        if (!is_memory)
          *ss << "                \"Total\": "
              << std::setprecision(4)
              << MicroToMilli(data.total_aggregate_)
              << "," << std::endl;
        *ss << "                \"Min\": "
            << std::setprecision(4)
            << (is_memory ?
                ByteToKilobyte(data.min_aggregate_) :
                MicroToMilli(data.min_aggregate_))
            << "," << std::endl
            << "                \"Max\": "
            << std::setprecision(4)
            << (is_memory ?
                ByteToKilobyte(data.max_aggregate_) :
                MicroToMilli(data.max_aggregate_))
            << "," << std::endl
            << "                \"Avg\": "
            << std::setprecision(4)
            << (data.type_ == AggregateStats::StatData::kCounter ?
                 ByteToKilobyte((data.max_aggregate_ - data.min_aggregate_) / 2) :
                 MicroToMilli(static_cast<double>(data.total_aggregate_) /  data.total_count_))
            << std::endl
            << "            }" << std::endl;
      }
      heap.pop();
    }
    *ss << "        }" << std::endl;
  }
  os << "{" << std::endl
     << "    \"Time\": {" << std::endl
     << time_ss.str()
     << "    }" << std::endl
     << "    ," << std::endl
     << "    \"Memory\": {" << std::endl
     << memory_ss.str()
     << "    }" << std::endl
     << "," << std::endl
     << "    \"Unit\": {" << std::endl
     << "        \"Time\": \"ms\"," << std::endl
     << "        \"Memory\": \"kB\"" << std::endl
     << "    }" << std::endl
     << "}" << std::endl
     << std::flush;
  os.copyfmt(state);
}

void AggregateStats::clear() {
  std::unique_lock<std::mutex> lk(m_);
  stats_.clear();
}

}  // namespace profiler
}  // namespace mxnet
