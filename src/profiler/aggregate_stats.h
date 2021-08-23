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

#ifndef MXNET_PROFILER_AGGREGATE_STATS_H_
#define MXNET_PROFILER_AGGREGATE_STATS_H_

#include <string>
#include <map>
#include <cstdint>
#include <ostream>
#include <mutex>
#include "./profiler.h"

namespace mxnet {
namespace profiler {

struct ProfileStat;

class AggregateStats {
 public:
  struct StatData {
    /*!
     * \brief Types that the console printer knows how to format
     */
    enum StatType {
      kDuration = 1,
      kCounter = 2,
      kOther = 4
    };

    StatType  type_ = kOther;
    size_t    total_count_ = 0;
    uint64_t  total_aggregate_ = 0;
    uint64_t  max_aggregate_ = 0;
    uint64_t  min_aggregate_ = INT_MAX;
  };

  /*!
   * \brief Record aggregate profile data
   * \param stat SIngle profile statistics to add to the accumulates statistics
   */
  void OnProfileStat(const ProfileStat& stat);
  /*!
   * \brief Print profliing statistics to console in a tabular format
   * \param sort_by by which stat to sort the entries, can be "avg", "min", "max", or "count"
   * \param ascending whether to sort ascendingly
   */
  void DumpTable(std::ostream& os, int sort_by, int ascending);
  /*!
   * \brief Print profliing statistics to console in json format
   *    * \param sort_by by which stat to sort the entries, can be "avg", "min", "max", or "count"
   * \param ascending whether to sort ascendingly
   */
  void DumpJson(std::ostream& os, int sort_by, int ascending);
  /*!
   * \brief Delete all of the current statistics
   */
  void clear();
  /* !\brief by which stat to sort */
  enum class SortBy {
    Total, Avg, Min, Max, Count
  };

 private:
  /*! \brief Should rarely collide, so most locks should occur only in user-space (futex) */
  std::mutex m_;
  /* !\brief Stat type -> State name -> Stats */
  std::map<std::string, std::unordered_map<std::string, StatData>> stats_;
};

}  // namespace profiler
}  // namespace mxnet
#endif  // MXNET_PROFILER_AGGREGATE_STATS_H_
