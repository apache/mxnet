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
 * Copyright (c) 2020 by Contributors
 * \file simple_partition_pass.cc
 * \brief Utilities used in simple partition pass
 * \author Przemyslaw Tredak
 */

#include "./simple_partition_pass.h"
#include <memory>
#include <utility>

namespace mxnet {
namespace exec {

namespace detail {

void MergeSets(std::vector<Interval>** my_set,
               std::vector<Interval>* other_set,
               std::vector<std::unique_ptr<std::vector<Interval>>>* storage) {
  if ((*my_set == nullptr) || (*my_set)->size() == 0) {
    *my_set = other_set;
    return;
  }
  if (other_set == nullptr || other_set->size() == 0 || other_set == (*my_set)) {
    return;
  }
  size_t current_interval = 0, current_other_interval = 0;
  storage->emplace_back(std::make_unique<std::vector<Interval>>());
  auto& new_set = storage->back();
  int last_end = -10;  // less than -1
  while (current_interval < (*my_set)->size() &&
         current_other_interval < other_set->size()) {
    const auto& mine = (**my_set)[current_interval];
    const auto& other = (*other_set)[current_other_interval];
    if (other.second < mine.first - 1) {
      // other interval is before ours
      if (last_end >= other.first - 1) {
        new_set->back().second = other.second;
      } else {
        new_set->emplace_back(other);
      }
      last_end = other.second;
      ++current_other_interval;
    } else if (other.first > mine.second + 1) {
      // other interval is after ours
      if (last_end >= mine.first - 1) {
        new_set->back().second = mine.second;
      } else {
        new_set->emplace_back(mine);
      }
      last_end = mine.second;
      ++current_interval;
    } else {
      // Intervals can be merged together
      Interval n(std::min(mine.first, other.first),
                 std::max(mine.second, other.second));
      if (last_end >= n.first - 1) {
        new_set->back().second = n.second;
      } else {
        new_set->emplace_back(n);
      }
      last_end = n.second;
      ++current_interval;
      ++current_other_interval;
    }
  }
  // Add the rest of entries
  new_set->insert(new_set->end(), (*my_set)->begin() + current_interval,
                 (*my_set)->end());
  new_set->insert(new_set->end(), other_set->begin() + current_other_interval,
                 other_set->end());
  *my_set = new_set.get();
}

bool Intersect(const std::vector<Interval>& checked_sets,
               const std::vector<Interval>& excluded_sets) {
  size_t current_interval = 0, current_other_interval = 0;
  while (current_interval < checked_sets.size() &&
         current_other_interval < excluded_sets.size()) {
    const auto& mine = checked_sets[current_interval];
    const auto& other = excluded_sets[current_other_interval];
    if (other.second < mine.first) {
      // other interval is before ours
      ++current_other_interval;
    } else if (other.first > mine.second) {
      // other interval is after ours
      ++current_interval;
    } else {
      // Intervals intersect
      return true;
    }
  }
  return false;
}

void AddSet(std::vector<Interval>** sets, const int set_to_add,
            std::vector<std::unique_ptr<std::vector<Interval>>>* storage) {
  if (*sets != nullptr && (*sets)->size() != 0) {
    bool found = false;
    for (auto& interval : (**sets)) {
      if (set_to_add >= interval.first - 1 ||
          set_to_add <= interval.second + 1) {
        interval.first = std::min(set_to_add, interval.first);
        interval.second = std::max(set_to_add, interval.second);
        found = true;
        break;
      }
    }
    if (found) return;
  }
  storage->emplace_back(
      std::make_unique<std::vector<Interval>>(1, std::make_pair(set_to_add, set_to_add)));
  MergeSets(sets, storage->back().get(), storage);
}

int GetSetMapping(int set, std::vector<int>* set_mapping) {
  if (set == -1) return -1;
  int temp = set;
  while ((*set_mapping)[temp] != temp) {
    temp = (*set_mapping)[temp];
  }
  (*set_mapping)[set] = temp;
  return temp;
}

}  // namespace detail

}  // namespace exec
}  // namespace mxnet
