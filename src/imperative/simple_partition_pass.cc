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

const IntervalVec* LargerSet(const IntervalVec* const first,
                             const IntervalVec* const second) noexcept {
  const IntervalVec* ret = nullptr;
  auto first_iter = first->begin();
  auto second_iter = second->begin();
  while (first_iter != first->end() &&
         second_iter != second->end()) {
    if (*first_iter == *second_iter) {
      ++first_iter;
      ++second_iter;
    } else {
      // Entry in first set not seen in the second set
      if (first_iter->second < second_iter->first) {
        if (ret == first || ret == nullptr) {
          ret = first;
          ++first_iter;
        } else {
          return nullptr;
        }
        continue;
      }
      // Entry in second set not seen in the first set
      if (second_iter->second < first_iter->first) {
        if (ret == second || ret == nullptr) {
          ret = second;
          ++second_iter;
        } else {
          return nullptr;
        }
        continue;
      }
      // Entry in first set fully encloses the entry in the second set
      if (first_iter->first <= second_iter->first &&
          first_iter->second >= second_iter->second) {
        if (ret == first || ret == nullptr) {
          ret = first;
          ++second_iter;
        } else {
          return nullptr;
        }
        continue;
      }
      // Entry in second set fully encloses the entry in the first set
      if (second_iter->first <= first_iter->first &&
          second_iter->second >= first_iter->second) {
        if (ret == second || ret == nullptr) {
          ret = second;
          ++first_iter;
        } else {
          return nullptr;
        }
        continue;
      }
      // Entries intersect but one is not fully enclosed in the other
      return nullptr;
    }
  }
  if (ret == nullptr) {
    // The common part is the same
    return second_iter == second->end() ? first : second;
  } else {
    if ((ret == first && second_iter == second->end()) ||
        (ret == second && first_iter == first->end())) {
      return ret;
    }
  }
  return nullptr;
}

void MergeSets(const IntervalVec** const my_set,
               const IntervalVec* const other_set,
               std::vector<std::unique_ptr<const IntervalVec>>* const storage) noexcept {
  if ((*my_set == nullptr) || (*my_set)->size() == 0) {
    *my_set = other_set;
    return;
  }
  if (other_set == nullptr || other_set->size() == 0) {
    return;
  }
  auto* larger_set = LargerSet(*my_set, other_set);
  if (larger_set != nullptr) {
    *my_set = larger_set;
    return;
  }
  auto my_iter = (*my_set)->cbegin();
  auto other_iter = other_set->cbegin();
  auto new_set = IntervalVec();
  int last_end = -10;  // less than -1
  while (my_iter != (*my_set)->cend() &&
         other_iter != other_set->cend()) {
    const auto& mine = *my_iter;
    const auto& other = *other_iter;
    if (other.second < mine.first - 1) {
      // other interval is before ours
      if (last_end >= other.first - 1) {
        new_set.back().second = other.second;
      } else {
        new_set.emplace_back(other);
      }
      last_end = other.second;
      ++other_iter;
    } else if (other.first > mine.second + 1) {
      // other interval is after ours
      if (last_end >= mine.first - 1) {
        new_set.back().second = mine.second;
      } else {
        new_set.emplace_back(mine);
      }
      last_end = mine.second;
      ++my_iter;
    } else {
      // Intervals can be merged together
      Interval n(std::min(mine.first, other.first),
                 std::max(mine.second, other.second));
      if (last_end >= n.first - 1) {
        new_set.back().second = n.second;
      } else {
        new_set.emplace_back(n);
      }
      last_end = n.second;
      if (other.second >= mine.second) {
        ++my_iter;
      }
      if (mine.second >= other.second) {
        ++other_iter;
      }
    }
  }
  auto remaining_iter = my_iter == (*my_set)->cend() ? other_iter : my_iter;
  auto remaining_end = my_iter == (*my_set)->cend() ? other_set->cend() : (*my_set)->cend();
  // Add the rest of entries
  for (; remaining_iter != remaining_end; ++remaining_iter) {
    auto& mine = new_set.back();
    const auto& other = *remaining_iter;
    if (other.second < mine.first - 1) {
      // other interval is before ours, should never happen
      continue;
    } else if (other.first > mine.second + 1) {
      // other interval is after ours
      new_set.emplace_back(other);
    } else {
      // Intervals can be merged together
      mine.first = std::min(mine.first, other.first);
      mine.second = std::max(mine.second, other.second);
    }
  }
  storage->emplace_back(std::make_unique<IntervalVec>(std::move(new_set)));
  *my_set = storage->back().get();
}

bool Intersect(const IntervalVec& checked_sets,
               const IntervalVec& excluded_sets) noexcept {
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

void AddSet(const IntervalVec** const sets, const int set_to_add,
            std::vector<std::unique_ptr<const IntervalVec>>* const storage) noexcept {
  if (*sets != nullptr && (*sets)->size() != 0) {
    for (auto& interval : (**sets)) {
      if (set_to_add >= interval.first &&
          set_to_add <= interval.second) {
        return;
      }
    }
  }
  storage->emplace_back(
      std::make_unique<IntervalVec>(1, std::make_pair(set_to_add, set_to_add)));
  MergeSets(sets, storage->back().get(), storage);
}

int GetSetMapping(const int set, std::vector<int>* const set_mapping) noexcept {
  if (set == -1) return -1;
  int temp = set;
  while ((*set_mapping)[temp] != temp) {
    temp = (*set_mapping)[temp];
  }
  (*set_mapping)[set] = temp;
  return temp;
}

void CheckAndUpdateCombinedExcludedSets(const IntervalVec** const combined_excluded_sets_ptr,
                                        const IntervalVec* const new_excluded_sets,
                                        std::vector<const IntervalVec*>* const excluded_sets_ptr,
                                        const int set_id,
                                        const int first_node_in_set,
                                        const size_t new_node_id,
                                        const std::vector<int>& set_assignment,
                                        std::vector<int>* const set_mapping_ptr,
                                        const IntervalVec& inverse_set_mapping,
                                        std::vector<std::unique_ptr<const IntervalVec>>* const
                                          storage) noexcept {
  const auto* previous_excluded_sets = *combined_excluded_sets_ptr;
  MergeSets(combined_excluded_sets_ptr, new_excluded_sets, storage);
  if (new_excluded_sets != nullptr) {
    if (previous_excluded_sets == nullptr ||
        *previous_excluded_sets != **(combined_excluded_sets_ptr)) {
      // Their set's excluded sets list got larger, need to update the descendants
      // of their set
      auto& excluded_sets = *excluded_sets_ptr;
      for (size_t j = first_node_in_set; j < new_node_id; ++j) {
        if (GetSetMapping(set_assignment[j], set_mapping_ptr) == set_id ||
            (excluded_sets[j] != nullptr &&
             Intersect(inverse_set_mapping, *excluded_sets[j]))) {
          MergeSets(&excluded_sets[j], *combined_excluded_sets_ptr, storage);
        }
      }
    }
  }
}

}  // namespace detail

}  // namespace exec
}  // namespace mxnet
