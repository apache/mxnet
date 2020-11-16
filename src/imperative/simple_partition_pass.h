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
 * Copyright (c) 2019-2020 by Contributors
 * \file simple_partition_pass.h
 * \brief Simple pass for partitioning a graph.
 * \author Clement Fuji Tsang, Przemyslaw Tredak
 */
#ifndef MXNET_IMPERATIVE_SIMPLE_PARTITION_PASS_H_
#define MXNET_IMPERATIVE_SIMPLE_PARTITION_PASS_H_

#include <mxnet/base.h>
#include <mxnet/op_attr_types.h>
#include <mxnet/operator.h>
#include <nnvm/graph_attr_types.h>
#include <utility>
#include <deque>
#include <algorithm>
#include <vector>
#include <tuple>

#include "exec_pass.h"

namespace mxnet {
namespace exec {

namespace detail {

using Interval = std::pair<int, int>;
using IntervalVec = std::vector<Interval>;

/* \brief Return the set that fully contains the other set, or nullptr
 *        if neither set is a subset of another.
 */
const IntervalVec*  LargerSet(const IntervalVec* const first,
                              const IntervalVec* const second) noexcept;

/* \brief Compute the sum of the 2 sets and store it in my_set.
 */
void MergeSets(const IntervalVec** const my_set,
               const IntervalVec* const other_set,
               std::vector<std::unique_ptr<const IntervalVec>>* const storage) noexcept;

/* \brief Returns true if there is non-empty intersection
 *        between the 2 sets.
 */
bool Intersect(const IntervalVec& checked_sets,
               const IntervalVec& excluded_sets) noexcept;

/* \brief Add a single entry to the sets.
 */
void AddSet(const IntervalVec** const sets, const int set_to_add,
            std::vector<std::unique_ptr<const IntervalVec>>* const storage) noexcept;

/* \brief Get the true mapping of the set (which could change
 *        due to merging of multiple sets.
 */
int GetSetMapping(const int set, std::vector<int>* const set_mapping) noexcept;

/* \brief Check if 2 ids are on the same side of the cutoff
 *        (so either both on the FWD side or the BWD side).
 */
inline bool IsSamePass(const int my_id, const int their_id, const int cutoff) noexcept {
  return (my_id > cutoff && their_id > cutoff) ||
         (my_id <= cutoff && their_id <= cutoff);
}

/* \brief Check if adding a new node to the set changes the excluded set of the future
 *        fused node. If so, update all descendants of the fused node.
 *
 * \param combined_excluded_sets_ptr pointer to the set's list of excluded sets
 *                                   before adding the new node
 * \param new_excluded_sets list of excluded sets of the new node
 * \param excluded_sets_ptr pointer to the lists of excluded sets of all the nodes
 * \param set_id number of the set, to which the new node is added
 * \param first_node_in_set id of the first node in the set, according to topological ordering
 * \param new_node_id id of the node added to the set
 * \param set_assignment assignment of sets
 * \param set_mapping_ptr pointer to the mappings of sets
 * \param inverse_set_mapping inverse mapping of the set
 * \param storage memory storage
 */
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
                                          storage) noexcept;

}  // namespace detail


/* \brief Get all subsets of nodes, where:
 *  - graph constructed from nodes in each subset is a connected graph
 *  - every node fulfills a predicate is_compatible
 *  - if nodes u and v are part of a subset, then for each path between
 *    u and v in the original directed graph, all nodes on those paths
 *    are also part of the subset
 * \param g NNVM graph
 * \param num_forward_outputs Number of outputs from the graph that come
 *                            from the forward pass
 * \param is_compatible A function taking nnvm::Node* and returning bool
 *                      which identifies which nodes could be included in
 *                      subsets.
 * \param is_input_only_compatible A function taking nnvm::Node* and
 *                                 returning bool which identifies which
 *                                 nodes could be included in subsets only
 *                                 as the first operations (their inputs
 *                                 need to be excluded).
 * \return tuple (subset assignment, number of found subsets)
 */
template<typename FCompatible, typename FInputOnlyCompatible>
std::tuple<std::vector<int>, int> GetCompatibleSubsets(
    const Graph& g,
    const size_t num_forward_outputs,
    FCompatible is_compatible,
    FInputOnlyCompatible is_input_only_compatible) {

  using namespace detail;
  const auto& idx = g.indexed_graph();
  std::vector<int> set_assignment(idx.num_nodes(), -1);
  std::vector<const std::vector<Interval>*> excluded_sets(idx.num_nodes());
  std::vector<int> set_mapping;
  std::vector<const std::vector<Interval>*> combined_excluded_sets;
  std::vector<int> first_node_in_set;
  std::vector<const std::vector<Interval>*> inverse_set_mapping;
  std::vector<std::unique_ptr<const std::vector<Interval>>> storage;

  int last_forward_node = -1;
  for (size_t i = 0; i < num_forward_outputs; ++i) {
    const int output_id = idx.outputs()[i].node_id;
    if (last_forward_node < output_id) {
      last_forward_node = output_id;
    }
  }

  int num_sets = 0;
  for (size_t i = 0; i < idx.num_nodes(); ++i) {
    const auto& node = idx[i];
    auto& my_excluded_sets = excluded_sets[i];
    for (const auto& input : node.inputs) {
      MergeSets(&my_excluded_sets, excluded_sets[input.node_id], &storage);
    }
    if (is_compatible(node.source)) {
      int my_set = -1;
      for (const auto& input : node.inputs) {
        int their_set = GetSetMapping(set_assignment[input.node_id], &set_mapping);
        if (their_set != -1 &&
            their_set != my_set &&
            IsSamePass(i, input.node_id, last_forward_node) &&
            (my_excluded_sets == nullptr ||
            !Intersect(*inverse_set_mapping[their_set], *my_excluded_sets))) {
          if (my_set == -1) {
            my_set = their_set;
            CheckAndUpdateCombinedExcludedSets(&(combined_excluded_sets[their_set]),
                                               my_excluded_sets,
                                               &excluded_sets,
                                               their_set,
                                               first_node_in_set[their_set],
                                               i,
                                               set_assignment,
                                               &set_mapping,
                                               *(inverse_set_mapping[their_set]),
                                               &storage);
          } else {
            MergeSets(&inverse_set_mapping[my_set],
                      inverse_set_mapping[their_set],
                      &storage);
            set_mapping[their_set] = my_set;
            first_node_in_set[my_set] = std::min(first_node_in_set[my_set],
                                                 first_node_in_set[their_set]);
            CheckAndUpdateCombinedExcludedSets(&(combined_excluded_sets[their_set]),
                                               combined_excluded_sets[my_set],
                                               &excluded_sets,
                                               my_set,
                                               first_node_in_set[my_set],
                                               i,
                                               set_assignment,
                                               &set_mapping,
                                               *(inverse_set_mapping[my_set]),
                                               &storage);
          }
        }
      }
      if (my_set == -1) {
        set_mapping.emplace_back(num_sets);
        combined_excluded_sets.emplace_back(my_excluded_sets);
        first_node_in_set.emplace_back(i);
        storage.emplace_back(std::make_unique<std::vector<Interval>>(
                               1, std::make_pair(num_sets,
                                                 num_sets)));
        inverse_set_mapping.emplace_back(storage.back().get());
        my_set = num_sets++;
      }
      set_assignment[i] = my_set;
    } else {
      for (const auto& input : node.inputs) {
        int their_set = GetSetMapping(set_assignment[input.node_id], &set_mapping);
        if (their_set != -1) {
          AddSet(&my_excluded_sets, their_set, &storage);
        }
      }
      if ((is_input_only_compatible != nullptr) &&
          is_input_only_compatible(node.source)) {
        set_mapping.emplace_back(num_sets);
        combined_excluded_sets.emplace_back(my_excluded_sets);
        first_node_in_set.emplace_back(i);
        storage.emplace_back(std::make_unique<std::vector<Interval>>(
                               1, std::make_pair(num_sets,
                                                 num_sets)));
        inverse_set_mapping.emplace_back(storage.back().get());
        set_assignment[i] = num_sets++;
      }
    }
  }

  for (int& set : set_assignment) {
    set = GetSetMapping(set, &set_mapping);
  }

  std::vector<int> set_reorder(num_sets, 0);
  // First count the number of elements in each set.
  for (int& set : set_assignment) {
    if (set != -1) {
      ++set_reorder[set];
    }
  }
  // Then reorder them, removing sets that have
  // only a single element.
  int final_num_sets = 0;
  for (int& set : set_reorder) {
    if (set > 1) {
      set = final_num_sets++;
    } else {
      set = -1;
    }
  }

  for (int& set : set_assignment) {
    if (set != -1) {
      set = set_reorder[set];
    }
  }

  return {set_assignment, final_num_sets};
}

}  // namespace exec
}  // namespace mxnet
#endif  // MXNET_IMPERATIVE_SIMPLE_PARTITION_PASS_H_
