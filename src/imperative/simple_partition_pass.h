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
 * Copyright (c) 2019 by Contributors
 * \file simple_partition_pass.h
 * \brief Simple pass for partitioning a graph.
 * \author Clement Fuji Tsang
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

void MergeSets(std::vector<Interval>** my_set,
               std::vector<Interval>* other_set,
               std::vector<std::unique_ptr<std::vector<Interval>>>* storage);

bool Intersect(const std::vector<Interval>& checked_sets,
               const std::vector<Interval>& excluded_sets);

void AddSet(std::vector<Interval>** sets, const int set_to_add,
            std::vector<std::unique_ptr<std::vector<Interval>>>* storage);

int GetSetMapping(int set, std::vector<int>* set_mapping);

inline bool IsSamePass(int my_id, int their_id, int cutoff) {
  return (my_id > cutoff && their_id > cutoff) ||
         (my_id <= cutoff && their_id <= cutoff);
}

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
  std::vector<int> sets(idx.num_nodes(), -1);
  std::vector<std::vector<Interval>*> excluded_sets(idx.num_nodes());
  std::vector<int> set_mapping;
  std::vector<std::vector<Interval>*> inverse_set_mapping;
  std::vector<std::unique_ptr<std::vector<Interval>>> storage;

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
        int their_set = GetSetMapping(sets[input.node_id], &set_mapping);
        if (their_set != -1 &&
            their_set != my_set &&
            IsSamePass(i, input.node_id, last_forward_node) &&
            (my_excluded_sets == nullptr ||
            !Intersect(*inverse_set_mapping[their_set], *my_excluded_sets))) {
          if (my_set == -1) {
            my_set = their_set;
          } else {
            MergeSets(&inverse_set_mapping[my_set],
                      inverse_set_mapping[their_set],
                      &storage);
            set_mapping[their_set] = my_set;
          }
        }
      }
      if (my_set == -1) {
        set_mapping.emplace_back(num_sets);
        storage.emplace_back(std::make_unique<std::vector<Interval>>(
                               1, std::make_pair(num_sets,
                                                 num_sets)));
        inverse_set_mapping.emplace_back(storage.back().get());
        my_set = num_sets++;
      }
      sets[i] = my_set;
    } else {
      for (const auto& input : node.inputs) {
        int their_set = GetSetMapping(sets[input.node_id], &set_mapping);
        if (their_set != -1) {
          AddSet(&my_excluded_sets, their_set, &storage);
        }
      }
      if ((is_input_only_compatible != nullptr) &&
          is_input_only_compatible(node.source)) {
        set_mapping.emplace_back(num_sets);
        storage.emplace_back(std::make_unique<std::vector<Interval>>(
                               1, std::make_pair(num_sets,
                                                 num_sets)));
        inverse_set_mapping.emplace_back(storage.back().get());
        sets[i] = num_sets++;
      }
    }
  }

  for (int& set : sets) {
    set = GetSetMapping(set, &set_mapping);
  }

  std::vector<int> set_reorder(num_sets, 0);
  // First count the number of elements in each set.
  for (int& set : sets) {
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

  for (int& set : sets) {
    if (set != -1) {
      set = set_reorder[set];
    }
  }

  return {sets, final_num_sets};
}

}  // namespace exec
}  // namespace mxnet
#endif  // MXNET_IMPERATIVE_SIMPLE_PARTITION_PASS_H_
