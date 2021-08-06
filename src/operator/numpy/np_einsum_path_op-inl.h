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

/*
 * Copyright (c) 2005-2019, NumPy Developers.
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *  * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *
 *  * Neither the name of the NumPy Developers nor the names of any
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*!
 * \file np_einsum_path_op-inl.h
 * \brief Function definition of numpy-compatible einsum_path operator
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_EINSUM_PATH_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_EINSUM_PATH_OP_INL_H_

#include <mxnet/operator_util.h>
#include <functional>
#include <algorithm>
#include <string>
#include <vector>
#include <bitset>

namespace mxnet {
namespace op {

const int MAXAXIS = 128;

typedef std::vector<std::bitset<MAXAXIS> > SetVector;

struct Contraction {
  std::bitset<MAXAXIS> new_result;
  std::vector<std::bitset<MAXAXIS> > remaining;
  std::bitset<MAXAXIS> idx_removed;
  std::bitset<MAXAXIS> idx_contract;
};

struct Alternative {
  int64_t cost[2];
  std::vector<int> positions;
  SetVector new_input_sets;
};

struct Step {
  std::vector<int> contract_inds;
  std::bitset<MAXAXIS> idx_removed;
  std::string einsum_str, blas2einsum_str, einsum2blas_str;
  std::vector<std::string> input_list;
  bool do_blas, do_einsum;
  TShape oshape, tshape;
  Tuple<int> left_pos, right_pos;
};

inline size_t _compute_size_by_dict(const std::string& indices,
                                    const dim_t idx_dict[]) {
  size_t ret = 1;
  for (const char& c : indices) {
    ret *= idx_dict[static_cast<int>(c)];
  }
  return ret;
}

inline size_t _compute_size_by_dict(const std::bitset<MAXAXIS>& indices,
                                    const dim_t idx_dict[]) {
  size_t ret = 1;
  for (int i = 0; i < MAXAXIS; ++i) {
    if (indices[i]) {
      ret *= idx_dict[i];
    }
  }
  return ret;
}

inline int64_t _flop_count(const std::string& idx_contraction,
                           bool inner,
                           int num_terms,
                           const dim_t size_dictionary[]) {
  size_t overall_size = _compute_size_by_dict(idx_contraction, size_dictionary);
  int op_factor = std::max(1, num_terms - 1);
  if (inner) {
    ++op_factor;
  }
  return static_cast<int64_t>(overall_size) * op_factor;
}

inline int64_t _flop_count(const std::bitset<MAXAXIS>& idx_contraction,
                           bool inner,
                           int num_terms,
                           const dim_t size_dictionary[]) {
  size_t overall_size = _compute_size_by_dict(idx_contraction, size_dictionary);
  int op_factor = std::max(1, num_terms - 1);
  if (inner) {
    ++op_factor;
  }
  return static_cast<int64_t>(overall_size) * op_factor;
}

inline Contraction _find_contraction(const std::vector<int>& positions,
                                     const SetVector& input_sets,
                                     const std::bitset<MAXAXIS>& output_set) {
  Contraction ret;
  std::bitset<MAXAXIS> idx_remain(output_set);
  size_t size = input_sets.size();
  for (size_t i = 0; i < size; ++i) {
    if (std::find(positions.begin(), positions.end(), i) != positions.end()) {
      ret.idx_contract |= input_sets[i];
    } else {
      ret.remaining.push_back(input_sets[i]);
      idx_remain |= input_sets[i];
    }
  }
  ret.new_result = idx_remain & ret.idx_contract;
  ret.idx_removed = (ret.idx_contract & ~ret.new_result);
  ret.remaining.push_back(ret.new_result);

  return ret;
}

inline int _parse_possible_contraction(const std::vector<int>& positions,
                                       const SetVector& input_sets,
                                       const std::bitset<MAXAXIS>& output_set,
                                       const dim_t idx_dict[],
                                       size_t memory_limit,
                                       int64_t path_cost,
                                       int64_t naive_cost,
                                       Alternative* ret) {
  // Find the contraction
  Contraction contract = _find_contraction(positions, input_sets, output_set);

  // Sieve the results based on memory_limit
  size_t new_size = _compute_size_by_dict(contract.new_result, idx_dict);
  if (new_size > memory_limit) {
    return -1;
  }

  // Build sort tuple
  size_t old_sizes = 0;
  for (auto p : positions) {
    old_sizes += _compute_size_by_dict(input_sets[p], idx_dict);
  }
  int64_t remove_size = static_cast<int64_t>(old_sizes) - static_cast<int64_t>(new_size);

  int64_t cost = _flop_count(contract.idx_contract, contract.idx_removed.any(),
                             positions.size(), idx_dict);
  ret->cost[0] = -remove_size;
  ret->cost[1] = cost;

  // Sieve based on total cost as well
  if (path_cost + cost > naive_cost) {
    return -1;
  }

  // Add contraction to possible choices
  ret->positions = positions;
  ret->new_input_sets = contract.remaining;
  return 0;
}

inline void _update_other_results(std::vector<Alternative>* results,
                                  const Alternative& best) {
  const std::vector<int>& best_con = best.positions;
  int bx = best_con[0], by = best_con[1];
  size_t size = results->size();

  for (int i = static_cast<int>(size) - 1; i >= 0; --i) {
    int x = results->at(i).positions[0], y = results->at(i).positions[1];

    // Ignore results involving tensors just contracted
    if (x == bx || x == by || y == bx || y == by) {
      results->erase(results->begin() + i);
      continue;
    }

    // Update the input_sets
    CHECK_GT(by, bx)
      << "by must be greater than bx";
    results->at(i).new_input_sets.erase(results->at(i).new_input_sets.begin() +
                                        by - static_cast<int>(by > x) - static_cast<int>(by > y));
    results->at(i).new_input_sets.erase(results->at(i).new_input_sets.begin() +
                                        bx - static_cast<int>(bx > x) - static_cast<int>(bx > y));
    results->at(i).new_input_sets.push_back(best.new_input_sets.back());

    // Update the position indices
    results->at(i).positions[0] = x - static_cast<int>(x > bx) - static_cast<int>(x > by);
    results->at(i).positions[1] = y - static_cast<int>(y > bx) - static_cast<int>(y > by);
  }
}

inline std::vector<std::vector<int> > _greedy_path(const SetVector* input_sets,
                                                   const std::bitset<MAXAXIS>& output_set,
                                                   const dim_t idx_dict[],
                                                   size_t memory_limit) {
  int isize = static_cast<int>(input_sets->size());
  int iteration_num = isize;
  // Handle trivial cases that leaked through
  if (isize == 1) {
    return std::vector<std::vector<int> >{std::vector<int>{0}};
  } else if (isize == 2) {
    return std::vector<std::vector<int> >{std::vector<int>{0, 1}};
  }

  // Build up a naive cost
  std::vector<int> range(isize);
  for (int i = 0; i < isize; ++i) {
    range[i] = i;
  }
  Contraction contract = _find_contraction(range, *input_sets, output_set);
  int64_t naive_cost = _flop_count(contract.idx_contract, contract.idx_removed.any(),
                                   isize, idx_dict);

  // Initially iterate over all pairs
  std::vector<Alternative> known_contractions;
  Alternative best;
  int64_t path_cost = 0;
  std::vector<std::vector<int> > ret;

  for (int iteration = 0; iteration + 1 < iteration_num; ++iteration) {
    if (iteration == 0) {
      for (int x = 0; x < isize; ++x) {
        for (int y = x + 1; y < isize; ++y) {
          if (!((input_sets->at(x) & input_sets->at(y)).any())) {
            continue;
          }
          Alternative alternative;
          int result = _parse_possible_contraction(std::vector<int>{x, y},
                                                   *input_sets,
                                                   output_set,
                                                   idx_dict,
                                                   memory_limit,
                                                   path_cost,
                                                   naive_cost,
                                                   &alternative);
          if (result != -1) {
            known_contractions.push_back(alternative);
          }
        }
      }
    } else {
      for (int x = 0; x < isize - 1; ++x) {
        int y = isize - 1;
        if (!((input_sets->at(x) & input_sets->at(y)).any())) {
            continue;
          }
          Alternative alternative;
          int result = _parse_possible_contraction(std::vector<int>{x, y},
                                                   *input_sets,
                                                   output_set,
                                                   idx_dict,
                                                   memory_limit,
                                                   path_cost,
                                                   naive_cost,
                                                   &alternative);
          if (result != -1) {
            known_contractions.push_back(alternative);
          }
      }
    }

    // If we do not have a inner contraction, rescan pairs including outer products
    if (known_contractions.size() == 0) {
      // Then check the outer productsj
      for (int x = 0; x < isize; ++x) {
        for (int y = x + 1; y < isize; ++y) {
          Alternative alternative;
          int result = _parse_possible_contraction(std::vector<int>{x, y},
                                                   *input_sets,
                                                   output_set,
                                                   idx_dict,
                                                   memory_limit,
                                                   path_cost,
                                                   naive_cost,
                                                   &alternative);
          if (result != -1) {
            known_contractions.push_back(alternative);
          }
        }
      }

      // If we still did not find any remaining contractions, default back to einsum like behavior
      if (known_contractions.size() == 0) {
        std::vector<int> range(isize);
        for (int i = 0; i < isize; ++i) {
          range[i] = i;
        }
        ret.push_back(range);
        break;
      }
    }

    // Sort based on first index
    int64_t best_cost[2];
    int idx = -1, size = static_cast<int>(known_contractions.size());
    for (int i = 0; i < size; ++i) {
      auto x = known_contractions[i];
      if (idx == -1) {
        best_cost[0] = x.cost[0];
        best_cost[1] = x.cost[1];
        idx = i;
      } else if (x.cost[0] < best_cost[0] ||
                 (x.cost[0] == best_cost[0] &&
                  x.cost[1] < best_cost[1])) {
        best_cost[0] = x.cost[0];
        best_cost[1] = x.cost[1];
        idx = i;
      }
    }
    best = known_contractions[idx];

    // Now propagate as many unused contractions as possible to next iteration
    _update_other_results(&known_contractions, best);

    // Next iteration only compute contractions with the new tensor
    // All other contractions have been accounted for
    input_sets = &best.new_input_sets;
    isize = static_cast<int>(input_sets->size());

    // Update path and total cost
    ret.push_back(best.positions);
    path_cost += best.cost[1];
  }
  return ret;
}

inline bool _can_dot(const std::vector<std::string>& inputs,
                     const std::bitset<MAXAXIS>& result,
                     const std::bitset<MAXAXIS>& idx_removed) {
  // All `dot` calls remove indices
  if (!idx_removed.any()) {
    return false;
  }

  // BLAS can only handle two operands
  if (inputs.size() != 2) {
    return false;
  }

  const std::string& input_left = inputs[0];
  const std::string& input_right = inputs[1];

  if (input_left.size() == 0 || input_right.size() == 0) {
    return false;
  }

  for (int i = 0; i < 2; ++i) {
    for (const char& c : inputs[i]) {
      // can't deal with repeated indices on same input or more than 2 total
      size_t nl = std::count(input_left.begin(), input_left.end(), c);
      size_t nr = std::count(input_right.begin(), input_right.end(), c);
      if (nl > 1 || nr > 1 || nl + nr > 2) {
        return false;
      }

      // can't do implicit summation or dimension collapse e.g.
      // "ab,bc->c" (implicitly sum over 'a')
      // "ab,ca->ca" (take diagonal of 'a')
      if (nl + nr == static_cast<size_t>(result.test(c)) + 1) {
        return false;
      }
    }
  }

  // Build a few temporaries
  std::bitset<MAXAXIS> set_left;
  std::bitset<MAXAXIS> set_right;
  for (const char& c : input_left) {
    set_left.set(c);
  }
  for (const char& c : input_right) {
    set_right.set(c);
  }
  std::bitset<MAXAXIS> keep_left = set_left & ~idx_removed;
  std::bitset<MAXAXIS> keep_right = set_right & ~idx_removed;
  size_t rs = idx_removed.count();

  // At this point we are a DOT, GEMV, or GEMM operation

  // Handle inner products

  // DDOT with aligned data
  if (input_left == input_right)
    return true;

  // DDOT without aligned data (better to use einsum)
  if (set_left == set_right)
    return false;

  // Handle the 4 possible (aligned) GEMV or GEMM cases

  // GEMM or GEMV no transpose
  if (std::equal(input_left.end() - rs,
                 input_left.end(),
                 input_right.begin())) {
    return true;
  }

  // GEMM or GEMV transpose both
  if (std::equal(input_left.begin(),
                 input_left.begin() + rs,
                 input_right.end() - rs)) {
    return true;
  }

  // GEMM or GEMV transpose right
  if (std::equal(input_left.end() - rs,
                 input_left.end(),
                 input_right.end() - rs)) {
    return true;
  }

  // GEMM or GEMV transpose left
  if (std::equal(input_left.begin(),
                 input_left.begin() + rs,
                 input_right.begin())) {
    return true;
  }

  // Einsum is faster than GEMV if we have to copy data
  if (!keep_left.any() || !keep_right.any()) {
    return false;
  }

  // We are a matrix-matrix product, but we need to copy data
  return true;
}


inline int _count_substring(const std::string& str,
                            const std::string& sub) {
  int count = 0;
  std::string::size_type pos = 0;
  while ((pos = str.find(sub, pos)) != std::string::npos) {
    ++count;
    pos += sub.length();
  }
  return count;
}

inline std::bitset<MAXAXIS> str2set(const std::string& str) {
  std::bitset<MAXAXIS> ret;
  for (const char& c : str) {
    ret.set(static_cast<int>(c));
  }
  return ret;
}

inline std::string set2str(const std::bitset<MAXAXIS>& set) {
  std::string ret;
  for (int i = 0; i < MAXAXIS; ++i) {
    if (set.test(i)) {
      ret.append(1, static_cast<char>(i));
    }
  }
  return ret;
}

inline std::vector<std::string> split(const std::string& str,
                                      const std::string& sub) {
  std::string::size_type pos = 0;
  std::string::size_type start = 0;
  std::vector<std::string> ret;
  while ((pos = str.find(sub, start)) != std::string::npos) {
    ret.push_back(str.substr(start, pos - start));
    start = pos + sub.length();
  }
  ret.push_back(str.substr(start));
  return ret;
}

inline std::vector<std::string> _parse_einsum_input(
  std::string subscripts,
  const std::vector<TBlob>& operands) {
  const std::string einsum_symbols =
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
  std::bitset<MAXAXIS> einsum_symbols_set;
  for (const char& c : einsum_symbols) {
    einsum_symbols_set.set(c);
  }

  CHECK_NE(operands.size(), 0U)
    << "No input operands";

  auto end_pos = std::remove(subscripts.begin(), subscripts.end(), ' ');
  subscripts.erase(end_pos, subscripts.end());

  // Ensure all characters are valid
  for (const char& c : subscripts) {
    if (c == '.' || c == ',' || c == '-' || c == '>') {
      continue;
    }
    CHECK(einsum_symbols_set.test(c))
      << "Character " << c
      << " is not a valid symbol.";
  }

  // Check for proper "->"
  if (subscripts.find('-') != std::string::npos ||
      subscripts.find('>') != std::string::npos) {
    bool invalid = (std::count(subscripts.begin(), subscripts.end(), '-') > 1 ||
                    std::count(subscripts.begin(), subscripts.end(), '>') > 1);
    CHECK(!invalid && _count_substring(subscripts, "->") == 1)
      << "Subscripts can only contain one '->'.";
  }

  // Parse ellipses
  if (subscripts.find('.') != std::string::npos) {
    std::string used = subscripts;
    used.erase(std::remove_if(used.begin(),
                              used.end(),
                              [](const char& c){return c == '.' ||
                                                       c == ',' ||
                                                       c == '-' ||
                                                       c == '>';}),
               used.end());

    std::bitset<MAXAXIS> used_set = str2set(used);
    std::string ellipse_inds = "";
    for (const char& c : einsum_symbols) {
      if (!used_set.test(static_cast<int>(c))) {
        ellipse_inds.append(1, c);
      }
    }
    int longest = 0;
    std::string input_tmp, output_sub;
    std::vector<std::string> split_subscripts;
    bool out_sub;

    if (subscripts.find("->") != std::string::npos) {
      std::vector<std::string> tmp = split(subscripts, "->");
      input_tmp = tmp[0];
      output_sub = tmp[1];
      split_subscripts = split(input_tmp, ",");
      out_sub = true;
    } else {
      split_subscripts = split(subscripts, ",");
      out_sub = false;
    }

    size_t size_split_subscripts = split_subscripts.size();
    subscripts = "";
    for (size_t i = 0; i < size_split_subscripts; ++i) {
      const std::string& sub = split_subscripts[i];
      if (sub.find('.') != std::string::npos) {
        CHECK_EQ(std::count(sub.begin(), sub.end(), '.'), 3)
          << "Invalid Ellipses";
        CHECK_EQ(_count_substring(sub, "..."), 1)
          << "Invalid Ellipses";

        // Take into account numerical values
        int ellipse_count = 0;
        if (operands[i].shape_.ndim() == 0) {
          ellipse_count = 0;
        } else {
          ellipse_count = std::max(operands[i].shape_.ndim(), 1);
          ellipse_count -= sub.length() - 3;
        }

        if (ellipse_count > longest) {
          longest = ellipse_count;
        }

        CHECK_GE(ellipse_count, 0)
          << "Ellipses lengths do not match.";
        if (ellipse_count == 0) {
          split_subscripts[i].erase(sub.find("..."), 3);
        } else {
          std::string rep_inds = ellipse_inds.substr(ellipse_inds.length() - ellipse_count);
          split_subscripts[i].replace(sub.find("..."), 3, rep_inds);
        }
      }
      subscripts += split_subscripts[i];
      if (i + 1 < size_split_subscripts) {
        subscripts += ",";
      }
    }
    std::string out_ellipse;
    if (longest == 0) {
      out_ellipse = "";
    } else {
      out_ellipse = ellipse_inds.substr(ellipse_inds.length() - longest);
    }

    if (out_sub) {
      output_sub.replace(output_sub.find("..."), 3, out_ellipse);
      subscripts += "->" + output_sub;
    } else {
      // Special care for outputless ellipses
      std::bitset<MAXAXIS> out_ellipse_set = str2set(out_ellipse);
      std::string tmp_subscripts = subscripts, output_subscript = "";
      size_t len_tmp_subscripts = tmp_subscripts.length();
      std::sort(tmp_subscripts.begin(), tmp_subscripts.end());
      for (size_t i = 0; i < len_tmp_subscripts; ++i) {
        const char& c = tmp_subscripts[i];
        if (c == ',') {
          continue;
        }
        CHECK(einsum_symbols_set.test(c))
          << "Character " << c
          << " is not a valid symbol.";
        if ((i == 0 || tmp_subscripts[i - 1] != c) &&
            (i == len_tmp_subscripts - 1 || tmp_subscripts[i + 1] != c) &&
            !out_ellipse_set.test(c)) {
          output_subscript.append(1, c);
        }
      }
      subscripts += "->" + out_ellipse + output_subscript;
    }
  }

  // Build output string if does not exist
  std::vector<std::string> ret(2);
  if (subscripts.find("->") != std::string::npos) {
    ret = split(subscripts, "->");
  } else {
    ret[0] = subscripts;
    ret[1] = "";
    // Build output subscripts
    std::string tmp_subscripts = subscripts;
    size_t len_tmp_subscripts = tmp_subscripts.length();
    std::sort(tmp_subscripts.begin(), tmp_subscripts.end());
    for (size_t i = 0; i < len_tmp_subscripts; ++i) {
      const char& c = tmp_subscripts[i];
      if (c == ',') {
        continue;
      }
      CHECK(einsum_symbols_set.test(c))
        << "Character " << c
        << " is not a valid symbol.";
      if ((i == 0 || tmp_subscripts[i - 1] != c) &&
          (i == len_tmp_subscripts - 1 || tmp_subscripts[i + 1] != c)) {
        ret[1].append(1, c);
      }
    }
  }

  // Make sure output subscripts are in the input
  std::bitset<MAXAXIS> input_subscripts_set = str2set(ret[0]);
  for (const char& c : ret[1]) {
    CHECK(input_subscripts_set.test(c))
      << "Output character " << c
      << " did not appear in the input";
  }

  // Make sure number operands is equivalent to the number of terms
  CHECK_EQ(std::count(ret[0].begin(), ret[0].end(), ',') + 1, operands.size())
    << "Number of einsum subscripts must be equal to the "
    << "number of operands.";

  return ret;
}

inline bool _tensordot_type_check(int type_flag_, const RunContext& run_ctx) {
  return type_flag_ == kFloat32 || type_flag_ == kFloat64 ||
         (type_flag_ == kFloat16 && run_ctx.ctx.dev_mask() == mshadow::gpu::kDevMask);
}

inline std::vector<Step> einsum_path(const std::string& subscripts,
                                     const std::vector<TBlob>& operands,
                                     bool optimize,
                                     const RunContext& run_ctx,
                                     std::vector<std::vector<int> >* ret_path,
                                     std::string* ret_string_repr) {
  // Parsing
  std::vector<std::string> parsed_subscripts = _parse_einsum_input(subscripts, operands);

  // Build a few useful list and sets
  std::vector<std::string> input_list = split(parsed_subscripts[0], ",");
  int isize = static_cast<int>(input_list.size());
  SetVector input_sets;
  for (int i = 0; i < isize; ++i) {
    input_sets.push_back(str2set(input_list[i]));
  }
  std::bitset<MAXAXIS> output_set = str2set(parsed_subscripts[1]);
  std::bitset<MAXAXIS> indices = str2set(parsed_subscripts[0]);
  indices.set(',', false);

  // Get length of each unique dimension and ensure all dimensions are correct
  dim_t dimension_dict[MAXAXIS];
  SetVector broadcast_indices(isize);
  memset(dimension_dict, -1, sizeof(dimension_dict));
  for (int i = 0; i < isize; ++i) {
    const std::string& term = input_list[i];
    const TShape& sh = operands[i].shape_;
    CHECK_EQ(sh.ndim(), term.length())
      << "Einstein sum subscript " << input_list[i]
      << " does not contain the "
      << "correct number of indices for operand " << i << ".";
    size_t len_term = term.length();
    for (size_t j = 0; j < len_term; ++j) {
      dim_t dim = sh[j];
      const char& c = term[j];
      // Build out broadcast indices
      if (dim == 1) {
        broadcast_indices[i].set(c);
      }

      if (dimension_dict[static_cast<int>(c)] != -1) {
        // For broadcasting cases we always want the largest dim size
        if (dimension_dict[static_cast<int>(c)] == 1) {
          dimension_dict[static_cast<int>(c)] = dim;
        }
        CHECK(dim == 1 || dim == dimension_dict[static_cast<int>(c)])
          << "Size of label '" << c
          << "' for operand  " << i
          << " (" << dimension_dict[static_cast<int>(c)]
          << ") does not match previous terms ("
          << dim << ").";
      } else {
        dimension_dict[static_cast<int>(c)] = dim;
      }
    }
  }

  // Compute size of each input array plus the output array
  std::vector<size_t> size_list(isize + 1);
  size_t max_size = 0, memory_arg;
  for (int i = 0; i < isize; ++i) {
    size_list[i] = _compute_size_by_dict(input_list[i], dimension_dict);
    max_size = std::max(max_size, size_list[i]);
  }
  size_list[isize] = _compute_size_by_dict(parsed_subscripts[1], dimension_dict);
  max_size = std::max(max_size, size_list[isize]);
  memory_arg = max_size;

  // Compute naive cost
  // This isn't quite right, need to look into exactly how einsum does this
  size_t sum_len_input_sets = 0;
  for (auto x : input_sets) {
    sum_len_input_sets += x.count();
  }
  bool inner_product = (sum_len_input_sets > indices.count());
  int naive_cost = _flop_count(indices, inner_product, isize, dimension_dict);

  // Compute the path
  std::vector<std::vector<int> > path;
  if (optimize == false) {
    path.push_back(std::vector<int>());
    for (int i = 0; i < isize; ++i) {
      path[0].push_back(i);
    }
  } else {
    path = _greedy_path(&input_sets, output_set, dimension_dict, memory_arg);
  }

  std::vector<int> cost_list;
  std::vector<size_t> scale_list;
  int opt_cost = 1;
  size_t max_i = 0, max_scale = 0, size_path = path.size();
  std::vector<Step> ret(size_path);
  size_list.clear();

  // Build contraction tuple (positions, gemm, einsum_str, remaining)
  for (size_t i = 0; i < size_path; ++i) {
    // Make sure we remove inds from right to left
    std::vector<int> contract_inds = path[i];
    std::sort(contract_inds.begin(), contract_inds.end(), std::greater<int>());

    Contraction contract = _find_contraction(contract_inds, input_sets, output_set);
    input_sets = contract.remaining;

    int64_t cost = _flop_count(contract.idx_contract,
                           contract.idx_removed.any(),
                           contract_inds.size(),
                           dimension_dict);
    opt_cost += cost;
    cost_list.push_back(cost);
    scale_list.push_back(contract.idx_contract.count());
    size_list.push_back(_compute_size_by_dict(contract.new_result, dimension_dict));
    max_i = std::max(max_i, size_list.back());
    max_scale = std::max(max_scale, scale_list.back());

    std::bitset<MAXAXIS> bcast;
    std::vector<std::string> tmp_inputs;
    for (const int& x : contract_inds) {
      tmp_inputs.push_back(input_list[x]);
      input_list.erase(input_list.begin() + x);
      bcast |= broadcast_indices[x];
      broadcast_indices.erase(broadcast_indices.begin() + x);
    }

    std::bitset<MAXAXIS> new_bcast_inds = bcast & ~contract.idx_removed;

    // If we're broadcasting, nix blas
    bool do_blas;
    if ((contract.idx_removed & bcast).any() ||
        !_tensordot_type_check(operands[0].type_flag_, run_ctx)) {
      do_blas = false;
    } else {
      do_blas = _can_dot(tmp_inputs, contract.new_result, contract.idx_removed);
    }

    // Last contraction
    std::string idx_result;
    if (i + 1 == size_path) {
      idx_result = parsed_subscripts[1];
    } else {
      idx_result = set2str(contract.new_result);
      std::sort(idx_result.begin(), idx_result.end(),
                [&dimension_dict](const char& a, const char& b) -> bool {
                  return dimension_dict[static_cast<int>(a)] <
                         dimension_dict[static_cast<int>(b)] ||
                         (dimension_dict[static_cast<int>(a)] ==
                         dimension_dict[static_cast<int>(b)] &&
                         a < b);
                });
    }
    int len_idx_result = static_cast<int>(idx_result.length());
    ret[i].oshape = TShape(len_idx_result, -1);
    for (int j = 0; j < len_idx_result; ++j) {
      ret[i].oshape[j] = dimension_dict[static_cast<int>(idx_result[j])];
    }

    if (do_blas) {
      CHECK_EQ(tmp_inputs.size(), 2U)
        << "BLAS accepts exactly 2 inputs";
      std::string tensor_result = tmp_inputs[0] + tmp_inputs[1];
      tensor_result.erase(std::remove_if(tensor_result.begin(),
                                         tensor_result.end(),
                                         [&](const char& c) {
                                           return contract.idx_removed.test(static_cast<int>(c));}),
                          tensor_result.end());

      // Find indices to contract over
      std::vector<int> left_pos, right_pos;
      left_pos.reserve(MAXAXIS);
      right_pos.reserve(MAXAXIS);
      int tmp[MAXAXIS] = {0};
      int length_left_input = static_cast<int>(tmp_inputs[0].length());
      int length_right_input = static_cast<int>(tmp_inputs[1].length());
      for (int j = 0; j < length_right_input; ++j) {
        if (contract.idx_removed.test(static_cast<int>(tmp_inputs[1][j]))) {
          tmp[static_cast<int>(tmp_inputs[1][j])] = j;
        }
      }
      for (int j = 0; j < length_left_input; ++j) {
        if (contract.idx_removed.test(static_cast<int>(tmp_inputs[0][j]))) {
          left_pos.push_back(j);
          right_pos.push_back(tmp[static_cast<int>(tmp_inputs[0][j])]);
        }
      }
      // Calculate left_pos and right_pos
      ret[i].left_pos = Tuple<int>(left_pos);
      ret[i].right_pos = Tuple<int>(right_pos);
      // Calculate do_einsum
      ret[i].do_einsum = (tensor_result != idx_result);
      // Calculate tshape
      CHECK_EQ(static_cast<int>(tensor_result.length()), len_idx_result)
        << "tensordot produces dim " << tensor_result.length()
        << ", while einsum produces dim " << len_idx_result << ".";
      ret[i].tshape = TShape(len_idx_result, -1);
      for (int j = 0; j < len_idx_result; ++j) {
        ret[i].tshape[j] = dimension_dict[static_cast<int>(tensor_result[j])];
      }
      // Calculate blas2einsum_str
      ret[i].blas2einsum_str = tensor_result + "->" + idx_result;
      ret[i].einsum2blas_str = idx_result + "->" + tensor_result;
    }
    input_list.push_back(idx_result);
    broadcast_indices.push_back(new_bcast_inds);
    size_t len_tmp_inputs = tmp_inputs.size();
    for (size_t j = 0; j < len_tmp_inputs; ++j) {
      ret[i].einsum_str += tmp_inputs[j];
      if (j + 1 < len_tmp_inputs) {
        ret[i].einsum_str += ",";
      }
    }
    ret[i].einsum_str += "->" + idx_result;
    ret[i].contract_inds = contract_inds;
    ret[i].idx_removed = contract.idx_removed;
    ret[i].input_list = input_list;
    ret[i].do_blas = do_blas;
  }

  if (ret_path == nullptr || ret_string_repr == nullptr) {
    return ret;
  }

  // Return the path along with a nice string representation
  std::string overall_contraction = parsed_subscripts[0] + "->" + parsed_subscripts[1];
  std::string header[3] = {"scaling", "current", "remaining"};

  double speedup = 1.0 * naive_cost / (1.0 * opt_cost);
  std::ostringstream ss;
  ss << "  Complete contraction:  " << overall_contraction << std::endl;
  ss << "         Naive scaling:  " <<  indices.count() << std::endl;
  ss << "     Optimized scaling:  " << max_scale << std::endl;
  ss.precision(3);
  ss << "      Naive FLOP count:  " << std::scientific << naive_cost << std::endl;
  ss << "  Optimized FLOP count:  " << std::scientific << opt_cost << std::endl;
  ss << "   Theoretical speedup:  " << std::scientific << speedup << std::endl;
  ss << "  Largest intermediate:  " << std::scientific << max_i << "elements" << std::endl;
  ss << std::string(74, '-') << std::endl;
  ss << std::setw(6) << header[0] << " ";
  ss << std::setw(24) << header[1] << " ";
  ss << std::setw(40) << header[2] << std::endl;
  ss << std::string(74, '-');

  for (size_t i = 0; i < size_path; ++i) {
    ss << std::endl;
    ss << std::setw(4) << scale_list[i] << "    ";
    ss << std::setw(24) << ret[i].einsum_str << " ";
    std::string remaining_str;
    size_t len_input_list = ret[i].input_list.size();
    for (size_t j = 0; j < len_input_list; ++j) {
      remaining_str += ret[i].input_list[j];
      if (j + 1 < len_input_list) {
        remaining_str += ",";
      }
    }
    remaining_str += "->" + parsed_subscripts[1];
    ss << std::setw(40) << remaining_str;
  }
  *ret_string_repr = ss.str();
  *ret_path = path;
  return ret;
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_EINSUM_PATH_OP_INL_H_
