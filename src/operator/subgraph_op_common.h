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

#ifndef MXNET_OPERATOR_SUBGRAPH_OP_COMMON_H_
#define MXNET_OPERATOR_SUBGRAPH_OP_COMMON_H_

#include <mxnet/io.h>
#include <mxnet/base.h>
#include <mxnet/op_attr_types.h>
#include <vector>
#include <utility>
#include <string>
#include "../imperative/cached_op.h"
#include "../imperative/imperative_utils.h"

namespace mxnet {
namespace op {

/*
 * Infer the data types of inputs and outputs of an operator that contains a
 * subgraph.
 */
bool InferSubgraphDataType(const nnvm::Symbol &subgraph, std::vector<int> *in_type,
                           std::vector<int> *out_type);

/*
 * Infer the shape of inputs and outputs of an operator that contains a
 * subgraph.
 */
bool InferSubgraphShape(const nnvm::Symbol &subgraph,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape);

/*
 * Infer the storage types of inputs and outputs of an operator that contains a
 * subgraph.
 */
bool InferSubgraphStorage(const nnvm::Symbol &subgraph,
                          const int dev_mask,
                          DispatchMode* dispatch_mode,
                          std::vector<int> *in_attrs,
                          std::vector<int> *out_attrs);

bool as_bool_scalar(const NDArray &a);

bool is_shape_udf(const TShape &x);

bool is_stype_udf(const int &x);

bool is_type_udf(const int &x);

template <typename T>
void extract_by_loc(const std::vector<T> &array,
                    const nnvm::Tuple<dim_t> input_locs,
                    std::vector<T> *out) {
  out->clear();
  out->reserve(input_locs.ndim());
  for (dim_t i : input_locs) {
    out->push_back(array[i]);
  }
}

template <typename T>
bool fill_value(T *x, T *y, bool x_empty, bool y_empty) {
  if (*x == *y || (x_empty && y_empty)) {
    return true;
  }
  if (!x_empty && !y_empty) {
    return false;
  }
  if (x_empty) {
    *x = *y;
  }
  if (y_empty) {
    *y = *x;
  }
  return true;
}

template <typename T>
bool sync_in_in(const nnvm::Tuple<dim_t> &input_locs,
                         std::vector<T> *in,
                         std::vector<T> *subg_in,
                         std::function<bool(const T &)> is_empty) {
  for (size_t i = 0; i < input_locs.ndim(); ++i) {
    T &x = in->at(input_locs[i]);
    T &y = subg_in->at(i);
    fill_value(&x, &y, is_empty(x), is_empty(y));
  }
  return true;
}

template <typename T>
bool sync_out_out(std::vector<T> *out_1,
                  std::vector<T> *out_2,
                  std::function<bool(const T &)> is_empty) {
  CHECK_EQ(out_1->size(), out_2->size());
  for (size_t i = 0; i < out_1->size(); ++i) {
    T &x = out_1->at(i);
    T &y = out_2->at(i);
    fill_value(&x, &y, is_empty(x), is_empty(y));
  }
  return true;
}

/*
 * This contains the states for running a loop and provides methods
 * of running the subgraph computation for an iteration.
 */
class LoopState {
  // These are output arrays from all iterations.
  // They also contain the Op state for each CachedOp.
  std::vector<std::vector<NDArray> > all_outputs;
  std::vector<std::vector<NDArray> > all_inputs;
  // For inference, there should be only one cached op because we
  // want to share the memory in iterations.
  // For training, each iteration has a cached op because each iteration
  // needs to maintain a set of memory buffers for all computation states,
  // which will be used in the backward.
  std::vector<OpStatePtr> all_states;
  CachedOpPtr iter_op;
  Symbol subgraph_sym;
  nnvm::Graph subgraph;

 public:
  explicit LoopState(const Symbol &g);

  void Forward(int iter_no,
               const std::vector<NDArray> &inputs,
               const std::vector<OpReqType>& req,
               const std::vector<NDArray> &outputs,
               bool is_recording);
  void Backward(int iter_no,
                const std::vector<NDArray> &ograds,
                const std::vector<OpReqType> &req,
                const std::vector<NDArray> &igrads);
  void Cleanup() {
    all_outputs.clear();
    all_inputs.clear();
    all_states.clear();
  }
  static CachedOpPtr MakeSharedOp(const Symbol &sym) {
    // We turn on static_alloc for two reasons.
    // It avoids the overhead of unnecessary memory allocation.
    // only static_alloc supports nested call of CachedOp.
    std::vector<std::pair<std::string, std::string> > kwargs = {
      {"inline_limit", "0"},
      {"static_alloc", "1"}
    };
    return std::make_shared<CachedOp>(sym, kwargs);
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_SUBGRAPH_OP_COMMON_H_
