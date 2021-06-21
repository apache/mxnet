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

#ifndef MXNET_OPERATOR_NPX_CONTROL_FLOW_H_
#define MXNET_OPERATOR_NPX_CONTROL_FLOW_H_

#include <mxnet/io.h>
#include <mxnet/base.h>
#include <dmlc/optional.h>

#include <string>
#include <vector>

namespace mxnet {
namespace op {

struct NPXForeachParam : public dmlc::Parameter<NPXForeachParam> {
  int num_args;
  int num_outputs;
  int num_out_data;
  // The location of states in the subgraph inputs.
  mxnet::Tuple<dim_t> in_state_locs;
  // The location of data arrays in the subgraph inputs.
  mxnet::Tuple<dim_t> in_data_locs;
  // The location of remaining arrays in the subgraph inputs.
  mxnet::Tuple<dim_t> remain_locs;
  // The index mapping from out_states to in_states.
  mxnet::Tuple<dim_t> in_state_index;
  DMLC_DECLARE_PARAMETER(NPXForeachParam) {
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(1)
    .describe("Number of inputs.");
    DMLC_DECLARE_FIELD(num_outputs)
    .describe("The number of outputs of the subgraph.");
    DMLC_DECLARE_FIELD(num_out_data)
    .describe("The number of output data of the subgraph.");
    DMLC_DECLARE_FIELD(in_state_locs)
    .describe("The locations of loop states among the inputs.");
    DMLC_DECLARE_FIELD(in_data_locs)
    .describe("The locations of input data among the inputs.");
    DMLC_DECLARE_FIELD(remain_locs)
    .describe("The locations of remaining data among the inputs.");
    DMLC_DECLARE_FIELD(in_state_index)
    .describe("The index mapping from out_states to in_states.");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream num_args_s, num_outputs_s, num_out_data_s, in_state_locs_s,
                       in_data_locs_s, remain_locs_s, in_state_index_s;
    num_args_s << num_args;
    num_outputs_s << num_outputs;
    num_out_data_s << num_out_data;
    in_state_locs_s << in_state_locs;
    in_data_locs_s << in_data_locs;
    remain_locs_s << remain_locs;
    in_state_index_s << in_state_index;
  }
};  // struct NPXForeachParam

struct NPXWhileLoopParam : public dmlc::Parameter<NPXWhileLoopParam> {
  int num_args;
  int num_outputs;
  int num_out_data;
  int max_iterations;
  // `cond' and `func' each takes a subset of while_loop's inputs as that to their subgraphs
  // `cond_input_locs' contains indices of inputs fed to `cond', and
  // `func_input_locs' contains indices of inputs fed to `func'.
  // `func_var_locs' are indices in which input "variables" are stored in func's inputs.
  mxnet::Tuple<dim_t> cond_input_locs;
  mxnet::Tuple<dim_t> func_input_locs;
  mxnet::Tuple<dim_t> func_var_locs;
  DMLC_DECLARE_PARAMETER(NPXWhileLoopParam) {
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(2)
    .describe("Number of input arguments, including cond and func as two symbol inputs.");
    DMLC_DECLARE_FIELD(num_outputs).set_lower_bound(1)
    .describe("The number of outputs of the subgraph.");
    DMLC_DECLARE_FIELD(num_out_data).set_lower_bound(0)
    .describe("The number of outputs from the function body.");
    DMLC_DECLARE_FIELD(max_iterations).set_lower_bound(1)
    .describe("Maximum number of iterations.");
    DMLC_DECLARE_FIELD(cond_input_locs)
    .describe("The locations of cond's inputs in the given inputs.");
    DMLC_DECLARE_FIELD(func_input_locs)
    .describe("The locations of func's inputs in the given inputs.");
    DMLC_DECLARE_FIELD(func_var_locs)
    .describe("The locations of loop_vars among func's inputs.");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream num_args_s, num_outputs_s, num_out_data_s, max_iterations_s,
                       cond_input_locs_s, func_input_locs_s, func_var_locs_s;
    num_args_s << num_args;
    num_outputs_s << num_outputs;
    num_out_data_s << num_out_data;
    max_iterations_s << max_iterations;
    cond_input_locs_s << cond_input_locs;
    func_input_locs_s << func_input_locs;
    func_var_locs_s << func_var_locs;
  }
  template <typename T>
  bool sync_in_out(std::vector<T> *in,
                   std::vector<T> *out,
                   std::function<bool(const T &)> is_empty) const {
    for (int i = this->num_out_data; i < this->num_outputs; ++i) {
      // each out->at(i) is a params, loop_var
      T &x = in->at(this->func_input_locs[this->func_var_locs[i - this->num_out_data]]);
      T &y = out->at(i);
      fill_value(&x, &y, is_empty(x), is_empty(y));
    }
    return true;
  }
};  // struct NPXWhileLoopParam

struct NPXCondParam : public dmlc::Parameter<NPXCondParam> {
  int num_args;
  int num_outputs;
  mxnet::Tuple<dim_t> cond_input_locs;
  mxnet::Tuple<dim_t> then_input_locs;
  mxnet::Tuple<dim_t> else_input_locs;
  DMLC_DECLARE_PARAMETER(NPXCondParam) {
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(3)
    .describe("Number of input arguments, including cond, then and else as three symbol inputs.");
    DMLC_DECLARE_FIELD(num_outputs).set_lower_bound(1)
    .describe("The number of outputs of the subgraph.");
    DMLC_DECLARE_FIELD(cond_input_locs)
    .describe("The locations of cond's inputs in the given inputs.");
    DMLC_DECLARE_FIELD(then_input_locs)
    .describe("The locations of then's inputs in the given inputs.");
    DMLC_DECLARE_FIELD(else_input_locs)
    .describe("The locations of else's inputs in the given inputs.");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream num_args_s, num_outputs_s,
                       cond_input_locs_s, then_input_locs_s, else_input_locs_s;
    num_args_s << num_args;
    num_outputs_s << num_outputs;
    cond_input_locs_s << cond_input_locs;
    then_input_locs_s << then_input_locs;
    else_input_locs_s << else_input_locs;
  }
};  // struct NPXCondParam

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NPX_CONTROL_FLOW_H_
