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
 * \file np_einsum_op.cc
 * \brief CPU Implementation of numpy-compatible einsum
 */

#include "./np_einsum_op-inl.h"
#include <stdlib.h>
#include <string.h>

namespace mxnet {
namespace op {

inline std::vector<std::string> _parse_einsum_input(std::string subscripts,
                                                    const mxnet::ShapeVector& shapes) {
  const std::string einsum_symbols =
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
  std::bitset<MAXAXIS> einsum_symbols_set;
  for (const char& c : einsum_symbols) {
    einsum_symbols_set.set(c);
  }

  CHECK_NE(shapes.size(), 0U)
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
        if (shapes[i].ndim() == 0) {
          ellipse_count = 0;
        } else {
          ellipse_count = std::max(shapes[i].ndim(), 1);
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
  CHECK_EQ(std::count(ret[0].begin(), ret[0].end(), ',') + 1, shapes.size())
    << "Number of einsum subscripts must be equal to the "
    << "number of operands.";

  return ret;
}


bool NumpyEinsumShape(const nnvm::NodeAttrs& attrs,
                      mxnet::ShapeVector *in_attrs,
                      mxnet::ShapeVector *out_attrs) {
  const NumpyEinsumParam &param = nnvm::get<NumpyEinsumParam>(attrs.parsed);
  const std::string& subscripts = param.subscripts;
  int num_args = param.num_args;
  CHECK_EQ(in_attrs->size(), num_args);
  CHECK_EQ(out_attrs->size(), 1U);
  for (int i = 0; i < num_args; i++) {
    if (!shape_is_known(in_attrs->at(i))) {
      return false;
    }
  }

  // Parsing
  std::vector<std::string> parsed_subscripts = _parse_einsum_input(subscripts, *in_attrs);

  // Build a few useful list and sets
  std::vector<std::string> input_list = split(parsed_subscripts[0], ",");
  size_t isize = input_list.size();

  // Get length of each unique dimension and ensure all dimensions are correct
  dim_t dimension_dict[MAXAXIS];
  memset(dimension_dict, -1, sizeof(dimension_dict));
  for (size_t i = 0; i < isize; ++i) {
    const std::string& term = input_list[i];
    const TShape& sh = in_attrs->at(i);
    CHECK_EQ(sh.ndim(), term.length())
      << "Einstein sum subscript " << input_list[i]
      << " does not contain the "
      << "correct number of indices for operand " << i << ".";
    size_t len_term = term.length();
    for (size_t j = 0; j < len_term; ++j) {
      dim_t dim = sh[j];
      const char& c = term[j];

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

  // Get oshape
  const std::string& output_str = parsed_subscripts[1];
  size_t odim = output_str.size();
  TShape oshape(odim, -1);
  for (size_t i = 0; i < odim; ++i) {
    oshape[i] = dimension_dict[static_cast<int>(output_str[i])];
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  size_t lim = static_cast<size_t>(std::numeric_limits<index_t>::max());
  for (int i = 0; i < num_args; ++i) {
    CHECK_LE(in_attrs->at(i).Size(), lim)
      << "Size of operand " << i
      << " exceeds the maximum index."
      << " Try setting `USE_INT64_TENSOR_SIZE`.";
  }
  CHECK_LE(oshape.Size(), lim)
    << "Size of output"
    << " exceeds the maximum index."
    << " Try setting `USE_INT64_TENSOR_SIZE`.";
  return shape_is_known(oshape);
}

OpStatePtr CreateEinsumState(const NodeAttrs& attrs,
                             Context ctx,
                             const mxnet::ShapeVector& in_shapes,
                             const std::vector<int>& in_types) {
  const NumpyEinsumParam& param = dmlc::get<NumpyEinsumParam>(attrs.parsed);
  return OpStatePtr::Create<EinsumOp>(param.num_args, param.optimize, param.subscripts);
}

DMLC_REGISTER_PARAMETER(NumpyEinsumParam);

NNVM_REGISTER_OP(_npi_einsum)
.describe(R"doc()doc" ADD_FILELINE)
.set_attr_parser(ParamParser<NumpyEinsumParam>)
.set_num_inputs([](const nnvm::NodeAttrs& attrs) {
  const NumpyEinsumParam& param = dmlc::get<NumpyEinsumParam>(attrs.parsed);
  return static_cast<uint32_t>(param.num_args);
})
.set_num_outputs(1)
.set_attr<std::string>("key_var_num_args", "num_args")
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const nnvm::NodeAttrs& attrs) {
    int num_args = dmlc::get<NumpyEinsumParam>(attrs.parsed).num_args;
    std::vector<std::string> ret;
    for (int i = 0; i < num_args; i++) {
      ret.push_back(std::string("arg") + std::to_string(i));
    }
    return ret;
})
.set_attr<mxnet::FInferShape>("FInferShape", NumpyEinsumShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<-1, 1>)
.set_attr<FCreateOpState>("FCreateOpState", CreateEinsumState)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>(1, ResourceRequest::kTempSpace);
  })
.set_attr<FStatefulCompute>("FStatefulCompute<cpu>", NumpyEinsumForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_einsum"})
.add_argument("data", "NDArray-or-Symbol[]", "List of eimsum operands")
.add_arguments(NumpyEinsumParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_npi_einsum)
.set_attr_parser(ParamParser<NumpyEinsumParam>)
.set_num_inputs([](const nnvm::NodeAttrs& attrs) {
  const NumpyEinsumParam& param = dmlc::get<NumpyEinsumParam>(attrs.parsed);
  return static_cast<uint32_t>(param.num_args + 1);
})
.set_num_outputs([](const nnvm::NodeAttrs& attrs) {
  const NumpyEinsumParam& param = dmlc::get<NumpyEinsumParam>(attrs.parsed);
  return static_cast<uint32_t>(param.num_args);
})
.set_attr<bool>("TIsLayerOpBackward", true)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>(1, ResourceRequest::kTempSpace);
  })
.set_attr<FStatefulCompute>("FStatefulCompute<cpu>", NumpyEinsumBackward<cpu>);

}  // namespace op
}  // namespace mxnet
