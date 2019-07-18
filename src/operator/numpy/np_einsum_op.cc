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

bool NumpyEinsumShape(const nnvm::NodeAttrs& attrs,
                             mxnet::ShapeVector *in_attrs,
                             mxnet::ShapeVector *out_attrs) {
  const NumpyEinsumParam &param = nnvm::get<NumpyEinsumParam>(attrs.parsed);
  const char* subscripts = param.subscripts.c_str();
  int num_args = param.num_args;
  CHECK_EQ(in_attrs->size(), num_args);
  CHECK_EQ(out_attrs->size(), 1U);
  for (int i = 0; i < num_args; i++) {
    if (!shape_is_known(in_attrs->at(i))) {
      return false;
    }
  }

  int iop, label, min_label = 127, max_label = 0;
  int nop = num_args;
  char label_counts[128];
  int label_size[128], max_broadcast = -1;
  char op_labels[NPY_MAXARGS][NPY_MAXDIMS];
  char output_labels[NPY_MAXDIMS];
  int idim, ndim_output, ndim_broadcast;

  /* Parse the subscripts string into label_counts and op_labels */
  memset(label_counts, 0, sizeof(label_counts));
  for (iop = 0; iop < nop; ++iop) {
    int length = static_cast<int>(strcspn(subscripts, ",-"));
    CHECK(!(iop == nop-1 && subscripts[length] == ','))
      << "more operands provided to einstein sum function "
         "than specified in the subscripts string";
    CHECK(!(iop < nop-1 && subscripts[length] != ','))
      << "fewer operands provided to einstein sum function "
         "than specified in the subscripts string";
    CHECK_GE(parse_operand_subscripts(subscripts, length,
                                      in_attrs->at(iop).ndim(),
                                      iop, op_labels[iop], label_counts,
                                      &min_label, &max_label), 0);

    /* Move subscripts to the start of the labels for the next op */
    subscripts += length;
    if (iop < nop - 1) {
      subscripts++;
    }
  }

  /*
   * Find the number of broadcast dimensions, which is the maximum
   * number of labels == 0 in an op_labels array.
   */
  ndim_broadcast = 0;
  for (iop = 0; iop < nop; ++iop) {
    int count_zeros = 0;
    int ndim;
    char *labels = op_labels[iop];
    ndim = in_attrs->at(iop).ndim();
    for (idim = 0; idim < ndim; ++idim) {
      if (labels[idim] == 0) {
        ++count_zeros;
      } else if (labels[idim] > 0) {
        label_size[static_cast<int>(labels[idim])] = in_attrs->at(iop)[idim];
      }
    }
    if (count_zeros > ndim_broadcast) {
      ndim_broadcast = count_zeros;
      max_broadcast = iop;
    }
  }

  /*
   * If there is no output signature, fill output_labels and ndim_output
   * using each label that appeared once, in alphabetical order.
   */
  if (subscripts[0] == '\0') {
    /* If no output was specified, always broadcast left, as usual. */
    for (ndim_output = 0; ndim_output < ndim_broadcast; ++ndim_output) {
      output_labels[ndim_output] = 0;
    }
    for (label = min_label; label <= max_label; ++label) {
      if (label_counts[label] == 1) {
        CHECK(ndim_output < NPY_MAXDIMS)
          << "einstein sum subscript string has too many "
          << "distinct labels";
        output_labels[ndim_output++] = label;
      }
    }
  } else {
    CHECK(subscripts[0] == '-' && subscripts[1] == '>')
      << "einstein sum subscript string does not "
      << "contain proper '->' output specified";
    subscripts += 2;

    /* Parse the output subscript string. */
    ndim_output = parse_output_subscripts(subscripts, strlen(subscripts),
                                          ndim_broadcast, label_counts,
                                          output_labels);
    CHECK_GE(ndim_output, 0);
  }

  TShape oshape(ndim_output, -1);
  for (int i = 0, j = 0; i < ndim_output; i++) {
    if (output_labels[i] > 0) {
      oshape[i] = label_size[static_cast<int>(output_labels[i])];
    } else {
      while (op_labels[max_broadcast][j] != 0) {
        j++;
      }
      oshape[i] = in_attrs->at(max_broadcast)[j++];
    }
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  return shape_is_known(oshape);
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
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>(3, ResourceRequest::kTempSpace);
  })
.set_attr<FCompute>("FCompute<cpu>", NumpyEinsumForward<cpu>)
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
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>(3, ResourceRequest::kTempSpace);
  })
.set_attr<FCompute>("FCompute<cpu>", NumpyEinsumBackward<cpu>);

}  // namespace op
}  // namespace mxnet
