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
 * Copyright (c) 2019, The Apache Software Foundation.
 *
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
 * \file np_einsum_op-inl.h
 * \brief Function definition of numpy-compatible einsum operator
 * modified by Haozheng Fan(@hzfan) from:
 * https://github.com/numpy/numpy/blob/master/numpy/core/src/multiarray/einsum.c.src
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_EINSUM_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_EINSUM_OP_INL_H_

#include <mxnet/operator_util.h>
#include <string>
#include <vector>
#include <algorithm>
#include "./np_tensordot_op-inl.h"
#include "./np_einsum_path_op-inl.h"
#include "../../common/static_array.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

#define NPY_MAXDIMS 16
#define NPY_MAXARGS 16

inline TShape get_stride(const TShape& shape) {
  int ndim = shape.ndim(), prod = 1;
  TShape stride = TShape(ndim, -1);
  for (int i = ndim - 1; i >= 0; i--) {
    stride[i] = shape[i] > 1 ? prod : 0;
    prod = prod * shape[i];
  }
  return stride;
}

inline TShape pad(const TShape& shape, int odim) {
  int ndim = shape.ndim();
  CHECK_GE(odim, ndim);
  TShape ret(odim, 1);
  for (int idim = 0; idim < ndim; ++idim) {
    ret[idim] = shape[idim];
  }
  return ret;
}

/*
 * Parses the subscripts for one operand into an output of 'ndim'
 * labels. The resulting 'op_labels' array will have:
 *  - the ASCII code of the label for the first occurrence of a label;
 *  - the (negative) offset to the first occurrence of the label for
 *    repeated labels;
 *  - zero for broadcast dimensions, if subscripts has an ellipsis.
 * For example:
 *  - subscripts="abbcbc",  ndim=6 -> op_labels=[97, 98, -1, 99, -3, -2]
 *  - subscripts="ab...bc", ndim=6 -> op_labels=[97, 98, 0, 0, -3, 99]
 */
inline int parse_operand_subscripts(const char *subscripts, int length,
                                    int ndim, int iop, char *op_labels,
                                    char *label_counts, int *min_label, int *max_label) {
  using namespace mxnet_op;
  int i;
  int idim = 0;
  int ellipsis = -1;

  /* Process all labels for this operand */
  for (i = 0; i < length; ++i) {
    int label = subscripts[i];

    /* A proper label for an axis. */
    if (label > 0 && isalpha(label)) {
      /* Check we don't exceed the operator dimensions. */
      CHECK(idim < ndim)
        << "einstein sum subscripts string contains "
        << "too many subscripts for operand "
        << iop;

      op_labels[idim++] = label;
      if (label < *min_label) {
        *min_label = label;
      }
      if (label > *max_label) {
        *max_label = label;
      }
      label_counts[label]++;
    } else if (label == '.') {
      /* The beginning of the ellipsis. */
      /* Check it's a proper ellipsis. */
      CHECK(!(ellipsis != -1 || i + 2 >= length
              || subscripts[++i] != '.' || subscripts[++i] != '.'))
        << "einstein sum subscripts string contains a "
        << "'.' that is not part of an ellipsis ('...') "
        << "in operand "
        << iop;

      ellipsis = idim;
    } else {
        CHECK(label == ' ')
          << "invalid subscript '" << static_cast<char>(label)
          << "' in einstein sum "
          << "subscripts string, subscripts must "
          << "be letters";
    }
  }

  /* No ellipsis found, labels must match dimensions exactly. */
  if (ellipsis == -1) {
    CHECK(idim == ndim)
      << "operand has more dimensions than subscripts "
      << "given in einstein sum, but no '...' ellipsis "
      << "provided to broadcast the extra dimensions.";
  } else if (idim < ndim) {
    /* Ellipsis found, may have to add broadcast dimensions. */
    /* Move labels after ellipsis to the end. */
    for (i = 0; i < idim - ellipsis; ++i) {
      op_labels[ndim - i - 1] = op_labels[idim - i - 1];
    }
    /* Set all broadcast dimensions to zero. */
    for (i = 0; i < ndim - idim; ++i) {
      op_labels[ellipsis + i] = 0;
    }
  }

  /*
   * Find any labels duplicated for this operand, and turn them
   * into negative offsets to the axis to merge with.
   *
   * In C, the char type may be signed or unsigned, but with
   * twos complement arithmetic the char is ok either way here, and
   * later where it matters the char is cast to a signed char.
   */
  for (idim = 0; idim < ndim - 1; ++idim) {
    int label = op_labels[idim];
    /* If it is a proper label, find any duplicates of it. */
    if (label > 0) {
      /* Search for the next matching label. */
      char *next = reinterpret_cast<char*>(memchr(op_labels + idim + 1, label, ndim - idim - 1));

      while (next != nullptr) {
        /* The offset from next to op_labels[idim] (negative). */
        *next = static_cast<char>((op_labels + idim) - next);
        /* Search for the next matching label. */
        next = reinterpret_cast<char*>(memchr(next + 1, label, op_labels + ndim - 1 - next));
      }
    }
  }
  return 0;
}

/*
 * Parses the subscripts for the output operand into an output that
 * includes 'ndim_broadcast' unlabeled dimensions, and returns the total
 * number of output dimensions, or -1 if there is an error. Similarly
 * to parse_operand_subscripts, the 'out_labels' array will have, for
 * each dimension:
 *  - the ASCII code of the corresponding label;
 *  - zero for broadcast dimensions, if subscripts has an ellipsis.
 */
inline int parse_output_subscripts(const char *subscripts, int length,
                                   int ndim_broadcast,
                                   const char *label_counts, char *out_labels) {
  using namespace mxnet_op;
  int i, bdim;
  int ndim = 0;
  int ellipsis = 0;

  /* Process all the output labels. */
  for (i = 0; i < length; ++i) {
    int label = subscripts[i];

    /* A proper label for an axis. */
    if (label > 0 && isalpha(label)) {
      /* Check that it doesn't occur again. */
      CHECK(memchr(subscripts + i + 1, label, length - i - 1) == nullptr)
        << "einstein sum subscripts string includes "
        << "output subscript '" << static_cast<char>(label)
        << "' multiple times";

      /* Check that it was used in the inputs. */
      CHECK(label_counts[label] != 0)
        << "einstein sum subscripts string included "
        << "output subscript '" << static_cast<char>(label)
        << "' which never appeared "
        << "in an input";

      /* Check that there is room in out_labels for this label. */
      CHECK(ndim < NPY_MAXDIMS)
        << "einstein sum subscripts string contains "
        << "too many subscripts in the output";

      out_labels[ndim++] = label;
    } else if (label == '.') {
      /* The beginning of the ellipsis. */
      /* Check it is a proper ellipsis. */
      CHECK(!(ellipsis || i + 2 >= length
              || subscripts[++i] != '.' || subscripts[++i] != '.'))
        << "einstein sum subscripts string "
        << "contains a '.' that is not part of "
        << "an ellipsis ('...') in the output";

      /* Check there is room in out_labels for broadcast dims. */
      CHECK(ndim + ndim_broadcast <= NPY_MAXDIMS)
        << "einstein sum subscripts string contains "
        << "too many subscripts in the output";

      ellipsis = 1;
      for (bdim = 0; bdim < ndim_broadcast; ++bdim) {
        out_labels[ndim++] = 0;
      }
    } else {
      CHECK(label == ' ')
        << "invalid subscript '" << static_cast<char>(label)
        << "' in einstein sum "
        << "subscripts string, subscripts must "
        << "be letters";
    }
  }

  /* If no ellipsis was found there should be no broadcast dimensions. */
  CHECK(!(!ellipsis && ndim_broadcast > 0))
    << "output has more dimensions than subscripts "
    << "given in einstein sum, but no '...' ellipsis "
    << "provided to broadcast the extra dimensions.";

  return ndim;
}

inline void get_combined_dims_view(const TBlob& op, int iop,
                                   char *labels,
                                   TShape* newshape,
                                   TShape* newstride) {
  using namespace mxnet_op;
  int idim, ndim, icombine, combineoffset;
  int icombinemap[NPY_MAXDIMS];
  int newdim;

  const TShape& shape = op.shape_;
  TShape stride = get_stride(shape);
  ndim = op.shape_.ndim();
  newdim = newshape->ndim();

  /* Initialize the dimensions and strides to zero */
  for (idim = 0; idim < newdim; ++idim) {
    (*newshape)[idim] = 0;
    (*newstride)[idim] = 0;
  }

  /* Copy the dimensions and strides, except when collapsing */
  icombine = 0;
  for (idim = 0; idim < ndim; ++idim) {
    /*
     * The char type may be either signed or unsigned, we
     * need it to be signed here.
     */
    int label = (signed char)labels[idim];
    /* If this label says to merge axes, get the actual label */
    if (label < 0) {
      combineoffset = label;
      label = labels[idim+label];
    } else {
      combineoffset = 0;
      if (icombine != idim) {
        labels[icombine] = labels[idim];
      }
      icombinemap[idim] = icombine;
    }
    /* If the label is 0, it's an unlabeled broadcast dimension */
    if (label == 0) {
      (*newshape)[icombine] = shape[idim];
      (*newstride)[icombine] = stride[idim];
    } else {
      /* Update the combined axis dimensions and strides */
      int i = icombinemap[idim + combineoffset];
      CHECK(!(combineoffset < 0 && (*newshape)[i] != 0 &&
              (*newshape)[i] != shape[idim]))
        << "dimensions in operand " << iop
        << " for collapsing index '" << label
        << "' don't match (" << static_cast<int>((*newshape)[i])
        << " != " << shape[idim] << ")";
      (*newshape)[i] = shape[idim];
      (*newstride)[i] += stride[idim];
    }

    /* If the label didn't say to combine axes, increment dest i */
    if (combineoffset == 0) {
      icombine++;
    }
  }
}

inline static int prepare_op_axes(int ndim, int iop, char *labels,
                                  int *axes, int ndim_iter, char *iter_labels) {
  using namespace mxnet_op;
  int i, label, ibroadcast;

  ibroadcast = ndim-1;
  for (i = ndim_iter-1; i >= 0; --i) {
    label = iter_labels[i];
    /*
     * If it's an unlabeled broadcast dimension, choose
     * the next broadcast dimension from the operand.
     */
    if (label == 0) {
      while (ibroadcast >= 0 && labels[ibroadcast] != 0) {
        --ibroadcast;
      }
      /*
       * If we used up all the operand broadcast dimensions,
       * extend it with a "newaxis"
       */
      if (ibroadcast < 0) {
        axes[i] = -1;
      } else {
        /* Otherwise map to the broadcast axis */
        axes[i] = ibroadcast;
        --ibroadcast;
      }
    } else {
      /* It's a labeled dimension, find the matching one */
      char *match = reinterpret_cast<char*>(memchr(labels, label, ndim));
      /* If the op doesn't have the label, broadcast it */
      if (match == nullptr) {
        axes[i] = -1;
      } else {
        /* Otherwise use it */
        axes[i] = match - labels;
      }
    }
  }
  return 0;
}

struct NumpyEinsumParam: public dmlc::Parameter<NumpyEinsumParam> {
  int num_args;
  int optimize;
  std::string  subscripts;
  DMLC_DECLARE_PARAMETER(NumpyEinsumParam) {
    DMLC_DECLARE_FIELD(num_args)
      .set_lower_bound(1)
      .describe("Number of input arrays.");
    DMLC_DECLARE_FIELD(subscripts)
      .set_default("")
      .describe("Specifies the subscripts for summation as comma separated list"
      " of subscript labels. An implicit (classical Einstein summation) calculation"
      " is performed unless the explicit indicator '->' is included as well as"
      " subscript labels of the precise output form.");
    DMLC_DECLARE_FIELD(optimize)
      .set_default(0);
  }
};

class EinsumOp {
 public:
  int num_args;
  int optimize;
  std::string subscripts;
  std::shared_ptr<NDArray> tempspace;
  std::vector<Step> paths;
  explicit EinsumOp(int num_args, int optimize, std::string subscripts) {
    this->num_args = num_args;
    this->optimize = optimize;
    this->subscripts = subscripts;
  }
};  // class EinsumOp

template<int dimension, int req, bool back, typename AType>
struct numpy_einsum{
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType* out,
                                  common::StaticArray<DType*, NPY_MAXARGS> op,
                                  mshadow::Shape<dimension> oshape,
                                  common::StaticArray<mshadow::Shape<dimension>,
                                                      NPY_MAXARGS> ostride,
                                  mshadow::Shape<dimension> reduceshape,
                                  common::StaticArray<mshadow::Shape<dimension>,
                                                      NPY_MAXARGS> rstride,
                                  int nop,
                                  int iop0,
                                  const DType* out_grad) {
    using namespace mxnet_op;
    mshadow::Shape<dimension> oidx = unravel(i, oshape);
    i = back ? dot(oidx, ostride[iop0]) : i;
    if (req == kWriteTo) {
      out[i] = (DType)0;
    }
    for (int rdim = 0; rdim < dimension; ++rdim) {
      if (reduceshape[rdim] == 0) {
        return;
      }
    }
    mshadow::Shape<dimension> ridx = unravel(0, reduceshape);
    AType sum = 0;
    do {
      AType tmp = back ? static_cast<AType>(out_grad[dot(oidx, ostride[nop]) +
                                                     dot(ridx, rstride[nop])]): (AType)1;
      for (int iop = 0; iop < nop; ++iop) {
        if (iop != iop0) {
          index_t k = dot(oidx, ostride[iop]) + dot(ridx, rstride[iop]);
          tmp = tmp * static_cast<AType>(op[iop][k]);
        }
      }
      sum = sum + tmp;
    }while (inc(&ridx, reduceshape));
    out[i] = out[i] + static_cast<DType>(sum);
  }
};

template<typename xpu, bool back>
inline void NumpyEinsumProcess(const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs,
                               const char *subscripts, int nop,
                               const OpContext& ctx) {
  using namespace mxnet_op;

  /* nop+1 (+1 is for the output) must fit in NPY_MAXARGS */
  CHECK(nop < NPY_MAXARGS)
    << "too many operands provided to einstein sum function";
  CHECK(nop >= 1)
    << "not enough operands provided to einstein sum function";

  /* Step 1: Parse the subscripts string into label_counts and op_labels */
  int iop, idim, min_label = 127, max_label = 0;
  char label_counts[128], op_labels[NPY_MAXARGS][NPY_MAXDIMS];
  memset(label_counts, 0, sizeof(label_counts));
  for (iop = 0; iop < nop; ++iop) {
    int length = static_cast<int>(strcspn(subscripts, ",-"));

    CHECK(!(iop == nop - 1 && subscripts[length] == ','))
      << "more operands provided to einstein sum function "
      << "than specified in the subscripts string";
    CHECK(!(iop < nop-1 && subscripts[length] != ','))
      << "fewer operands provided to einstein sum function "
      << "than specified in the subscripts string";
    CHECK_GE(parse_operand_subscripts(subscripts, length,
                                      inputs[iop + back].shape_.ndim(),
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
  int ndim_broadcast = 0;
  for (iop = 0; iop < nop; ++iop) {
    int count_zeros = 0;
    int ndim;
    char *labels = op_labels[iop];

    ndim = inputs[iop + back].shape_.ndim();
    for (idim = 0; idim < ndim; ++idim) {
      if (labels[idim] == 0) {
        ++count_zeros;
      }
    }

    if (count_zeros > ndim_broadcast) {
      ndim_broadcast = count_zeros;
    }
  }

  /*
   * If there is no output signature, fill output_labels and ndim_output
   * using each label that appeared once, in alphabetical order.
   */
  int label, ndim_output;
  char output_labels[NPY_MAXDIMS];
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

  /*
   * Step 2:
   * Process all the input ops, combining dimensions into their
   * diagonal where specified.
   */
  std::vector<TShape> opshape(nop), opstride_true(nop);
  for (iop = 0; iop < nop; ++iop) {
    char *labels = op_labels[iop];
    int combine, ndim;

    ndim = inputs[iop + back].shape_.ndim();

    /*
     * Check whether any dimensions need to be combined
     *
     * The char type may be either signed or unsigned, we
     * need it to be signed here.
     */
    combine = 0;
    for (idim = 0; idim < ndim; ++idim) {
      if ((signed char)labels[idim] < 0) {
        combine++;
      }
    }

    /* If any dimensions are combined, create a view which combines them */
    if (combine) {
      TShape tshape(ndim - combine, -1);
      TShape tstride(ndim - combine, -1);
      get_combined_dims_view(inputs[iop + back], iop, labels,
                             &tshape, &tstride);
      opshape[iop] = tshape;
      opstride_true[iop] = tstride;
    } else {
      /* No combining needed */
      opshape[iop] = inputs[iop + back].shape_;
      opstride_true[iop] = get_stride(opshape[iop]);
    }
  }

  /*
   * Step 3:
   * Set up the labels for the iterator (output + combined labels).
   * Can just share the output_labels memory, because iter_labels
   * is output_labels with some more labels appended.
   */
  char *iter_labels = output_labels;
  int ndim_iter = ndim_output;
  for (label = min_label; label <= max_label; ++label) {
    if (label_counts[label] > 0 &&
        memchr(output_labels, label, ndim_output) == nullptr) {
      CHECK(ndim_iter < NPY_MAXDIMS)
        << "too many subscripts in einsum";
      iter_labels[ndim_iter++] = label;
    }
  }

  /* Step 4: Set up the op_axes for the iterator */
  TShape itershape(ndim_iter, -1);
  std::vector<TShape> iterstride(nop + 1, TShape(ndim_iter, 0));
  TShape oshape = back ? inputs[0].shape_ : outputs[0].shape_;
  TShape ostride_true = get_stride(oshape);
  TShape reduceshape;
  std::vector<TShape> remainshape(nop);
  int op_axes_arrays[NPY_MAXARGS][NPY_MAXDIMS];
  int *op_axes[NPY_MAXARGS];

  for (iop = 0; iop < nop; ++iop) {
    op_axes[iop] = op_axes_arrays[iop];
    CHECK_GE(prepare_op_axes(opshape[iop].ndim(), iop, op_labels[iop],
             op_axes[iop], ndim_iter, iter_labels), 0);
    for (idim = 0; idim < ndim_iter; idim++) {
      if (op_axes[iop][idim] != -1) {
        iterstride[iop][idim] = opstride_true[iop][op_axes[iop][idim]];
        if (itershape[idim] != -1) {
          if (itershape[idim] == 1) {
            itershape[idim] = opshape[iop][op_axes[iop][idim]];
          }
        } else {
          itershape[idim] = opshape[iop][op_axes[iop][idim]];
        }
      }
    }
  }
  for (idim = 0; idim < ndim_output; ++idim) {
    iterstride[nop][idim] = ostride_true[idim];
  }
  reduceshape = TShape(ndim_iter - ndim_output, 0);
  for (idim = ndim_output; idim < ndim_iter; ++idim) {
    reduceshape[idim - ndim_output] = itershape[idim];
  }
  for (iop = 0; iop < nop; iop++) {
    std::vector<size_t> rsh;
    for (idim = 0; idim < ndim_iter; idim++) {
      if (op_axes_arrays[iop][idim] == -1 ||
          itershape[idim] != opshape[iop][op_axes_arrays[iop][idim]]) {
        rsh.push_back(itershape[idim]);
      }
    }
    remainshape[iop] = TShape(rsh.begin(), rsh.end());
  }

  // exclude the 0-dim case
  if (ndim_iter == 0) {
    ndim_iter = 1;
  }
  itershape = pad(itershape, ndim_iter);
  for (iop = 0; iop <= nop; ++iop) {
    iterstride[iop] = pad(iterstride[iop], ndim_iter);
  }
  oshape = pad(oshape, ndim_iter);
  reduceshape = pad(reduceshape, ndim_iter);
  for (iop = 0; iop < nop; ++iop) {
    opshape[iop] = pad(opshape[iop], ndim_iter);
    remainshape[iop] = pad(remainshape[iop], ndim_iter);
  }

  if (!back) {
    if (oshape.Size() == 0) {
      return;
    }
    const TBlob &out_data = outputs[0];
    MXNET_ACC_TYPE_SWITCH(out_data.type_flag_, DType, AType, {
      mxnet::common::StaticArray<DType*, NPY_MAXARGS> op;
      for (iop = 0; iop < nop; ++iop) {
        op[iop] = inputs[iop].dptr<DType>();
      }
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        MXNET_NDIM_SWITCH_EX(ndim_iter, dimension, {
          mxnet::common::StaticArray<mshadow::Shape<dimension>, NPY_MAXARGS> ostride_arr;
          mxnet::common::StaticArray<mshadow::Shape<dimension>, NPY_MAXARGS> rstride_arr;
          for (iop = 0; iop < nop; ++iop) {
            mshadow::Shape<dimension> otmp, rtmp;
            for (idim = 0; idim < dimension; ++idim) {
              otmp[idim] = idim < ndim_output ? iterstride[iop][idim] : 1;
              rtmp[idim] = idim < dimension - ndim_output ? iterstride[iop][idim + ndim_output] : 1;
            }
            ostride_arr[iop] = otmp;
            rstride_arr[iop] = rtmp;
          }
          Kernel<numpy_einsum<dimension, req_type, 0, AType>,
                 xpu>::Launch(ctx.get_stream<xpu>(),
                              oshape.Size(),
                              out_data.dptr<DType>(),
                              op,
                              oshape.get<dimension>(),
                              ostride_arr,
                              reduceshape.get<dimension>(),
                              rstride_arr,
                              nop,
                              -1,
                              reinterpret_cast<DType*>(NULL));
        })
      })
    })
  } else {
    if (oshape.Size() == 0) {
      for (iop = 0; iop < nop; ++iop) {
        const TBlob& out_data = outputs[iop];
        if (opshape[iop].Size() > 0) {
          MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
            MXNET_ASSIGN_REQ_SWITCH(req[iop], req_type, {
              if (req_type == kWriteTo) {
                out_data.FlatTo1D<xpu, DType>(ctx.get_stream<xpu>()) = 0;
              }
            })
          })
        }
      }
      return;
    }
    for (int i = 0; i < nop; ++i) {
      const TBlob &out_data = outputs[i];
      const TBlob &out_grad = inputs[0];
      std::vector<TShape> opstride(nop + 1, TShape(ndim_iter, 0));
      std::vector<TShape> remainstride(nop + 1, TShape(ndim_iter, 0));
      for (iop = 0; iop <= nop; ++iop) {
        int j = 0;
        for (idim = 0; idim < ndim_iter; ++idim) {
          if (op_axes_arrays[i][idim] == -1 ||
              (iop != nop && opshape[i][op_axes_arrays[i][idim]] == 1 &&
              op_axes_arrays[iop][idim] != -1 &&
              opshape[iop][op_axes_arrays[iop][idim]] != 1)) {
            remainstride[iop][j++] = iterstride[iop][idim];
          } else {
            opstride[iop][op_axes_arrays[i][idim]] = iterstride[iop][idim];
          }
        }
      }
      MXNET_ACC_TYPE_SWITCH(out_data.type_flag_, DType, AType, {
        mxnet::common::StaticArray<DType*, NPY_MAXARGS> op;
        for (iop = 0; iop < nop; ++iop) {
          op[iop] = inputs[iop + back].dptr<DType>();
        }
        MXNET_ASSIGN_REQ_SWITCH(req[i], req_type, {
          MXNET_NDIM_SWITCH_EX(ndim_iter, dimension, {
            mxnet::common::StaticArray<mshadow::Shape<dimension>, NPY_MAXARGS> opstride_arr;
            mxnet::common::StaticArray<mshadow::Shape<dimension>, NPY_MAXARGS> remainstride_arr;
            for (iop = 0; iop <= nop; ++iop) {
              opstride_arr[iop] = opstride[iop].get<dimension>();
              remainstride_arr[iop] = remainstride[iop].get<dimension>();
            }
            Kernel<numpy_einsum<dimension, req_type, 1, AType>,
                  xpu>::Launch(ctx.get_stream<xpu>(),
                               opshape[i].Size(),
                               out_data.dptr<DType>(),
                               op,
                               opshape[i].get<dimension>(),
                               opstride_arr,
                               remainshape[i].get<dimension>(),
                               remainstride_arr,
                               nop,
                               i,
                               out_grad.dptr<DType>());
          })
        })
      })
    }
  }
}

template<typename xpu>
inline void NumpyEinsumForward(const OpStatePtr& state_ptr,
                               const OpContext& ctx,
                               const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  EinsumOp& state = state_ptr.get_state<EinsumOp>();
  int num_args = state.num_args;
  int optimize = state.optimize;
  const char* subscripts = state.subscripts.c_str();
  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(inputs.size(), num_args);
  CHECK_EQ(outputs.size(), 1U);
  if (optimize == 0) {
    NumpyEinsumProcess<xpu, 0>(inputs, req, outputs, subscripts, num_args, ctx);
    return;
  }
  std::vector<Step>& paths = state.paths;
  std::vector<std::vector<int> > pos;
  std::string string_repr;
  paths = einsum_path(state.subscripts, inputs, true, ctx.run_ctx, &pos, &string_repr);
  int paths_len = paths.size();
  size_t temp_space_size = 0, max_temp_space_size = 0;
  std::vector<TBlob> operands(inputs), tmp_operands, temp_space_vec(paths_len - 1);
  for (int i = 0; i + 1 < paths_len; ++i) {
    temp_space_size += paths[i].oshape.Size();
  }
  for (int i = 0; i < paths_len; ++i) {
    max_temp_space_size = std::max(max_temp_space_size, paths[i].oshape.Size());
  }
  temp_space_size += max_temp_space_size;
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    state.tempspace.reset<NDArray>(new NDArray(TShape(Shape1(temp_space_size)),
                                               ctx.run_ctx.ctx,
                                               false,
                                               outputs[0].type_flag_));
    Tensor<xpu, 1, DType> temp_space = state.tempspace->data().FlatTo1D<xpu, DType>();
    size_t begin = max_temp_space_size;
    for (int i = 0; i < paths_len - 1; ++i) {
      TBlob tblob = TBlob(temp_space.Slice(begin, begin + paths[i].oshape.Size()));
      temp_space_vec[i] = tblob.reshape(paths[i].oshape);
      begin = begin + paths[i].oshape.Size();
    }
    for (int i = 0; i < paths_len; ++i) {
      tmp_operands.clear();

      // We remove inds from right to left
      for (const int& p : paths[i].contract_inds) {
        tmp_operands.push_back(operands[p]);
        operands.erase(operands.begin() + p);
      }
      bool handle_out = (i == paths_len - 1);
      // Call tensordot if still possible
      if (paths[i].do_blas) {
        // Contract!
        if (paths[i].do_einsum || handle_out) {
          TBlob max_temp_space = TBlob(temp_space.Slice(0, paths[i].tshape.Size()));
          max_temp_space.FlatTo1D<xpu, DType>(s) = 0;
          max_temp_space = max_temp_space.reshape(paths[i].tshape);
          size_t tensordot_tempspace_size =
            TensordotWorkspaceSize<xpu>(paths[i].left_pos,
                                        paths[i].right_pos,
                                        tmp_operands[0],
                                        tmp_operands[1],
                                        max_temp_space,
                                        std::vector<OpReqType>{OpReqType::kWriteTo});
          Tensor<xpu, 1, char> tensordot_tempspace =
            ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(tensordot_tempspace_size), s);
          TensordotImpl<xpu>(paths[i].left_pos,
                             paths[i].right_pos,
                             ctx,
                             tmp_operands[0],
                             tmp_operands[1],
                             max_temp_space,
                             std::vector<OpReqType>{OpReqType::kWriteTo},
                             tensordot_tempspace);
          NumpyEinsumProcess<xpu, 0>(std::vector<TBlob>{max_temp_space},
            handle_out ? req : std::vector<OpReqType>{OpReqType::kWriteTo},
            handle_out ? outputs : std::vector<TBlob>{temp_space_vec[i]},
            paths[i].blas2einsum_str.c_str(),
            1, ctx);
        } else {
          size_t tensordot_tempspace_size =
            TensordotWorkspaceSize<xpu>(paths[i].left_pos,
                                        paths[i].right_pos,
                                        tmp_operands[0],
                                        tmp_operands[1],
                                        temp_space_vec[i],
                                        std::vector<OpReqType>{OpReqType::kWriteTo});
          Tensor<xpu, 1, char> tensordot_tempspace = ctx.requested[0].get_space_typed<xpu, 1, char>(
            Shape1(tensordot_tempspace_size), s);
          TensordotImpl<xpu>(paths[i].left_pos,
                             paths[i].right_pos,
                             ctx,
                             tmp_operands[0],
                             tmp_operands[1],
                             temp_space_vec[i],
                             std::vector<OpReqType>{OpReqType::kWriteTo},
                             tensordot_tempspace);
        }
      } else {
        NumpyEinsumProcess<xpu, 0>(tmp_operands,
        handle_out ? req : std::vector<OpReqType>{OpReqType::kWriteTo},
        handle_out ? outputs : std::vector<TBlob>{temp_space_vec[i]},
        paths[i].einsum_str.c_str(), tmp_operands.size(), ctx);
      }
      if (!handle_out) {
        operands.push_back(temp_space_vec[i]);
      }
    }
  });
}

template<typename xpu>
inline void NumpyEinsumBackward(const OpStatePtr& state_ptr,
                                const OpContext& ctx,
                                const std::vector<TBlob>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow_op;
  const EinsumOp& state = state_ptr.get_state<EinsumOp>();
  int num_args = state.num_args;
  int optimize = state.optimize;
  const char* subscripts = state.subscripts.c_str();
  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(inputs.size(), 1 + num_args);
  CHECK_EQ(outputs.size(), num_args);
  if (optimize == 0) {
    NumpyEinsumProcess<xpu, 1>(inputs, req, outputs, subscripts, num_args, ctx);
    return;
  }
  // calculate temporary space size for temp_grad
  const std::vector<Step>& paths = state.paths;
  int paths_len = paths.size();
  size_t temp_space_size = 0, max_temp_space_size = 0;
  for (int i = 0; i < paths_len - 1; ++i) {
    temp_space_size += paths[i].oshape.Size();
  }
  for (int i = 0; i < paths_len; ++i) {
    max_temp_space_size = std::max(max_temp_space_size, paths[i].oshape.Size());
  }
  temp_space_size += max_temp_space_size;
  // replay the forward process
  std::vector<std::vector<int> > op_idx(paths_len + 1);
  for (int i = 0; i <= paths_len; ++i) {
    if (i == 0) {
      op_idx[i].reserve(num_args);
      for (int j = 0; j < num_args; ++j) {
        op_idx[i].push_back(j + 1);
      }
    } else {
      op_idx[i] = op_idx[i - 1];
      // We remove inds from right to left
      for (const int& p : paths[i - 1].contract_inds) {
        op_idx[i].erase(op_idx[i].begin() + p);
      }
      op_idx[i].push_back(-static_cast<int>(i - 1));
    }
  }
  // calculate temporary space size for tensordot
  size_t tensordot_max_tempspace_size = 0;
  size_t begin_tensordot_tempspace = 0;
  std::vector<TBlob> temp_inputs, temp_outputs;
  std::vector<OpReqType> temp_req;
  std::vector<size_t> tensordot_tempspace_size;
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    for (int i = 0; i < paths_len; ++i) {
      temp_inputs.clear();
      temp_outputs.clear();
      temp_req.clear();
      bool handle_out = (i == paths_len - 1);

      if (handle_out) {
        temp_inputs.push_back(inputs[0]);
      } else {
        temp_inputs.push_back(TBlob(reinterpret_cast<DType*>(NULL),
                                    paths[i].oshape,
                                    xpu::kDevMask));
      }
      for (auto p : paths[i].contract_inds) {
        int idx = op_idx[i][p];
        if (idx >= 1) {
          temp_inputs.push_back(inputs[idx]);
          temp_outputs.push_back(outputs[idx - 1]);
          temp_req.push_back(req[idx - 1]);
        } else {
          temp_inputs.push_back(TBlob(reinterpret_cast<DType*>(NULL),
                                      paths[-idx].oshape,
                                      xpu::kDevMask));
          temp_outputs.push_back(TBlob(reinterpret_cast<DType*>(NULL),
                                      paths[-idx].oshape,
                                      xpu::kDevMask));
          temp_req.push_back(OpReqType::kWriteTo);
        }
      }
      size_t cur_tensordot_tempspace_size = 0;
      if (paths[i].do_blas) {
        if (paths[i].do_einsum) {
          cur_tensordot_tempspace_size =
            TensordotBackwardWorkspaceSize<xpu>(paths[i].left_pos,
                                                paths[i].right_pos,
                                                TBlob(reinterpret_cast<DType*>(NULL),
                                                      paths[i].tshape,
                                                      xpu::kDevMask),
                                                temp_inputs[1],
                                                temp_inputs[2],
                                                temp_outputs[0],
                                                temp_outputs[1],
                                                temp_req);
        } else {
          cur_tensordot_tempspace_size =
            TensordotBackwardWorkspaceSize<xpu>(paths[i].left_pos,
                                                paths[i].right_pos,
                                                temp_inputs[0],
                                                temp_inputs[1],
                                                temp_inputs[2],
                                                temp_outputs[0],
                                                temp_outputs[1],
                                                temp_req);
        }
      }
      tensordot_tempspace_size.push_back(cur_tensordot_tempspace_size);
      tensordot_max_tempspace_size = std::max(tensordot_max_tempspace_size,
                                              cur_tensordot_tempspace_size);
    }
    begin_tensordot_tempspace = temp_space_size;
    temp_space_size += (tensordot_max_tempspace_size + sizeof(DType) - 1) / sizeof(DType);
  });
  // allocate temporary space and propagate
  std::vector<TBlob> temp_grad(paths_len - 1), temp_data(paths_len - 1);
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    // allocate temporary space for gradients of intermediate results
    Tensor<xpu, 1, DType> temp_space = ctx.requested[0].get_space_typed<xpu, 1, DType>
      (Shape1(temp_space_size), s);
    size_t begin = max_temp_space_size;
    for (int i = 0; i + 1 < paths_len; ++i) {
      TBlob tblob = TBlob(temp_space.Slice(begin, begin + paths[i].oshape.Size()));
      temp_grad[i] = tblob.reshape(paths[i].oshape);
      begin = begin + paths[i].oshape.Size();
    }

    // reinterprete ndarray for intermediate results
    Tensor<xpu, 1, DType> ndarray_space = state.tempspace->data().FlatTo1D<xpu, DType>();
    begin = max_temp_space_size;
    for (int i = 0; i + 1 < paths_len; ++i) {
      TBlob tblob = TBlob(ndarray_space.Slice(begin, begin + paths[i].oshape.Size()));
      temp_data[i] = tblob.reshape(paths[i].oshape);
      begin = begin + paths[i].oshape.Size();
    }

    // go through the paths in the reversed order
    for (int i = paths_len - 1; i >= 0; i--) {
      temp_inputs.clear();
      temp_outputs.clear();
      temp_req.clear();
      bool handle_out = (i == paths_len - 1);

      if (handle_out) {
        temp_inputs.push_back(inputs[0]);
      } else {
        temp_inputs.push_back(temp_grad[i]);
      }
      for (auto p : paths[i].contract_inds) {
        int idx = op_idx[i][p];
        if (idx >= 1) {
          temp_inputs.push_back(inputs[idx]);
          temp_outputs.push_back(outputs[idx - 1]);
          temp_req.push_back(req[idx - 1]);
        } else {
          temp_inputs.push_back(temp_data[-idx]);
          temp_outputs.push_back(temp_grad[-idx]);
          temp_req.push_back(OpReqType::kWriteTo);
        }
      }
      if (paths[i].do_blas) {
        CHECK_EQ(temp_inputs.size(), 3U);
        CHECK_EQ(temp_outputs.size(), 2U);
        CHECK_EQ(temp_req.size(), 2U);
        Tensor<xpu, 1, DType> tensordot_tempspace = temp_space.Slice(begin_tensordot_tempspace,
                                                                     temp_space_size);
        Tensor<xpu, 1, char> char_tempspace =
          Tensor<xpu, 1, char>(reinterpret_cast<char*>(tensordot_tempspace.dptr_),
                                                       Shape1(tensordot_tempspace_size[i]),
                                                       tensordot_tempspace.stream_);
        if (paths[i].do_einsum) {
          TBlob max_temp_space = TBlob(temp_space.Slice(0, paths[i].tshape.Size()));
          max_temp_space = max_temp_space.reshape(paths[i].tshape);
          NumpyEinsumProcess<xpu, 0>(std::vector<TBlob>{temp_inputs[0]},
                                     std::vector<OpReqType>{kWriteTo},
                                     std::vector<TBlob>{max_temp_space},
                                     paths[i].einsum2blas_str.c_str(),
                                     1, ctx);
          TensordotBackwardImpl<xpu>(paths[i].left_pos, paths[i].right_pos, ctx,
                                     max_temp_space, temp_inputs[1], temp_inputs[2],
                                     temp_outputs[0], temp_outputs[1], temp_req, char_tempspace);
        } else {
          TensordotBackwardImpl<xpu>(paths[i].left_pos, paths[i].right_pos, ctx,
                                     temp_inputs[0], temp_inputs[1], temp_inputs[2],
                                     temp_outputs[0], temp_outputs[1], temp_req, char_tempspace);
        }
      } else {
        NumpyEinsumProcess<xpu, 1>(temp_inputs, temp_req, temp_outputs,
                                   paths[i].einsum_str.c_str(),
                                   temp_outputs.size(),
                                   ctx);
      }
    }
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_EINSUM_OP_INL_H_
