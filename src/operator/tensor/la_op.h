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
 * Copyright (c) 2017 by Contributors
 * \file la_op.h
 * \brief Function definition of Operators for advanced linear algebra.
 */
#ifndef MXNET_OPERATOR_TENSOR_LA_OP_H_
#define MXNET_OPERATOR_TENSOR_LA_OP_H_

#include <mxnet/operator_util.h>
#include <mxnet/imperative.h>
#include <vector>
#include <algorithm>
#include <string>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

// Parameters for general matrix-matrix multiply-accumulate (mac)
struct LaMatrixMacParam : public dmlc::Parameter<LaMatrixMacParam> {
  bool transpose_a, transpose_b;
  double alpha, beta;
  int axis;
  DMLC_DECLARE_PARAMETER(LaMatrixMacParam) {
    DMLC_DECLARE_FIELD(transpose_a)
      .set_default(false)
      .describe("Multiply with transposed of first input (A).");
    DMLC_DECLARE_FIELD(transpose_b)
      .set_default(false)
      .describe("Multiply with transposed of second input (B).");
    DMLC_DECLARE_FIELD(alpha)
      .set_default(1.0)
      .describe("Scalar factor multiplied with A*B.");
    DMLC_DECLARE_FIELD(beta)
      .set_default(1.0)
      .describe("Scalar factor multiplied with C.");
    DMLC_DECLARE_FIELD(axis)
      .set_default(-2)
      .describe("Axis corresponding to the matrix rows.");
  }
};

// Parameters for general matrix-matrix multiply
struct LaMatrixMultParam : public dmlc::Parameter<LaMatrixMultParam> {
  bool transpose_a, transpose_b;
  double alpha;
  int axis;
  DMLC_DECLARE_PARAMETER(LaMatrixMultParam) {
    DMLC_DECLARE_FIELD(transpose_a)
      .set_default(false)
      .describe("Multiply with transposed of first input (A).");
    DMLC_DECLARE_FIELD(transpose_b)
      .set_default(false)
      .describe("Multiply with transposed of second input (B).");
    DMLC_DECLARE_FIELD(alpha)
      .set_default(1.0)
      .describe("Scalar factor multiplied with A*B.");
    DMLC_DECLARE_FIELD(axis)
      .set_default(-2)
      .describe("Axis corresponding to the matrix row indices.");
  }
};

// Parameters for Cholesky factorization and matrix inversion
struct LaCholeskyParam : public dmlc::Parameter<LaCholeskyParam> {
  bool lower;
  DMLC_DECLARE_PARAMETER(LaCholeskyParam) {
    DMLC_DECLARE_FIELD(lower)
      .set_default(true)
      .describe
         ("True if the triangular matrix is lower triangular, false if it is upper triangular.");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream lower_s;
    lower_s << lower;
    (*dict)["lower"] = lower_s.str();
  }
};

// Parameters for matrix-matrix multiplication where one is a triangular matrix.
struct LaTriangMatrixMultParam : public dmlc::Parameter<LaTriangMatrixMultParam> {
  bool transpose;
  bool rightside;
  bool lower;
  double alpha;
  DMLC_DECLARE_PARAMETER(LaTriangMatrixMultParam) {
    DMLC_DECLARE_FIELD(transpose)
      .set_default(false)
      .describe("Use transposed of the triangular matrix");
    DMLC_DECLARE_FIELD(rightside)
      .set_default(false)
      .describe("Multiply triangular matrix from the right to non-triangular one.");
    DMLC_DECLARE_FIELD(lower)
      .set_default(true)
      .describe
         ("True if the triangular matrix is lower triangular, false if it is upper triangular.");
    DMLC_DECLARE_FIELD(alpha)
      .set_default(1.0)
      .describe("Scalar factor to be applied to the result.");
  }
};

// Parameters for syrk
struct LaSyrkParam : public dmlc::Parameter<LaSyrkParam> {
  bool transpose;
  double alpha;
  DMLC_DECLARE_PARAMETER(LaSyrkParam) {
    DMLC_DECLARE_FIELD(transpose)
      .set_default(false)
      .describe("Use transpose of input matrix.");
    DMLC_DECLARE_FIELD(alpha)
      .set_default(1.0)
      .describe("Scalar factor to be applied to the result.");
  }
};

// Parameters for diag extraction/creation.
struct LaDiagParam : public dmlc::Parameter<LaDiagParam> {
  int offset;
  DMLC_DECLARE_PARAMETER(LaDiagParam) {
    DMLC_DECLARE_FIELD(offset)
      .set_default(0)
      .describe("Offset of the diagonal versus the main diagonal. 0 corresponds to the main "
                "diagonal, a negative/positive value to diagonals below/above the main diagonal.");
  }
};

// Parameters for trian extraction/creation.
struct LaTrianParam : public dmlc::Parameter<LaTrianParam> {
  int  offset;
  bool lower;
  DMLC_DECLARE_PARAMETER(LaTrianParam) {
    DMLC_DECLARE_FIELD(offset)
      .set_default(0)
      .describe("Offset of the diagonal versus the main diagonal. 0 corresponds to the main "
                "diagonal, a negative/positive value to diagonals below/above the main diagonal.");
    DMLC_DECLARE_FIELD(lower)
      .set_default(true)
      .describe("Refer to the lower triangular matrix if lower=true, refer to the upper otherwise."
                 " Only relevant when offset=0");
  }
};

// Common function for shape inference for matrix mult and matrix mac.
inline bool LaMatrixMultMacOpShape(const nnvm::NodeAttrs& attrs,
                                   mxnet::ShapeVector* in_attrs,
                                   mxnet::ShapeVector* out_attrs) {
  CHECK_GE(in_attrs->size(), 2);
  CHECK_EQ(out_attrs->size(), 1);
  bool transpose_a(false), transpose_b(false);
  int axis_param(-2);
  if ( in_attrs->size() == 2 ) {
     // Matrix-Matrix mult
     transpose_a = nnvm::get<LaMatrixMultParam>(attrs.parsed).transpose_a;
     transpose_b = nnvm::get<LaMatrixMultParam>(attrs.parsed).transpose_b;
     axis_param  = nnvm::get<LaMatrixMultParam>(attrs.parsed).axis;
  } else {
     // Matrix-Matrix mac
     transpose_a = nnvm::get<LaMatrixMacParam>(attrs.parsed).transpose_a;
     transpose_b = nnvm::get<LaMatrixMacParam>(attrs.parsed).transpose_b;
     axis_param  = nnvm::get<LaMatrixMacParam>(attrs.parsed).axis;
  }
  if ( (*in_attrs)[0].ndim() >= 2 && (*in_attrs)[0].ndim() == (*in_attrs)[1].ndim() ) {
    // Forward shape inference.
    const int ndim((*in_attrs)[0].ndim()), axis(axis_param < 0 ? ndim + axis_param : axis_param);
    CHECK(axis >= 0 && axis < ndim-1)
      << "Invalid row axis (" << axis_param << ")";
    std::vector<int> oshape(ndim);
    for ( int i = 0; i < ndim-1; ++i ) {
      if (i != axis) {
        // Both inputs must have same shape except for row/col dimensions.
        CHECK_EQ((*in_attrs)[0][i], (*in_attrs)[1][i])
          << "Shapes of inputs 0, 1 must be the same, except on row/col axis";
      }
      oshape[i] = (*in_attrs)[0][i];
    }
    CHECK_EQ((transpose_a ? (*in_attrs)[0][axis] : (*in_attrs)[0][ndim-1]),
             (transpose_b ? (*in_attrs)[1][ndim-1] : (*in_attrs)[1][axis]))
             << "Incompatible matrix dimensions for multiplication";
    oshape[axis] = (transpose_a ? (*in_attrs)[0][ndim-1] : (*in_attrs)[0][axis]);
    oshape[ndim-1] = (transpose_b ? (*in_attrs)[1][axis] : (*in_attrs)[1][ndim-1]);
    mxnet::TShape tshape(oshape.begin(), oshape.end());
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, tshape);
    if ( in_attrs->size() > 2 ) {
       // Infer/check shape of third operand of a mac.
       SHAPE_ASSIGN_CHECK(*in_attrs, 2, tshape);
    }
    return true;
  }
  // Can't do backward inference of shapes for this operator.
  return false;
}

inline bool LaTriangMatrixMultOpShape(const nnvm::NodeAttrs& attrs,
                                      mxnet::ShapeVector* in_attrs,
                                      mxnet::ShapeVector* out_attrs) {
  const LaTriangMatrixMultParam& param = nnvm::get<LaTriangMatrixMultParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 2);
  CHECK_EQ(out_attrs->size(), 1);
  if ( (*in_attrs)[0].ndim() >= 2 && (*in_attrs)[0].ndim() == (*in_attrs)[1].ndim() ) {
    // Forward shape inference.
    const int ndim((*in_attrs)[0].ndim());
    CHECK_EQ((*in_attrs)[0][ndim-2], (*in_attrs)[0][ndim-1])
      << "First operand must be a tensor of square matrices";
    std::vector<int> oshape(ndim);
    for ( int i = 0; i < ndim-2; ++i ) {
      // Must have same shape except for last two dimensions.
      CHECK_EQ((*in_attrs)[0][i], (*in_attrs)[1][i])
        << "Shapes of inputs 0, 1 must be the same, except on last two dimensions";
      oshape[i] = (*in_attrs)[0][i];
    }
    if ( param.rightside ) {
      // We compute B * A where A is the first and B the second input.
      CHECK_EQ((*in_attrs)[0][ndim-2], (*in_attrs)[1][ndim-1])
        << "Incompatible matrix dimensions for multiplication";
      oshape[ndim-2] = (*in_attrs)[1][ndim-2];
      oshape[ndim-1] = (param.transpose ? (*in_attrs)[0][ndim-2] : (*in_attrs)[0][ndim-1]);
    } else {
      // We compute A * B where A is the first and B the second input.
      CHECK_EQ((*in_attrs)[1][ndim-2], (*in_attrs)[0][ndim-1])
        << "Incompatible matrix dimensions for multiplication";
      oshape[ndim-2] = (param.transpose ? (*in_attrs)[0][ndim-1] : (*in_attrs)[0][ndim-2]);
      oshape[ndim-1] = (*in_attrs)[1][ndim-1];
    }
    mxnet::TShape tshape(oshape.begin(), oshape.end());
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, tshape);
    return true;
  }
  if ( (*out_attrs)[0].ndim() >= 2 ) {
    // Backward shape inference.
    const int odim((*out_attrs)[0].ndim());
    std::vector<int> ishape1(odim), ishape2(odim);
    for ( int i = 0; i < odim-2; ++i ) {
      ishape1[i] = ishape2[i] = (*out_attrs)[0][i];
    }
    if ( param.rightside ) {
      // We compute B * A where A is the first and B the second input.
      ishape2[odim-2] = (*out_attrs)[0][odim-2];
      ishape1[odim-2] = ishape1[odim-1] = ishape2[odim-1] = (*out_attrs)[0][odim-1];
    } else {
      // We compute A * B where A is the first and B the second input.
      ishape2[odim-1] = (*out_attrs)[0][odim-1];
      ishape1[odim-2] = ishape1[odim-1] = ishape2[odim-2] = (*out_attrs)[0][odim-2];
    }
    mxnet::TShape tshape1(ishape1.begin(), ishape1.end());
    SHAPE_ASSIGN_CHECK(*in_attrs, 0, tshape1);
    mxnet::TShape tshape2(ishape2.begin(), ishape2.end());
    SHAPE_ASSIGN_CHECK(*in_attrs, 1, tshape2);
    return true;
  }
  return false;
}

template<int dim>
inline bool LaReduceShape(const nnvm::NodeAttrs& attrs,
                          mxnet::ShapeVector* in_attrs,
                          mxnet::ShapeVector* out_attrs) {
  // Shape for reduction of the dim lowest dimensions to a scalar.
  // Can only deduct in forward direction.
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 1);
  const int ndim((*in_attrs)[0].ndim());
  if (ndim < dim) {
    return false;
  }
  std::vector<int> oshape(std::max(1, ndim-dim));
  oshape[0] = 1;
  for ( int i = 0; i < ndim - dim; ++i ) {
    oshape[i] = (*in_attrs)[0][i];
  }
  // Will reduce all matrices/vectors to a scalar.
  mxnet::TShape tshape(oshape.begin(), oshape.end());
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, tshape);
  return true;
}

template<bool diag, bool extract>
inline bool LaDiagTrianShape(const nnvm::NodeAttrs& attrs,
                             mxnet::ShapeVector* in_attrs,
                             mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 1);
  const int ndim((*in_attrs)[0].ndim());
  // Only infer in forward direction
  if (ndim == 0) {
    return false;
  }
  const int offset = (diag ? nnvm::get<LaDiagParam>(attrs.parsed).offset
                           : nnvm::get<LaTrianParam>(attrs.parsed).offset);
  std::vector<int> oshape(extract ? ndim-1 : ndim+1);
  for (int i = 0; i < ndim-1; ++i) {
    oshape[i] = (*in_attrs)[0][i];
  }
  if (extract) {
    CHECK_GE(ndim, 2)
      << "Input operand must be a tensor of matrices";
    CHECK_EQ((*in_attrs)[0][ndim-2], (*in_attrs)[0][ndim-1])
      << "Input operand must be a tensor of square matrices";
    const int n((*in_attrs)[0][ndim-1]-abs(offset));
    CHECK_GT(n, 0)
      << "Illegal offset " << offset << " for diag/trian extraction of matrix with dimension "
      << ndim;
    oshape[ndim-2] = (diag ? n : (n*(n+1))/2);
  } else if (diag) {
    oshape[ndim] = oshape[ndim-1] = (*in_attrs)[0][ndim-1]+abs(offset);
  } else {
    const int n((*in_attrs)[0][ndim-1]);
    const int m(std::floor(0.5+(std::sqrt(8*n+1)-1.0)*0.5));
    CHECK_EQ((m*(m+1))/2, n)
      << "Input tensor of maketrian has an invalid dimension for the last axis.";
    oshape[ndim] = oshape[ndim-1] = m+abs(offset);
  }
  mxnet::TShape tshape(oshape.begin(), oshape.end());
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, tshape);
  return true;
}

// Shape inference function for linalg_syrk
inline bool LaSyrkShape(const nnvm::NodeAttrs& attrs,
                        mxnet::ShapeVector* in_attrs,
                        mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 1);
  const mxnet::TShape& in_attr = (*in_attrs)[0];
  bool transpose = nnvm::get<LaSyrkParam>(attrs.parsed).transpose;
  const int ndim = in_attr.ndim();
  if ( ndim >= 2 ) {
    // Forward shape inference.
    std::vector<int> oshape(ndim);
    for ( int i = 0; i < ndim-2; ++i ) {
      oshape[i] = in_attr[i];
    }
    oshape[ndim-2] = (transpose ? in_attr[ndim-1] : in_attr[ndim-2]);
    oshape[ndim-1] = oshape[ndim-2];
    mxnet::TShape tshape(oshape.begin(), oshape.end());
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, tshape);
    return true;
  }
  // Can't do backward inference of shapes for this operator.
  return false;
}

// Shape inference function for linalg_gelqf
// Inputs: A. Outputs: Q, L
inline bool LaLQFactShape(const nnvm::NodeAttrs& attrs,
                          mxnet::ShapeVector* in_attrs,
                          mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 2);
  const mxnet::TShape& in_a = (*in_attrs)[0];
  const mxnet::TShape& out_q = (*out_attrs)[0];
  const mxnet::TShape& out_l = (*out_attrs)[1];
  if ( in_a.ndim() >= 2 ) {
    // Forward shape inference.
    const int ndim(in_a.ndim());
    CHECK_LE(in_a[ndim-2], in_a[ndim-1])
      << "Input A shape wrong: Last dimension must be >= than second to last";
    // Q must have same shape as A
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_a);
    std::vector<int> oshape_l(ndim);
    for ( int i = 0; i < ndim-1; ++i ) {
      oshape_l[i] = in_a[i];
    }
    oshape_l[ndim-1] = in_a[ndim-2];
    mxnet::TShape tshape_l(oshape_l.begin(), oshape_l.end());
    SHAPE_ASSIGN_CHECK(*out_attrs, 1, tshape_l);
    return true;
  }
  if ( out_q.ndim() >= 2 && out_q.ndim() == out_l.ndim() ) {
    // Backward shape inference.
    const int ndim(out_q.ndim());
    for ( int i = 0; i < ndim-1; ++i ) {
      CHECK_EQ(out_q[i], out_l[i])
        << "Outputs Q, L must have same dimensions except for last";
    }
    CHECK_LE(out_q[ndim-2], out_q[ndim-1])
      << "Output Q shape wrong: Last dimension must be >= than second to last";
    CHECK_EQ(out_l[ndim-2], out_l[ndim-1])
      << "Output L shape wrong: Last two dimensions must be equal";
    SHAPE_ASSIGN_CHECK(*in_attrs, 0, out_q);
    return true;
  }
  return false;
}

// Shape inference function for linalg_inverse
// Inputs: A. Outputs: inverse(A)
inline bool InverseShape(const nnvm::NodeAttrs& attrs,
                         mxnet::ShapeVector* in_attrs,
                         mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 1);
  const mxnet::TShape& in = (*in_attrs)[0];
  if (!ndim_is_known(in)) return false;
  const int ndim(in.ndim());
  CHECK_GE(ndim, 2) << "Input A's dimension must be >= 2";
  CHECK_EQ(in[ndim-2], in[ndim-1]) << "Input A's last two dimension must be equal";
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, in);
  return shape_is_known(in);
}

// Shape inference function for det functions in linalg
template<int onum>
inline bool DetShape(const nnvm::NodeAttrs& attrs,
                     mxnet::ShapeVector* in_attrs,
                     mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), onum + 2);
  const mxnet::TShape& in = (*in_attrs)[0];
  if (!ndim_is_known(in)) return false;
  const int ndim(in.ndim());
  CHECK_GE(ndim, 2) << "Input A's dimension must be >= 2";
  CHECK_EQ(in[ndim-2], in[ndim-1]) << "Input A's last two dimension must be equal";
  mxnet::TShape out;
  if (ndim == 2) {
    if (Imperative::Get()->is_np_shape() || in.Size() == 0U) {
      out = mxnet::TShape(0, 1);
    } else {
      out = mxnet::TShape(1, 1);
    }
  } else {
    out = mxnet::TShape(in.begin(), in.end() - 2);
  }
  for (int i = 0; i < onum; ++i) {
    SHAPE_ASSIGN_CHECK(*out_attrs, i, out); /* sign or det or logdet */
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, onum, in); /* LU */
  SHAPE_ASSIGN_CHECK(*out_attrs, onum + 1, mxnet::TShape(in.begin(), in.end() - 1)); /* pivot */
  return shape_is_known(in);
}

// Type inference function for det functions in linalg
template<int onum>
inline bool DetType(const nnvm::NodeAttrs& attrs,
                    std::vector<int>* in_type,
                    std::vector<int>* out_type) {
  using namespace mshadow;
  CHECK_EQ(in_type->size(), 1);
  CHECK_EQ(out_type->size(), onum + 2);
  const int dtype = (*in_type)[0];
  if (dtype == -1) return false;
  CHECK(dtype == kFloat32 || dtype == kFloat64)
    << "This operation only supports 32-bit and 64-bit floating point";
  for (int i = 0; i < onum; ++i) {
    TYPE_ASSIGN_CHECK(*out_type, i, dtype);  /* sign or det or logdet */
  }
  TYPE_ASSIGN_CHECK(*out_type, onum, dtype);  /* LU */
  TYPE_ASSIGN_CHECK(*out_type, onum + 1, index_type_flag); /* pivot */
  return true;
}

// Shape inference function for linalg_syevd
// Inputs: A. Outputs: U, L
inline bool LaEigFactShape(const nnvm::NodeAttrs& attrs,
                           mxnet::ShapeVector* in_attrs,
                           mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 2);
  const mxnet::TShape& in_a = (*in_attrs)[0];
  const mxnet::TShape& out_u = (*out_attrs)[0];
  const mxnet::TShape& out_l = (*out_attrs)[1];
  if ( in_a.ndim() >= 2 ) {
    // Forward shape inference.
    const int ndim(in_a.ndim());
    CHECK_EQ(in_a[ndim-2], in_a[ndim-1])
      << "Input A shape wrong: Last two dimensions must be equal";
    // U must have same shape as A
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_a);
    std::vector<int> oshape_l(ndim-1);
    for ( int i = 0; i < ndim-1; ++i ) {
      oshape_l[i] = in_a[i];
    }
    mxnet::TShape tshape_l(oshape_l.begin(), oshape_l.end());
    SHAPE_ASSIGN_CHECK(*out_attrs, 1, tshape_l);
    return true;
  }
  if ( out_u.ndim() >= 2 && out_u.ndim() == out_l.ndim()+1 ) {
    // Backward shape inference.
    const int ndim(out_u.ndim());
    for ( int i = 0; i < ndim-1; ++i ) {
      CHECK_EQ(out_u[i], out_l[i])
        << "Outputs U, L must have same dimensions except for last";
    }
    CHECK_EQ(out_u[ndim-2], out_u[ndim-1])
      << "Output U shape wrong: Last two dimensions must be equal";
    // A must have same shape as U
    SHAPE_ASSIGN_CHECK(*in_attrs, 0, out_u);
    return true;
  }
  return false;
}

// Flattener for following adaptors.
template<typename xpu, int dim, typename DType>
mshadow::Tensor<xpu, dim, DType> LaOpFlatten(const TBlob& blob,
                                             mshadow::Stream<xpu> *s, int axis = -2) {
  if (axis < 0) {
    axis = blob.ndim() + axis;
  }
  if (axis >= blob.ndim()-2) {
    // Leave highest axis, collapse rest.
    return blob.FlatToKD<xpu, dim, DType>(s);
  }
  // Collapse ranges [0,axis-1] and [axis+1,ndim-2].
  CHECK_EQ(dim, 4);
  mxnet::TShape shape(dim, -1);
  shape[0] = 1;
  for (int i = 0; i < axis; ++i) {
    shape[0] *= blob.shape_[i];
  }
  shape[1] = blob.shape_[axis];
  shape[2] = 1;
  for (int i = axis+1; i < blob.ndim()-1; ++i) {
    shape[2] *= blob.shape_[i];
  }
  shape[3] = blob.shape_[blob.ndim()-1];
  return blob.get_with_shape<xpu, dim, DType>(shape.get<dim>(), s);
}

// Adapters for calling the various operators with appropriate signatures.

template<typename xpu, typename DType, int idim, int odim, int inum, int onum, typename laop>
struct LaOpCaller {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<TBlob>& outputs,
                 const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx, int axis = -2) {
    CHECK(false) << "no specialized LaOpCaller defined for template parameters";
  }
};
template<typename xpu, typename DType, int idim, int odim, typename laop>
struct LaOpCaller<xpu, DType, idim, odim, 1, 1, laop> {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<TBlob>& outputs,
                 const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx, int axis = -2) {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    laop::op(LaOpFlatten<xpu, idim+1, DType>(inputs[0], s, axis),
             LaOpFlatten<xpu, odim+1, DType>(outputs[0], s, axis), ctx, attrs);
  }
};
template<typename xpu, typename DType, int idim, int odim, typename laop>
struct LaOpCaller<xpu, DType, idim, odim, 1, 2, laop> {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<TBlob>& outputs,
                 const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx, int axis = -2) {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    laop::op(LaOpFlatten<xpu, idim+1, DType>(inputs[0], s, axis),
             LaOpFlatten<xpu, odim+1, DType>(outputs[0], s, axis),
             LaOpFlatten<xpu, odim+1, DType>(outputs[1], s, axis), ctx, attrs);
  }
};
template<typename xpu, typename DType, int idim, int odim, typename laop>
struct LaOpCaller<xpu, DType, idim, odim, 2, 1, laop> {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<TBlob>& outputs,
                 const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx, int axis = -2) {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    laop::op(LaOpFlatten<xpu, idim+1, DType>(inputs[0], s, axis),
             LaOpFlatten<xpu, idim+1, DType>(inputs[1], s, axis),
             LaOpFlatten<xpu, odim+1, DType>(outputs[0], s, axis), ctx, attrs);
  }
};
template<typename xpu, typename DType, int idim, int odim, typename laop>
struct LaOpCaller<xpu, DType, idim, odim, 3, 1, laop> {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<TBlob>& outputs,
                 const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx, int axis = -2) {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    laop::op(LaOpFlatten<xpu, idim+1, DType>(inputs[0], s, axis),
             LaOpFlatten<xpu, idim+1, DType>(inputs[1], s, axis),
             LaOpFlatten<xpu, idim+1, DType>(inputs[2], s, axis),
             LaOpFlatten<xpu, odim+1, DType>(outputs[0], s, axis), ctx, attrs);
  }
};
template<typename xpu, typename DType, int idim, int odim, typename laop>
struct LaOpCaller<xpu, DType, idim, odim, 3, 2, laop> {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<TBlob>& outputs,
                 const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx, int axis = -2) {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    laop::op(LaOpFlatten<xpu, idim+1, DType>(inputs[0], s, axis),
             LaOpFlatten<xpu, idim+1, DType>(inputs[1], s, axis),
             LaOpFlatten<xpu, idim+1, DType>(inputs[2], s, axis),
             LaOpFlatten<xpu, odim+1, DType>(outputs[0], s, axis),
             LaOpFlatten<xpu, odim+1, DType>(outputs[1], s, axis), ctx, attrs);
  }
};
template<typename xpu, typename DType, int idim, int odim, typename laop>
struct LaOpCaller<xpu, DType, idim, odim, 4, 1, laop> {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<TBlob>& outputs,
                 const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx, int axis = -2) {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    laop::op(LaOpFlatten<xpu, idim+1, DType>(inputs[0], s, axis),
             LaOpFlatten<xpu, idim+1, DType>(inputs[1], s, axis),
             LaOpFlatten<xpu, idim+1, DType>(inputs[2], s, axis),
             LaOpFlatten<xpu, idim+1, DType>(inputs[3], s, axis),
             LaOpFlatten<xpu, odim+1, DType>(outputs[0], s, axis), ctx, attrs);
  }
};
template<typename xpu, typename DType, int idim, int odim, typename laop>
struct LaOpCaller<xpu, DType, idim, odim, 4, 2, laop> {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<TBlob>& outputs,
                 const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx, int axis = -2) {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    laop::op(LaOpFlatten<xpu, idim+1, DType>(inputs[0], s, axis),
             LaOpFlatten<xpu, idim+1, DType>(inputs[1], s, axis),
             LaOpFlatten<xpu, idim+1, DType>(inputs[2], s, axis),
             LaOpFlatten<xpu, idim+1, DType>(inputs[3], s, axis),
             LaOpFlatten<xpu, odim+1, DType>(outputs[0], s, axis),
             LaOpFlatten<xpu, odim+1, DType>(outputs[1], s, axis), ctx, attrs);
  }
};
template<typename xpu, typename DType, int idim, int odim, typename laop>
struct LaOpCaller<xpu, DType, idim, odim, 4, 3, laop> {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<TBlob>& outputs,
                 const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx, int axis = -2) {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    laop::op(LaOpFlatten<xpu, idim+1, DType>(inputs[0], s, axis),
             LaOpFlatten<xpu, idim+1, DType>(inputs[1], s, axis),
             LaOpFlatten<xpu, idim+1, DType>(inputs[2], s, axis),
             LaOpFlatten<xpu, idim+1, DType>(inputs[3], s, axis),
             LaOpFlatten<xpu, odim+1, DType>(outputs[0], s, axis),
             LaOpFlatten<xpu, odim+1, DType>(outputs[1], s, axis),
             LaOpFlatten<xpu, odim+1, DType>(outputs[2], s, axis), ctx, attrs);
  }
};


template<typename xpu, int idim, int odim, int inum, int onum, typename laop>
void LaOpForward(const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx,
                 const std::vector<TBlob>& inputs,
                 const std::vector<OpReqType>& req,
                 const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), inum);
  CHECK_EQ(outputs.size(), onum);
  MSHADOW_SGL_DBL_TYPE_SWITCH(outputs[0].type_flag_, OType, {
    LaOpCaller<xpu, OType, idim, odim, inum, onum, laop>::op(inputs, outputs,
                                                             attrs, ctx);
  });
}

template<typename xpu, int idim, int odim, int inum, int onum, typename laop>
void LaOpBackward(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(inputs.size(), inum);
  CHECK_EQ(outputs.size(), onum);
  MSHADOW_SGL_DBL_TYPE_SWITCH(outputs[0].type_flag_, OType, {
    std::vector<TBlob> tspace(outputs);
    for ( int i = 0; i < onum; ++i ) {
      if ( req[i] == kAddTo ) {
        tspace[i].dptr_ = ctx.requested[0]
                             .get_space_typed<xpu, 1, OType>(Shape1(outputs[i].Size()), s).dptr_;
      }
    }
    LaOpCaller<xpu, OType, idim, odim, inum, onum, laop>::op(inputs, tspace,
                                                             attrs, ctx);
    for ( int i = 0; i < onum; ++i ) {
      if ( req[i] == kAddTo ) {
        Tensor<xpu, 1, OType> out = outputs[i].FlatTo1D<xpu, OType>(s);
        out += tspace[i].FlatTo1D<xpu, OType>(s);
      }
    }
  });
}

template<typename xpu, int idim, int odim, int inum, int onum, typename laop>
void LaOpGemmForward(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), inum);
  CHECK_EQ(outputs.size(), onum);
  const int axis(inputs.size() == 2 ? nnvm::get<LaMatrixMultParam>(attrs.parsed).axis
                                    : nnvm::get<LaMatrixMacParam>(attrs.parsed).axis);
  MSHADOW_SGL_DBL_TYPE_SWITCH(outputs[0].type_flag_, OType, {
    if (axis == -2 || axis == inputs[0].ndim()-2) {
      LaOpCaller<xpu, OType, idim, odim, inum, onum, laop>::op(inputs, outputs,
                                                               attrs, ctx);
    } else {
      LaOpCaller<xpu, OType, idim+1, odim+1, inum, onum, laop>::op(inputs, outputs,
                                                                   attrs, ctx, axis);
    }
  });
}

template<typename xpu, int idim, int odim, int inum, int onum, typename laop>
void LaOpGemmBackward(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(inputs.size(), inum);
  CHECK_EQ(outputs.size(), onum);
  const int axis(inputs.size() == 3 ? nnvm::get<LaMatrixMultParam>(attrs.parsed).axis
                                    : nnvm::get<LaMatrixMacParam>(attrs.parsed).axis);
  MSHADOW_SGL_DBL_TYPE_SWITCH(outputs[0].type_flag_, OType, {
    std::vector<TBlob> tspace(outputs);
    for ( int i = 0; i < onum; ++i ) {
      if ( req[i] == kAddTo ) {
        tspace[i].dptr_ = ctx.requested[0]
                             .get_space_typed<xpu, 1, OType>(Shape1(outputs[i].Size()), s).dptr_;
      }
    }
    if (axis == -2 || axis == inputs[0].ndim()-2) {
      LaOpCaller<xpu, OType, idim, odim, inum, onum, laop>::op(inputs, outputs,
                                                               attrs, ctx);
    } else {
      LaOpCaller<xpu, OType, idim+1, odim+1, inum, onum, laop>::op(inputs, outputs,
                                                                   attrs, ctx, axis);
    }
    for ( int i = 0; i < onum; ++i ) {
      if ( req[i] == kAddTo ) {
        Tensor<xpu, 1, OType> out = outputs[i].FlatTo1D<xpu, OType>(s);
        out += tspace[i].FlatTo1D<xpu, OType>(s);
      }
    }
  });
}

// Specific wrapper for syevd (cannot use the default ones, because A, U have
// different dimensionality than L

// (A) => (U, L)
template<typename xpu, typename laop>
void LaOpForwSyevd(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(outputs.size(), 2);
  MSHADOW_SGL_DBL_TYPE_SWITCH(outputs[0].type_flag_, OType, {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    laop::op(inputs[0].FlatToKD<xpu, 3, OType>(s),
             outputs[0].FlatToKD<xpu, 3, OType>(s),
             outputs[1].FlatToKD<xpu, 2, OType>(s), ctx, attrs);
  });
}

// (dU, dL, U, L) => (dA)
template<typename xpu, typename laop>
void LaOpBackwSyevd(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(inputs.size(), 4);
  CHECK_EQ(outputs.size(), 1);
  MSHADOW_SGL_DBL_TYPE_SWITCH(outputs[0].type_flag_, OType, {
    std::vector<TBlob> tspace(outputs);
    if ( req[0] == kAddTo ) {
      tspace[0].dptr_ = ctx.requested[0]
        .get_space_typed<xpu, 1, OType>(Shape1(outputs[0].Size()), s).dptr_;
    }
    laop::op(inputs[0].FlatToKD<xpu, 3, OType>(s),
             inputs[1].FlatToKD<xpu, 2, OType>(s),
             inputs[2].FlatToKD<xpu, 3, OType>(s),
             inputs[3].FlatToKD<xpu, 2, OType>(s),
             tspace[0].FlatToKD<xpu, 3, OType>(s), ctx, attrs);
    if ( req[0] == kAddTo ) {
      Tensor<xpu, 1, OType> out = outputs[0].FlatTo1D<xpu, OType>(s);
      out += tspace[0].FlatTo1D<xpu, OType>(s);
    }
  });
}


template<typename xpu, typename DType, int onum, typename laop, typename IndexT>
struct LaOpDetForwardCaller {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<TBlob>& outputs,
                 const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx) {
      CHECK(false) << "no specialized LaOpDetForward defined for template parameters";
  }
};
template<typename xpu, typename DType, typename laop, typename IndexT>
struct LaOpDetForwardCaller<xpu, DType, 1, laop, IndexT> {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<TBlob>& outputs,
                 const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx) {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    laop::op(inputs[0].FlatToKD<xpu, 3, DType>(s),
             outputs[0].FlatToKD<xpu, 1, DType>(s),
             outputs[1].FlatToKD<xpu, 3, DType>(s),
             outputs[2].FlatToKD<xpu, 2, IndexT>(s), ctx, attrs);
  }
};
template<typename xpu, typename DType, typename laop, typename IndexT>
struct LaOpDetForwardCaller<xpu, DType, 2, laop, IndexT> {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<TBlob>& outputs,
                 const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx) {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    laop::op(inputs[0].FlatToKD<xpu, 3, DType>(s),
             outputs[0].FlatToKD<xpu, 1, DType>(s),
             outputs[1].FlatToKD<xpu, 1, DType>(s),
             outputs[2].FlatToKD<xpu, 3, DType>(s),
             outputs[3].FlatToKD<xpu, 2, IndexT>(s), ctx, attrs);
  }
};
template<typename xpu, int onum, typename laop>
void LaOpDetForward(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using IndexT = lapack_index_t;
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(outputs.size(), onum + 2);
  MSHADOW_SGL_DBL_TYPE_SWITCH(outputs[0].type_flag_, OType, {
    LaOpDetForwardCaller<xpu, OType, onum, laop, IndexT>::op(inputs, outputs, attrs, ctx);
  });
}

template<typename xpu, typename DType, int onum, typename laop, typename IndexT>
struct LaOpDetBackwardCaller {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<TBlob>& outputs,
                 const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx) {
      CHECK(false) << "no specialized LaOpDetBackward defined for template parameters";
  }
};
template<typename xpu, typename DType, typename laop, typename IndexT>
struct LaOpDetBackwardCaller<xpu, DType, 1, laop, IndexT> {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<TBlob>& outputs,
                 const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx) {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    laop::op(inputs[0].FlatToKD<xpu, 1, DType>(s),
             inputs[1].FlatToKD<xpu, 1, DType>(s),
             inputs[2].FlatToKD<xpu, 3, DType>(s),
             inputs[3].FlatToKD<xpu, 2, IndexT>(s),
             outputs[0].FlatToKD<xpu, 3, DType>(s), ctx, attrs);
  }
};
template<typename xpu, typename DType, typename laop, typename IndexT>
struct LaOpDetBackwardCaller<xpu, DType, 2, laop, IndexT> {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<TBlob>& outputs,
                 const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx) {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    laop::op(inputs[0].FlatToKD<xpu, 1, DType>(s),
             inputs[1].FlatToKD<xpu, 1, DType>(s),
             inputs[2].FlatToKD<xpu, 1, DType>(s),
             inputs[3].FlatToKD<xpu, 3, DType>(s),
             inputs[4].FlatToKD<xpu, 2, IndexT>(s),
             outputs[0].FlatToKD<xpu, 3, DType>(s), ctx, attrs);
  }
};
template<typename xpu, int onum, typename laop>
void LaOpDetBackward(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using IndexT = lapack_index_t;
  if (outputs[0].shape_.Size() == 0U) {
    return;
  }
  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(inputs.size(), onum + 3);
  CHECK_EQ(outputs.size(), 1);
  MSHADOW_SGL_DBL_TYPE_SWITCH(outputs[0].type_flag_, OType, {
    std::vector<TBlob> tspace(outputs);
    for ( size_t i = 0; i < outputs.size(); ++i ) {
      if ( req[i] == kAddTo ) {
        tspace[i].dptr_ = ctx.requested[0]
                             .get_space_typed<xpu, 1, OType>(Shape1(outputs[i].Size()), s).dptr_;
      }
    }
    LaOpDetBackwardCaller<xpu, OType, onum, laop, IndexT>::op(inputs, tspace, attrs, ctx);
    for ( size_t i = 0; i < outputs.size(); ++i ) {
      if ( req[i] == kAddTo ) {
        Tensor<xpu, 1, OType> out = outputs[i].FlatTo1D<xpu, OType>(s);
        out += tspace[i].FlatTo1D<xpu, OType>(s);
      }
    }
  });
}

// Only transfer ddet and outputs to gradient
template<int onum>
struct ReduceDetGrad {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::ObjectPtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) {
    std::vector<nnvm::NodeEntry> heads;
    heads.push_back(ograds[onum - 1]);
    uint32_t n_out = n->num_outputs();
    for (uint32_t i = 0; i < n_out; ++i) {
      heads.emplace_back(nnvm::NodeEntry{n, i, 0});
    }
    return MakeGradNode(op_name, n, heads, n->attrs.dict);
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_LA_OP_H_
