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
 * \brief Operators for advanced linear algebra.
 */
#ifndef MXNET_OPERATOR_TENSOR_LA_OP_H_
#define MXNET_OPERATOR_TENSOR_LA_OP_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <algorithm>
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
  }
};

// Parameters for general matrix-matrix multiply
struct LaMatrixMultParam : public dmlc::Parameter<LaMatrixMultParam> {
  bool transpose_a, transpose_b;
  double alpha;
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
  }
};

// Parameters for matrix-matrix multiplication where one is a triangular matrix.
struct LaTriangMatrixMultParam : public dmlc::Parameter<LaTriangMatrixMultParam> {
  bool transpose;
  bool rightside;
  double alpha;
  DMLC_DECLARE_PARAMETER(LaTriangMatrixMultParam) {
    DMLC_DECLARE_FIELD(transpose)
      .set_default(false)
      .describe("Use transposed of the triangular matrix");
    DMLC_DECLARE_FIELD(rightside)
      .set_default(false)
      .describe("Multiply triangular matrix from the right to non-triangular one.");
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

// Common function for shape inference for matrix mult and matrix mac.
inline bool LaMatrixMultMacOpShape(const nnvm::NodeAttrs& attrs,
                                   std::vector<TShape>* in_attrs,
                                   std::vector<TShape>* out_attrs) {
  CHECK_GE(in_attrs->size(), 2);
  CHECK_EQ(out_attrs->size(), 1);
  bool transpose_a(false), transpose_b(false);
  if ( in_attrs->size() == 2 ) {
     // Matrix-Matrix mult
     transpose_a = nnvm::get<LaMatrixMultParam>(attrs.parsed).transpose_a;
     transpose_b = nnvm::get<LaMatrixMultParam>(attrs.parsed).transpose_b;
  } else {
     // Matrix-Matrix mac
     transpose_a = nnvm::get<LaMatrixMacParam>(attrs.parsed).transpose_a;
     transpose_b = nnvm::get<LaMatrixMacParam>(attrs.parsed).transpose_b;
  }
  if ( (*in_attrs)[0].ndim() >= 2 && (*in_attrs)[0].ndim() == (*in_attrs)[1].ndim() ) {
    // Forward shape inference.
    const int ndim((*in_attrs)[0].ndim());
    std::vector<int> oshape(ndim);
    for ( int i = 0; i < ndim-2; ++i ) {
      // Both inputs must have same shape except for last two dimensions.
      CHECK_EQ((*in_attrs)[0][i], (*in_attrs)[1][i])
        << "Shapes of inputs 0, 1 must be the same, except on last two dimensions";
      oshape[i] = (*in_attrs)[0][i];
    }
    CHECK_EQ((transpose_a ? (*in_attrs)[0][ndim-2] : (*in_attrs)[0][ndim-1]),
             (transpose_b ? (*in_attrs)[1][ndim-1] : (*in_attrs)[1][ndim-2]))
             << "Incompatible matrix dimensions for multiplication";
    oshape[ndim-2] = (transpose_a ? (*in_attrs)[0][ndim-1] : (*in_attrs)[0][ndim-2]);
    oshape[ndim-1] = (transpose_b ? (*in_attrs)[1][ndim-2] : (*in_attrs)[1][ndim-1]);
    TShape tshape(oshape.begin(), oshape.end());
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
                                      std::vector<TShape>* in_attrs,
                                      std::vector<TShape>* out_attrs) {
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
    TShape tshape(oshape.begin(), oshape.end());
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
    TShape tshape1(ishape1.begin(), ishape1.end());
    SHAPE_ASSIGN_CHECK(*in_attrs, 0, tshape1);
    TShape tshape2(ishape2.begin(), ishape2.end());
    SHAPE_ASSIGN_CHECK(*in_attrs, 1, tshape2);
    return true;
  }
  return false;
}

template<int dim>
inline bool LaReduceShape(const nnvm::NodeAttrs& attrs,
                          std::vector<TShape>* in_attrs,
                          std::vector<TShape>* out_attrs) {
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
  TShape tshape(oshape.begin(), oshape.end());
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, tshape);
  return true;
}

// Shape inference function for linalg_syrk
inline bool LaSyrkShape(const nnvm::NodeAttrs& attrs,
                        std::vector<TShape>* in_attrs,
                        std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 1);
  const TShape& in_attr = (*in_attrs)[0];
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
    TShape tshape(oshape.begin(), oshape.end());
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, tshape);
    return true;
  }
  // Can't do backward inference of shapes for this operator.
  return false;
}

// Shape inference function for linalg_gelqf
// Inputs: A. Outputs: Q, L
inline bool LaLQFactShape(const nnvm::NodeAttrs& attrs,
                          std::vector<TShape>* in_attrs,
                          std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 2);
  const TShape& in_a = (*in_attrs)[0];
  const TShape& out_q = (*out_attrs)[0];
  const TShape& out_l = (*out_attrs)[1];
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
    TShape tshape_l(oshape_l.begin(), oshape_l.end());
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

// Shape inference function for linalg_syevd
// Inputs: A. Outputs: U, L
inline bool LaEigFactShape(const nnvm::NodeAttrs& attrs,
                           std::vector<TShape>* in_attrs,
                           std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 2);
  const TShape& in_a = (*in_attrs)[0];
  const TShape& out_u = (*out_attrs)[0];
  const TShape& out_l = (*out_attrs)[1];
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
    TShape tshape_l(oshape_l.begin(), oshape_l.end());
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

// Adapters for calling the various operators with appropriate signatures.

template<typename xpu, typename DType, int idim, int odim, int inum, int onum, typename laop>
struct LaOpCaller {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<TBlob>& outputs,
                 const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx) {
    CHECK(false) << "no specialized LaOpCaller defined for template parameters";
  }
};
template<typename xpu, typename DType, int idim, int odim, typename laop>
struct LaOpCaller<xpu, DType, idim, odim, 1, 1, laop> {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<TBlob>& outputs,
                 const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx) {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    laop::op(inputs[0].FlatToKD<xpu, idim+1, DType>(s),
             outputs[0].FlatToKD<xpu, odim+1, DType>(s), ctx, attrs);
  }
};
template<typename xpu, typename DType, int idim, int odim, typename laop>
struct LaOpCaller<xpu, DType, idim, odim, 1, 2, laop> {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<TBlob>& outputs,
                 const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx) {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    laop::op(inputs[0].FlatToKD<xpu, idim+1, DType>(s),
             outputs[0].FlatToKD<xpu, odim+1, DType>(s),
             outputs[1].FlatToKD<xpu, odim+1, DType>(s), ctx, attrs);
  }
};
template<typename xpu, typename DType, int idim, int odim, typename laop>
struct LaOpCaller<xpu, DType, idim, odim, 2, 1, laop> {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<TBlob>& outputs,
                 const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx) {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    laop::op(inputs[0].FlatToKD<xpu, idim+1, DType>(s),
             inputs[1].FlatToKD<xpu, idim+1, DType>(s),
             outputs[0].FlatToKD<xpu, odim+1, DType>(s), ctx, attrs);
  }
};
template<typename xpu, typename DType, int idim, int odim, typename laop>
struct LaOpCaller<xpu, DType, idim, odim, 3, 1, laop> {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<TBlob>& outputs,
                 const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx) {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    laop::op(inputs[0].FlatToKD<xpu, idim+1, DType>(s),
             inputs[1].FlatToKD<xpu, idim+1, DType>(s),
             inputs[2].FlatToKD<xpu, idim+1, DType>(s),
             outputs[0].FlatToKD<xpu, odim+1, DType>(s), ctx, attrs);
  }
};
template<typename xpu, typename DType, int idim, int odim, typename laop>
struct LaOpCaller<xpu, DType, idim, odim, 3, 2, laop> {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<TBlob>& outputs,
                 const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx) {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    laop::op(inputs[0].FlatToKD<xpu, idim+1, DType>(s),
             inputs[1].FlatToKD<xpu, idim+1, DType>(s),
             inputs[2].FlatToKD<xpu, idim+1, DType>(s),
             outputs[0].FlatToKD<xpu, odim+1, DType>(s),
             outputs[1].FlatToKD<xpu, odim+1, DType>(s), ctx, attrs);
  }
};
template<typename xpu, typename DType, int idim, int odim, typename laop>
struct LaOpCaller<xpu, DType, idim, odim, 4, 1, laop> {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<TBlob>& outputs,
                 const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx) {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    laop::op(inputs[0].FlatToKD<xpu, idim+1, DType>(s),
             inputs[1].FlatToKD<xpu, idim+1, DType>(s),
             inputs[2].FlatToKD<xpu, idim+1, DType>(s),
             inputs[3].FlatToKD<xpu, idim+1, DType>(s),
             outputs[0].FlatToKD<xpu, odim+1, DType>(s), ctx, attrs);
  }
};
template<typename xpu, typename DType, int idim, int odim, typename laop>
struct LaOpCaller<xpu, DType, idim, odim, 4, 2, laop> {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<TBlob>& outputs,
                 const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx) {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    laop::op(inputs[0].FlatToKD<xpu, idim+1, DType>(s),
             inputs[1].FlatToKD<xpu, idim+1, DType>(s),
             inputs[2].FlatToKD<xpu, idim+1, DType>(s),
             inputs[3].FlatToKD<xpu, idim+1, DType>(s),
             outputs[0].FlatToKD<xpu, odim+1, DType>(s),
             outputs[1].FlatToKD<xpu, odim+1, DType>(s), ctx, attrs);
  }
};
template<typename xpu, typename DType, int idim, int odim, typename laop>
struct LaOpCaller<xpu, DType, idim, odim, 4, 3, laop> {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<TBlob>& outputs,
                 const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx) {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    laop::op(inputs[0].FlatToKD<xpu, idim+1, DType>(s),
             inputs[1].FlatToKD<xpu, idim+1, DType>(s),
             inputs[2].FlatToKD<xpu, idim+1, DType>(s),
             inputs[3].FlatToKD<xpu, idim+1, DType>(s),
             outputs[0].FlatToKD<xpu, odim+1, DType>(s),
             outputs[1].FlatToKD<xpu, odim+1, DType>(s),
             outputs[2].FlatToKD<xpu, odim+1, DType>(s), ctx, attrs);
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

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_LA_OP_H_
