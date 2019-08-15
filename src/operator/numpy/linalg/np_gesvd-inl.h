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
 * \file np_gesvd-inl.h
 * \brief Function definition of the SVD Operator.
 */
#ifndef MXNET_OPERATOR_NUMPY_LINALG_NP_GESVD_INL_H_
#define MXNET_OPERATOR_NUMPY_LINALG_NP_GESVD_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "../../tensor/la_op.h"
#include "../../tensor/la_op-inl.h"

namespace mxnet {
namespace op {

struct GesvdVecSign {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, int m, int n, DType* UT,
                                  DType* V, int ldut, int ldv) {
    DType* vrow(V + i * ldv);
    DType maxval(fabs(vrow[0])), vval(0.0);
    int maxind(0);
    for (int i = 1; i < n; ++i) {
      vval = fabs(vrow[i]);
      if (vval > maxval) {
        maxval = vval;
        maxind = i;
      }
    }
    if (vrow[maxind] < 0) {
      DType* utcol(UT + i % m + (i / m) * ldut * m);
      for (int i = 0; i < n; ++i) {
        vrow[i] = -vrow[i];
        if (i < m) {
          utcol[i * ldut] = -utcol[i * ldut];
        }
      }
    }
  }
};

// (UT, L, V) = gesvd(A) [singular value decomposition]
// - V can overwrite A
// - Needs workspace (both DType and int), size of which is determined by a
//   workspace query
struct gesvd {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& A,
                 const Tensor<xpu, 3, DType>& UT,
                 const Tensor<xpu, 2, DType>& L,
                 const Tensor<xpu, 3, DType>& V,
                 const OpContext& ctx,
                 const nnvm::NodeAttrs& attrs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    if (A.dptr_ != V.dptr_) Copy(V, A, s);
    // From here on, we work on V only
    // Reserve workspace (size determined by query)
    int lwork(linalg_gesvd_workspace_query(UT[0], L[0], V[0], s));
    Tensor<xpu, 1, DType> work = ctx.requested[0]
      .get_space_typed<xpu, 1, DType>(Shape1(lwork), s);
    // Loop over items in batch
    for (index_t i = 0; i < UT.size(0); ++i) {
      linalg_gesvd(UT[i], L[i], V[i], work, s);
    }
    // Set signs in a deterministic way
    using namespace mxnet_op;
    Kernel<GesvdVecSign, xpu>::Launch
      (s, V.size(0) * V.size(1), V.size(1), V.size(2),
       UT.dptr_, V.dptr_, UT.stride_, V.stride_);
  }
};

// (A) => (UT, L, V)
template<typename xpu, typename laop>
void NumpyLaGesvdForward(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(outputs.size(), 3);
  if (inputs[0].shape_.Size() == 0) {
    return;
  }
  MSHADOW_SGL_DBL_TYPE_SWITCH(outputs[0].type_flag_, OType, {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    laop::op(inputs[0].FlatToKD<xpu, 3, OType>(s),
             outputs[0].FlatToKD<xpu, 3, OType>(s),
             outputs[1].FlatToKD<xpu, 2, OType>(s),
             outputs[2].FlatToKD<xpu, 3, OType>(s), ctx, attrs);
  });
}

// Helper for gesvd_backward. See technical report for details
template<typename DType>
DType gesvd_back_helper_eps(DType* X);

template<>
MSHADOW_XINLINE float gesvd_back_helper_eps(float* X) {
  return 1e-30;
}

template<>
MSHADOW_XINLINE double gesvd_back_helper_eps(double* X) {
  return 1e-100;
}

struct GesvdBackHelper_dV {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int k, int m, int n, DType* L, int ldl,
                                  DType* dA, int ldda) {
    const int offl(k * ldl);
    const int offda(k * m * ldda);
    DType denom(0.0);
    const DType eps(gesvd_back_helper_eps(dA));
    for (int i = 0; i < m; ++i) {
      denom = L[offl + i];
      if (denom < eps) denom = eps;
      for (int j = 0; j < n; ++j) {
        dA[offda + i * ldda + j] /= denom;
      }
    }
  }
};

struct GesvdBackHelper_G1 {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int k, int m, int n, DType* X, int ldx,
                                  DType* L, int ldl) {
    const int offl(k * ldl);
    const int offx(k * m * ldx);
    DType numer(0.0);
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < m; ++j) {
        numer = L[offl + j];
        X[offx + i * ldx + j] *= numer;
      }
    }
  }
};

struct GesvdBackHelper_G2 {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int k, int m, int n, DType* X, int ldx,
                                  DType* L, int ldl, DType* dL, int lddl,
                                  DType* dA, int ldda, DType* V, int ldv) {
    const int offx(k * m * ldx);
    const int offl(k * ldl);
    const int offdl(k * lddl);
    const int offda(k * m * ldda);
    const int offv(k * m * ldv);
    const DType eps(gesvd_back_helper_eps(X));
    DType denom1(0.0), denom2(0.0), elem(0.0);

    for (int i = 0; i < m - 1; ++i) {
      for (int j = i + 1; j < m; ++j) {
        denom1 = L[offl + i] - L[offl + j];
        denom2 = L[offl + i] + L[offl + j];
        if (denom1 < eps) denom1 = eps;
        if (denom2 < eps) denom2 = eps;
        elem = (X[offx + i * ldx + j] - X[offx + j * ldx + i]) / denom1 / denom2;
        X[offx + i * ldx + j] = elem * L[offl + j];
        X[offx + j * ldx + i] = elem * L[offl + i];
      }
    }
    for (int i = 0; i < m; ++i) {
      elem = DType(0.0);
      for (int j = 0; j < n; ++j) {
        elem += dA[offda + i * ldda + j] * V[offv + i * ldv + j];
      }
      elem = -elem + dL[offdl + i];
      X[offx + i * ldx + i] = elem;
    }
  }
};

struct gesvd_backward {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& dUT,
                 const Tensor<xpu, 2, DType>& dL,
                 const Tensor<xpu, 3, DType>& dV,
                 const Tensor<xpu, 3, DType>& UT,
                 const Tensor<xpu, 2, DType>& L,
                 const Tensor<xpu, 3, DType>& V,
                 const Tensor<xpu, 3, DType>& dA,
                 const Tensor<xpu, 3, DType>& tempMs,
                 const Tensor<xpu, 3, DType>& tempMr,
                 Stream<xpu>* s, const nnvm::NodeAttrs& attrs) {
    // Backward of (UT, L, V) = gesvd(A)
    using namespace mxnet_op;
    if (dA.dptr_ != dV.dptr_) {
      Copy(dA, dV, s);
    }
    // From here on, we work on dA only

    // Need temporal space, same shape as dUT
    // invdV:
    Kernel<GesvdBackHelper_dV, xpu>::Launch
      (s, dA.size(0), dA.size(1), dA.size(2), L.dptr_, L.stride_, dA.dptr_, dA.stride_);

    // G1:
    // This copy is just to make sure there are no invalid values (NaN, infinity) in tempM
    Copy(tempMs, dUT, s);
    Copy(tempMr, dA, s);
    gemm::op(dA, V, tempMs, DType(1.0), DType(0.0), false, true, s);
    Kernel<GesvdBackHelper_G1, xpu>::Launch
      (s, dA.size(0), dA.size(1), dA.size(2), tempMs.dptr_, tempMs.stride_,
       L.dptr_, L.stride_);
    gemm::op(dUT, UT, tempMs, DType(1.0), DType(1.0), true, false, s);

    // G2:
    Kernel<GesvdBackHelper_G2, xpu>::Launch
      (s, dA.size(0), dA.size(1), dA.size(2), tempMs.dptr_, tempMs.stride_,
       L.dptr_, L.stride_, dL.dptr_, dL.stride_, dA.dptr_, dA.stride_,
       V.dptr_, V.stride_);

    // G3:
    gemm::op(tempMs, V, dA, DType(1.0), DType(1.0), false, false, s);
    gemm::op(UT, dA, tempMr, DType(1.0), DType(0.0), false, false, s);
    Copy(dA, tempMr, s);
  }
};

// (dUT, dL, dV, UT, L, V) => (dA)
template<typename xpu, typename laop>
void NumpyLaGesvdBackward(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(inputs.size(), 6);
  CHECK_EQ(outputs.size(), 1);
  if (outputs[0].shape_.Size() == 0) {
    return;
  }
  MSHADOW_SGL_DBL_TYPE_SWITCH(outputs[0].type_flag_, OType, {
    TBlob tspace(outputs[0]);
    TBlob tempMs, tempMr;
    if (req[0] == kAddTo) {
      Tensor<xpu, 1, OType> tempspace = ctx.requested[0]
        .get_space_typed<xpu, 1, OType>(Shape1(2 * outputs[0].shape_.Size()), s);
      tspace = TBlob(tempspace.Slice(0, outputs[0].shape_.Size()))
        .reshape(outputs[0].shape_);
      tempMs = TBlob(tempspace.Slice(outputs[0].shape_.Size(),
                                     outputs[0].shape_.Size() + inputs[0].shape_.Size()))
        .reshape(inputs[0].shape_);
      tempMr = TBlob(tempspace.Slice(outputs[0].shape_.Size(),
                                     2 * outputs[0].shape_.Size()))
        .reshape(outputs[0].shape_);
    } else {
      Tensor<xpu, 1, OType> tempspace = ctx.requested[0]
        .get_space_typed<xpu, 1, OType>(Shape1(outputs[0].shape_.Size()), s);
      tempMs = TBlob(tempspace.Slice(0, inputs[0].shape_.Size())).reshape(inputs[0].shape_);
      tempMr = TBlob(tempspace.Slice(0, outputs[0].shape_.Size())).reshape(outputs[0].shape_);
    }
    laop::op(inputs[0].FlatToKD<xpu, 3, OType>(s),  // dUT
             inputs[1].FlatToKD<xpu, 2, OType>(s),  // dL
             inputs[2].FlatToKD<xpu, 3, OType>(s),  // dV
             inputs[3].FlatToKD<xpu, 3, OType>(s),  // UT
             inputs[4].FlatToKD<xpu, 2, OType>(s),  // L
             inputs[5].FlatToKD<xpu, 3, OType>(s),  // V
             tspace.FlatToKD<xpu, 3, OType>(s),  // dA
             tempMs.FlatToKD<xpu, 3, OType>(s),  // tempMs
             tempMr.FlatToKD<xpu, 3, OType>(s),  // tempMr
             s, attrs);
    if (req[0] == kAddTo) {
      Tensor<xpu, 1, OType> out = outputs[0].FlatTo1D<xpu, OType>(s);
      out += tspace.FlatTo1D<xpu, OType>(s);
    }
  });
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NUMPY_LINALG_NP_GESVD_INL_H_
