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
 * \file np_eig-inl.h
 * \brief Placeholder for eig
 */
#ifndef MXNET_OPERATOR_NUMPY_LINALG_NP_EIG_INL_H_
#define MXNET_OPERATOR_NUMPY_LINALG_NP_EIG_INL_H_

#include <vector>
#include "./np_eigvals-inl.h"

namespace mxnet {
namespace op {

using namespace mshadow;

template<int req>
struct eigvec_assign_helper {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, const DType *in_data, DType *out_data,
                                  const int nrow, const int ld, const int step) {
    int idx = i / step, row = (i % step) / ld, col = (i % step) % ld;
    KERNEL_ASSIGN(out_data[idx * step + row + col * ld], req, in_data[i]);
  }
};

// Calculates workspace size of eig forward op.
// The dimension of the array WORK in LAPACKE_#GEEV should >= max(1,3*N), and
// if JOBVL = 'V' or JOBVR = 'V', LWORK >= 4*N.
// For good performance, LWORK must generally be larger.
template<typename xpu>
size_t EigForwardWorkspaceSize(const TBlob& a,
                               const TBlob& w,
                               const TBlob& v,
                               const std::vector<OpReqType>& req) {
  if (kNullOp == req[0] && kNullOp == req[1]) { return 0U; }

  // Zero-size input, no need to launch kernel
  if (0U == a.Size()) { return 0U; }

  MSHADOW_SGL_DBL_TYPE_SWITCH(w.type_flag_, DType, {
    size_t work_space_size = 0;
    size_t n = a.size(a.ndim() - 1);
    work_space_size += a.Size();      // For matrix.
    work_space_size += 2 * w.Size();  // For eigenvalues' real and image component.
    work_space_size += n * n;         // For left eigenvectors temp memory
    work_space_size += v.Size();      // For right eigenvectors real and image component.
    work_space_size += 4 * n;         // For workspace size in LAPACKE_#GEEV.
    work_space_size *= sizeof(DType);
    return work_space_size;
  });
  LOG(FATAL) << "InternalError: cannot reach here";
  return 0U;
}

template<typename xpu>
void EigOpForwardImpl(const TBlob& a,
                      const TBlob& w,
                      const TBlob& v,
                      const std::vector<OpReqType>& req,
                      std::vector<char> *workspace,
                      mshadow::Stream<xpu> *s) {
  if (kNullOp == req[0] && kNullOp == req[1]) { return; }
  const mxnet::TShape& a_shape = a.shape_;
  const mxnet::TShape& w_shape = w.shape_;
  const mxnet::TShape& v_shape = v.shape_;
  const int a_ndim = a_shape.ndim();

  // Zero-size output, no need to launch kernel
  if (0U == a.Size()) { return; }

  MSHADOW_SGL_DBL_TYPE_SWITCH(w.type_flag_, DType, {
    const int N = a_shape[a_ndim - 1];
    DType *a_ptr =
      reinterpret_cast<DType*>(workspace->data());
    DType *wr_ptr =
      reinterpret_cast<DType*>(workspace->data() + a.Size() * sizeof(DType));
    DType *wi_ptr =
      reinterpret_cast<DType*>(workspace->data() + (w.Size() + a.Size()) * sizeof(DType));
    DType *vl_ptr =
      reinterpret_cast<DType*>(workspace->data() + (2 * w.Size() + a.Size()) * sizeof(DType));
    DType *vr_ptr =
      reinterpret_cast<DType*>(
        workspace->data() + (2 * w.Size() + N * N + a.Size()) * sizeof(DType));
    DType *work_ptr =
      reinterpret_cast<DType*>(
        workspace->data() + (2 * w.Size() + v.Size() + N * N + a.Size()) * sizeof(DType));
    MSHADOW_TYPE_SWITCH(a.type_flag_, AType, {
      // Cast type and transpose.
      mxnet_op::Kernel<SolveTypeTransposeHelper, xpu>::Launch(
        s, a_shape.Size(), a.dptr<AType>(), a_ptr, N, N, N * N);
    });
    char jobvl = 'N', jobvr = 'V';
    mxnet::TBlob a_trans_data(a_ptr, a_shape, a.dev_mask(), a.dev_id());
    mxnet::TBlob wr_data(wr_ptr, w_shape, w.dev_mask(), w.dev_id());
    mxnet::TBlob wi_data(wi_ptr, w_shape, w.dev_mask(), w.dev_id());
    mxnet::TBlob vl_data(vl_ptr, Shape3(1, N, N), v.dev_mask(), v.dev_id());
    mxnet::TBlob vr_data(vr_ptr, v_shape, v.dev_mask(), v.dev_id());
    mxnet::TBlob work_data(work_ptr, Shape1(4 * N), a.dev_mask(), a.dev_id());
    eig_eigvals::op(jobvl, jobvr,
                    a_trans_data.FlatToKD<xpu, 3, DType>(s),
                    wr_data.FlatToKD<xpu, 2, DType>(s),
                    wi_data.FlatToKD<xpu, 2, DType>(s),
                    vl_data.get<xpu, 3, DType>(s),
                    vr_data.FlatToKD<xpu, 3, DType>(s),
                    work_data.get<xpu, 1, DType>(s));
    for (size_t i = 0; i < wi_data.Size(); ++i) {
      CHECK_LE(fabs(wi_ptr[i]), 1e-15)
        << "Complex eigvals is unsupported in linalg temporary.";
    }
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      mxnet_op::Kernel<eigvals_assign_helper<req_type>, xpu>::Launch(
        s, w.Size(), wr_ptr, w.dptr<DType>());
    });
    MXNET_ASSIGN_REQ_SWITCH(req[1], req_type, {
      mxnet_op::Kernel<eigvec_assign_helper<req_type>, xpu>::Launch(
        s, v.Size(), vr_ptr, v.dptr<DType>(), N, N, N * N);
    });
  });
}

template<typename xpu>
void EigOpForward(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 2U);
  CHECK_EQ(req.size(), 2U);
  const TBlob& a = inputs[0];
  const TBlob& w = outputs[0];
  const TBlob& v = outputs[1];

  // Calculate workspace size.
  size_t workspace_size = EigForwardWorkspaceSize<cpu>(a, w, v, req);
  std::vector<char> workspace(workspace_size, 0);

  MSHADOW_SGL_DBL_TYPE_SWITCH(w.type_flag_, WType, {
    MSHADOW_SGL_DBL_TYPE_SWITCH(a.type_flag_, AType, {
      if (xpu::kDevCPU) {
        // Op forward implement.
        EigOpForwardImpl<cpu>(a, w, v, req, &workspace, ctx.get_stream<cpu>());
      } else {
#if __CUDACC__
        mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
        cudaStream_t stream = Stream<gpu>::GetStream(s);
        std::vector<AType> a_vec(a.Size(), 0);
        std::vector<WType> w_vec(w.Size(), 0);
        std::vector<WType> v_vec(v.Size(), 0);
        AType* a_cp_ptr = a_vec.data();
        WType* w_cp_ptr = w_vec.data();
        WType* v_cp_ptr = v_vec.data();
        CUDA_CALL(cudaMemcpyAsync(a_cp_ptr, a.dptr<AType>(), sizeof(AType) * a.Size(),
                                  cudaMemcpyDeviceToHost, stream));
        CUDA_CALL(cudaMemcpyAsync(w_cp_ptr, w.dptr<WType>(), sizeof(WType) * w.Size(),
                                  cudaMemcpyDeviceToHost, stream));
        CUDA_CALL(cudaMemcpyAsync(v_cp_ptr, v.dptr<WType>(), sizeof(WType) * v.Size(),
                                  cudaMemcpyDeviceToHost, stream));
        CUDA_CALL(cudaStreamSynchronize(stream));
        mxnet::TBlob a_data(a_cp_ptr, a.shape_, cpu::kDevMask);
        mxnet::TBlob w_data(w_cp_ptr, w.shape_, cpu::kDevMask);
        mxnet::TBlob v_data(v_cp_ptr, v.shape_, cpu::kDevMask);
        // Op forward implement on cpu.
        EigOpForwardImpl<cpu>(a_data, w_data, v_data, req, &workspace, ctx.get_stream<cpu>());
        // Copy back to gpu.
        CUDA_CALL(cudaMemcpyAsync(w.dptr<WType>(), w_cp_ptr, sizeof(WType) * w.Size(),
                                  cudaMemcpyHostToDevice, stream));
        CUDA_CALL(cudaMemcpyAsync(v.dptr<WType>(), v_cp_ptr, sizeof(WType) * v.Size(),
                                  cudaMemcpyHostToDevice, stream));
        CUDA_CALL(cudaStreamSynchronize(stream));
#else
        LOG(FATAL) << "Please build with USE_CUDA=1 to enable GPU";
#endif  // __CUDACC__
      }
    });
  });
}

struct EighParam : public dmlc::Parameter<EighParam> {
  char UPLO;
  DMLC_DECLARE_PARAMETER(EighParam) {
    DMLC_DECLARE_FIELD(UPLO)
    .set_default('L')
    .describe("Specifies whether the calculation is done with the lower or upper triangular part.");
  }
};

template<typename xpu>
void EighOpForward(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 2U);
  CHECK_EQ(req.size(), 2U);
  const TBlob& a = inputs[0];
  const TBlob& w = outputs[0];
  const TBlob& v = outputs[1];
  const char UPLO = nnvm::get<EighParam>(attrs.parsed).UPLO;
  Stream<xpu> *s = ctx.get_stream<xpu>();

  if (kNullOp == req[0] && kNullOp == req[0]) { return; }
  // Zero-size output, no need to launch kernel
  if (0U == a.Size()) { return; }

  // Calculate workspace size.
  size_t workspace_size = EighEigvalshForwardWorkspaceSize<xpu>(a, w, req, ctx);
  Tensor<xpu, 1, char> workspace =
    ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);

  EighEigvalshOpForwardImpl(a, w, UPLO, attrs, ctx, req, workspace);

  MSHADOW_SGL_DBL_TYPE_SWITCH(w.type_flag_, DType, {
    DType *a_ptr = reinterpret_cast<DType*>(workspace.dptr_);
    DType *w_ptr = reinterpret_cast<DType*>(workspace.dptr_ + a.Size() * sizeof(DType));
    TBlob a_data(a_ptr, a.shape_, a.dev_mask(), a.dev_id());
    Tensor<xpu, 3, DType> A = a_data.FlatToKD<xpu, 3, DType>(s);

    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      mxnet_op::Kernel<eigvals_assign_helper<req_type>, xpu>::Launch(
        s, w.Size(), w_ptr, w.dptr<DType>());
    });

    // Set signs of eigenvectors in a deterministic way
    mxnet_op::Kernel<SyevdEigenVecSigns, xpu>::Launch(
      s, A.size(0) * A.size(1), A.size(1), A.dptr_, A.stride_);

    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      mxnet_op::Kernel<eigvec_assign_helper<req_type>, xpu>::Launch(
        s, v.Size(), a_ptr, v.dptr<DType>(),
        A.size(1), A.stride_,
        A.size(1) * A.stride_);
    });
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_LINALG_NP_EIG_INL_H_
