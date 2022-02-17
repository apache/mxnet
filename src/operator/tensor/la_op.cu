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
 * \file la_op.cu
 * \brief GPU implementation of Operators for advanced linear algebra.
 */
#include "./la_op.h"
#include "./la_op-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_linalg_gemm)
    .set_attr<FCompute>("FCompute<gpu>", LaOpGemmForward<gpu, 2, 2, 3, 1, gemm>);

NNVM_REGISTER_OP(_backward_linalg_gemm)
    .set_attr<FCompute>("FCompute<gpu>", LaOpGemmBackward<gpu, 2, 2, 4, 3, gemm_backward>);

NNVM_REGISTER_OP(_linalg_gemm2)
    .set_attr<FCompute>("FCompute<gpu>", LaOpGemmForward<gpu, 2, 2, 2, 1, gemm2>);

NNVM_REGISTER_OP(_backward_linalg_gemm2)
    .set_attr<FCompute>("FCompute<gpu>", LaOpGemmBackward<gpu, 2, 2, 3, 2, gemm2_backward>);

NNVM_REGISTER_OP(_linalg_trmm)
    .set_attr<FCompute>("FCompute<gpu>", LaOpForward<gpu, 2, 2, 2, 1, trmm>);

NNVM_REGISTER_OP(_backward_linalg_trmm)
    .set_attr<FCompute>("FCompute<gpu>", LaOpBackward<gpu, 2, 2, 3, 2, trmm_backward>);

NNVM_REGISTER_OP(_linalg_trsm)
    .set_attr<FCompute>("FCompute<gpu>", LaOpForward<gpu, 2, 2, 2, 1, trsm>);

NNVM_REGISTER_OP(_backward_linalg_trsm)
    .set_attr<FCompute>("FCompute<gpu>", LaOpBackward<gpu, 2, 2, 4, 2, trsm_backward>);

NNVM_REGISTER_OP(_linalg_syrk)
    .set_attr<FCompute>("FCompute<gpu>", LaOpForward<gpu, 2, 2, 1, 1, syrk>);

NNVM_REGISTER_OP(_backward_linalg_syrk)
    .set_attr<FCompute>("FCompute<gpu>", LaOpBackward<gpu, 2, 2, 2, 1, syrk_backward>);

NNVM_REGISTER_OP(_linalg_sumlogdiag)
    .set_attr<FCompute>("FCompute<gpu>", LaOpForward<gpu, 2, 0, 1, 1, sumlogdiag>);

NNVM_REGISTER_OP(_backward_linalg_sumlogdiag)
    .set_attr<FCompute>("FCompute<gpu>", LaOpBackward<gpu, 2, 2, 2, 1, sumlogdiag_backward>);

NNVM_REGISTER_OP(_linalg_extractdiag)
    .set_attr<FCompute>("FCompute<gpu>", LaOpForward<gpu, 2, 1, 1, 1, copydiag>);

NNVM_REGISTER_OP(_backward_linalg_extractdiag)
    .set_attr<FCompute>("FCompute<gpu>", LaOpBackward<gpu, 1, 2, 1, 1, copydiag>);

NNVM_REGISTER_OP(_linalg_makediag)
    .set_attr<FCompute>("FCompute<gpu>", LaOpForward<gpu, 1, 2, 1, 1, copydiag>);

NNVM_REGISTER_OP(_backward_linalg_makediag)
    .set_attr<FCompute>("FCompute<gpu>", LaOpBackward<gpu, 2, 1, 1, 1, copydiag>);

NNVM_REGISTER_OP(_linalg_extracttrian)
    .set_attr<FCompute>("FCompute<gpu>", LaOpForward<gpu, 2, 1, 1, 1, copytrian>);

NNVM_REGISTER_OP(_backward_linalg_extracttrian)
    .set_attr<FCompute>("FCompute<gpu>", LaOpBackward<gpu, 1, 2, 1, 1, copytrian>);

NNVM_REGISTER_OP(_linalg_maketrian)
    .set_attr<FCompute>("FCompute<gpu>", LaOpForward<gpu, 1, 2, 1, 1, copytrian>);

NNVM_REGISTER_OP(_backward_linalg_maketrian)
    .set_attr<FCompute>("FCompute<gpu>", LaOpBackward<gpu, 2, 1, 1, 1, copytrian>);

NNVM_REGISTER_OP(_linalg_potri)
    .set_attr<FIsCUDAGraphsCompatible>("FIsCUDAGraphsCompatible",
                                       [](const NodeAttrs&, const bool) { return false; })
    .set_attr<FCompute>("FCompute<gpu>", LaOpForward<gpu, 2, 2, 1, 1, potri>);

NNVM_REGISTER_OP(_backward_linalg_potri)
    .set_attr<FCompute>("FCompute<gpu>", LaOpBackward<gpu, 2, 2, 3, 1, potri_backward>);

NNVM_REGISTER_OP(_linalg_inverse)
    .set_attr<FCompute>("FCompute<gpu>", LaOpForward<gpu, 2, 2, 1, 1, inverse>);

NNVM_REGISTER_OP(_backward_linalg_inverse)
    .set_attr<FCompute>("FCompute<gpu>", LaOpBackward<gpu, 2, 2, 2, 1, inverse_backward>);

NNVM_REGISTER_OP(_linalg_det)
    // Incompatibility comes from allocs made in linalg_batch_getrf(), called by det::op()
    // see https://github.com/apache/incubator-mxnet/issues/19353
    .set_attr<FIsCUDAGraphsCompatible>("FIsCUDAGraphsCompatible",
                                       [](const NodeAttrs&, const bool) { return false; })
    .set_attr<FCompute>("FCompute<gpu>", LaOpDetForward<gpu, 1, det>);

NNVM_REGISTER_OP(_backward_linalg_det)
    // Incompatibility comes from allocs made in linalg_batch_getri(),
    // called by linalg_batch_det_backward_helper, called by det_backward::op()
    .set_attr<FIsCUDAGraphsCompatible>("FIsCUDAGraphsCompatible",
                                       [](const NodeAttrs&, const bool) { return false; })
    .set_attr<FCompute>("FCompute<gpu>", LaOpDetBackward<gpu, 1, det_backward>);

NNVM_REGISTER_OP(_linalg_slogdet)
    // Incompatibility comes from allocs made in linalg_batch_getrf(),
    // called by slogdet::op().
    .set_attr<FIsCUDAGraphsCompatible>("FIsCUDAGraphsCompatible",
                                       [](const NodeAttrs&, const bool) { return false; })
    .set_attr<FCompute>("FCompute<gpu>", LaOpDetForward<gpu, 2, slogdet>);

NNVM_REGISTER_OP(_backward_linalg_slogdet)
    // Incompatibility comes from allocs made in linalg_batch_getri(),
    // called by linalg_batch_det_backward_helper, called by slogdet_backward::op()
    .set_attr<FIsCUDAGraphsCompatible>("FIsCUDAGraphsCompatible",
                                       [](const NodeAttrs&, const bool) { return false; })
    .set_attr<FCompute>("FCompute<gpu>", LaOpDetBackward<gpu, 2, slogdet_backward>);

#if MXNET_USE_CUSOLVER == 1

NNVM_REGISTER_OP(_linalg_potrf)
    .set_attr<FIsCUDAGraphsCompatible>("FIsCUDAGraphsCompatible",
                                       [](const NodeAttrs&, const bool) { return false; })
    .set_attr<FCompute>("FCompute<gpu>", LaOpForward<gpu, 2, 2, 1, 1, potrf>);

NNVM_REGISTER_OP(_backward_linalg_potrf)
    .set_attr<FCompute>("FCompute<gpu>", LaOpBackward<gpu, 2, 2, 2, 1, potrf_backward>);

NNVM_REGISTER_OP(_linalg_gelqf)
    .set_attr<FIsCUDAGraphsCompatible>("FIsCUDAGraphsCompatible",
                                       [](const NodeAttrs&, const bool) { return false; })
    .set_attr<FCompute>("FCompute<gpu>", LaOpForward<gpu, 2, 2, 1, 2, gelqf>);

NNVM_REGISTER_OP(_backward_linalg_gelqf)
    .set_attr<FCompute>("FCompute<gpu>", LaOpBackward<gpu, 2, 2, 4, 1, gelqf_backward>);

NNVM_REGISTER_OP(_linalg_syevd)
    .set_attr<FIsCUDAGraphsCompatible>("FIsCUDAGraphsCompatible",
                                       [](const NodeAttrs&, const bool) { return false; })
    .set_attr<FCompute>("FCompute<gpu>", LaOpForwSyevd<gpu, syevd>);

NNVM_REGISTER_OP(_backward_linalg_syevd)
    .set_attr<FCompute>("FCompute<gpu>", LaOpBackwSyevd<gpu, syevd_backward>);

#endif

}  // namespace op
}  // namespace mxnet
