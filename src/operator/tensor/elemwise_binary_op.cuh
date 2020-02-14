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
 *  Copyright (c) 2020 by Contributors
 * \file elemwise_binary_op.cuh
 * \brief GPU helpers for elementwise operators
 */

#ifndef MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_OP_CUH_
#define MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_OP_CUH_

#include <cuda_runtime.h>
#include "../operator_common.h"
#include "../../common/cuda_vectorization.cuh"

#include <vector>

#if MXNET_USE_CUDA

namespace mxnet {
namespace op {

namespace binary {

using common::cuda::VectorizedKernelLauncher;
using common::cuda::VectorizedLoader;
using common::cuda::VectorizedStorer;

template <typename DType, int NumInputs, int NumOutputs>
struct VectorizedBinaryKernelParams {
  const DType* inputs[NumInputs];
  DType* outputs[NumOutputs];
};

template <bool aligned, typename DType, typename LType, typename OP, int req>
__global__ void VectorizedBinaryKernelFwd(const VectorizedBinaryKernelParams<DType, 2, 1> params,
                                          const index_t N) {
  VectorizedLoader<DType, LType, aligned> loader0(params.inputs[0], N);
  VectorizedLoader<DType, LType, aligned> loader1(params.inputs[1], N);
  VectorizedStorer<DType, LType, aligned> storer(params.outputs[0], N);

  const index_t M = loader0.num_aligned_elements();

  for (index_t tid = blockIdx.x * blockDim.x + threadIdx.x;
       tid < M;
       tid += gridDim.x * blockDim.x) {
    loader0.load(tid, N);
    loader1.load(tid, N);
    if (req == kAddTo) {
      storer.load(tid, N);
    }
#pragma unroll
    for (int i = 0; i < loader0.nvec(); ++i) {
      DType temp = OP::Map(loader0.separate()[i],
                           loader1.separate()[i]);

      if (req == kAddTo) {
        storer.separate()[i] += temp;
      } else {
        storer.separate()[i] = temp;
      }
    }
    storer.store(tid, N);
  }
}

template <bool aligned, typename DType, typename LType,
          typename LOP, typename ROP, int lreq, int rreq>
__global__ void VectorizedBinaryKernelBwdUseNone(
    const VectorizedBinaryKernelParams<DType, 1, 2> params,
    const index_t N) {
  VectorizedLoader<DType, LType, aligned> loader(params.inputs[0], N);
  VectorizedStorer<DType, LType, aligned> lstorer(params.outputs[0], N);
  VectorizedStorer<DType, LType, aligned> rstorer(params.outputs[1], N);

  const index_t M = loader.num_aligned_elements();

  for (index_t tid = blockIdx.x * blockDim.x + threadIdx.x;
       tid < M;
       tid += gridDim.x * blockDim.x) {
    loader.load(tid, N);
    if (lreq == kAddTo) {
      lstorer.load(tid, N);
    }
    if (rreq == kAddTo) {
      rstorer.load(tid, N);
    }
#pragma unroll
    for (int i = 0; i < loader.nvec(); ++i) {
      DType inp = loader.separate()[i];
      if (!((std::is_same<LOP, mshadow_op::identity>::value && lreq == kWriteInplace) ||
            lreq == kNullOp)) {
        DType ltemp = LOP::Map(inp);
        if (lreq == kAddTo) {
          lstorer.separate()[i] += ltemp;
        } else {
          lstorer.separate()[i] = ltemp;
        }
        lstorer.store(tid, N);
      }
      if (!((std::is_same<ROP, mshadow_op::identity>::value && rreq == kWriteInplace) ||
            rreq == kNullOp)) {
        DType rtemp = ROP::Map(inp);

        if (rreq == kAddTo) {
          rstorer.separate()[i] += rtemp;
        } else {
          rstorer.separate()[i] = rtemp;
        }
        rstorer.store(tid, N);
      }
    }
  }
}

template <bool aligned, typename DType, typename LType,
          typename LOP, typename ROP, int lreq, int rreq>
__global__ void VectorizedBinaryKernelBwdUseIn(
    const VectorizedBinaryKernelParams<DType, 3, 2> params,
    const index_t N) {
  VectorizedLoader<DType, LType, aligned> ograd_loader(params.inputs[0], N);
  VectorizedLoader<DType, LType, aligned> linput_loader(params.inputs[1], N);
  VectorizedLoader<DType, LType, aligned> rinput_loader(params.inputs[2], N);
  VectorizedStorer<DType, LType, aligned> lstorer(params.outputs[0], N);
  VectorizedStorer<DType, LType, aligned> rstorer(params.outputs[1], N);

  const index_t M = ograd_loader.num_aligned_elements();

  for (index_t tid = blockIdx.x * blockDim.x + threadIdx.x;
       tid < M;
       tid += gridDim.x * blockDim.x) {
    ograd_loader.load(tid, N);
    linput_loader.load(tid, N);
    rinput_loader.load(tid, N);
    if (lreq == kAddTo) {
      lstorer.load(tid, N);
    }
    if (rreq == kAddTo) {
      rstorer.load(tid, N);
    }
#pragma unroll
    for (int i = 0; i < ograd_loader.nvec(); ++i) {
      DType ograd = ograd_loader.separate()[i];
      DType linput = linput_loader.separate()[i];
      DType rinput = rinput_loader.separate()[i];
      if (!(lreq == kNullOp)) {
        DType ltemp = ograd * LOP::Map(linput, rinput);
        if (lreq == kAddTo) {
          lstorer.separate()[i] += ltemp;
        } else {
          lstorer.separate()[i] = ltemp;
        }
        lstorer.store(tid, N);
      }
      if (!(rreq == kNullOp)) {
        DType rtemp = ograd * ROP::Map(linput, rinput);

        if (rreq == kAddTo) {
          rstorer.separate()[i] += rtemp;
        } else {
          rstorer.separate()[i] = rtemp;
        }
        rstorer.store(tid, N);
      }
    }
  }
}

template <typename DType, typename OP, int req>
class VectorizedBinaryFwd {
 public:
  using ParamType = VectorizedBinaryKernelParams<DType, 2, 1>;

  template <bool aligned, typename LType>
  static void Launch(const index_t blocks, const index_t threads,
                     cudaStream_t stream,
                     const ParamType params, const index_t lead_dim,
                     const index_t /* other_dim */) {
    VectorizedBinaryKernelFwd<aligned, DType, LType, OP, req>
      <<<blocks, threads, 0, stream>>>(params, lead_dim);
  }
};

template <typename DType, typename LOP, typename ROP, int lreq, int rreq>
class VectorizedBinaryBwdUseNone {
 public:
  using ParamType = VectorizedBinaryKernelParams<DType, 1, 2>;

  template <bool aligned, typename LType>
  static void Launch(const index_t blocks, const index_t threads,
                     cudaStream_t stream,
                     const ParamType params, const index_t lead_dim,
                     const index_t /* other_dim */) {
    VectorizedBinaryKernelBwdUseNone<aligned, DType, LType, LOP, ROP, lreq, rreq>
      <<<blocks, threads, 0, stream>>>(params, lead_dim);
  }
};

template <typename DType, typename LOP, typename ROP, int lreq, int rreq>
class VectorizedBinaryBwdUseIn {
 public:
  using ParamType = VectorizedBinaryKernelParams<DType, 3, 2>;

  template <bool aligned, typename LType>
  static void Launch(const index_t blocks, const index_t threads,
                     cudaStream_t stream,
                     const ParamType params, const index_t lead_dim,
                     const index_t /* other_dim */) {
    VectorizedBinaryKernelBwdUseIn<aligned, DType, LType, LOP, ROP, lreq, rreq>
      <<<blocks, threads, 0, stream>>>(params, lead_dim);
  }
};

}  // namespace binary

template<typename OP>
void ElemwiseBinaryOp::Compute_(const nnvm::NodeAttrs &attrs,
                                mshadow::Stream<gpu> *s,
                                const std::vector<TBlob> &inputs,
                                const std::vector<OpReqType> &req,
                                const std::vector<TBlob> &outputs) {
  using namespace binary;
  if (req[0] == kNullOp) return;
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      using LType = uint4;
      using Kernel = VectorizedBinaryFwd<DType, OP, Req>;

      const index_t size = outputs[0].Size();
      typename Kernel::ParamType params;
      params.inputs[0] = inputs[0].dptr<DType>();
      params.inputs[1] = inputs[1].dptr<DType>();
      params.outputs[0] = outputs[0].dptr<DType>();

      VectorizedKernelLauncher<DType, LType, Kernel>(size, 1, s, params);
    });
  });
}

template<typename LOP, typename ROP>
void ElemwiseBinaryOp::BackwardUseNone_(const nnvm::NodeAttrs &attrs,
                                        mshadow::Stream<gpu>* s,
                                        const std::vector<TBlob> &inputs,
                                        const std::vector<OpReqType> &req,
                                        const std::vector<TBlob> &outputs) {
  using namespace binary;
  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);

  MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    const index_t size = inputs[0].Size();
    if (req[0] != kNullOp || req[1] != kNullOp) {
      MXNET_REQ_TYPE_SWITCH(req[0], lreq, {
        MXNET_REQ_TYPE_SWITCH(req[1], rreq, {
          using LType = uint4;
          using Kernel = VectorizedBinaryBwdUseNone<DType, LOP, ROP, lreq, rreq>;

          typename Kernel::ParamType params;
          params.inputs[0] = inputs[0].dptr<DType>();
          params.outputs[0] = outputs[0].dptr<DType>();
          params.outputs[1] = outputs[1].dptr<DType>();

          VectorizedKernelLauncher<DType, LType, Kernel>(size, 1, s, params);
        });
      });
    }
  });
}

template<typename LOP, typename ROP>
void ElemwiseBinaryOp::BackwardUseIn_(const nnvm::NodeAttrs &attrs,
                                      mshadow::Stream<gpu>* s,
                                      const std::vector<TBlob> &inputs,
                                      const std::vector<OpReqType> &req,
                                      const std::vector<TBlob> &outputs) {
  using namespace binary;
  if (req[0] != kNullOp || req[1] != kNullOp) {
    MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
      MXNET_REQ_TYPE_SWITCH(req[0], lreq, {
        MXNET_REQ_TYPE_SWITCH(req[1], rreq, {
          const index_t size = inputs[0].Size();
          // Using 64 bit loads to reduce register pressure
          using LType = uint2;
          using Kernel = VectorizedBinaryBwdUseIn<DType, LOP, ROP, lreq, rreq>;

          typename Kernel::ParamType params;
          params.inputs[0] = inputs[0].dptr<DType>();
          params.inputs[1] = inputs[1].dptr<DType>();
          params.inputs[2] = inputs[2].dptr<DType>();
          params.outputs[0] = outputs[0].dptr<DType>();
          params.outputs[1] = outputs[1].dptr<DType>();

          VectorizedKernelLauncher<DType, LType, Kernel>(size, 1, s, params);
        });
      });
    });
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_CUDA
#endif  // MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_OP_CUH_
