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
 * Copyright (c) 2015 by Contributors
 * \file dropout-inl.h
 * \brief
 * \author Bing Xu, Da Zheng, Hang Zhang
*/

#ifndef MXNET_OPERATOR_NN_DROPOUT_INL_H_
#define MXNET_OPERATOR_NN_DROPOUT_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include "../mxnet_op.h"
#include "../mshadow_op.h"
#include "../random/sampler.h"
#include "../tensor/elemwise_binary_broadcast_op.h"

#if defined(USE_MKL) && defined(_OPENMP)
#include <omp.h>

#include <mkl_vml_functions.h>
#include <mkl_vsl.h>
#endif  // USE_MKL && _OPENMP

namespace dropout {
enum DropoutOpInputs {kData};
enum DropoutOpOutputs {kOut, kMask};
enum DropoutOpForwardResource {kRandom};
enum DropoutOpMode {kTraining, kAlways};
}  // namespace dropout

namespace mxnet {
namespace op {

const int MAX_DIM = 5;

struct DropoutParam : public dmlc::Parameter<DropoutParam> {
  float p;
  int mode;
  TShape axes;
  DMLC_DECLARE_PARAMETER(DropoutParam) {
    DMLC_DECLARE_FIELD(p).set_default(0.5)
    .set_range(0, 1)
    .describe("Fraction of the input that gets dropped out during training time.");
    DMLC_DECLARE_FIELD(mode)
    .add_enum("training", dropout::kTraining)
    .add_enum("always", dropout::kAlways)
    .set_default(dropout::kTraining)
    .describe("Whether to only turn on dropout during training or to also turn on for inference.");
    DMLC_DECLARE_FIELD(axes).set_default(TShape())
    .describe("Axes for variational dropout kernel.");
  }
};  // struct DropoutParam

template<typename xpu, typename DType>
class DropoutOp {
#if defined(USE_MKL) && defined(_OPENMP)
  static void BernoulliGenerate(common::random::RandGenerator<cpu, DType> gen,
                                int n, double p, int* r) {
    typename RandGenerator<xpu, DType>::Impl genImpl(&gen, 1);
    const int seed = 17 + abs(genImpl.rand() % 4096);
    CHECK_GE(seed, 0);
    const int nthr = engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
#pragma omp parallel num_threads(nthr)
    {
      const int ithr = omp_get_thread_num();
      const int avg_amount = (n + nthr - 1) / nthr;
      const int my_offset = ithr * avg_amount;
      const int my_amount = std::min(my_offset + avg_amount, n) - my_offset;
      if (my_amount > 0) {
        VSLStreamStatePtr stream;
        vslNewStream(&stream, VSL_BRNG_MCG31, seed);
        vslSkipAheadStream(stream, my_offset);
        viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF, stream, my_amount, r + my_offset, p);
        vslDeleteStream(&stream);
      }
    }
  }

  // MKL forward pass
  static bool MSHADOW_CINLINE MKLForward(mshadow::Stream<cpu> *s, RandGenerator<cpu, DType> *pgen,
                                         const double pkeep,
                                         const std::vector<TBlob> &in_data,
                                         const std::vector<TBlob> &out_data) {
    // BernoulliGenerate expects an array int, so for types smaller than int, the mask buffer
    // will be too small, so we can;t use MKL in those cases
    if (sizeof(DType) >= sizeof(int)) {
      Tensor<xpu, 2, DType> mask = out_data[dropout::kMask].FlatTo2D<xpu, DType>(s);
      Tensor<xpu, 2, DType> data = in_data[dropout::kData].FlatTo2D<xpu, DType>(s);
      Tensor<xpu, 2, DType> out = out_data[dropout::kOut].FlatTo2D<xpu, DType>(s);
      DType *outptr = out.dptr_;
      DType *dataptr = data.dptr_;
      auto maskptr = reinterpret_cast<int *>(mask.dptr_);
      int count = mask.shape_[0] * mask.shape_[1];
      BernoulliGenerate(*pgen, count, pkeep, maskptr);
      const float pk_1 = 1.0f / pkeep;
#pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
      for (int i = 0; i < count; ++i) {
        outptr[i] = dataptr[i] * maskptr[i] * pk_1;
      }
      return true;
    }
    return false;
  }

  // MKL backward pass
  static bool MSHADOW_CINLINE MKLBackward(mshadow::Stream<cpu> *s, const double pkeep,
                                          const std::vector<TBlob> &in_grad,
                                          const std::vector<TBlob> &out_data,
                                          const std::vector<TBlob> &out_grad) {
    if (sizeof(DType) >= sizeof(int)) {
      Tensor<xpu, 2, DType> grad = out_grad[dropout::kOut].FlatTo2D<xpu, DType>(s);
      Tensor<xpu, 2, DType> mask = out_data[dropout::kMask].FlatTo2D<xpu, DType>(s);
      Tensor<xpu, 2, DType> gdata = in_grad[dropout::kData].FlatTo2D<xpu, DType>(s);
      DType *ingradptr = gdata.dptr_;
      const DType *outgradptr = grad.dptr_;
      auto maskptr = reinterpret_cast<int *>(mask.dptr_);
      int count = mask.shape_[0] * mask.shape_[1];
      const float pk_1 = 1.0f / pkeep;
#pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
      for (int i = 0; i < count; ++i) {
        ingradptr[i] = outgradptr[i] * maskptr[i] * pk_1;
      }
      return true;
    }
    return false;
  }

#ifdef __CUDACC__
  // GPU never uses MKL
  static bool MSHADOW_CINLINE MKLForward(mshadow::Stream<gpu> *s, RandGenerator<gpu, DType> *pgen,
                                         const double pkeep,
                                         const std::vector<TBlob> &in_data,
                                         const std::vector<TBlob> &out_data) {
    return false;
  }
  static bool MSHADOW_CINLINE MKLBackward(mshadow::Stream<gpu> *s, const double pkeep,
                                          const std::vector<TBlob> &in_grad,
                                          const std::vector<TBlob> &out_data,
                                          const std::vector<TBlob> &out_grad) {
    return false;
  }
#endif  // __CUDACC__

#else  // #if defined(USE_MKL) && defined(_OPENMP)
  static bool MSHADOW_CINLINE MKLForward(mshadow::Stream<xpu> *s, RandGenerator<xpu, DType> *pgen,
                                const double pkeep,
                                const std::vector<TBlob> &in_data,
                                const std::vector<TBlob> &out_data) {
    return false;
  }
  static bool MSHADOW_CINLINE MKLBackward(mshadow::Stream<xpu> *s, const double pkeep,
                                          const std::vector<TBlob> &in_grad,
                                          const std::vector<TBlob> &out_data,
                                          const std::vector<TBlob> &out_grad) {
    return false;
  }
#endif  // #if defined(USE_MKL) && defined(_OPENMP)

 public:
  /*!
   * \brief Dropout kernel, compute dropout tensor
   */
  struct DropoutKernel {
    /*!
     * \brief Dropout kernel function
     * \param id Thread number (0-based representing count)
     * \param gen Random number generator
     * \param N Total number of items in the output
     * \param step Step between items, related to parallelism
     * \param dropout_out Output dropout values
     * \param mask_out  Output mask (is multiplied to create dropout output, may be 0)
     * \param input_data Input data to perform the dropout on
     * \param pkeep Dropout rate (keep when the generated random number is less than this value)
     */
    MSHADOW_XINLINE static void Map(int id,
                                    RandGenerator<xpu, DType> gen,
                                    const int N,
                                    const int step,
                                    DType *dropout_out,
                                    DType *mask_out,
                                    const DType *input_data,
                                    const real_t pkeep) {
      RNG_KERNEL_LOOP(xpu, DType, id, gen, N, step, {
        const real_t rand_num = static_cast<real_t>(genImpl.uniform());
        mask_out[i] = mshadow_op::threshold_eq::Map<real_t>(rand_num, pkeep) * (1.0f / pkeep);
        dropout_out[i] = input_data[i] * mask_out[i];
      });
    }
  };
  struct BernoulliKernel {
    /*! \brief Bernoulli kernel for generating mask */
    MSHADOW_XINLINE static void Map(int id,
                                    RandGenerator<xpu, DType> gen,
                                    const int N,
                                    const int step,
                                    DType *mask_out,
                                    const real_t pkeep) {
      RNG_KERNEL_LOOP(xpu, DType, id, gen, N, step, {
        const real_t rand_num = static_cast<real_t>(genImpl.uniform());
        mask_out[i] = mshadow_op::threshold::Map<real_t>(rand_num, pkeep) * (1.0f / pkeep);
      });
    }
  };

  void Init(const DropoutParam &param) {
    this->pkeep_ = 1.0f - param.p;
    this->mode_ = static_cast<dropout::DropoutOpMode>(param.mode);
    this->axes_ = param.axes;
  }

  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data) {
    if (req[dropout::kOut] != kNullOp) {
      CHECK_EQ(in_data.size(), 1U);
      if (ctx.is_train) {
        CHECK_EQ(out_data.size(), 2U);
      }
      Stream<xpu> *s = ctx.get_stream<xpu>();
      const TBlob &out = out_data[dropout::kOut];
      if (ctx.is_train || this->mode_ == dropout::kAlways) {
        RandGenerator<xpu, DType> *pgen = ctx.requested[0].get_parallel_random<xpu, DType>();
        CHECK_NOTNULL(pgen);
        if (this->axes_.ndim() != 0 || !MKLForward(s, pgen, this->pkeep_, in_data, out_data)) {
          const TBlob &mask = out_data[dropout::kMask];
          CHECK(req[dropout::kOut] != kAddTo);
          if (this->axes_.ndim() == 0) {
            // standard case for dropout
            LaunchRNG<DropoutKernel, xpu>(s, pgen, out.Size(),
                                        out.dptr<DType>(),
                                        mask.dptr<DType>(),
                                        in_data[dropout::kData].dptr<DType>(),
                                        this->pkeep_);
            return;
          }

          // initialize the mask
          LaunchRNG<BernoulliKernel, xpu>(s, pgen, mask.Size(),
                                          mask.dptr<DType>(),
                                          this->pkeep_);
          // broadcast mul
          TShape new_lshape, new_rshape, new_oshape;
          int ndim = BinaryBroadcastShapeCompact(in_data[dropout::kData].shape_,
                                                 mask.shape_, out.shape_,
                                                 &new_lshape, &new_rshape, &new_oshape);
          if (!ndim) {
            MXNET_ASSIGN_REQ_SWITCH(req[dropout::kOut], Req, {
              mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::mul, Req>, xpu>::Launch(
                s, out.Size(), out.dptr<DType>(), in_data[dropout::kData].dptr<DType>(),
                mask.dptr<DType>());
            });
          } else {
            BROADCAST_NDIM_SWITCH(ndim, NDim, {
              mshadow::Shape<NDim> oshape = new_oshape.get<NDim>();
              mshadow::Shape<NDim> lstride = mxnet_op::calc_stride(new_lshape.get<NDim>());
              mshadow::Shape<NDim> rstride = mxnet_op::calc_stride(new_rshape.get<NDim>());
              mxnet_op::Kernel<mxnet_op::binary_broadcast_kernel<NDim, DType,
                               mshadow_op::mul>, xpu>::
              template LaunchEx(s, new_oshape.Size(), req[dropout::kOut],
              lstride, rstride, oshape,
              in_data[dropout::kData].dptr<DType>(),
              mask.dptr<DType>(), out.dptr<DType>());
            });
          }
        }
      } else {
        const TBlob& data = in_data[dropout::kData];
        if (req[dropout::kOut] == kWriteTo) {
          mxnet_op::copy(s, out, data);
        } else {
          MXNET_ASSIGN_REQ_SWITCH(req[dropout::kOut], Req, {
            mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::identity, Req>, xpu>::Launch(
              s, out.Size(), out.dptr<DType>(), data.dptr<DType>());
          });
        }
      }
    }
  }

  void Backward(const OpContext &ctx,
                const std::vector<TBlob> &out_grad,
                const std::vector<TBlob> &out_data,
                const std::vector<OpReqType> &req,
                const std::vector<TBlob> &in_grad) {
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    if (ctx.is_train || mode_ == dropout::kAlways) {
      if (this->axes_.ndim() != 0 || !MKLBackward(s, this->pkeep_, in_grad, out_data, out_grad)) {
        const TBlob &gdata = in_grad[dropout::kData];
        const TBlob &grad = out_grad[dropout::kOut];
        const TBlob &mask = out_data[dropout::kMask];
        if (this->axes_.ndim() == 0) {
          // standard case for dropout
          CHECK_EQ(grad.Size(), mask.Size());
          MXNET_ASSIGN_REQ_SWITCH(req[dropout::kData], Req, {
            mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::mul, Req>, xpu>::Launch(
              s, gdata.Size(), gdata.dptr<DType>(), grad.dptr<DType>(), mask.dptr<DType>());
          });
          return;
        }
        // broardcast mul
        TShape new_lshape, new_rshape, new_oshape;
        int ndim = BinaryBroadcastShapeCompact(grad.shape_,
                                               mask.shape_, gdata.shape_,
                                               &new_lshape, &new_rshape, &new_oshape);
        if (!ndim) {
          MXNET_ASSIGN_REQ_SWITCH(req[dropout::kData], Req, {
            mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::mul, Req>, xpu>::Launch(
              s, gdata.Size(), gdata.dptr<DType>(), grad.dptr<DType>(), mask.dptr<DType>());
          });
        } else {
          BROADCAST_NDIM_SWITCH(ndim, NDim, {
            mshadow::Shape<NDim> oshape = new_oshape.get<NDim>();
            mshadow::Shape<NDim> lstride = mxnet_op::calc_stride(new_lshape.get<NDim>());
            mshadow::Shape<NDim> rstride = mxnet_op::calc_stride(new_rshape.get<NDim>());
            mxnet_op::Kernel<mxnet_op::binary_broadcast_kernel<NDim, DType,
                             mshadow_op::mul>, xpu>::
            template LaunchEx(s, new_oshape.Size(), req[0], lstride, rstride, oshape,
            grad.dptr<DType>(), mask.dptr<DType>(), gdata.dptr<DType>());
          });
        }
      }
    } else {
      const TBlob& gdata = in_grad[dropout::kData];
      const TBlob& grad = out_grad[dropout::kOut];
      if (req[dropout::kData] == kWriteTo) {
        mxnet_op::copy(s, gdata, grad);
      } else {
        MXNET_ASSIGN_REQ_SWITCH(req[dropout::kData], Req, {
          mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::identity, Req>, xpu>::Launch(
            s, gdata.Size(), gdata.dptr<DType>(), grad.dptr<DType>());
        });
      }
    }
  }

 private:
  /*! \brief Dropout rate (keep when the generated random number is less than this value) */
  real_t pkeep_;
  /*! \brief Dropout mode */
  dropout::DropoutOpMode mode_;
  TShape axes_;
};  // class DropoutOp

template<typename xpu>
void DropoutCompute(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  const DropoutParam& param = nnvm::get<DropoutParam>(attrs.parsed);
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    DropoutOp<xpu, DType> op;
    op.Init(param);
    op.Forward(ctx, inputs, req, outputs);
  });
}

template<typename xpu>
void DropoutGradCompute(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  const DropoutParam& param = nnvm::get<DropoutParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1);
  CHECK_EQ(req.size(), 1);
  std::vector<TBlob> out_grads(2);
  std::vector<TBlob> out_data(2);
  out_grads[dropout::kOut] = inputs[0];
  out_data[dropout::kMask] = inputs[1];

  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    DropoutOp<xpu, DType> op;
    op.Init(param);
    op.Backward(ctx, out_grads, out_data, req, outputs);
  });
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NN_DROPOUT_INL_H_
