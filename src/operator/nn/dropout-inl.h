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

#if (MSHADOW_USE_MKL == 1) && defined(_OPENMP) && !defined(__CUDACC__)
#define MXNET_USE_MKL_DROPOUT 1
#endif

#if MXNET_USE_MKL_DROPOUT
#include <omp.h>

#include <mkl_vml_functions.h>
#include <mkl_vsl.h>
#endif  // MXNET_USE_MKL_DROPOUT

#define MXNET_USE_CUDNN_DROPOUT MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 7

namespace dropout {
enum DropoutOpInputs { kData };
enum DropoutOpOutputs { kOut, kMask };
enum DropoutOpForwardResource { kRandom };
enum DropoutOpMode { kTraining, kAlways };
}  // namespace dropout

namespace mxnet {
namespace op {

const int MAX_DIM = 5;

struct DropoutParam : public dmlc::Parameter<DropoutParam> {
  float p;
  int mode;
  mxnet::TShape axes;
  dmlc::optional<bool> cudnn_off;
  DMLC_DECLARE_PARAMETER(DropoutParam) {
    DMLC_DECLARE_FIELD(p).set_default(0.5).set_range(0, 1).describe(
        "Fraction of the input that gets dropped out during training time.");
    DMLC_DECLARE_FIELD(mode)
        .add_enum("training", dropout::kTraining)
        .add_enum("always", dropout::kAlways)
        .set_default(dropout::kTraining)
        .describe(
            "Whether to only turn on dropout during training or to also turn on for inference.");
    DMLC_DECLARE_FIELD(axes)
        .set_default(mxnet::TShape(0, 0))
        .describe("Axes for variational dropout kernel.");
    DMLC_DECLARE_FIELD(cudnn_off)
        .set_default(dmlc::optional<bool>(false))
        .describe(
            "Whether to turn off cudnn in dropout operator. "
            "This option is ignored if axes is specified.");
  }
  std::string Mode2String(int mode) {
    switch (mode) {
      case dropout::kTraining:
        return "training";
      case dropout::kAlways:
        return "always";
      default:
        LOG(FATAL) << "Unknown mode enum " << mode;
    }
    LOG(FATAL) << "should not reach here ";
    return "";
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream p_s, mode_s, axes_s, cudnn_off_s;
    p_s << p;
    mode_s << mode;
    axes_s << axes;
    cudnn_off_s << cudnn_off;
    (*dict)["p"]         = p_s.str();
    (*dict)["mode"]      = Mode2String(mode);
    (*dict)["axes"]      = axes_s.str();
    (*dict)["cudnn_off"] = cudnn_off_s.str();
  }
};  // struct DropoutParam

template <typename xpu, typename DType>
class DropoutOp {
#if MXNET_USE_MKL_DROPOUT
  static void BernoulliGenerate(common::random::RandGenerator<cpu, DType> gen,
                                int n,
                                double p,
                                int* r) {
    typename RandGenerator<xpu, DType>::Impl genImpl(&gen, 1);
    const int seed = 17 + abs(genImpl.rand() % 4096);
    CHECK_GE(seed, 0);
    const int nthr = engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
#pragma omp parallel num_threads(nthr)
    {
      const int ithr       = omp_get_thread_num();
      const int avg_amount = (n + nthr - 1) / nthr;
      const int my_offset  = ithr * avg_amount;
      const int my_amount  = std::min(my_offset + avg_amount, n) - my_offset;
      if (my_amount > 0) {
        VSLStreamStatePtr stream;
        vslNewStream(&stream, VSL_BRNG_MCG31, seed);
        vslSkipAheadStream(stream, my_offset);
        viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF, stream, my_amount, r + my_offset, p);
        vslDeleteStream(&stream);
      }
    }
  }
  static inline bool MKLAvailable() {
    // BernoulliGenerate expects an array int, so for types smaller than int, the mask buffer
    // will be too small, so we can;t use MKL in those cases
    return sizeof(DType) >= sizeof(int);
  }

  // MKL forward pass
  inline void MKLForward(const OpContext& ctx,
                         const std::vector<TBlob>& in_data,
                         const std::vector<TBlob>& out_data) {
    Stream<xpu>* s                  = ctx.get_stream<xpu>();
    RandGenerator<xpu, DType>* pgen = ctx.requested[0].get_parallel_random<xpu, DType>();
    CHECK_NOTNULL(pgen);
    Tensor<xpu, 2, DType> mask = out_data[dropout::kMask].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> data = in_data[dropout::kData].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out  = out_data[dropout::kOut].FlatTo2D<xpu, DType>(s);
    DType* outptr              = out.dptr_;
    DType* dataptr             = data.dptr_;
    auto maskptr               = reinterpret_cast<int*>(mask.dptr_);
    int count                  = mask.shape_[0] * mask.shape_[1];
    if (sizeof(DType) > sizeof(int)) {
      // allocating new buffer to avoiding memory overlapping between `mask.dptr_` and `maskptr`
      Tensor<xpu, 1, int> temp = ctx.requested[1].get_space_typed<xpu, 1, int>(Shape1(count), s);
      maskptr                  = temp.dptr_;
    }
    BernoulliGenerate(*pgen, count, this->pkeep_, maskptr);
    const float pk_1 = 1.0f / this->pkeep_;
#pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
    for (int i = 0; i < count; ++i) {
      const DType maskVal = static_cast<DType>(maskptr[i]) * pk_1;
      outptr[i]           = dataptr[i] * maskVal;
      mask.dptr_[i]       = maskVal;
    }
  }

  // MKL backward pass
  inline void MKLBackward(const OpContext& ctx,
                          const std::vector<TBlob>& in_grad,
                          const std::vector<TBlob>& out_data,
                          const std::vector<TBlob>& out_grad) {
    Stream<xpu>* s              = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> grad  = out_grad[dropout::kOut].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> mask  = out_data[dropout::kMask].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> gdata = in_grad[dropout::kData].FlatTo2D<xpu, DType>(s);
    DType* ingradptr            = gdata.dptr_;
    const DType* outgradptr     = grad.dptr_;
    const DType* maskptr        = mask.dptr_;
    const int count             = mask.shape_[0] * mask.shape_[1];
#pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
    for (int i = 0; i < count; ++i) {
      ingradptr[i] = outgradptr[i] * maskptr[i];
    }
  }

#endif  // #if MXNET_USE_MKL_DROPOUT

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
    MSHADOW_XINLINE static void Map(index_t id,
                                    RandGenerator<xpu, DType> gen,
                                    const index_t N,
                                    const index_t step,
                                    DType* dropout_out,
                                    DType* mask_out,
                                    const DType* input_data,
                                    const real_t pkeep) {
      RNG_KERNEL_LOOP(xpu, DType, id, gen, N, step, {
        const real_t rand_num = static_cast<real_t>(genImpl.uniform());
        mask_out[i]    = mshadow_op::threshold_eq::Map<real_t>(rand_num, pkeep) * (1.0f / pkeep);
        dropout_out[i] = input_data[i] * mask_out[i];
      });
    }
  };
  struct BernoulliKernel {
    /*! \brief Bernoulli kernel for generating mask */
    MSHADOW_XINLINE static void Map(index_t id,
                                    RandGenerator<xpu, DType> gen,
                                    const index_t N,
                                    const index_t step,
                                    DType* mask_out,
                                    const real_t pkeep) {
      RNG_KERNEL_LOOP(xpu, DType, id, gen, N, step, {
        const real_t rand_num = static_cast<real_t>(genImpl.uniform());
        mask_out[i] = mshadow_op::threshold::Map<real_t>(rand_num, pkeep) * (1.0f / pkeep);
      });
    }
  };

  explicit DropoutOp(const DropoutParam& param, Context ctx) {
    this->pkeep_               = 1.0f - param.p;
    this->mode_                = static_cast<dropout::DropoutOpMode>(param.mode);
    this->axes_                = param.axes;
    this->dropout_passthrough_ = true;
#if MXNET_USE_CUDNN_DROPOUT
    this->cudnn_off_ = param.cudnn_off && param.cudnn_off.value();
    this->ctx_       = ctx;
    if (ctx.dev_type == kGPU && this->pkeep_ > 0 && !this->cudnn_off_) {
      dtype_ = mshadow::DataType<DType>::kCudnnFlag;
      CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc_));
      CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc_));
      CUDNN_CALL(cudnnCreateTensorDescriptor(&dx_desc_));
      CUDNN_CALL(cudnnCreateTensorDescriptor(&dy_desc_));
      CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc_));
    }
#endif  // MXNET_USE_CUDNN_DROPOUT
  }

  ~DropoutOp() {
#if MXNET_USE_CUDNN_DROPOUT
    if (this->ctx_.dev_type == kGPU && this->pkeep_ > 0 && !this->cudnn_off_) {
      CUDNN_CALL(cudnnDestroyTensorDescriptor(x_desc_));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(y_desc_));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(dx_desc_));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(dy_desc_));
      CUDNN_CALL(cudnnDestroyDropoutDescriptor(dropout_desc_));
    }
#endif  // MXNET_USE_CUDNN_DROPOUT
  }

#if MXNET_USE_CUDNN_DROPOUT && defined(__CUDACC__)
  inline bool CuDNNAvailable() {
    return this->pkeep_ > 0 && !this->cudnn_off_;
  }

  inline void CuDNNForward(const OpContext& ctx,
                           const TBlob& in,
                           const TBlob& mask,
                           const TBlob& out) {
    Stream<xpu>* s = ctx.get_stream<xpu>();

    // set dropout state.
    ctx.requested[0].get_cudnn_dropout_desc(&dropout_desc_, s, 1.0f - this->pkeep_);

    // describe input/output tensor
    int dim[4], stride[4];
    dim[0]    = 1;
    dim[1]    = 1;
    dim[2]    = 1;
    dim[3]    = out.Size();
    stride[0] = out.Size();
    stride[1] = out.Size();
    stride[2] = out.Size();
    stride[3] = 1;
    CUDNN_CALL(cudnnSetTensorNdDescriptor(x_desc_, dtype_, 4, dim, stride));
    CUDNN_CALL(cudnnSetTensorNdDescriptor(y_desc_, dtype_, 4, dim, stride));

    // perform dropout with cudnn
    CUDNN_CALL(cudnnDropoutGetReserveSpaceSize(x_desc_, &dropout_reserve_byte_));
    // cudnn uses bits to record the positions that are dropped, so reserve bytes is always
    // 1/8 of input size.
    CHECK_GE(mask.Size() * sizeof(DType), dropout_reserve_byte_)
        << "The size of the mask space is smaller than the required cudnn reserved space.";
    CUDNN_CALL(cudnnDropoutForward(s->dnn_handle_,
                                   dropout_desc_,
                                   x_desc_,
                                   in.dptr<DType>(),
                                   y_desc_,
                                   out.dptr<DType>(),
                                   mask.dptr<DType>(),
                                   dropout_reserve_byte_));
  }

  inline void CuDNNBackward(const OpContext& ctx,
                            const TBlob& out_grad,
                            const TBlob& mask,
                            const TBlob& in_grad) {
    Stream<xpu>* s = ctx.get_stream<xpu>();

    // describe input/output tensor
    int dim[4], stride[4];
    dim[0]    = 1;
    dim[1]    = 1;
    dim[2]    = 1;
    dim[3]    = in_grad.Size();
    stride[0] = in_grad.Size();
    stride[1] = in_grad.Size();
    stride[2] = in_grad.Size();
    stride[3] = 1;
    CUDNN_CALL(cudnnSetTensorNdDescriptor(dy_desc_, dtype_, 4, dim, stride));
    CUDNN_CALL(cudnnSetTensorNdDescriptor(dx_desc_, dtype_, 4, dim, stride));

    // perform dropout with cudnn
    CUDNN_CALL(cudnnDropoutBackward(s->dnn_handle_,
                                    dropout_desc_,
                                    dy_desc_,
                                    out_grad.dptr<DType>(),
                                    dx_desc_,
                                    in_grad.dptr<DType>(),
                                    mask.dptr<DType>(),
                                    dropout_reserve_byte_));
  }
#endif  // MXNET_USE_CUDNN_DROPOUT && defined(__CUDACC__)

  void Forward(const OpContext& ctx,
               const std::vector<TBlob>& in_data,
               const std::vector<OpReqType>& req,
               const std::vector<TBlob>& out_data) {
    this->dropout_passthrough_ = true;
    if (req[dropout::kOut] != kNullOp) {
      CHECK_EQ(in_data.size(), 1U);
      if (ctx.is_train) {
        CHECK_EQ(out_data.size(), 2U);
      }
      Stream<xpu>* s    = ctx.get_stream<xpu>();
      const TBlob& in   = in_data[dropout::kData];
      const TBlob& out  = out_data[dropout::kOut];
      const TBlob& mask = out_data[dropout::kMask];
      if (this->pkeep_ < 1 && (ctx.is_train || this->mode_ == dropout::kAlways)) {
        this->dropout_passthrough_ = false;
        if (this->axes_.ndim() == 0) {
#if MXNET_USE_MKL_DROPOUT
          if (MKLAvailable()) {
            MKLForward(ctx, in_data, out_data);
            return;
          }
#endif  // MXNET_USE_MKL_DROPOUT
#if MXNET_USE_CUDNN_DROPOUT && defined(__CUDACC__)
          if (CuDNNAvailable()) {
            CuDNNForward(ctx, in, mask, out);
            return;
          }
#endif  // MXNET_USE_CUDNN_DROPOUT && defined(__CUDACC__)
          RandGenerator<xpu, DType>* pgen = ctx.requested[0].get_parallel_random<xpu, DType>();
          CHECK_NOTNULL(pgen);
          CHECK(req[dropout::kOut] != kAddTo);
          LaunchRNG<DropoutKernel, xpu>(s,
                                        pgen,
                                        out.Size(),
                                        out.dptr<DType>(),
                                        mask.dptr<DType>(),
                                        in.dptr<DType>(),
                                        this->pkeep_);
          return;
        } else {
          RandGenerator<xpu, DType>* pgen = ctx.requested[0].get_parallel_random<xpu, DType>();
          CHECK_NOTNULL(pgen);
          // initialize the mask
          LaunchRNG<BernoulliKernel, xpu>(s, pgen, mask.Size(), mask.dptr<DType>(), this->pkeep_);
          // broadcast mul
          mxnet::TShape new_lshape, new_rshape, new_oshape;
          int ndim = BinaryBroadcastShapeCompact(
              in.shape_, mask.shape_, out.shape_, &new_lshape, &new_rshape, &new_oshape);
          if (!ndim) {
            MXNET_ASSIGN_REQ_SWITCH(req[dropout::kOut], Req, {
              mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::mul, Req>, xpu>::Launch(
                  s, out.Size(), out.dptr<DType>(), in.dptr<DType>(), mask.dptr<DType>());
            });
          } else {
            BROADCAST_NDIM_SWITCH(ndim, NDim, {
              mshadow::Shape<NDim> oshape  = new_oshape.get<NDim>();
              mshadow::Shape<NDim> lstride = mxnet_op::calc_stride(new_lshape.get<NDim>());
              mshadow::Shape<NDim> rstride = mxnet_op::calc_stride(new_rshape.get<NDim>());
              mxnet_op::Kernel<mxnet_op::binary_broadcast_kernel<NDim, mshadow_op::mul>,
                               xpu>::template LaunchEx(s,
                                                       new_oshape.Size(),
                                                       req[dropout::kOut],
                                                       lstride,
                                                       rstride,
                                                       oshape,
                                                       in.dptr<DType>(),
                                                       mask.dptr<DType>(),
                                                       out.dptr<DType>());
            });
          }
        }
      } else {
        if (req[dropout::kOut] == kWriteInplace)
          return;

        MXNET_ASSIGN_REQ_SWITCH(req[dropout::kOut], Req, {
          mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::identity, Req>, xpu>::Launch(
              s, out.Size(), out.dptr<DType>(), in.dptr<DType>());
        });
      }
    }
  }

  void Backward(const OpContext& ctx,
                const std::vector<TBlob>& out_grad,
                const std::vector<TBlob>& out_data,
                const std::vector<OpReqType>& req,
                const std::vector<TBlob>& in_grad) {
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<xpu>* s = ctx.get_stream<xpu>();
    if (!this->dropout_passthrough_) {
      const TBlob& gdata         = in_grad[dropout::kData];
      const TBlob& grad          = out_grad[dropout::kOut];
      const TBlob& mask          = out_data[dropout::kMask];
      if (this->axes_.ndim() == 0) {
#if MXNET_USE_MKL_DROPOUT
        if (MKLAvailable()) {
          MKLBackward(ctx, in_grad, out_data, out_grad);
          return;
        }
#endif  // MXNET_USE_MKL_DROPOUT
#if MXNET_USE_CUDNN_DROPOUT && defined(__CUDACC__)
        if (CuDNNAvailable()) {
          CuDNNBackward(ctx, grad, mask, gdata);
          return;
        }
#endif  // MXNET_USE_CUDNN_DROPOUT && defined(__CUDACC__)
        // standard case for dropout
        CHECK_EQ(grad.Size(), mask.Size());
        MXNET_ASSIGN_REQ_SWITCH(req[dropout::kData], Req, {
          mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::mul, Req>, xpu>::Launch(
              s, gdata.Size(), gdata.dptr<DType>(), grad.dptr<DType>(), mask.dptr<DType>());
        });
        return;
      } else {
        // broardcast mul
        mxnet::TShape new_lshape, new_rshape, new_oshape;
        int ndim = BinaryBroadcastShapeCompact(
            grad.shape_, mask.shape_, gdata.shape_, &new_lshape, &new_rshape, &new_oshape);
        if (!ndim) {
          MXNET_ASSIGN_REQ_SWITCH(req[dropout::kData], Req, {
            mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::mul, Req>, xpu>::Launch(
                s, gdata.Size(), gdata.dptr<DType>(), grad.dptr<DType>(), mask.dptr<DType>());
          });
        } else {
          BROADCAST_NDIM_SWITCH(ndim, NDim, {
            mshadow::Shape<NDim> oshape  = new_oshape.get<NDim>();
            mshadow::Shape<NDim> lstride = mxnet_op::calc_stride(new_lshape.get<NDim>());
            mshadow::Shape<NDim> rstride = mxnet_op::calc_stride(new_rshape.get<NDim>());
            mxnet_op::Kernel<mxnet_op::binary_broadcast_kernel<NDim, mshadow_op::mul>,
                             xpu>::template LaunchEx(s,
                                                     new_oshape.Size(),
                                                     req[0],
                                                     lstride,
                                                     rstride,
                                                     oshape,
                                                     grad.dptr<DType>(),
                                                     mask.dptr<DType>(),
                                                     gdata.dptr<DType>());
          });
        }
      }
    } else {
      const TBlob& gdata = in_grad[dropout::kData];
      const TBlob& grad  = out_grad[dropout::kOut];
      MXNET_ASSIGN_REQ_SWITCH(req[dropout::kData], Req, {
        mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::identity, Req>, xpu>::Launch(
            s, gdata.Size(), gdata.dptr<DType>(), grad.dptr<DType>());
      });
    }
  }

 private:
  /*! \brief Dropout rate (keep when the generated random number is less than this value) */
  real_t pkeep_;
  /*! \brief Dropout mode */
  dropout::DropoutOpMode mode_;
  /*! \brief Axes on which dropout mask is shared in the form of broadcast multiply */
  mxnet::TShape axes_;
  /*! \brief Flag to record whether forward is executed in pass-through mode */
  bool dropout_passthrough_;
#if MXNET_USE_CUDNN_DROPOUT
  bool cudnn_off_;
  Context ctx_;
  cudnnDataType_t dtype_;
  cudnnDropoutDescriptor_t dropout_desc_;
  size_t dropout_reserve_byte_;
  cudnnTensorDescriptor_t x_desc_, y_desc_, dx_desc_, dy_desc_;
#endif  // MXNET_USE_CUDNN_DROPOUT
};      // class DropoutOp

template <typename xpu>
void DropoutCompute(const OpStatePtr& state,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    DropoutOp<xpu, DType>& op = state.get_state<DropoutOp<xpu, DType>>();
    op.Forward(ctx, inputs, req, outputs);
  });
}

template <typename xpu>
void DropoutGradCompute(const OpStatePtr& state,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1);
  CHECK_EQ(req.size(), 1);
  std::vector<TBlob> out_grads(2);
  std::vector<TBlob> out_data(2);
  out_grads[dropout::kOut] = inputs[0];
  out_data[dropout::kMask] = inputs[1];

  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    DropoutOp<xpu, DType>& op = state.get_state<DropoutOp<xpu, DType>>();
    op.Backward(ctx, out_grads, out_data, req, outputs);
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NN_DROPOUT_INL_H_
