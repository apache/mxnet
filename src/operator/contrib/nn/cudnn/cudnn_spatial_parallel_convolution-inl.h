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
 * \file cudnn_convolution-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_CONTRIB_NN_CUDNN_CUDNN_SPATIAL_PARALLEL_CONVOLUTION_INL_H_
#define MXNET_OPERATOR_CONTRIB_NN_CUDNN_CUDNN_SPATIAL_PARALLEL_CONVOLUTION_INL_H_

#include <mshadow/base.h>
#include <mxnet/storage.h>
#include <dmlc/parameter.h>
#include <mxnet/tuple.h>
#if MXNET_USE_NCCL
#include <nccl.h>
#endif
#include <algorithm>
#include <vector>
#include <set>
#include <mutex>
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <utility>
#include "../spatial_parallel_convolution-inl.h"
#include "../../../nn/cudnn/cudnn_algoreg-inl.h"
#include "../../../../common/cuda/utils.h"
#include "../../spatial_parallel_support.h"
#include "../../../nn/fully_connected-inl.h"

namespace mxnet {
namespace op {
#if MXNET_USE_CUDNN == 1 && MXNET_USE_NCCL == 1

typedef CuDNNAlgoReg<SpatialParallelConvolutionParam> CuDNNSPConvAlgoReg;

// Equivalent algo performance threshhold (e.g. 1.01 == 1% performance difference)
// Used to prune Tensor Core algos with no appreciable performance benefit.
#define ALGO_PERF_THRESHOLD 1.01

namespace {

template <typename DType>
__global__ void ExtractHaloFilter_kernel(DType* out, const DType* in, const index_t slice_begin,
                                  const index_t slice_end, const index_t stride,
                                  const index_t out_dim, const index_t in_dim,
                                  const index_t slice_offset, const index_t N) {
  const index_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N) return;
  const index_t my_slice = (tid / stride) % out_dim;
  DType val = 0;
  if (my_slice >= slice_begin && my_slice < slice_end) {
    const index_t base_x = tid % stride;
    const index_t base_y = tid / (stride * out_dim);
    const index_t base = base_x + (my_slice + slice_offset + base_y * in_dim) * stride;
    val = in[base];
  }
  out[tid] = val;
}

template <typename DType>
void ExtractHaloFilter(mshadow::Stream<gpu>* s, DType* out, const DType* in,
                       const index_t slice_begin, const index_t slice_end,
                       const index_t slice_offset, const mxnet::TShape& halo_wshape,
                       const mxnet::TShape& wshape) {
    index_t n_elements = halo_wshape.Size();
    const int threads = 512;
    const index_t blocks = common::div_round(n_elements, threads);
    cudaStream_t stream = Stream<gpu>::GetStream(s);
    index_t stride = wshape.Size() / (wshape[0] * wshape[1]);
    const index_t filter_halo_dim = halo_wshape.ndim() == wshape.ndim() ? halo_wshape[1] : 1;
    ExtractHaloFilter_kernel<<<blocks, threads, 0, stream>>>(out, in, slice_begin, slice_end,
                                                             stride, filter_halo_dim, wshape[1],
                                                             slice_offset, n_elements);
}

template <typename DType>
__global__ void AddHaloWGrad_kernel(const DType* in, DType* out, const index_t slice_begin,
                                    const index_t slice_end, const index_t stride,
                                    const index_t in_dim, const index_t out_dim,
                                    const index_t slice_offset, const index_t N) {
  const index_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N) return;
  const index_t my_slice = (tid / stride) % in_dim;
  if (my_slice >= slice_begin && my_slice < slice_end) {
    const index_t base_x = tid % stride;
    const index_t base_y = tid / (stride * in_dim);
    const index_t base = base_x + (my_slice + slice_offset + base_y * out_dim) * stride;
    out[base] += in[tid];
  }
}

template <typename DType>
void AddHaloWGrad(mshadow::Stream<gpu>* s, DType* out, const DType* in,
                  const index_t slice_begin, const index_t slice_end,
                  const index_t slice_offset, const mxnet::TShape& halo_wshape,
                  const mxnet::TShape& wshape) {
    index_t n_elements = halo_wshape.Size();
    const int threads = 512;
    const index_t blocks = common::div_round(n_elements, threads);
    cudaStream_t stream = Stream<gpu>::GetStream(s);
    index_t stride = wshape.Size() / (wshape[0] * wshape[1]);
    const index_t filter_halo_dim = halo_wshape.ndim() == wshape.ndim() ? halo_wshape[1] : 1;
    AddHaloWGrad_kernel<<<blocks, threads, 0, stream>>>(in, out, slice_begin, slice_end, stride,
                                                        filter_halo_dim, wshape[1], slice_offset,
                                                        n_elements);
}

void SetTensorDescriptor(cudnnTensorDescriptor_t desc,
                         cudnnDataType_t dtype,
                         const mxnet::TShape& shape,
                         const mxnet::TShape& strides) {
  std::vector<int> shape_buffer(shape.ndim());
  nnvm::ShapeTypeCast(shape.begin(), shape.end(), shape_buffer.data());
  std::vector<int> stride_buffer(strides.ndim());
  nnvm::ShapeTypeCast(strides.begin(), strides.end(), stride_buffer.data());

  CUDNN_CALL(cudnnSetTensorNdDescriptor(desc,
                                        dtype,
                                        static_cast<int>(shape.ndim()),
                                        shape_buffer.data(),
                                        stride_buffer.data()));
}

int *CastTShapeToIntPtr(const mxnet::TShape& s, std::vector<int> *buffer) {
  buffer->resize(s.ndim());
  nnvm::ShapeTypeCast(s.begin(), s.end(), buffer->data());
  return buffer->data();
}

// Converts a mxnet::TShape to a Shape<> of strides.
// e.g. {shape[0], shape[1], shape[2]} -> {shape[1]*shape[2], shape[2], 1}
template <int dim>
inline Shape<dim> Strides(const mxnet::TShape &s) {
  int ndim = s.ndim();
  mxnet::TShape strides(ndim, -1);
  for (int i = 0; i != ndim; ++i)
    strides[i] = s.ProdShape(i+1, ndim);
  return strides.get<dim>();
}

}  // namespace

/*!
 * \brief The Operator used to perform convolution using cuDNN kernels.
 */
template<typename DType>
class CuDNNSPConvolutionOp {
  STATIC_ASSERT_CUDNN_VERSION_GE(7000);

  struct CUDNNConvolution {
   public:
    cudnnConvolutionDescriptor_t forward_desc_;
    // Convolution descriptor for back-prop operations to the data
    cudnnConvolutionDescriptor_t dgrad_desc_;
    // Convolution descriptor for back-prop operations to the weights
    cudnnConvolutionDescriptor_t wgrad_desc_;
    cudnnTensorDescriptor_t in_desc_;
    cudnnTensorDescriptor_t out_desc_;
    cudnnTensorDescriptor_t bias_desc_;
    cudnnFilterDescriptor_t filter_desc_;

    // Algorithm for the forward inference operation
    CuDNNAlgo<cudnnConvolutionFwdAlgo_t> forward_algo_;
    // Algorithm for the back-prop operation to the data
    CuDNNAlgo<cudnnConvolutionBwdDataAlgo_t> dgrad_algo_;
    // Algorithm for the back-prop operation to the weights
    CuDNNAlgo<cudnnConvolutionBwdFilterAlgo_t> wgrad_algo_;
    // Temp workspace size in bytes needed for Forward() operation.
    size_t forward_workspace_byte_;
    // Temp workspace size in bytes needed for Backward() dgrad (data gradient) operation.
    size_t dgrad_workspace_byte_;
    // Temp workspace size in bytes needed for Backward() wgrad (weight gradient) operation.
    size_t wgrad_workspace_byte_;
    // Temp workspace size in bytes needed for Backward() bgrad (bias gradient) operation.
    size_t bias_workspace_byte_;


    CUDNNConvolution() {
      CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc_));
      CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc_));
      CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc_));
      CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc_));
      CUDNN_CALL(cudnnCreateConvolutionDescriptor(&forward_desc_));
      CUDNN_CALL(cudnnCreateConvolutionDescriptor(&dgrad_desc_));
      CUDNN_CALL(cudnnCreateConvolutionDescriptor(&wgrad_desc_));
      bias_workspace_byte_ = 0;
      forward_workspace_byte_ = 0;
      dgrad_workspace_byte_ = 0;
      wgrad_workspace_byte_ = 0;
    }

    ~CUDNNConvolution() {
      CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc_));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc_));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(bias_desc_));
      CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc_));
      CUDNN_CALL(cudnnDestroyConvolutionDescriptor(forward_desc_));
      CUDNN_CALL(cudnnDestroyConvolutionDescriptor(wgrad_desc_));
      CUDNN_CALL(cudnnDestroyConvolutionDescriptor(dgrad_desc_));
    }

    void Init(const mxnet::TShape& original_dshape,
              const mxnet::TShape& original_wshape,
              const mxnet::TShape& original_oshape,
              const mshadow::LayoutFlag original_layout,
              const cudnnDataType_t dtype,
              const mxnet::TShape& stride_shape,
              const mxnet::TShape& dilate_shape,
              const mxnet::TShape& pad_shape,
              const cudnnDataType_t forward_compute_type,
              const cudnnDataType_t backward_compute_type,
              const bool cudnn_tensor_core) {
      mxnet::TShape dstride, ostride;
      mxnet::TShape dshape_channel_first, oshape_channel_first;
      std::vector<int> stride, dilate, pad;
      CastTShapeToIntPtr(stride_shape, &stride);
      CastTShapeToIntPtr(dilate_shape, &dilate);
      CastTShapeToIntPtr(pad_shape, &pad);
      auto layout = original_layout;
      auto dshape = original_dshape;
      auto wshape = original_wshape;
      auto oshape = original_oshape;
      if (pad.size() == 1) {
        // Convert 1D conv to 2D with first dimension being 1
        switch (layout) {
          case mshadow::kNCW:
          case mshadow::kCWN:
            LOG(FATAL) << "Only channels-last layouts are supported for now!";
            break;
          case mshadow::kNWC:
            layout = mshadow::kNHWC;
            pad = {0, pad[0]};
            dilate = {1, dilate[0]};
            stride = {1, stride[0]};
            dshape = mxnet::TShape{dshape[0], 1, dshape[1], dshape[2]};
            wshape = mxnet::TShape{wshape[0], 1, wshape[1], wshape[2]};
            oshape = mxnet::TShape{oshape[0], 1, oshape[1], oshape[2]};
          default: break;
        }
      }
      cudnnTensorFormat_t format;
      MSHADOW_LAYOUT_SWITCH(layout, Layout, {
        format = LayoutType<Layout>::kCudnnFlag;
      });
      CHECK_GE(dshape.ndim(), 4) << "Only 2D and 3D convolutions are supported for now!";
      if (dshape.ndim() == 4) {
        // 1d or 2d conv
        CUDNN_CALL(cudnnSetConvolution2dDescriptor(forward_desc_,
                                                   pad[0],
                                                   pad[1],
                                                   stride[0],
                                                   stride[1],
                                                   dilate[0],
                                                   dilate[1],
                                                   CUDNN_CROSS_CORRELATION,
                                                   forward_compute_type));
        CUDNN_CALL(cudnnSetConvolution2dDescriptor(dgrad_desc_,
                                                   pad[0],
                                                   pad[1],
                                                   stride[0],
                                                   stride[1],
                                                   dilate[0],
                                                   dilate[1],
                                                   CUDNN_CROSS_CORRELATION,
                                                   backward_compute_type));
        CUDNN_CALL(cudnnSetConvolution2dDescriptor(wgrad_desc_,
                                                   pad[0],
                                                   pad[1],
                                                   stride[0],
                                                   stride[1],
                                                   dilate[0],
                                                   dilate[1],
                                                   CUDNN_CROSS_CORRELATION,
                                                   backward_compute_type));
        auto wshape_channel_first = ConvertLayout(wshape.get<4>(), layout, kNCHW);
        dstride = ConvertLayout(Strides<4>(dshape), layout, kNCHW);
        dshape_channel_first = ConvertLayout(dshape.get<4>(), layout, kNCHW);
        ostride = ConvertLayout(Strides<4>(oshape), layout, kNCHW);
        oshape_channel_first = ConvertLayout(oshape.get<4>(), layout, kNCHW);
        CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_desc_,
                                              dtype,
                                              format,
                                              wshape_channel_first[0],
                                              wshape_channel_first[1],
                                              wshape_channel_first[2],
                                              wshape_channel_first[3]));
        auto kernel_h = wshape_channel_first[2];
        auto kernel_w = wshape_channel_first[3];
        // The 5x5 non-fused Winograd kernel is fast, but because of its reduced numerical
        // accuracy compared to other algos, users must opt-in to its use.
        bool exclude_nonfused_winograd_5x5 =
          !dmlc::GetEnv("MXNET_CUDNN_ENABLE_WINOGRAD_NONFUSED_5X5", false);
        if (exclude_nonfused_winograd_5x5 && kernel_h == 5 && kernel_w == 5) {
          // excluded_forward_algos_.insert(CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED);
          // excluded_back_algos_.insert(CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED);
          // excluded_back_algos_w_.insert(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED);
        }
      } else if (dshape.ndim() == 5) {
        // 3d conv
        std::vector<int> wshape_buffer(wshape.ndim());
        mxnet::TShape wshape_channel_first = ConvertLayout(wshape.get<5>(), layout, kNCDHW);

        CUDNN_CALL(cudnnSetFilterNdDescriptor(filter_desc_,
                                              dtype,
                                              format,
                                              static_cast<int>(wshape.ndim()),
                                              CastTShapeToIntPtr(wshape_channel_first,
                                                                 &wshape_buffer)));
        CUDNN_CALL(cudnnSetConvolutionNdDescriptor(forward_desc_,
                                                   3,
                                                   pad.data(),
                                                   stride.data(),
                                                   dilate.data(),
                                                   CUDNN_CROSS_CORRELATION,
                                                   forward_compute_type));

        CUDNN_CALL(cudnnSetConvolutionNdDescriptor(dgrad_desc_,
                                                   3,
                                                   pad.data(),
                                                   stride.data(),
                                                   dilate.data(),
                                                   CUDNN_CROSS_CORRELATION,
                                                   backward_compute_type));

        CUDNN_CALL(cudnnSetConvolutionNdDescriptor(wgrad_desc_,
                                                   3,
                                                   pad.data(),
                                                   stride.data(),
                                                   dilate.data(),
                                                   CUDNN_CROSS_CORRELATION,
                                                   backward_compute_type));

        dstride = ConvertLayout(Strides<5>(dshape), layout, kNCDHW);
        dshape_channel_first = ConvertLayout(dshape.get<5>(), layout, kNCDHW);

        ostride = ConvertLayout(Strides<5>(oshape), layout, kNCDHW);
        oshape_channel_first = ConvertLayout(oshape.get<5>(), layout, kNCDHW);
      }
      // Set "allow tensor core" flag in convolution descriptors, if available.
      cudnnMathType_t math_type = cudnn_tensor_core ? CUDNN_TENSOR_OP_MATH
                                                    : CUDNN_DEFAULT_MATH;
#if CUDNN_VERSION >= 7200
      if (GetEnvAllowTensorCore() && GetEnvAllowTensorCoreConversion() &&
          (DataType<DType>::kFlag != kFloat16))
        math_type = CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
#endif
      CUDNN_CALL(cudnnSetConvolutionMathType(forward_desc_, math_type));
      CUDNN_CALL(cudnnSetConvolutionMathType(dgrad_desc_, math_type));
      CUDNN_CALL(cudnnSetConvolutionMathType(wgrad_desc_, math_type));
      CUDNN_CALL(cudnnSetConvolutionGroupCount(forward_desc_, 1));
      CUDNN_CALL(cudnnSetConvolutionGroupCount(dgrad_desc_, 1));
      CUDNN_CALL(cudnnSetConvolutionGroupCount(wgrad_desc_, 1));

      SetTensorDescriptor(in_desc_, dtype, dshape_channel_first, dstride);
      SetTensorDescriptor(out_desc_, dtype, oshape_channel_first, ostride);
    }

    void InitBias(const mxnet::TShape& bias,
                  const int ndim,
                  const cudnnDataType_t dtype) {
      int bias_dim = static_cast<int>(bias[0]);
      std::vector<int> bias_shape(ndim, 1);
      std::vector<int> bias_stride(ndim, bias_dim);
      bias_shape[1] = bias_dim;
      bias_stride[1] = 1;
      CUDNN_CALL(cudnnSetTensorNdDescriptor(bias_desc_,
                                            dtype,
                                            static_cast<int>(bias_shape.size()),
                                            bias_shape.data(),
                                            bias_stride.data()));
    }

    void GetWorkspaceSize(const RunContext& rctx) {
      mshadow::Stream<gpu> *s = rctx.get_stream<gpu>();
      CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(s->dnn_handle_,
                                                              filter_desc_,
                                                              out_desc_,
                                                              dgrad_desc_,
                                                              in_desc_,
                                                              dgrad_algo_.AlgoNumber(),
                                                              &dgrad_workspace_byte_));
      CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(s->dnn_handle_,
                                                                in_desc_,
                                                                out_desc_,
                                                                wgrad_desc_,
                                                                filter_desc_,
                                                                wgrad_algo_.AlgoNumber(),
                                                                &wgrad_workspace_byte_));

      CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(s->dnn_handle_,
                                                         in_desc_,
                                                         filter_desc_,
                                                         forward_desc_,
                                                         out_desc_,
                                                         forward_algo_.AlgoNumber(),
                                                         &forward_workspace_byte_));
    }

    template <typename ScaleType>
    void Forward(mshadow::Stream<gpu>* s, ScaleType alpha, void* input,
                 void* filter, void* wsp, ScaleType beta, void* output) {
      CUDNN_CALL(cudnnConvolutionForward(s->dnn_handle_, &alpha, in_desc_,
                                         input, filter_desc_, filter,
                                         forward_desc_, forward_algo_.AlgoNumber(),
                                         wsp, forward_workspace_byte_,
                                         &beta, out_desc_, output));
    }

    template <typename ScaleType>
    void DGrad(mshadow::Stream<gpu>* s, ScaleType alpha, void* input,
                 void* filter, void* wsp, ScaleType beta, void* output) {
      CUDNN_CALL(cudnnConvolutionBackwardData(s->dnn_handle_, &alpha, filter_desc_,
                                              filter, out_desc_, input, dgrad_desc_,
                                              dgrad_algo_.AlgoNumber(), wsp,
                                              dgrad_workspace_byte_, &beta,
                                              in_desc_, output));
    }

    template <typename ScaleType>
    void WGrad(mshadow::Stream<gpu>* s, ScaleType alpha, void* input,
                 void* in_grad, void* wsp, ScaleType beta, void* dfilter) {
      CUDNN_CALL(cudnnConvolutionBackwardFilter(s->dnn_handle_, &alpha, in_desc_,
                                                input, out_desc_, in_grad, wgrad_desc_,
                                                wgrad_algo_.AlgoNumber(), wsp,
                                                wgrad_workspace_byte_, &beta,
                                                filter_desc_, dfilter));
    }
  };

 public:
  CuDNNSPConvolutionOp() : back_bias_get_workspace_performed_(false) {
    parallelize_backward_kernels_ = Context::GetGPUStreamsPerWorker() >= 2;
  }

  void Init(const SpatialParallelConvolutionParam& param,
            int forward_compute_type,
            int backward_compute_type,
            const mxnet::ShapeVector& in_shape,
            const mxnet::ShapeVector& out_shape,
            const RunContext& rctx,
            bool add_to_weight) {
    using namespace mshadow;
    this->param_ = param;
    cudnn_tensor_core_ = DataType<DType>::kFlag == kFloat16 && GetEnvAllowTensorCore();
    this->add_to_weight_ = add_to_weight;
    auto cudnn_forward_compute_type = convertToCuDNNDataType(forward_compute_type);
    auto cudnn_backward_compute_type = convertToCuDNNDataType(backward_compute_type);
    // convert MB to words
    param_.workspace = (param_.workspace << 20) / sizeof(DType);

    // Double check to make sure this class supports the operation
    if (!Supports(param, forward_compute_type, backward_compute_type, rctx.ctx.dev_id))
      LOG(FATAL) << "Convolution parameters not supported by cuDNN implementation.";

    InitDescriptors(in_shape, out_shape,
                    cudnn_forward_compute_type, cudnn_backward_compute_type);

    if (!param_.cudnn_tune) {
      param_.cudnn_tune = dmlc::GetEnv("MXNET_CUDNN_AUTOTUNE_DEFAULT", 1);
    }

    // In cuDNN_v6, dilated convolution descriptors are compatible with only a
    // single convolution algorithm.  Despite this, we go through the algorithm
    // selection process, which will return the only algorithm supported.  This
    // approach keeps the treatment of convolution cases uniform and will
    // naturally respond to more algorithms supporting dilated convolutions in
    // future cuDNN releases.
    SelectMainAlgo(rctx, in_shape, out_shape,
                   cudnn_forward_compute_type, cudnn_backward_compute_type);
    mxnet::ShapeVector halo_in_shape(2);
    const index_t input_halo_dim = (param_.kernel[0] - 1) / 2;
    const index_t output_halo_dim = (input_halo_dim + 2 * param_.pad[0] - param_.kernel[0])
                                    / param_.stride[0] + 1;
    const bool reduced_dim = param_.kernel[0] == 3;
    halo_in_shape[0] = GetHaloShape(in_shape[0], input_halo_dim, reduced_dim);
    halo_in_shape[1] = GetHaloFilterShape(in_shape[1], param.kernel[0]);
    mxnet::ShapeVector halo_out_shape(1);
    halo_out_shape[0] = GetHaloShape(out_shape[0], output_halo_dim, reduced_dim);

    SelectHaloAlgo(rctx, halo_in_shape, halo_out_shape,
                   cudnn_forward_compute_type, cudnn_backward_compute_type);
    GetTempSize(rctx);
    SpatialParallelParam p;
    p.num_gpus = param_.num_gpus;
    p.rank = param_.rank;
    p.nccl_unique_id = param_.nccl_unique_id;
    NCCLCommContainer::Init(p);
  }

  ~CuDNNSPConvolutionOp() {}

  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1U);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    auto dshape = in_data[0].shape_;
    auto oshape = out_data[0].shape_;
    const index_t input_halo_dim = (param_.kernel[0] - 1) / 2;
    const index_t output_halo_dim = (input_halo_dim + 2 * param_.pad[0] - param_.kernel[0])
                                    / param_.stride[0] + 1;
    const bool reduced_dim = param_.kernel[0] == 3;
    auto halo_dshape = GetHaloShape(dshape, input_halo_dim, reduced_dim);
    size_t halo_size = halo_dshape.Size() * sizeof(DType);
    mxnet::TShape wshape = in_data[spconv::kWeight].shape_;
    mxnet::TShape halo_wshape = GetHaloFilterShape(wshape, param_.kernel[0]);
    size_t halo_filter_size = halo_wshape.Size() * sizeof(DType);

    // I/O's should have 2 more dims than the kernel dim
    DType *data_ptr = GetNdPtr(in_data[spconv::kData], param_.kernel.ndim() + 2, s);
    DType *wmat_ptr = GetNdPtr(in_data[spconv::kWeight], param_.kernel.ndim() + 2, s);
    DType *out_ptr = GetNdPtr(out_data[spconv::kOut], param_.kernel.ndim() + 2, s);

    typename DataType<DType>::ScaleType alpha = 1.0f;
    typename DataType<DType>::ScaleType beta = 0.0f;
    typename DataType<DType>::ScaleType beta_add = 1.0f;

    index_t first_index = param_.rank * dshape[1];
    index_t last_index = (param_.rank + 1) * dshape[1] - 1;
    const bool needs_recv_previous_halo = param_.rank > 0 && (param_.pad[0] > 0);
    const bool needs_recv_next_halo = (param_.rank < param_.num_gpus - 1) &&
                                      ((param_.kernel[0] - 1)/2 >
                                       ((last_index - first_index + param_.pad[0] -
                                         (param_.kernel[0] - 1) / 2)
                                        % param_.stride[0]));
    const bool needs_send_first_halo = param_.rank > 0 &&
                                       ((param_.kernel[0] - 1)/2 >
                                        ((last_index - first_index + param_.pad[0] -
                                          (param_.kernel[0] - 1) / 2)
                                         % param_.stride[0]));
    const bool needs_send_last_halo = (param_.rank < param_.num_gpus - 1) &&
                                      (param_.pad[0] > 0);
    const bool needs_communicate = needs_recv_next_halo || needs_recv_previous_halo ||
                                   needs_send_first_halo || needs_send_last_halo;

    const index_t input_slice_size = RemoveFirstSpatialDim(dshape).Size();
    const index_t output_slice_size = RemoveFirstSpatialDim(oshape).Size();

    std::vector<std::vector<size_t>> stages;
    using WorkspaceType = mshadow::Tensor<gpu, 1, DType>;
    WorkspaceType previous_halo_data;
    WorkspaceType next_halo_data;
    WorkspaceType main_fwd_wsp;
    WorkspaceType halo_prev_fwd_wsp;
    WorkspaceType halo_next_fwd_wsp;
    WorkspaceType halo_prev_filter;
    WorkspaceType halo_next_filter;
    if (needs_communicate) {
      size_t prev_halo_size = needs_recv_previous_halo ? halo_size : 0;
      size_t next_halo_size = needs_recv_next_halo ? halo_size : 0;
      if (parallelize_backward_kernels_) {
        std::pair<size_t, size_t> prev_halo_wsp_size = needs_recv_previous_halo
          ? std::make_pair(halo_conv_.forward_workspace_byte_, halo_filter_size)
          : std::make_pair(0ul, 0ul);
        std::pair<size_t, size_t> next_halo_wsp_size = needs_recv_next_halo
          ? std::make_pair(halo_conv_.forward_workspace_byte_, halo_filter_size)
          : std::make_pair(0ul, 0ul);
        stages = {{prev_halo_size, next_halo_size, main_conv_.forward_workspace_byte_},
                  {prev_halo_size, next_halo_size,
                   prev_halo_wsp_size.first, prev_halo_wsp_size.second,
                   next_halo_wsp_size.first, next_halo_wsp_size.second}};
        auto workspace_stages = AllocateTempWorkspaces(ctx, stages);
        previous_halo_data = workspace_stages[0][0];
        next_halo_data = workspace_stages[0][1];
        main_fwd_wsp = workspace_stages[0][2];
        halo_prev_fwd_wsp = workspace_stages[1][2];
        halo_prev_filter = workspace_stages[1][3];
        halo_next_fwd_wsp = workspace_stages[1][4];
        halo_next_filter = workspace_stages[1][5];
      } else {
        stages = {{prev_halo_size, next_halo_size, halo_filter_size,
                   std::max(main_conv_.forward_workspace_byte_,
                            halo_conv_.forward_workspace_byte_)}};
        auto workspace_stages = AllocateTempWorkspaces(ctx, stages);
        previous_halo_data = workspace_stages[0][0];
        next_halo_data = workspace_stages[0][1];
        halo_prev_filter = workspace_stages[0][2];
        halo_next_filter = workspace_stages[0][2];
        main_fwd_wsp = workspace_stages[0][3];
        halo_prev_fwd_wsp = workspace_stages[0][3];
        halo_next_fwd_wsp = workspace_stages[0][3];
      }
    } else {
      stages = {{main_conv_.forward_workspace_byte_}};
      auto workspace_stages = AllocateTempWorkspaces(ctx, stages);
      main_fwd_wsp = workspace_stages[0][0];
    }
    if (needs_communicate) {
      {
        // First phase: Fwd + NCCL
        SyncedGPUAuxStream s_nccl = ctx.get_gpu_aux_stream();
        main_conv_.Forward(s, alpha, data_ptr, wmat_ptr, main_fwd_wsp.dptr_,
                           req[spconv::kOut] == kAddTo ? beta_add : beta,
                           out_ptr);
        {
          cudaStream_t nccl_stream = Stream<gpu>::GetStream(s_nccl.GetStream());
          std::lock_guard<std::mutex> l(Storage::Get()->GetMutex(Context::kGPU));
          ncclComm_t comm = *(NCCLCommContainer::comm_map.at(param_.num_gpus));
          ncclGroupStart();
          if (needs_recv_previous_halo) {
            ncclRecv(previous_halo_data.dptr_, halo_dshape.Size(), DataType<DType>::kNCCLFlag,
                     param_.rank - 1, comm, nccl_stream);
          }
          if (needs_recv_next_halo) {
            ncclRecv(next_halo_data.dptr_, halo_dshape.Size(), DataType<DType>::kNCCLFlag,
                     param_.rank + 1, comm, nccl_stream);
          }
          if (needs_send_first_halo) {
            ncclSend(data_ptr, halo_dshape.Size(), DataType<DType>::kNCCLFlag,
                     param_.rank - 1, comm, nccl_stream);
          }
          if (needs_send_last_halo) {
            ncclSend(data_ptr + dshape.Size() - halo_dshape.Size(),
                     halo_dshape.Size(), DataType<DType>::kNCCLFlag,
                     param_.rank + 1, comm, nccl_stream);
          }
          ncclGroupEnd();
        }
      }
      {
        // Second phase - halo convolutions
        SyncedGPUAuxStream s_aux = ctx.get_gpu_aux_stream();
        if (needs_recv_previous_halo) {
          ExtractHaloFilter(s, halo_prev_filter.dptr_, wmat_ptr, input_halo_dim - 1,
                            2 * input_halo_dim - 1, 1 - input_halo_dim, halo_wshape, wshape);
          halo_conv_.Forward(s, alpha, previous_halo_data.dptr_, halo_prev_filter.dptr_,
                             halo_prev_fwd_wsp.dptr_, beta_add, out_ptr);
        }
        if (needs_recv_next_halo) {
          const index_t stride_adjustment = (param_.stride[0] -
                                             (input_halo_dim % param_.stride[0])) %
                                            param_.stride[0];
          // If the output regions do not overlap use the secondary stream
          mshadow::Stream<gpu>* second_stream =
            oshape.Size() >= 2 * output_slice_size * output_halo_dim ? s_aux.GetStream() : s;
          ExtractHaloFilter(second_stream, halo_next_filter.dptr_, wmat_ptr, 0,
                            input_halo_dim - stride_adjustment,
                            input_halo_dim + 1 + stride_adjustment,
                            halo_wshape, wshape);
          halo_conv_.Forward(second_stream, alpha, next_halo_data.dptr_, halo_next_filter.dptr_,
                             halo_next_fwd_wsp.dptr_, beta_add,
                             out_ptr + oshape.Size() - output_slice_size * output_halo_dim);
        }
      }
    } else {
      main_conv_.Forward(s, alpha, data_ptr, wmat_ptr, main_fwd_wsp.dptr_,
                         req[spconv::kOut] == kAddTo? beta_add : beta, out_ptr);
    }

    bool perform_forward_bias = !param_.no_bias;
    bool perform_cuda_forward_bias = perform_forward_bias &&
                                     FeaturesLastLayout() &&
                                     dmlc::GetEnv("MXNET_CONV_CUDA_FORWARD_BIAS", true);
    if (perform_forward_bias) {
      if (perform_cuda_forward_bias) {
        int output_features = static_cast<int>(Features(out_data[spconv::kOut].shape_));
        Tensor<gpu, 1, DType> bias =
          in_data[spconv::kBias].get_with_shape<gpu, 1, DType>(Shape1(output_features), s);
        Tensor<gpu, 2, DType> out = FlattenAs2DHead<gpu, DType>(out_data[spconv::kOut], ctx);
        auto &data = out;  // Only data.shape_[0] is used by AddBias()
        AddBias(bias, data, out, s);
      } else {
        Tensor<gpu, 1, DType> bias = in_data[spconv::kBias].get<gpu, 1, DType>(s);
        CUDNN_CALL(cudnnAddTensor(s->dnn_handle_,
                                  &alpha,
                                  main_conv_.bias_desc_,
                                  bias.dptr_,
                                  &beta_add,
                                  main_conv_.out_desc_,
                                  out_ptr));
      }
    }
  }

  void Backward(const OpContext &ctx,
                const std::vector<TBlob> &out_grad,
                const std::vector<TBlob> &in_data,
                const std::vector<OpReqType> &req,
                const std::vector<TBlob> &in_grad) {
    using namespace mshadow;
    using namespace mshadow::expr;
    size_t expected = param_.no_bias == 0 ? 3 : 2;
    CHECK_EQ(out_grad.size(), 1U);
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(in_grad.size(), expected);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    const index_t output_halo_dim = (param_.kernel[0] - 1) / 2;
    const index_t input_halo_dim = (output_halo_dim + 2 * param_.pad[0] - param_.kernel[0])
                                   / param_.stride[0] + 1;
    const auto& dshape = out_grad[spconv::kOut].shape_;
    const bool reduced_dim = param_.kernel[0] == 3;
    const auto& halo_dshape = GetHaloShape(dshape, input_halo_dim, reduced_dim);
    const auto& oshape = in_grad[spconv::kData].shape_;
    const size_t halo_size = halo_dshape.Size() * sizeof(DType);
    mxnet::TShape wshape = in_data[spconv::kWeight].shape_;
    mxnet::TShape halo_wshape = GetHaloFilterShape(wshape, param_.kernel[0]);
    size_t halo_filter_size = halo_wshape.Size() * sizeof(DType);

    // I/O's should have 2 more dims than the kernel dim
    DType *grad_ptr = GetNdPtr(out_grad[spconv::kOut], param_.kernel.ndim() + 2, s);
    DType *wmat_ptr = GetNdPtr(in_data[spconv::kWeight], param_.kernel.ndim() + 2, s);
    DType *gwmat_ptr = GetNdPtr(in_grad[spconv::kWeight], param_.kernel.ndim() + 2, s);
    DType *data_ptr = GetNdPtr(in_data[spconv::kData], param_.kernel.ndim() + 2, s);
    DType *gdata_ptr = GetNdPtr(in_grad[spconv::kData], param_.kernel.ndim() + 2, s);

    bool perform_backward_bias = !param_.no_bias && (req[spconv::kBias] != kNullOp);
    bool perform_cuda_backward_bias = perform_backward_bias &&
                                      FeaturesLastLayout() &&
                                      dmlc::GetEnv("MXNET_CONV_CUDA_BACKWARD_BIAS", true);
    if (perform_cuda_backward_bias && !back_bias_get_workspace_performed_) {
      auto y_grad = FlattenAs2DHead<gpu, DType>(out_grad[spconv::kOut], ctx);
      int output_features = static_cast<int>(Features(out_grad[spconv::kOut].shape_));
      main_conv_.bias_workspace_byte_ = AddBiasGradWorkspaceSizeBytes(in_grad[spconv::kBias],
                                                                      y_grad,
                                                                      req[spconv::kBias],
                                                                      output_features, ctx);
      back_bias_get_workspace_performed_ = true;
    }

    index_t first_index = param_.rank * dshape[1];
    index_t last_index = (param_.rank + 1) * dshape[1] - 1;
    const bool needs_recv_previous_halo = param_.rank > 0 &&
                                          ((param_.kernel[0] - 1)/2 >
                                           ((last_index - first_index + param_.pad[0] -
                                             (param_.kernel[0] - 1) / 2)
                                            % param_.stride[0]));
    const bool needs_recv_next_halo = (param_.rank < param_.num_gpus - 1) &&
                                      (param_.pad[0] > 0);
    const bool needs_send_first_halo = param_.rank > 0 && (param_.pad[0] > 0);
    const bool needs_send_last_halo = (param_.rank < param_.num_gpus - 1) &&
                                      ((param_.kernel[0] - 1)/2 >
                                       ((last_index - first_index + param_.pad[0] -
                                         (param_.kernel[0] - 1) / 2)
                                        % param_.stride[0]));
    bool needs_communicate = needs_recv_next_halo || needs_recv_previous_halo ||
                             needs_send_first_halo || needs_send_last_halo;

    const index_t output_slice_size = RemoveFirstSpatialDim(oshape).Size();
    std::vector<std::vector<size_t>> stages;
    using WorkspaceType = mshadow::Tensor<gpu, 1, DType>;
    WorkspaceType previous_halo_data;
    WorkspaceType next_halo_data;
    WorkspaceType main_dgrad_wsp;
    WorkspaceType main_wgrad_wsp;
    WorkspaceType main_bias_wsp;
    WorkspaceType halo_dgrad_wsp;
    WorkspaceType halo_wgrad_wsp;
    WorkspaceType halo_dfilter;
    WorkspaceType halo_filter;
    if (needs_communicate) {
      size_t prev_halo_size = needs_recv_previous_halo ? halo_size : 0;
      size_t next_halo_size = needs_recv_next_halo ? halo_size : 0;
      if (parallelize_backward_kernels_) {
        stages = {{prev_halo_size, next_halo_size, std::max(main_conv_.dgrad_workspace_byte_,
                                                            main_conv_.wgrad_workspace_byte_)},
                  {prev_halo_size, next_halo_size, std::max(halo_conv_.wgrad_workspace_byte_,
                                                            main_conv_.bias_workspace_byte_),
                   halo_filter_size, halo_conv_.dgrad_workspace_byte_, halo_filter_size}};
        CHECK_EQ(stages[0][0], prev_halo_size);
        CHECK_EQ(stages[0][1], next_halo_size);
        const auto& workspace_stages = AllocateTempWorkspaces(ctx, stages);
        previous_halo_data = workspace_stages[0][0];
        next_halo_data = workspace_stages[0][1];
        main_dgrad_wsp = workspace_stages[0][2];
        main_wgrad_wsp = workspace_stages[0][2];
        main_bias_wsp = workspace_stages[1][2];
        halo_wgrad_wsp = workspace_stages[1][2];
        halo_dfilter = workspace_stages[1][3];
        halo_dgrad_wsp = workspace_stages[1][4];
        halo_filter = workspace_stages[1][5];
      } else {
        stages = {{prev_halo_size, next_halo_size, halo_filter_size,
                   std::max({main_conv_.dgrad_workspace_byte_,
                             main_conv_.wgrad_workspace_byte_,
                             main_conv_.bias_workspace_byte_,
                             halo_conv_.dgrad_workspace_byte_,
                             halo_conv_.wgrad_workspace_byte_})}};
        const auto& workspace_stages = AllocateTempWorkspaces(ctx, stages);
        previous_halo_data = workspace_stages[0][0];
        next_halo_data = workspace_stages[0][1];
        halo_dfilter = workspace_stages[0][2];
        halo_filter = workspace_stages[0][2];
        main_dgrad_wsp = workspace_stages[0][3];
        main_wgrad_wsp = workspace_stages[0][3];
        main_bias_wsp = workspace_stages[0][3];
        halo_wgrad_wsp = workspace_stages[0][3];
        halo_dgrad_wsp = workspace_stages[0][3];
      }
    } else {
      if (parallelize_backward_kernels_) {
        stages = {{std::max(main_conv_.wgrad_workspace_byte_, main_conv_.bias_workspace_byte_),
                   main_conv_.dgrad_workspace_byte_}};
        const auto& workspace_stages = AllocateTempWorkspaces(ctx, stages);
        main_wgrad_wsp = workspace_stages[0][0];
        main_bias_wsp = workspace_stages[0][0];
        main_dgrad_wsp = workspace_stages[0][1];
      } else {
        stages = {{std::max({main_conv_.dgrad_workspace_byte_,
                             main_conv_.wgrad_workspace_byte_,
                             main_conv_.bias_workspace_byte_})}};
        const auto& workspace_stages = AllocateTempWorkspaces(ctx, stages);
        main_dgrad_wsp = workspace_stages[0][0];
        main_wgrad_wsp = workspace_stages[0][0];
        main_bias_wsp = workspace_stages[0][0];
      }
    }

    if (needs_communicate) {
      typename DataType<DType>::ScaleType alpha = 1.0f;
      typename DataType<DType>::ScaleType beta = 0.0f;
      typename DataType<DType>::ScaleType beta_add = 1.0f;
      {
        SyncedGPUAuxStream aux_stream = ctx.get_gpu_aux_stream();
        // First stage: [dgrad, wgrad] + [NCCL]
        if (req[spconv::kData] != kNullOp) {
          main_conv_.DGrad(s, alpha, grad_ptr, wmat_ptr, main_dgrad_wsp.dptr_,
                           req[spconv::kData] == kAddTo? beta_add : beta, gdata_ptr);
        }
        if (req[spconv::kWeight] != kNullOp) {
          CHECK_EQ(add_to_weight_, req[spconv::kWeight] == kAddTo);
          main_conv_.WGrad(s, alpha, data_ptr, grad_ptr, main_wgrad_wsp.dptr_,
                           req[spconv::kWeight] == kAddTo? beta_add : beta,
                           gwmat_ptr);
        }
        {
          cudaStream_t nccl_stream = Stream<gpu>::GetStream(aux_stream.GetStream());
          std::lock_guard<std::mutex> l(Storage::Get()->GetMutex(Context::kGPU));
          ncclComm_t comm = *(NCCLCommContainer::comm_map.at(param_.num_gpus));
          ncclGroupStart();
          if (needs_recv_previous_halo) {
            ncclRecv(previous_halo_data.dptr_, halo_dshape.Size(), DataType<DType>::kNCCLFlag,
                     param_.rank - 1, comm, nccl_stream);
          }
          if (needs_recv_next_halo) {
            ncclRecv(next_halo_data.dptr_, halo_dshape.Size(), DataType<DType>::kNCCLFlag,
                     param_.rank + 1, comm, nccl_stream);
          }
          if (needs_send_first_halo) {
            ncclSend(grad_ptr, halo_dshape.Size(), DataType<DType>::kNCCLFlag,
                     param_.rank - 1, comm, nccl_stream);
          }
          if (needs_send_last_halo) {
            ncclSend(grad_ptr + dshape.Size() - halo_dshape.Size(),
                     halo_dshape.Size(), DataType<DType>::kNCCLFlag,
                     param_.rank + 1, comm, nccl_stream);
          }
          ncclGroupEnd();
        }
      }
      {
        // Second stage: [halo_wgrad, halo_wgrad, bgrad] + [halo_dgrad, halo_dgrad]
        SyncedGPUAuxStream aux_stream = ctx.get_gpu_aux_stream();
        const index_t filter_halo_dim = halo_wshape.ndim() == wshape.ndim() ? halo_wshape[1] : 1;
        if (req[spconv::kWeight] != kNullOp) {
          if (needs_recv_previous_halo) {
            halo_conv_.WGrad(s, alpha, data_ptr, previous_halo_data.dptr_, halo_wgrad_wsp.dptr_,
                             beta, halo_dfilter.dptr_);
            const index_t stride_adjustment = (param_.stride[0] -
                                               (output_halo_dim % param_.stride[0])) %
                                              param_.stride[0];
            AddHaloWGrad(s, gwmat_ptr, halo_dfilter.dptr_, 0, output_halo_dim - stride_adjustment,
                         output_halo_dim + 1 + stride_adjustment, halo_wshape, wshape);
          }
          if (needs_recv_next_halo) {
            halo_conv_.WGrad(s, alpha, data_ptr + oshape.Size() -
                                                  output_slice_size * output_halo_dim,
                             next_halo_data.dptr_, halo_wgrad_wsp.dptr_, beta, halo_dfilter.dptr_);
            AddHaloWGrad(s, gwmat_ptr, halo_dfilter.dptr_, output_halo_dim - 1,
                         2 * output_halo_dim - 1, 1 - output_halo_dim, halo_wshape, wshape);
          }
        }
        if (perform_backward_bias) {
          if (perform_cuda_backward_bias) {
            auto y_grad = FlattenAs2DHead<gpu, DType>(out_grad[spconv::kOut], ctx);
            int output_features = static_cast<int>(Features(out_grad[spconv::kOut].shape_));
            Tensor<gpu, 1, uint8_t> workspace(reinterpret_cast<uint8_t*>(main_bias_wsp.dptr_),
                                              Shape1(main_bias_wsp.shape_.Size() * sizeof(DType)));
            AddBiasGrad(in_grad[spconv::kBias], y_grad, req[spconv::kBias], output_features,
                        ctx, spconv::kTempSpace, &workspace);
          } else {
            Tensor<gpu, 1, DType> gbias = in_grad[spconv::kBias].get<gpu, 1, DType>(s);
            CUDNN_CALL(cudnnConvolutionBackwardBias(s->dnn_handle_,
                                                &alpha,
                                                main_conv_.out_desc_,
                                                grad_ptr,
                                                req[spconv::kBias] == kAddTo ? &beta_add : &beta,
                                                main_conv_.bias_desc_,
                                                gbias.dptr_));
          }
        }
        if (needs_recv_previous_halo && req[spconv::kData] != kNullOp) {
          const index_t stride_adjustment = (param_.stride[0] -
                                             (output_halo_dim % param_.stride[0])) %
                                            param_.stride[0];
          ExtractHaloFilter(aux_stream.GetStream(), halo_filter.dptr_, wmat_ptr, 0,
                            output_halo_dim - stride_adjustment,
                            output_halo_dim + 1 + stride_adjustment,
                            halo_wshape, wshape);
          halo_conv_.DGrad(aux_stream.GetStream(), alpha, previous_halo_data.dptr_,
                           halo_filter.dptr_, halo_dgrad_wsp.dptr_, beta_add, gdata_ptr);
        }
        if (needs_recv_next_halo && req[spconv::kData] != kNullOp) {
          ExtractHaloFilter(aux_stream.GetStream(), halo_filter.dptr_, wmat_ptr,
                            output_halo_dim - 1, 2 * output_halo_dim - 1, 1 - output_halo_dim,
                            halo_wshape, wshape);
          halo_conv_.DGrad(aux_stream.GetStream(), alpha, next_halo_data.dptr_,
                           halo_filter.dptr_, halo_dgrad_wsp.dptr_, beta_add,
                           gdata_ptr + oshape.Size() - output_slice_size * output_halo_dim);
        }
      }
    } else {
      SyncedGPUAuxStream aux_stream = ctx.get_gpu_aux_stream();
      DType *workspace_dptr_wgrad = main_wgrad_wsp.dptr_;
      DType *workspace_dptr_dgrad = main_dgrad_wsp.dptr_;
      typename DataType<DType>::ScaleType alpha = 1.0f;
      typename DataType<DType>::ScaleType beta = 0.0f;
      typename DataType<DType>::ScaleType beta_add = 1.0f;
      if (req[spconv::kData] != kNullOp) {
        main_conv_.DGrad(aux_stream.GetStream(), alpha, grad_ptr, wmat_ptr, workspace_dptr_dgrad,
                         req[spconv::kData] == kAddTo? beta_add : beta, gdata_ptr);
      }
      if (req[spconv::kWeight] != kNullOp) {
          CHECK_EQ(add_to_weight_, req[spconv::kWeight] == kAddTo);
          main_conv_.WGrad(s, alpha, data_ptr, grad_ptr, workspace_dptr_wgrad,
                           req[spconv::kWeight] == kAddTo? beta_add : beta, gwmat_ptr);
      }
      if (perform_backward_bias) {
        if (perform_cuda_backward_bias) {
          auto y_grad = FlattenAs2DHead<gpu, DType>(out_grad[spconv::kOut], ctx);
          int output_features = static_cast<int>(Features(out_grad[spconv::kOut].shape_));
          Tensor<gpu, 1, uint8_t> workspace(reinterpret_cast<uint8_t*>(main_bias_wsp.dptr_),
                                            Shape1(main_bias_wsp.shape_.Size() * sizeof(DType)));
          // Will use the beginning of the workspace (so the one shared with wgrad)
          AddBiasGrad(in_grad[spconv::kBias], y_grad, req[spconv::kBias], output_features,
                      ctx, spconv::kTempSpace, &workspace);
        } else {
          Tensor<gpu, 1, DType> gbias = in_grad[spconv::kBias].get<gpu, 1, DType>(s);
          CUDNN_CALL(cudnnConvolutionBackwardBias(s->dnn_handle_,
                                              &alpha,
                                              main_conv_.out_desc_,
                                              grad_ptr,
                                              req[spconv::kBias] == kAddTo ? &beta_add : &beta,
                                              main_conv_.bias_desc_,
                                              gbias.dptr_));
        }
      }
    }
  }

/*!
 * \brief Returns whether the cuDNN library version supports the convolution
 * operation described by `param`: cuDNN v5 and earlier does not support
 * dilated convolutions.  Dilation only enabled after v6.0.20.
 */
  static bool Supports(SpatialParallelConvolutionParam param,
                       int forward_compute_type,
                       int backward_compute_type,
                       int dev_id) {
    using namespace mshadow;

    // NDHWC, NHWC, NHC not supported in true fp16
    auto layout_val = param.layout.value();
    auto true_fp16 = DataType<DType>::kFlag == kFloat16 &&
      (forward_compute_type == kFloat16 || backward_compute_type == kFloat16);
    if (true_fp16 &&
        (layout_val == kNDHWC || layout_val == kNHWC || layout_val == kNWC))
      return false;

    // Permits graceful fallback to pseudo-fp16 on heterogenous systems
    if (!SupportsFloat16Compute(dev_id) &&
        (forward_compute_type == kFloat16 || backward_compute_type == kFloat16)) {
      return false;
    }

    return true;
  }

 private:
/*!
 * \brief Translate an mxnet datatype to the corresponding cudnnDataType_t.
 */
  cudnnDataType_t convertToCuDNNDataType(int dtype) {
    cudnnDataType_t converted = CUDNN_DATA_FLOAT;
    // The following will always assign to `converted` or throw an exception.
    MSHADOW_REAL_TYPE_SWITCH(dtype, mxDType, {
      converted = mshadow::DataType<mxDType>::kCudnnFlag;
    })
    return converted;
  }

  void InitDescriptors(const mxnet::ShapeVector& in_shape,
                       const mxnet::ShapeVector& out_shape,
                       cudnnDataType_t cudnn_forward_compute_type,
                       cudnnDataType_t cudnn_backward_compute_type) {
    using namespace mshadow;
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_shape.size(), expected);
    CHECK_EQ(out_shape.size(), 1U);

    mxnet::TShape dshape = in_shape[spconv::kData];
    mxnet::TShape wshape = in_shape[spconv::kWeight];
    mxnet::TShape oshape = out_shape[spconv::kOut];

    CHECK(param_.layout.value() == kNDHWC ||
          param_.layout.value() == kNHWC) << "Supports only NHWC and NDHWC for now.";

    mxnet::TShape halo_wshape = GetHaloFilterShape(wshape, param_.kernel[0]);
    const index_t input_halo_dim = (param_.kernel[0] - 1) / 2;
    const index_t output_halo_dim = (input_halo_dim + 2 * param_.pad[0] - param_.kernel[0])
                                    / param_.stride[0] + 1;
    const bool reduced_dim = param_.kernel[0] == 3;
    mxnet::TShape halo_dshape = GetHaloShape(dshape, input_halo_dim, reduced_dim);
    mxnet::TShape halo_oshape = GetHaloShape(oshape, output_halo_dim, reduced_dim);

    CHECK_EQ(param_.num_group, 1) << "Does not support grouped convolutions for now.";
    auto dtype = DataType<DType>::kCudnnFlag;
    main_conv_.Init(dshape, wshape, oshape, mshadow::LayoutFlag(param_.layout.value()),
                    dtype, param_.stride, param_.dilate,
                    param_.pad, cudnn_forward_compute_type,
                    cudnn_backward_compute_type,
                    cudnn_tensor_core_);
    if (!param_.no_bias) {
      main_conv_.InitBias(in_shape[spconv::kBias], dshape.ndim(), dtype);
    }
    int ndim = param_.stride.ndim();
    if (reduced_dim) ndim -= 1;
    mxnet::TShape halo_stride(ndim, -1), halo_dilate(ndim, -1), halo_pad(ndim, -1);
    int count = 0;
    for (int i = 0; i < param_.stride.ndim(); ++i) {
      if (reduced_dim && i == 0) continue;
      halo_stride[count] = param_.stride[i];
      halo_dilate[count] = param_.dilate[i];
      if (!reduced_dim && i == 0) {
        halo_pad[count] = param_.pad[i] - 1;
      } else {
        halo_pad[count] = param_.pad[i];
      }
      ++count;
    }
    auto halo_layout = mshadow::LayoutFlag(param_.layout.value());
    if (reduced_dim) {
      switch (param_.layout.value()) {
        case mshadow::kNDHWC: halo_layout = mshadow::kNHWC; break;
        case mshadow::kNHWC: halo_layout = mshadow::kNWC; break;
        default: LOG(FATAL) << "Layout not supported!";
      }
    }
    halo_conv_.Init(halo_dshape, halo_wshape, halo_oshape, halo_layout,
                    dtype, halo_stride, halo_dilate, halo_pad,
                    cudnn_forward_compute_type,
                    cudnn_backward_compute_type,
                    cudnn_tensor_core_);
  }

  void CuDNNAlgoSetter(const RunContext& rctx,
                  const mxnet::ShapeVector& in_shape,
                  const mxnet::ShapeVector& out_shape,
                  cudnnDataType_t cudnn_forward_compute_type,
                  cudnnDataType_t cudnn_backward_compute_type,
                  CuDNNAlgo<cudnnConvolutionFwdAlgo_t> *fwd,
                  CuDNNAlgo<cudnnConvolutionBwdDataAlgo_t> *bwd,
                  CuDNNAlgo<cudnnConvolutionBwdFilterAlgo_t> *flt) {
    // Not in algo registry, must determine via *Get*() or *Find*()
    mshadow::Stream<gpu> *s = rctx.get_stream<gpu>();
    CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
    size_t workspace_byte = static_cast<size_t>(param_.workspace * sizeof(DType));

    // Since the function signature of *Get*_v7() matches that of *Find*(),
    // we can unify the find-vs-get logic by using function pointers.

    // Forward Algorithm Find/Get() v7
    std::vector<cudnnConvolutionFwdAlgoPerf_t> fwd_results(MaxForwardAlgos(s->dnn_handle_));
    int actual_fwd_algos = 0;
    auto fwd_algo_discoverer =
      param_.cudnn_tune.value() == spconv::kOff ? cudnnGetConvolutionForwardAlgorithm_v7
                                              : cudnnFindConvolutionForwardAlgorithm;
    CUDNN_CALL((*fwd_algo_discoverer)(s->dnn_handle_,
                                      main_conv_.in_desc_,
                                      main_conv_.filter_desc_,
                                      main_conv_.forward_desc_,
                                      main_conv_.out_desc_,
                                      fwd_results.size(),
                                      &actual_fwd_algos,
                                      fwd_results.data()));
    fwd_results.resize(actual_fwd_algos);
    AlgoFinalSelect<cudnnConvolutionFwdAlgoPerf_t,
                    cudnnConvolutionFwdAlgo_t>(fwd_results, "forward",
                                               workspace_byte,
                                               fwd, excluded_forward_algos_);

    // Backprop-to-Filter Algorithm Find/Get() v7
    auto max_bwd_filt_algos = MaxBackwardFilterAlgos(s->dnn_handle_);
    std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> bwd_filt_results(max_bwd_filt_algos);
    int actual_bwd_filter_algos = 0;
    // In cudnn v7.1.4, find() returned wgrad algos that could fail for large c if we
    // were summing into the output (i.e. beta != 0).  Get() returned OK algos though.
    auto bwd_filter_algo_discoverer =
      param_.cudnn_tune.value() == spconv::kOff ? cudnnGetConvolutionBackwardFilterAlgorithm_v7
                                              : cudnnFindConvolutionBackwardFilterAlgorithm;
    CUDNN_CALL((*bwd_filter_algo_discoverer)(s->dnn_handle_,
                                             main_conv_.in_desc_,
                                             main_conv_.out_desc_,
                                             main_conv_.wgrad_desc_,
                                             main_conv_.filter_desc_,
                                             bwd_filt_results.size(),
                                             &actual_bwd_filter_algos,
                                             bwd_filt_results.data()));
    bwd_filt_results.resize(actual_bwd_filter_algos);
    AlgoFinalSelect<cudnnConvolutionBwdFilterAlgoPerf_t,
                    cudnnConvolutionBwdFilterAlgo_t>(bwd_filt_results, "backprop-to-filter",
                                                     workspace_byte,
                                                     flt, excluded_back_algos_w_);

    // Backprop-to-Data Algorithm Find/Get() v7
    auto max_bwd_data_algos = MaxBackwardDataAlgos(s->dnn_handle_);
    std::vector<cudnnConvolutionBwdDataAlgoPerf_t> bwd_data_results(max_bwd_data_algos);
    int actual_bwd_data_algos = 0;
    auto bwd_data_algo_discoverer =
      param_.cudnn_tune.value() == spconv::kOff ? cudnnGetConvolutionBackwardDataAlgorithm_v7
                                              : cudnnFindConvolutionBackwardDataAlgorithm;
    CUDNN_CALL((*bwd_data_algo_discoverer)(s->dnn_handle_,
                                           main_conv_.filter_desc_,
                                           main_conv_.out_desc_,
                                           main_conv_.dgrad_desc_,
                                           main_conv_.in_desc_,
                                           bwd_data_results.size(),
                                           &actual_bwd_data_algos,
                                           bwd_data_results.data()));
    bwd_data_results.resize(actual_bwd_data_algos);
    AlgoFinalSelect<cudnnConvolutionBwdDataAlgoPerf_t,
                    cudnnConvolutionBwdDataAlgo_t>(bwd_data_results, "backprop-to-data",
                                                   workspace_byte,
                                                   bwd, excluded_back_algos_);

    // Fix for issue #11241
    int cudnn_find_issue_max_features = 64 * 1024;
    if (add_to_weight_ && Features(in_shape[spconv::kData]) >= cudnn_find_issue_max_features) {
      flt->Set(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, true);
    }
  }

  void SelectMainAlgo(const RunContext& rctx,
                      const mxnet::ShapeVector& in_shape,
                      const mxnet::ShapeVector& out_shape,
                      cudnnDataType_t cudnn_forward_compute_type,
                      cudnnDataType_t cudnn_backward_compute_type) {
    SelectAlgo(rctx, in_shape, out_shape, cudnn_forward_compute_type,
               cudnn_backward_compute_type, param_, DataType<DType>::kCudnnFlag, add_to_weight_,
               &main_conv_.forward_algo_, &main_conv_.dgrad_algo_, &main_conv_.wgrad_algo_,
               main_conv_.forward_desc_, main_conv_.dgrad_desc_, main_conv_.wgrad_desc_);
  }

  void SelectHaloAlgo(const RunContext& rctx,
                      const mxnet::ShapeVector& in_shape,
                      const mxnet::ShapeVector& out_shape,
                      cudnnDataType_t cudnn_forward_compute_type,
                      cudnnDataType_t cudnn_backward_compute_type) {
    SpatialParallelConvolutionParam halo_param = param_;
    if (param_.kernel[0] > 3) {
      halo_param.kernel[0] -= 2;
      halo_param.pad[0] -= 1;
    } else {
      halo_param.kernel = mxnet::TShape(param_.kernel.begin() + 1, param_.kernel.end());
      halo_param.stride = mxnet::TShape(param_.stride.begin() + 1, param_.stride.end());
      halo_param.dilate = mxnet::TShape(param_.dilate.begin() + 1, param_.dilate.end());
      halo_param.pad = mxnet::TShape(param_.pad.begin() + 1, param_.pad.end());
    }
    SelectAlgo(rctx, in_shape, out_shape, cudnn_forward_compute_type,
               cudnn_backward_compute_type, halo_param, DataType<DType>::kCudnnFlag, add_to_weight_,
               &halo_conv_.forward_algo_, &halo_conv_.dgrad_algo_, &halo_conv_.wgrad_algo_,
               halo_conv_.forward_desc_, halo_conv_.dgrad_desc_, halo_conv_.wgrad_desc_);
  }

  void SelectAlgo(const RunContext& rctx,
                  const mxnet::ShapeVector& in_shape,
                  const mxnet::ShapeVector& out_shape,
                  cudnnDataType_t cudnn_forward_compute_type,
                  cudnnDataType_t cudnn_backward_compute_type,
                  const SpatialParallelConvolutionParam& param,
                  cudnnDataType_t dtype,
                  bool add_to_weight,
                  CuDNNAlgo<cudnnConvolutionFwdAlgo_t>* forward_algo,
                  CuDNNAlgo<cudnnConvolutionBwdDataAlgo_t>* back_algo,
                  CuDNNAlgo<cudnnConvolutionBwdFilterAlgo_t>* back_algo_w,
                  cudnnConvolutionDescriptor_t forward_conv_desc,
                  cudnnConvolutionDescriptor_t back_conv_desc,
                  cudnnConvolutionDescriptor_t back_conv_desc_w) {
    auto algo_setter = [&](CuDNNAlgo<cudnnConvolutionFwdAlgo_t> *fwd,
                           CuDNNAlgo<cudnnConvolutionBwdDataAlgo_t> *bwd,
                           CuDNNAlgo<cudnnConvolutionBwdFilterAlgo_t> *flt) {
      if (param_.cudnn_tune.value() == spconv::kOff) {
        // The routine will only be calling cudnnGet, so no need to grab the Storage lock.
        this->CuDNNAlgoSetter(rctx, in_shape, out_shape,
                              cudnn_forward_compute_type,
                              cudnn_backward_compute_type,
                              fwd, bwd, flt);
      } else {
        // One potential problem is that cudnnFind() uses cudaMalloc() to directly allocate
        // I/O and workspace areas, and these allocations may result in an out-of-memory
        // error even though the StorageMangager free pool is not empty.  Ideally, cudnnFind
        // would use MXNet's storage allocator for its I/O and workspace areas, instead of using
        // the area carved out by MXNET_GPU_MEM_POOL_RESERVE.
        // To get somewhat the same effect as this, we can pre-allocate the areas needed for the
        // I/Os (possibly triggering a desirable StorageManager::ReleaseAll()), followed by a
        // DirectFree(), which makes these areas available for cudnn's subsequent cudaMalloc().

        // Allocate for x (or dx), w (or dw) and y (or dy).
        ReserveElements({in_shape[spconv::kData].Size(),
                         in_shape[spconv::kWeight].Size(),
                         out_shape[spconv::kOut].Size()});

        // We're about to call cudnnFind so we need to quiet the system by grabbing
        // the Storage lock.  Concurrent cudaMalloc's can disrupt the accurate timing
        // measurements of the algos, and can prevent the cuda driver's proper freeing
        // of cudnnFind's internal temporary allocations.  Grabbing the lock might also
        // impede other threads from launching work on the GPU.
        std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(Context::kGPU));
        this->CuDNNAlgoSetter(rctx, in_shape, out_shape,
                              cudnn_forward_compute_type,
                              cudnn_backward_compute_type,
                              fwd, bwd, flt);
      }
    };

    CuDNNSPConvAlgoReg::Get()->FindOrElseRegister(param, in_shape, out_shape, dtype,
                                                  cudnn_forward_compute_type,
                                                  cudnn_backward_compute_type,
                                                  SMArch(rctx.ctx.dev_id), add_to_weight,
                                                  forward_algo, back_algo,
                                                  back_algo_w, algo_setter);

    // If we're allowing Tensor Core variants of the algos to be considered in
    // *Find*() or *Get*(), but a non-Tensor-Core algo variant is the fastest,
    // we must change the descriptor to preclude Tensor Core.  Simplest is to
    // once again set the mathType in all cases.
    CUDNN_CALL(cudnnSetConvolutionMathType(forward_conv_desc, forward_algo->MathType()));
    CUDNN_CALL(cudnnSetConvolutionMathType(back_conv_desc, back_algo->MathType()));
    CUDNN_CALL(cudnnSetConvolutionMathType(back_conv_desc_w, back_algo_w->MathType()));
  }

  // Look over the results from *Find*() or *Get*() and pick the fastest algo given possible
  // workspace constraints and a possible user algo preference.
  template <typename PerfType, typename AlgoType>
  void AlgoFinalSelect(const std::vector<PerfType> &perf_results, std::string kernel_name,
                       size_t workspace_byte, CuDNNAlgo<AlgoType> *algo,
                       const std::set<AlgoType> &excluded_algos) {

    bool enforce_determinism = dmlc::GetEnv("MXNET_ENFORCE_DETERMINISM", false);
    for (decltype(perf_results.size()) i = 0; i != perf_results.size(); ++i) {
      const auto &result = perf_results[i];
      bool algo_is_tensor_core = result.mathType == CUDNN_TENSOR_OP_MATH;
      bool algo_exclusion = (excluded_algos.count(result.algo) != 0);
      if (result.status == CUDNN_STATUS_SUCCESS &&
          (!enforce_determinism || result.determinism == cudnnDeterminism_t::CUDNN_DETERMINISTIC) &&
          (param_.cudnn_tune.value() == spconv::kFastest || result.memory <= workspace_byte) &&
          !algo_exclusion) {
        // Fix for a current cuDNNv7 behavior where algos are reported twice
        // with equivalent performance (both as Tensor Core and not Tensor Core).
        if ((result.mathType == CUDNN_TENSOR_OP_MATH) &&
             (i != perf_results.size() - 1)) {
          const auto &next_result = perf_results[i+1];
          if (next_result.status == CUDNN_STATUS_SUCCESS &&
              next_result.algo == result.algo &&
              next_result.memory == result.memory &&
              next_result.mathType != CUDNN_TENSOR_OP_MATH &&
              next_result.time < ALGO_PERF_THRESHOLD * result.time) {
              // Skip over this result- it's not really a Tensor Core algo.
              // Prefer instead the next equivalent non-Tensor Core algo.
                continue;
          }
        }
        algo->Set(result.algo, algo_is_tensor_core);
        return;
      }
    }
    auto mode = param_.cudnn_tune.value() == spconv::kOff ? " get " : " find ";
    LOG(FATAL) << "Failed to" << mode << "any " << kernel_name << " convolution algorithm"
               << " with workspace size of " << workspace_byte << " bytes,"
               << " please consider reducing batch/model size or increasing the workspace size";
  }


  void GetTempSize(const RunContext& rctx) {
    main_conv_.GetWorkspaceSize(rctx);
    halo_conv_.GetWorkspaceSize(rctx);
  }

  // Converts a TBlob to a dptr, checking for the expected dim and that it's contiguous.
  DType *GetNdPtr(const TBlob& tb, int dim, Stream<gpu> *s) {
    DType *data_ptr = nullptr;
    if (dim == 3) {
      Tensor<gpu, 3, DType> data = tb.get<gpu, 3, DType>(s);
      CHECK_EQ(data.CheckContiguous(), true);
      data_ptr = data.dptr_;
    } else if (dim == 4) {
      Tensor<gpu, 4, DType> data = tb.get<gpu, 4, DType>(s);
      CHECK_EQ(data.CheckContiguous(), true);
      data_ptr = data.dptr_;
    } else if (dim == 5) {
      Tensor<gpu, 5, DType> data = tb.get<gpu, 5, DType>(s);
      CHECK_EQ(data.CheckContiguous(), true);
      data_ptr = data.dptr_;
    } else {
      LOG(FATAL) << "Unexpected Tensor size " << dim << ", supporting only 3, 4 or 5.";
    }
    return data_ptr;
  }

  // Round a value 'x' up to the next multiple of 'multiple'
  static size_t RoundToMultiple(size_t x, size_t multiple) {
    size_t retVal = ((x + multiple - 1) / multiple) * multiple;
    return retVal;
  }

  std::vector<std::vector<mshadow::Tensor<gpu, 1, DType>>>
  AllocateTempWorkspaces(const OpContext &ctx,
                         const std::vector<std::vector<size_t>> &sizes_bytes_stages) {
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    size_t max_total_size = 0;
    std::vector<std::vector<mshadow::Tensor<gpu, 1, DType>>> ret;
    std::vector<std::vector<size_t>> rounded_sizes_in_words_stages;
    // Get maximum size
    for (const auto& sizes_bytes : sizes_bytes_stages) {
      rounded_sizes_in_words_stages.emplace_back();
      auto& rounded_sizes_in_words = rounded_sizes_in_words_stages.back();
      std::transform(sizes_bytes.cbegin(), sizes_bytes.cend(),
                     std::back_inserter(rounded_sizes_in_words),
                     [](const size_t s) -> std::size_t {
                       const size_t dptr_alignment = 512 / sizeof(DType);
                       size_t size_in_words = std::max<size_t>(1,
                                                RoundToMultiple(s, sizeof(DType)) /
                                                sizeof(DType));
                       size_t aligned = RoundToMultiple(size_in_words, dptr_alignment);
                       return aligned;
                     });
      size_t total_size = 0;
      for (const size_t s : rounded_sizes_in_words) {
        total_size += s;
      }
      max_total_size = std::max(max_total_size, total_size);
    }
    auto total_storage = ctx.requested[spconv::kTempSpace].get_space_typed<gpu, 1, DType>(
        mshadow::Shape1(max_total_size), s);
    for (const auto& rounded_sizes_in_words : rounded_sizes_in_words_stages) {
      ret.emplace_back();
      auto& current = ret.back();
      DType *ptr = total_storage.dptr_;
      std::transform(rounded_sizes_in_words.cbegin(), rounded_sizes_in_words.cend(),
                     std::back_inserter(current),
                     [&ptr](const size_t s) -> mshadow::Tensor<gpu, 1, DType> {
                       mshadow::Tensor<gpu, 1, DType> t(ptr, mshadow::Shape1(s));
                       ptr += s;
                       return t;
                     });
    }
    return ret;
  }

  // Returns the size in bytes of the 1D Tensor of words.
  size_t TensorSizeBytes(const mshadow::Tensor<gpu, 1, DType> &tensor) {
    return tensor.MSize() * sizeof(DType);
  }

  // Given a tensor shape of this operation, return the number of features 'c'
  int64_t Features(const mxnet::TShape &dshape) {
    int c = 0;
    switch (dshape.ndim()) {
      case 3: c = ConvertLayout(dshape.get<3>(), param_.layout.value(), kNCW)[1]; break;
      case 4: c = ConvertLayout(dshape.get<4>(), param_.layout.value(), kNCHW)[1]; break;
      case 5: c = ConvertLayout(dshape.get<5>(), param_.layout.value(), kNCDHW)[1]; break;
      default:
        LOG(FATAL) << "Unexpected convolution data dimension " << dshape.ndim();
    }
    return c;
  }

  // Does the operation's data layout have the features dimension 'c' last?
  bool FeaturesLastLayout() {
    return param_.layout.value() == kNWC ||
           param_.layout.value() == kNHWC ||
           param_.layout.value() == kNDHWC;
  }

    // Give a tensor shape of this operation, return the N * H * W
  int64_t GetNHW(const TShape &dshape) {
    int nhw = 0;
    switch (dshape.ndim()) {
      case 3:
      {
         auto tmp = ConvertLayout(dshape.get<3>(), param_.layout.value(), kNCW);
         nhw = tmp[0] * tmp[2];
         break;
      }
      case 4:
      {
        auto tmp = ConvertLayout(dshape.get<4>(), param_.layout.value(), kNCHW);
        nhw = tmp[0] * tmp[2] * tmp[3];
        break;
      }
      case 5:
      {
        auto tmp = ConvertLayout(dshape.get<5>(), param_.layout.value(), kNCDHW);
        nhw = tmp[0] * tmp[3] * tmp[4];
        break;
      }
      default:
        LOG(FATAL) << "Unexpected convolution data dimension " << dshape.ndim();
    }
    return nhw;
  }

  // Make a number of allocations and directly free them, ensuring room for an equivalent set of
  // cudaMalloc() calls by (say) cudnnFind().  `elements` spec the alloc size in DTypes, not bytes.
  void ReserveElements(const std::vector<size_t> &elements) {
    std::vector<Storage::Handle> handles;
    for (size_t alloc_element : elements)
        handles.push_back(Storage::Get()->Alloc(alloc_element * sizeof(DType), Context::GPU()));
    for (auto &handle : handles)
        Storage::Get()->DirectFree(handle);
  }

  // Log that no suitable algo was found that met the workspace constraints, then exit.
  void LogNoSuitableAlgoAndExit(int num_algos_tried, size_t min_memory_needs,
                                size_t workspace_byte, std::string algo_kind) {
    LOG(FATAL) << num_algos_tried << " " << algo_kind << " with minimum memory requirement "
               << min_memory_needs << " bytes have been tried. Workspace size is set to "
               << workspace_byte << " bytes, please consider reducing the batch/model size, "
               << "or increasing workspace size.";
  }

  mxnet::TShape RemoveFirstSpatialDim(const mxnet::TShape& shape) {
    mxnet::TShape ret(shape.ndim() - 1, -1);

    for (int i = 0, count = 0; i < shape.ndim(); ++i) {
      // Remove D or H
      if (i != 1) {
        ret[count] = shape[i];
        ++count;
      }
    }

    return ret;
  }

  mxnet::TShape GetHaloShape(const mxnet::TShape& shape, const int halo_dim,
                             const bool reduced_dim) {
    const int ndim = reduced_dim ? shape.ndim() - 1 : shape.ndim();
    mxnet::TShape ret(ndim, -1);

    for (int i = 0, count = 0; i < shape.ndim(); ++i) {
      if (i != 1) {
        ret[count] = shape[i];
        ++count;
      } else {
        if (!reduced_dim) {
          CHECK_LE(halo_dim, shape[i])
            << "Tensor split into too many GPUs - the halo dimension (" << halo_dim
            << ") is larger than the per-GPU tensor dimension (" << shape[1] << ").";
          ret[count] = halo_dim;
          ++count;
        }
      }
    }

    return ret;
  }

  mxnet::TShape GetHaloFilterShape(const mxnet::TShape& shape, const int kernel_size) {
    const int ndim = kernel_size > 3 ? shape.ndim() : shape.ndim() - 1;
    mxnet::TShape ret(ndim, -1);

    for (int i = 0, count = 0; i < shape.ndim(); ++i) {
      if (i != 1) {
        ret[count] = shape[i];
        ++count;
      } else {
        if (kernel_size > 3) {
          ret[count] = kernel_size - 2;
          ++count;
        }
      }
    }

    return ret;
  }

  // If backward bias is needed, has the workspace size been requested
  bool back_bias_get_workspace_performed_;

  CUDNNConvolution main_conv_;
  CUDNNConvolution halo_conv_;

  // Should dgrad and wgrad be launched into separate streams
  bool parallelize_backward_kernels_;
  SpatialParallelConvolutionParam param_;
  // Is req[kWeight] == spconv::kAddTo ?
  bool add_to_weight_;
  // forward algos that should be avoided to work-around possible cuDNN issues.
  std::set<cudnnConvolutionFwdAlgo_t> excluded_forward_algos_;
  // dgrad algos that should be avoided to work-around possible cuDNN issues.
  std::set<cudnnConvolutionBwdDataAlgo_t> excluded_back_algos_;
  // wgrad algos that should be avoided to work-around possible cuDNN issues.
  std::set<cudnnConvolutionBwdFilterAlgo_t> excluded_back_algos_w_;
  // Allow TensorCore algo policy
  bool cudnn_tensor_core_;
};

#endif  // CUDNN && NCCL
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_NN_CUDNN_CUDNN_SPATIAL_PARALLEL_CONVOLUTION_INL_H_
