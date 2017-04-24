/*******************************************************************************
* Copyright 2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* \file mkl_pooling-inl.h
* \brief
* \author zhenlin.luo@intel.com
*         lingyan.guo@intel.com
*
*******************************************************************************/

#ifndef MXNET_OPERATOR_MKL_MKL_POOLING_INL_H_
#define MXNET_OPERATOR_MKL_MKL_POOLING_INL_H_
#include <vector>
#include <string>
#include <utility>
#include "../operator_common.h"
#include "../pooling-inl.h"
#include "./mkl_util-inl.h"

namespace mxnet {
namespace op {


template<typename xpu, typename DType>
class MKLPoolingOp : public Operator {
 public:
  static std::string getName() {
    return "MKLPoolingOp";
  }
  explicit MKLPoolingOp(PoolingParam p) {
    poolingFwd = static_cast<dnnPrimitive_t>(NULL);
    poolingBwd = static_cast<dnnPrimitive_t>(NULL);
    max_idx_data = static_cast<DType*>(NULL);
    fwd_top_data = MKLData<DType>::create();
    fwd_bottom_data = MKLData<DType>::create();
    bwd_top_diff = MKLData<DType>::create();
    bwd_bottom_diff = MKLData<DType>::create();
    this->param_ = p;
    init_mkldnn_ = false;
  }
  virtual ~MKLPoolingOp() {
    if (poolingFwd != NULL) {
      dnnDelete<DType>(poolingFwd);
      poolingFwd = NULL;
    }
    if (poolingBwd != NULL) {
      dnnDelete<DType>(poolingBwd);
      poolingBwd = NULL;
    }
    if (max_idx_data != NULL) {
      dnnReleaseBuffer<DType>(max_idx_data);
      max_idx_data = NULL;
    }
  }

 private:
  void LayerSetUp(const mshadow::Tensor<xpu, 4, DType> &data,
                  const mshadow::Tensor<xpu, 4, DType> &out) {
    channels_ = data.shape_[1];
    height_ = data.shape_[2];
    width_ = data.shape_[3];
    num_ = data.shape_[0];
    global_pooling_ = param_.global_pool;
    if (global_pooling_) {
      kernel_h_ = height_;
      kernel_w_ = width_;
    } else {
      kernel_h_ = param_.kernel[0];
      kernel_w_ = param_.kernel[1];
    }
    CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
    CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
    pad_h_ = param_.pad[0];
    pad_w_ = param_.pad[1];
    if (global_pooling_) {
      stride_h_ = stride_w_ = 1;
    } else {
      stride_h_ = param_.stride[0];
      stride_w_ = param_.stride[1];
    }
    if (global_pooling_) {
      CHECK(pad_h_ == 0 && pad_w_ == 0 && stride_h_ == 1 && stride_w_ == 1)
        << "With Global_pooling: true; only pad = 0 and stride = 1";
    }
    if (pad_h_ != 0 || pad_w_ != 0) {
      CHECK(param_.pool_type == pool_enum::kAvgPooling
          || param_.pool_type == pool_enum::kMaxPooling)
        << "Padding implemented only for average and max pooling.";
      CHECK_LT(pad_h_, kernel_h_);
      CHECK_LT(pad_w_, kernel_w_);
    }
    pooled_height_ = out.shape_[2];
    pooled_width_ = out.shape_[3];

    size_t dim = 4;
    size_t src_sizes[4], src_strides[4];
    size_t dst_sizes[4], dst_strides[4];
    src_sizes[0] = width_;
    src_sizes[1] = height_;
    src_sizes[2] = channels_;
    src_sizes[3] = num_;
    src_strides[0] = 1;
    src_strides[1] = src_sizes[0];
    src_strides[2] = src_sizes[0] * src_sizes[1];
    src_strides[3] = src_sizes[0] * src_sizes[1] * src_sizes[2];
    dst_sizes[0] = pooled_width_;
    dst_sizes[1] = pooled_height_;
    dst_sizes[2] = src_sizes[2];
    dst_sizes[3] = src_sizes[3];
    dst_strides[0] = 1;
    dst_strides[1] = dst_sizes[0];
    dst_strides[2] = dst_sizes[0] * dst_sizes[1];
    dst_strides[3] = dst_sizes[0] * dst_sizes[1] * dst_sizes[2];
    src_offset[0] = -pad_w_;
    src_offset[1] = -pad_h_;
    src_offset[2] = -pad_w_;
    src_offset[3] = -pad_h_;
    kernel_stride[0] = stride_w_;
    kernel_stride[1] = stride_h_;
    kernel_size[0] = kernel_w_;
    kernel_size[1] = kernel_h_;

    // Names are for debugging only
    fwd_bottom_data->name = "fwd_bottom_data   @ " + getName();
    fwd_top_data->name = "fwd_top_data      @ " + getName();
    bwd_top_diff->name = "bwd_top_diff      @ " + getName();
    bwd_bottom_diff->name = "bwd_bottom_diff   @ " + getName();

    fwd_bottom_data->create_user_layout(dim, src_sizes, src_strides);
    fwd_top_data->create_user_layout(dim, dst_sizes, dst_strides);
    bwd_bottom_diff->create_user_layout(dim, src_sizes, src_strides);
    bwd_top_diff->create_user_layout(dim, dst_sizes, dst_strides);

    // Primitives will be allocated during the first fwd pass
    poolingFwd = NULL;
    poolingBwd = NULL;
    max_idx_data = NULL;
  }

 public:
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    if (param_.kernel.ndim() >= 3) {
      LOG(FATAL) << "Not implmented";
    }
    Tensor<xpu, 4, DType> data = mkl_experimental_direct_get<xpu, 4, DType>(
      in_data[pool_enum::kData], s);
    Tensor<xpu, 4, DType> out = mkl_experimental_direct_get<xpu, 4, DType>(
      out_data[pool_enum::kOut], s);
    if (!init_mkldnn_) {
      LayerSetUp(data, out);
      init_mkldnn_ = true;
    }
    auto first_pass = false;
    if (poolingFwd == NULL) first_pass = true;

    dnnAlgorithm_t algorithm = dnnAlgorithmPoolingMax;

    switch (param_.pool_type) {
    case pool_enum::kMaxPooling:
      algorithm = dnnAlgorithmPoolingMax;
      break;
    case pool_enum::kAvgPooling:
      algorithm = (param_.pooling_convention == pool_enum::kValid) ?
          dnnAlgorithmPoolingAvgIncludePadding : dnnAlgorithmPoolingAvg;

      break;
    default:
      LOG(FATAL) << "Unknown pooling method.";
    }

    dnnError_t status;
    void* pooling_res[dnnResourceNumber];

    void* bottom_data = NULL;
#if MKL_EXPERIMENTAL == 1
    bottom_data =
          reinterpret_cast<void *>(mkl_prv_data<DType>(in_data[pool_enum::kData]));
#endif
    dnnBorder_t border_type = dnnBorderZerosAsymm;
    switch (param_.pooling_convention) {
    case pool_enum::kFull:
      border_type = dnnBorderZeros;
      break;
    case pool_enum::kValid:
      border_type = dnnBorderZerosAsymm;
      break;
    default:
      border_type = dnnBorderZerosAsymm;
      break;
    }
    if (NULL == bottom_data) {
      bottom_data = data.dptr_;
      if (NULL == poolingFwd) {
        status = dnnPoolingCreateForward<DType>(&poolingFwd, NULL,
                                                algorithm, fwd_bottom_data->layout_usr,
                                                kernel_size, kernel_stride,
                                                src_offset, border_type);
      CHECK_EQ(status, E_SUCCESS);
      // Now create poolingBwd
      status = dnnPoolingCreateBackward<DType>(&poolingBwd, NULL,
                                               algorithm, fwd_bottom_data->layout_usr,
                                               kernel_size, kernel_stride,
                                               src_offset, border_type);
      CHECK_EQ(status, E_SUCCESS);
      }
    }
#if MKL_EXPERIMENTAL == 1
    if (NULL != bottom_data) {
       if (NULL == poolingFwd) {
          std::shared_ptr<MKLMemHolder> bottom_data_mem = in_data[pool_enum::kData].Mkl_mem_;
          std::shared_ptr<PrvMemDescr> bottom_prv_descriptor =
            bottom_data_mem->get_prv_descriptor();
          CHECK_EQ(bottom_prv_descriptor->get_descr_type(),
                   PrvMemDescr::PRV_DESCR_MKL2017);
          std::shared_ptr<MKLData<DType> > mem_descr
            = std::static_pointer_cast<MKLData<DType>>(bottom_prv_descriptor);
          CHECK(mem_descr != nullptr);
          fwd_bottom_data = mem_descr;

          status = dnnPoolingCreateForward<DType>(&poolingFwd, NULL,
                                                  algorithm, fwd_bottom_data->layout_int,
                                                  kernel_size, kernel_stride,
                                                  src_offset, border_type);
          CHECK_EQ(status, E_SUCCESS);
          fwd_top_data->create_internal_layout(poolingFwd, dnnResourceDst);

          // Now create poolingBwd
          status = dnnPoolingCreateBackward<DType>(&poolingBwd, NULL,
                                                   algorithm, fwd_bottom_data->layout_int,
                                                   kernel_size, kernel_stride,
                                                   src_offset, border_type);
          CHECK_EQ(status, E_SUCCESS);
          bwd_top_diff->create_internal_layout(poolingFwd, dnnResourceDst);
          bwd_bottom_diff->create_internal_layout(poolingFwd, dnnResourceSrc);
        }
    }
#endif

    if (first_pass) {
      dnnLayout_t max_idx_datal = NULL;
      status = dnnLayoutCreateFromPrimitive<DType>(
          &max_idx_datal, poolingFwd, dnnResourceWorkspace);
      CHECK_EQ(status, E_SUCCESS);
      status = dnnAllocateBuffer<DType>(reinterpret_cast<void**>(&max_idx_data), max_idx_datal);
      CHECK_EQ(status, E_SUCCESS);
#if MKL_EXPERIMENTAL == 0
      fwd_bottom_data->create_internal_layout(poolingFwd, dnnResourceSrc);
      fwd_top_data->create_internal_layout(poolingFwd, dnnResourceDst);
      bwd_top_diff->create_internal_layout(poolingBwd, dnnResourceDiffDst);
      bwd_bottom_diff->create_internal_layout(poolingBwd, dnnResourceDiffSrc);
#endif
      dnnLayoutDelete<DType>(max_idx_datal);
      first_pass = false;
    }
    pooling_res[dnnResourceSrc] = bottom_data;
    pooling_res[dnnResourceWorkspace] = max_idx_data;

    pooling_res[dnnResourceDst] = fwd_top_data->get_output_ptr(
      out.dptr_, fwd_top_data, out_data[pool_enum::kOut]);
    status = dnnExecute<DType>(poolingFwd, pooling_res);
    CHECK_EQ(status, E_SUCCESS);
#if MKL_EXPERIMENTAL == 0
    if (fwd_top_data->conversion_needed()) {
      fwd_top_data->convert_from_prv(out.dptr_);
    }
#endif
  }
  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    if (!req[0]) {
      return;
    }
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1);
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(req.size(), 1);
    CHECK_EQ(in_grad.size(), 1);
    if (param_.kernel.ndim() >= 3) {
      LOG(FATAL) << "Not implmented";
    }
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> grad = mkl_experimental_direct_get<xpu, 4, DType>(
      out_grad[pool_enum::kOut], s);
    Tensor<xpu, 4, DType> input_grad = mkl_experimental_direct_get<xpu, 4, DType>(
      in_grad[pool_enum::kData], s);
    dnnError_t e;
    void* pooling_res[dnnResourceNumber];
    pooling_res[dnnResourceWorkspace] = reinterpret_cast<void *>(max_idx_data);

    pooling_res[dnnResourceDiffDst] =
      bwd_top_diff->get_converted_prv(grad.dptr_, true, out_grad[pool_enum::kOut]);

    pooling_res[dnnResourceDiffSrc] = bwd_bottom_diff->get_output_ptr(
      input_grad.dptr_, bwd_bottom_diff, in_grad[pool_enum::kData]);
    e = dnnExecute<DType>(poolingBwd, pooling_res);
    CHECK_EQ(e, E_SUCCESS);
#if MKL_EXPERIMENTAL == 0
    if (bwd_bottom_diff->conversion_needed()) {
      bwd_bottom_diff->convert_from_prv(input_grad.dptr_);
    }
#endif
  }

 private:
  PoolingParam param_;
  int kernel_h_, kernel_w_;
  int stride_h_, stride_w_;
  int pad_h_, pad_w_;
  int channels_, num_;
  int height_, width_;
  int pooled_height_, pooled_width_;
  bool global_pooling_;

 private:
  size_t kernel_size[2],
         kernel_stride[4];
  int src_offset[4];  // 2*(dimension-2)
  dnnPrimitive_t poolingFwd, poolingBwd;
  DType *max_idx_data;

  std::shared_ptr<MKLData<DType> > fwd_top_data;
  std::shared_ptr<MKLData<DType> > fwd_bottom_data;
  std::shared_ptr<MKLData<DType> > bwd_top_diff;
  std::shared_ptr<MKLData<DType> > bwd_bottom_diff;
  bool init_mkldnn_;
};  // class MKLPoolingOp
}   // namespace op
}   // namespace mxnet

#endif  // MXNET_OPERATOR_MKL_MKL_POOLING_INL_H_
