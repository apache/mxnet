/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
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
* \author young.jin.kim@intel.com
*         ashok.emani@intel.com
*         deepthi.karkada@intel.com
*         lingyan.guo@intel.com
*         zhenlin.luo@intel.com
*
*******************************************************************************/
#ifndef MXNET_OPERATOR_MKL_MKL_POOLING_INL_H_
#define MXNET_OPERATOR_MKL_MKL_POOLING_INL_H_
#include <vector>
#include <string>
#include <utility>
#if MXNET_USE_MKLDNN == 1
#include "mkldnn_base-inl.h"
#endif
#if MXNET_USE_MKL2017 == 1
#include "../operator_common.h"
#include "../pooling-inl.h"
#include "./mkl_util-inl.h"
#endif   //MXNET_USE_MKL2017

namespace mxnet {
namespace op {

#if MXNET_USE_MKLDNN == 1
template<typename xpu, typename Dtype>
class MKLDNNPoolingOp : public Operator, public MKLDNNLayer<Dtype> {
 public:
  std::string getName() {
     std::string name = "MKLDNNPoolingOp";
     return name;
  }
  explicit MKLDNNPoolingOp(PoolingParam p) : MKLDNNLayer<Dtype>()
    , num_(0), channels_(0), width_(0), height_(0), width_out_(0), height_out_(0)
    , kernel_w_(0), kernel_h_(0), stride_w_(0), stride_h_(0)
    , pad_t_(0), pad_b_(0), pad_l_(0), pad_r_(0) {
    this->param_ = p;
    this->init_mkldnn_ = false;
    switch (param_.pool_type) {
    case pool_enum::kMaxPooling:
      pooling_algorithm_ = pooling_max;
      break;
    case pool_enum::kAvgPooling:
      pooling_algorithm_ = pooling_avg;
      break;
    default:
      LOG(FATAL) << "Unknown pooling method.";
    }
  }
  virtual ~MKLDNNPoolingOp() {
  }

 private:
  void LayerSetUp(const mshadow::Tensor<xpu, 4, Dtype> &data,
                  const mshadow::Tensor<xpu, 4, Dtype> &out) {
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
    pad_t_ = pad_b_ = param_.pad[0];
    pad_l_ = pad_r_ = param_.pad[1];

    stride_h_ = param_.stride[0];
    stride_w_ = param_.stride[1];

    if (global_pooling_) {
      CHECK(pad_t_ == 0 && pad_l_ == 0 && stride_h_ == 1 && stride_w_ == 1)
            << "With Global_pooling: true; only pad = 0 and stride = 1";
    }
    if (pad_t_ != 0 || pad_l_ != 0) {
      CHECK(param_.pool_type == pool_enum::kAvgPooling
        || param_.pool_type == pool_enum::kMaxPooling)
        << "Padding implemented only for average and max pooling.";
      CHECK_LT(pad_t_, kernel_h_);
      CHECK_LT(pad_l_, kernel_w_);
    }
    height_out_ = out.shape_[2];
    width_out_ = out.shape_[3];
  }

 public:
  void InitPoolingFwd(const std::vector<TBlob> &in_data) {
      int32_t n = this->num_;
      int32_t c = this->channels_;
      int32_t ih = this->height_;
      int32_t iw = this->width_;
      int32_t oh = this->height_out_;
      int32_t ow = this->width_out_;

      int32_t kh = this->kernel_h_;
      int32_t kw = this->kernel_w_;

      int32_t sh = this->stride_h_;
      int32_t sw = this->stride_w_;

      int32_t pt = this->pad_t_;
      int32_t pb = this->pad_b_;
      int32_t pl = this->pad_l_;
      int32_t pr = this->pad_r_;

     bool bottom_data_is_prv =
       (const_cast<Dtype*>(mkl_prv_data<Dtype>(in_data[pool_enum::kData])) != NULL);
     mkldnn::engine cpu_engine = CpuEngine::Instance().get_engine();
     memory::data_type mpcsn = memory::data_type::f32;
     memory::dims bottom_tz = { n, c, ih, iw };
     memory::dims top_tz = { n, c, oh, ow };
     memory::format mfmt_nchw = memory::format::nchw;

     // ---- Initialize memory descriptors -------------
     typedef typename memory::primitive_desc MemPD;

     memory::format cmfmt = mfmt_nchw;
     if (bottom_data_is_prv) {
       std::shared_ptr<MKLDNNData<Dtype> > mem_descr
         = get_mkldnn_prv_descriptor<Dtype>(in_data[pool_enum::kData].Mkl_mem_);
       cmfmt = static_cast<memory::format>(mem_descr->prv_memory_pd()->desc().data.format);
     }
     std::shared_ptr<memory::desc> init_fwd_bottom_md(
       new memory::desc({ bottom_tz }, mpcsn, cmfmt));
     std::shared_ptr<memory::desc> init_fwd_top_md(new memory::desc({ top_tz }, mpcsn, cmfmt));
     std::shared_ptr<MemPD> usr_bottom_data_mpd(new MemPD({ { bottom_tz }, mpcsn, mfmt_nchw },
       cpu_engine));
     std::shared_ptr<MemPD> usr_top_data_mpd(
       new MemPD({ { top_tz }, mpcsn, mfmt_nchw }, cpu_engine));

     pooling_forward::desc poolingFwdInference_desc(prop_kind::forward_scoring,
        pooling_algorithm_, *init_fwd_bottom_md, *init_fwd_top_md
       , { sh, sw }, { kh, kw }, { pt, pl }, { pb, pr }, padding_kind::zero);
     pooling_forward::desc poolingFwdTraining_desc(prop_kind::forward_training
       , pooling_algorithm_, *init_fwd_bottom_md, *init_fwd_top_md
       , { sh, sw }, { kh, kw }, { pt, pl }, { pb, pr }, padding_kind::zero);
     poolingFwdInference_pd.reset(new pooling_forward::primitive_desc(
       poolingFwdInference_desc, cpu_engine));
     CHECK(poolingFwdInference_pd);
     poolingFwdTraining_pd.reset(new pooling_forward::primitive_desc(
       poolingFwdTraining_desc, cpu_engine));
     CHECK(poolingFwdTraining_pd);

     // ---- Initialize remaining memory descriptors -------------
     std::shared_ptr<MemPD> prv_fwd_bottom_data_mpd;
     std::shared_ptr<MemPD> prv_fwd_top_data_mpd;
     if (bottom_data_is_prv) {
       prv_fwd_bottom_data_mpd.reset(new MemPD(*init_fwd_bottom_md, cpu_engine));
       prv_fwd_top_data_mpd.reset(new MemPD(*init_fwd_top_md, cpu_engine));
     }

     fwd_bottom_data.reset(new MKLDNNData<Dtype>(usr_bottom_data_mpd, prv_fwd_bottom_data_mpd));
     fwd_bottom_data->name = "fwd_bottom_data   @ " + this->getName();

     fwd_top_data.reset(new MKLDNNData<Dtype>(usr_top_data_mpd, prv_fwd_top_data_mpd));
     fwd_top_data->name = "fwd_top_data   @ " + this->getName();
     // ---- Initialize pooling primitive descriptor -------------
     if (pooling_algorithm_ != algorithm::pooling_avg) {
       indices_pd.reset(
         new memory::primitive_desc(poolingFwdTraining_pd->workspace_primitive_desc()));
       indices_memory.reset(new memory(*indices_pd));
     }
  }
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    if (!init_mkldnn_) {
      using namespace mshadow;
      using namespace mshadow::expr;
      CHECK_EQ(in_data.size(), 1);
      CHECK_EQ(out_data.size(), 1);
      Stream<xpu> *s = ctx.get_stream<xpu>();
      if (param_.kernel.ndim() >= 3) {
        LOG(FATAL) << "Not implmented";
      }
      Tensor<xpu, 4, Dtype> data = mkl_experimental_direct_get<xpu, 4, Dtype>(
        in_data[pool_enum::kData], s);
      Tensor<xpu, 4, Dtype> out = mkl_experimental_direct_get<xpu, 4, Dtype>(
        out_data[pool_enum::kOut], s);
        LayerSetUp(data, out);
        init_mkldnn_ = true;
    
      if (poolingFwdInference_pd == NULL)
        InitPoolingFwd(in_data);
      // ---  init primitive and prv_memory descriptors ----------------------
      fwd_input_primitive = fwd_bottom_data->get_converted_prv(data.dptr_, false,
        in_data[pool_enum::kData]);
      fwd_output_memory = fwd_top_data->create_output_memory(out.dptr_, out_data[pool_enum::kOut],
        fwd_top_data);
      if (ctx.is_train && pooling_algorithm_ != algorithm::pooling_avg) {
        poolingFwd.reset(new pooling_forward(*poolingFwdTraining_pd, *fwd_input_primitive,
          *fwd_output_memory, *indices_memory));
      } else {
        poolingFwd.reset(new pooling_forward(*poolingFwdInference_pd, *fwd_input_primitive,
          *fwd_output_memory));
      }
    } else {
      fwd_bottom_data->sync_converted_prv(false,
        in_data[pool_enum::kData]);
      fwd_top_data->sync_output_memory(out_data[pool_enum::kOut],
        fwd_top_data);
    }
    poolingFwd.submit();
  }
  void InitPoolingBwd(const std::vector<TBlob> &out_grad) {
    int32_t n = this->num_;
    int32_t c = this->channels_;
    int32_t ih = this->height_;
    int32_t iw = this->width_;
    int32_t oh = this->height_out_;
    int32_t ow = this->width_out_;

    int32_t kh = this->kernel_h_;
    int32_t kw = this->kernel_w_;

    int32_t sh = this->stride_h_;
    int32_t sw = this->stride_w_;

    int32_t pt = this->pad_t_;
    int32_t pb = this->pad_b_;

    int32_t pr = this->pad_r_;
    int32_t pl = this->pad_l_;

    void * top_diff_data =
      const_cast<Dtype*>(mkl_prv_data<Dtype>(out_grad[pool_enum::kOut]));
    bool top_diff_is_prv = (top_diff_data != NULL);

    mkldnn::engine cpu_engine = CpuEngine::Instance().get_engine();
    memory::data_type mpcsn = memory::data_type::f32;
    memory::dims bottom_tz = { n, c, ih, iw };
    memory::dims top_tz = { n, c, oh, ow };
    memory::format mfmt_nchw = memory::format::nchw;

    // ---- Initialize memory descriptors -------------
    typedef typename memory::primitive_desc MemPD;

    memory::format bwd_cmfmt = mfmt_nchw;
    if (top_diff_is_prv) {
      std::shared_ptr<MKLDNNMemoryDescriptor<Dtype> > mem_descr
        = get_mkldnn_prv_descriptor<Dtype>(out_grad[pool_enum::kOut].Mkl_mem_);
      bwd_cmfmt = static_cast<memory::format>(mem_descr->prv_memory_pd()->desc().data.format);
    }

    std::shared_ptr<memory::desc> init_bwd_bottom_md(
      new memory::desc({ bottom_tz }, mpcsn, bwd_cmfmt));
    std::shared_ptr<memory::desc> init_bwd_top_md(
      new memory::desc({ top_tz }, mpcsn, bwd_cmfmt));
    std::shared_ptr<MemPD> usr_bottom_data_mpd(
      new MemPD({ { bottom_tz }, mpcsn, mfmt_nchw }, cpu_engine));
    std::shared_ptr<MemPD> usr_top_data_mpd(
      new MemPD({ { top_tz }, mpcsn, mfmt_nchw }, cpu_engine));
    // ---- Initialize pooling primitive descriptor -------------
    pooling_backward::desc poolingBwd_desc(this->pooling_algorithm_, *init_bwd_bottom_md,
      *init_bwd_top_md
      , { sh, sw }, { kh, kw }, { pt, pl }, { pb, pr }, padding_kind::zero);
    poolingBwd_pd.reset(new pooling_backward::primitive_desc(poolingBwd_desc,
      cpu_engine, *poolingFwdTraining_pd));
    CHECK(poolingBwd_pd);
    // ---- Initialize remaining memory descriptors -------------
    std::shared_ptr<MemPD> prv_bwd_bottom_diff_mpd, prv_bwd_top_diff_mpd;
    if (top_diff_is_prv) {
      prv_bwd_bottom_diff_mpd.reset(new MemPD(*init_bwd_bottom_md, cpu_engine));
      prv_bwd_top_diff_mpd.reset(new MemPD(*init_bwd_top_md, cpu_engine));
    }
    bwd_bottom_diff.reset(new MKLDNNData<Dtype>(usr_bottom_data_mpd, prv_bwd_bottom_diff_mpd));
    bwd_bottom_diff->name = "bwd_bottom_diff   @ " + getName();
    bwd_top_diff.reset(new MKLDNNData<Dtype>(usr_top_data_mpd, prv_bwd_top_diff_mpd));
    bwd_top_diff->name = "bwd_top_diff      @ " + getName();
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
    Tensor<xpu, 4, Dtype> grad = mkl_experimental_direct_get<xpu, 4, Dtype>(
      out_grad[pool_enum::kOut], s);
    Tensor<xpu, 4, Dtype> input_grad = mkl_experimental_direct_get<xpu, 4, Dtype>(
      in_grad[pool_enum::kData], s);
    if (poolingBwd_pd == NULL)
      InitPoolingBwd(out_grad);
    std::shared_ptr<memory> diff_dst_memory, diff_src_memory;
    diff_dst_memory = bwd_top_diff->get_converted_prv(grad.dptr_, true, out_grad[pool_enum::kOut]);
    diff_src_memory = bwd_bottom_diff->create_output_memory(input_grad.dptr_,
      in_grad[pool_enum::kData], bwd_bottom_diff);
    MKLDNNPrimitive<Dtype>  poolingBwd;
    if (param_.pool_type != pool_enum::kAvgPooling) {
      poolingBwd.reset(new pooling_backward(*poolingBwd_pd, *diff_dst_memory,
        *indices_memory, *diff_src_memory));
    } else {
      poolingBwd.reset(new pooling_backward(*poolingBwd_pd, *diff_dst_memory,
        *diff_src_memory));
    }
    poolingBwd.submit();
  }

 private:
  PoolingParam param_;
  int32_t num_, channels_, width_, height_, width_out_, height_out_;
  int32_t kernel_w_, kernel_h_, stride_w_, stride_h_;
  int32_t  pad_t_, pad_b_, pad_l_, pad_r_;
  bool global_pooling_;
  std::shared_ptr<pooling_forward::primitive_desc> poolingFwdInference_pd;
  std::shared_ptr<pooling_forward::primitive_desc> poolingFwdTraining_pd;
  std::shared_ptr<pooling_backward::primitive_desc> poolingBwd_pd;

  std::shared_ptr<MKLDNNData<Dtype> > fwd_bottom_data, fwd_top_data,
    bwd_top_diff, bwd_bottom_diff;
  std::shared_ptr<memory::primitive_desc> indices_pd;
  std::shared_ptr<memory> indices_memory;
  bool init_mkldnn_;
  algorithm pooling_algorithm_;
  MKLDNNPrimitive<Dtype> poolingFwd;
  std::shared_ptr<memory> fwd_input_primitive, fwd_output_memory;
};  // class MKLDNNPoolingOp
#endif

#if MXNET_USE_MKL2017 == 1
template<typename xpu, typename Dtype>
class MKLPoolingOp : public Operator {
 public:
  static std::string getName() {
    return "MKLPoolingOp";
  }
  explicit MKLPoolingOp(PoolingParam p) {
    poolingFwd = static_cast<dnnPrimitive_t>(NULL);
    poolingBwd = static_cast<dnnPrimitive_t>(NULL);
    max_idx_data = static_cast<Dtype*>(NULL);
    fwd_top_data = MKLData<Dtype>::create();
    fwd_bottom_data = MKLData<Dtype>::create();
    bwd_top_diff = MKLData<Dtype>::create();
    bwd_bottom_diff = MKLData<Dtype>::create();
    this->param_ = p;
    init_mkldnn_ = false;
  }
  virtual ~MKLPoolingOp() {
    if (poolingFwd != NULL) {
      dnnDelete<Dtype>(poolingFwd);
      poolingFwd = NULL;
    }
    if (poolingBwd != NULL) {
      dnnDelete<Dtype>(poolingBwd);
      poolingBwd = NULL;
    }
    if (max_idx_data != NULL) {
      dnnReleaseBuffer<Dtype>(max_idx_data);
      max_idx_data = NULL;
    }
  }

 private:
  void LayerSetUp(const mshadow::Tensor<xpu, 4, Dtype> &data,
                  const mshadow::Tensor<xpu, 4, Dtype> &out) {
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
    Tensor<xpu, 4, Dtype> data = mkl_experimental_direct_get<xpu, 4, Dtype>(
      in_data[pool_enum::kData], s);
    Tensor<xpu, 4, Dtype> out = mkl_experimental_direct_get<xpu, 4, Dtype>(
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
          reinterpret_cast<void *>(mkl_prv_data<Dtype>(in_data[pool_enum::kData]));
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
        status = dnnPoolingCreateForward<Dtype>(&poolingFwd, NULL,
                                                algorithm, fwd_bottom_data->layout_usr,
                                                kernel_size, kernel_stride,
                                                src_offset, border_type);
      CHECK_EQ(status, E_SUCCESS);
      // Now create poolingBwd
      status = dnnPoolingCreateBackward<Dtype>(&poolingBwd, NULL,
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
          std::shared_ptr<MKLData<Dtype> > mem_descr
            = std::static_pointer_cast<MKLData<Dtype>>(bottom_prv_descriptor);
          CHECK(mem_descr != nullptr);
          fwd_bottom_data = mem_descr;

          status = dnnPoolingCreateForward<Dtype>(&poolingFwd, NULL,
                                                  algorithm, fwd_bottom_data->layout_int,
                                                  kernel_size, kernel_stride,
                                                  src_offset, border_type);
          CHECK_EQ(status, E_SUCCESS);
          fwd_top_data->create_internal_layout(poolingFwd, dnnResourceDst);

          // Now create poolingBwd
          status = dnnPoolingCreateBackward<Dtype>(&poolingBwd, NULL,
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
      status = dnnLayoutCreateFromPrimitive<Dtype>(
          &max_idx_datal, poolingFwd, dnnResourceWorkspace);
      CHECK_EQ(status, E_SUCCESS);
      status = dnnAllocateBuffer<Dtype>(reinterpret_cast<void**>(&max_idx_data), max_idx_datal);
      CHECK_EQ(status, E_SUCCESS);
#if MKL_EXPERIMENTAL == 0
      fwd_bottom_data->create_internal_layout(poolingFwd, dnnResourceSrc);
      fwd_top_data->create_internal_layout(poolingFwd, dnnResourceDst);
      bwd_top_diff->create_internal_layout(poolingBwd, dnnResourceDiffDst);
      bwd_bottom_diff->create_internal_layout(poolingBwd, dnnResourceDiffSrc);
#endif
      dnnLayoutDelete<Dtype>(max_idx_datal);
      first_pass = false;
    }
    pooling_res[dnnResourceSrc] = bottom_data;
    pooling_res[dnnResourceWorkspace] = max_idx_data;

    pooling_res[dnnResourceDst] = fwd_top_data->get_output_ptr(
      out.dptr_, fwd_top_data, out_data[pool_enum::kOut]);
    status = dnnExecute<Dtype>(poolingFwd, pooling_res);
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
    Tensor<xpu, 4, Dtype> grad = mkl_experimental_direct_get<xpu, 4, Dtype>(
      out_grad[pool_enum::kOut], s);
    Tensor<xpu, 4, Dtype> input_grad = mkl_experimental_direct_get<xpu, 4, Dtype>(
      in_grad[pool_enum::kData], s);
    dnnError_t e;
    void* pooling_res[dnnResourceNumber];
    pooling_res[dnnResourceWorkspace] = reinterpret_cast<void *>(max_idx_data);

    pooling_res[dnnResourceDiffDst] =
      bwd_top_diff->get_converted_prv(grad.dptr_, true, out_grad[pool_enum::kOut]);

    pooling_res[dnnResourceDiffSrc] = bwd_bottom_diff->get_output_ptr(
      input_grad.dptr_, bwd_bottom_diff, in_grad[pool_enum::kData]);
    e = dnnExecute<Dtype>(poolingBwd, pooling_res);
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
  Dtype *max_idx_data;

  std::shared_ptr<MKLData<Dtype> > fwd_top_data;
  std::shared_ptr<MKLData<Dtype> > fwd_bottom_data;
  std::shared_ptr<MKLData<Dtype> > bwd_top_diff;
  std::shared_ptr<MKLData<Dtype> > bwd_bottom_diff;
  bool init_mkldnn_;
};  // class MKLPoolingOp
#endif

}   // namespace op
}   // namespace mxnet

#endif  // MXNET_OPERATOR_MKL_MKL_POOLING_INL_H_
