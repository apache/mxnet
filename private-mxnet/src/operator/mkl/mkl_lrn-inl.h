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
* \file mkl_lrn-inl.h
* \brief
* \author young.jin.kim@intel.com
*         ashok.emani@intel.com
*         deepthi.karkada@intel.com
*         lingyan.guo@intel.com
*         zhenlin.luo@intel.com
*
*******************************************************************************/
#ifndef MXNET_OPERATOR_MKL_MKL_LRN_INL_H_
#define MXNET_OPERATOR_MKL_MKL_LRN_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#if MXNET_USE_MKL2017 == 1
#include "../operator_common.h"
#include "../mshadow_op.h"
#include "./mkl_util-inl.h"
#endif   //MXNET_USE_MKL2017

namespace mxnet {
namespace op {
template<typename xpu, typename Dtype>
#if MXNET_USE_MKLDNN == 1
class MKLDNNLRNOp : public Operator, public MKLDNNLayer<Dtype> {
 public:
  std::string getName() {
    return "MKLDNNLRNOp";
  }

  explicit MKLDNNLRNOp(LRNParam param) :
    MKLDNNLayer<Dtype>()
    , fwd_bottom_data(NULL), fwd_top_data(NULL)
    , bwd_bottom_diff(NULL), bwd_top_diff(NULL)
    , lrnFwdInference_pd(NULL), lrnBwd_pd(NULL)
    , alpha_(0), beta_(0), k_(0)
    , size_(0), num_(0), width_(0), height_(0), channels_(0) {
    lrn_algorithm = algorithm::lrn_across_channels;
    this->param_ = param;
    init_mkldnn_ = false;
  }

  virtual ~MKLDNNLRNOp() {
  }

 private:
  void LayerSetup(const mshadow::Tensor<xpu, 4, Dtype> &data) {
    size_ = param_.nsize;
    CHECK_EQ(size_ % 2, 1) << "LRN only supports odd values for local size";
    alpha_ = param_.alpha;
    beta_ = param_.beta;
    k_ = param_.knorm;
    channels_ = data.shape_[1];
    height_ = data.shape_[2];
    width_ = data.shape_[3];
    num_ = data.shape_[0];
  }
  void InitLRNFwd(const std::vector<TBlob> &in_data) {
    int32_t n = this->num_;
    int32_t iw = this->width_;
    int32_t ih = this->height_;
    int32_t ic = this->channels_;

    bool bottom_data_is_prv =
      (const_cast<Dtype*>(mkl_prv_data<Dtype>(in_data[lrn_enum::kData])) != NULL);
    mkldnn::engine cpu_engine = CpuEngine::Instance().get_engine();
    memory::data_type mpcsn = memory::data_type::f32;
    // ---- Initialize memory descriptors -------------
    memory::dims tz = { n, ic, ih, iw };
    std::shared_ptr<memory::desc> bottom_md, top_md;
    std::shared_ptr<memory::primitive_desc> usr_mpd, prv_mpd;
    if (bottom_data_is_prv) {
      std::shared_ptr<MKLDNNData<Dtype> > mem_descr
        = get_mkldnn_prv_descriptor<Dtype>(in_data[lrn_enum::kData].Mkl_mem_);
      bottom_md.reset(new memory::desc(mem_descr->prv_memory_pd()->desc()));
      usr_mpd = mem_descr->usr_memory_pd();
      prv_mpd = mem_descr->prv_memory_pd();
    } else {
      bottom_md.reset(new memory::desc({ tz }, mpcsn, memory::format::nchw));
      usr_mpd.reset(new memory::primitive_desc(*bottom_md, cpu_engine));
    }
    top_md = bottom_md;

    // ---- Initialize LRN primitive descriptor -------------
    lrn_forward::desc lrnFwdInference_desc(prop_kind::forward_scoring, lrn_algorithm, *bottom_md,
      size_, alpha_, beta_);

    lrnFwdInference_pd.reset(new lrn_forward::primitive_desc(lrnFwdInference_desc, cpu_engine));
    CHECK(lrnFwdInference_pd);
    lrn_forward::desc lrnFwdTraining_desc(prop_kind::forward_training, lrn_algorithm, *bottom_md,
      size_, alpha_, beta_);
    lrnFwdTraining_pd.reset(new lrn_forward::primitive_desc(lrnFwdTraining_desc, cpu_engine));
    CHECK(lrnFwdTraining_pd);
    typedef typename memory::primitive_desc MemPD;
    std::shared_ptr<MemPD> prv_fwd_bottom_data_memory_pd(
      new MemPD(lrnFwdTraining_pd->src_primitive_desc()));
    std::shared_ptr<MemPD> prv_fwd_top_data_memory_pd(
      new MemPD(lrnFwdTraining_pd->dst_primitive_desc()));

    // ---- Create usr memory primitive descriptors -------------
    memory::format mfmt_nchw = memory::format::nchw;
    memory::format scratch_mfmt = memory::format::nchw;

    std::shared_ptr<MemPD> usr_data_memory_pd(new MemPD({ { tz }, mpcsn, mfmt_nchw }, cpu_engine));

    // ---  init primitive and prv_memory descriptors ----------------------
    fwd_bottom_data.reset(new MKLDNNData<Dtype>(usr_data_memory_pd, prv_fwd_bottom_data_memory_pd));
    fwd_bottom_data->name = "fwd_bottom_data   @ " + this->getName();
    fwd_top_data.reset(new MKLDNNData<Dtype>(usr_mpd, prv_fwd_top_data_memory_pd));
    fwd_top_data->name = "fwd_top_data   @ " + this->getName();
  }

 public:
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 2);
    CHECK_EQ(param_.nsize % 2, 1) << "LRN only supports odd values for local_size";
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, Dtype> data = mkl_experimental_direct_get<xpu, 4, Dtype>(
      in_data[lrn_enum::kData], s);
    Tensor<xpu, 4, Dtype> out = mkl_experimental_direct_get<xpu, 4, Dtype>(
      out_data[lrn_enum::kOut], s);
    if (!init_mkldnn_) {
      LayerSetup(data);
      init_mkldnn_ = true;
    }
    if (lrnFwdInference_pd == NULL) {
      InitLRNFwd(in_data);
    }
    MKLDNNPrimitive<Dtype> lrnFwd;
    fwd_bottom_data_primitive =
      fwd_bottom_data->get_converted_prv(data.dptr_, false, in_data[lrn_enum::kData]);
    std::shared_ptr<memory> fwd_top_data_memory = fwd_top_data->create_output_memory(
      out.dptr_, out_data[lrn_enum::kOut], fwd_top_data);
    if (ctx.is_train) {
      memory::primitive_desc scratch_mpd(lrnFwdTraining_pd->workspace_primitive_desc());
      scratch_memory.reset(new memory(scratch_mpd));
      lrnFwd.reset(new lrn_forward(*lrnFwdTraining_pd, *fwd_bottom_data_primitive, *scratch_memory,
        *fwd_top_data_memory));
    } else {
      lrnFwd.reset(new lrn_forward(*lrnFwdInference_pd, *fwd_bottom_data_primitive,
        *fwd_top_data_memory));
    }
    lrnFwd.submit();
  }
  void InitLRNBwd(const std::vector<TBlob> &out_grad) {
    int32_t n = this->num_;
    int32_t iw = this->width_;
    int32_t ih = this->height_;
    int32_t ic = this->channels_;
    void * top_diff_data =
      const_cast<Dtype*>(mkl_prv_data<Dtype>(out_grad[lrn_enum::kOut]));
    bool top_diff_is_prv = (top_diff_data != NULL);
    mkldnn::engine cpu_engine = CpuEngine::Instance().get_engine();
    memory::data_type mpcsn = memory::data_type::f32;
    // ---- Initialize memory descriptors -------------
    memory::dims tz = { n, ic, ih, iw };
    std::shared_ptr<memory::desc> bottom_diff_md, top_diff_md;
    std::shared_ptr<memory::primitive_desc> usr_diff_mpd, prv_diff_mpd;
    if (top_diff_is_prv) {
      std::shared_ptr<MKLDNNMemoryDescriptor<Dtype> > mem_descr
        = get_mkldnn_prv_descriptor<Dtype>(out_grad[lrn_enum::kOut].Mkl_mem_);
      top_diff_md.reset(new memory::desc(mem_descr->prv_memory_pd()->desc()));
      usr_diff_mpd = mem_descr->usr_memory_pd();
      prv_diff_mpd = mem_descr->prv_memory_pd();
    } else {
      top_diff_md.reset(new memory::desc({ tz }, mpcsn, memory::format::nchw));
      usr_diff_mpd.reset(new memory::primitive_desc(*top_diff_md, cpu_engine));
    }
    bottom_diff_md = top_diff_md;

    // ---- Initialize LRN primitive descriptor -------------
    lrn_backward::desc lrnBwd_desc(lrn_algorithm, *bottom_diff_md, *top_diff_md,
      size_, alpha_, beta_);
    lrnBwd_pd.reset(new lrn_backward::primitive_desc(lrnBwd_desc,
      cpu_engine, *lrnFwdTraining_pd));

    CHECK(lrnBwd_pd);
    // ---- Create priv memory primitive descriptors stored as class members -------------
    typedef typename memory::primitive_desc MemPD;
    std::shared_ptr<MemPD> prv_bwd_bottom_diff_memory_pd(
      new MemPD(lrnBwd_pd->diff_src_primitive_desc()));
    std::shared_ptr<MemPD> prv_bwd_top_diff_memory_pd(
      new MemPD(lrnBwd_pd->diff_dst_primitive_desc()));

    // ---- Create usr memory primitive descriptors -------------
    memory::format mfmt_nchw = memory::format::nchw;
    memory::format scratch_mfmt = memory::format::nchw;

    std::shared_ptr<MemPD> usr_data_memory_pd(new MemPD({ { tz }, mpcsn, mfmt_nchw }, cpu_engine));

    // ---  init primitive and prv_memory descriptors ----------------------
    bwd_bottom_diff.reset(new MKLDNNData<Dtype>(usr_data_memory_pd, prv_bwd_bottom_diff_memory_pd));
    bwd_bottom_diff->name = "bwd_bottom_diff_data   @ " + this->getName();
    bwd_top_diff.reset(new MKLDNNData<Dtype>(usr_diff_mpd, prv_bwd_top_diff_memory_pd));
    bwd_top_diff->name = "bwd_top_diff_data   @ " + this->getName();
  }
  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1);
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 2);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, Dtype> grad = mkl_experimental_direct_get<xpu, 4, Dtype>(
      out_grad[lrn_enum::kOut], s);
    Tensor<xpu, 4, Dtype> data = mkl_experimental_direct_get<xpu, 4, Dtype>(
      in_data[lrn_enum::kData], s);
    Tensor<xpu, 4, Dtype> grad_in = mkl_experimental_direct_get<xpu, 4, Dtype>(
      in_grad[lrn_enum::kData], s);
    if (lrnBwd_pd == NULL)
      InitLRNBwd(out_grad);
    MKLDNNPrimitive<Dtype> lrnBwd;
    std::shared_ptr<memory> bwd_bottom_diff_memory;
    std::shared_ptr<primitive> bwd_top_diff_primitive;
    bwd_top_diff_primitive =
      bwd_top_diff->get_converted_prv(grad.dptr_, false, out_grad[lrn_enum::kOut]);
    bwd_bottom_diff_memory = bwd_bottom_diff->create_output_memory(grad_in.dptr_,
      in_grad[lrn_enum::kData], bwd_bottom_diff);
    lrnBwd.reset(new lrn_backward(*lrnBwd_pd, *fwd_bottom_data_primitive,
      *bwd_top_diff_primitive, *scratch_memory, *bwd_bottom_diff_memory));
    lrnBwd.submit();
  }

 private:
  LRNParam param_;
  bool init_mkldnn_;
  std::shared_ptr<MKLDNNData<Dtype> > fwd_top_data, fwd_bottom_data,
    bwd_top_diff, bwd_bottom_diff;
  std::shared_ptr<lrn_forward::primitive_desc> lrnFwdInference_pd;
  std::shared_ptr<lrn_forward::primitive_desc> lrnFwdTraining_pd;
  std::shared_ptr<lrn_backward::primitive_desc> lrnBwd_pd;
  std::shared_ptr<primitive> fwd_bottom_data_primitive;
  std::shared_ptr<memory> scratch_memory;
  Dtype alpha_, beta_, k_;
  int size_, num_, width_, height_, channels_;
  algorithm  lrn_algorithm;
};  // class MKLDNNLRNOp
#endif

#if MXNET_USE_MKL2017 == 1
class MKLLRNOp : public Operator {
 public:
  static std::string getName() {
    return "MKLLRNOp";
  }

  explicit MKLLRNOp(LRNParam param) :
    lrnFwd(static_cast<dnnPrimitive_t>(NULL)),
    lrnBwd(static_cast<dnnPrimitive_t>(NULL)),
    lrn_buffer_(NULL) {
    this->param_ = param;
    fwd_top_data_ = MKLData<Dtype>::create();
    fwd_bottom_data_ = MKLData<Dtype>::create();
    bwd_top_diff_ = MKLData<Dtype>::create();
    bwd_bottom_diff_ = MKLData<Dtype>::create();
    init_mkldnn_ = false;
  }

  virtual ~MKLLRNOp() {
    if (lrnFwd != NULL) {
      dnnDelete<Dtype>(lrnFwd);
      lrnFwd = NULL;
    }
    if (lrnBwd != NULL) {
      dnnDelete<Dtype>(lrnBwd);
      lrnBwd = NULL;
    }
    dnnReleaseBuffer<Dtype>(lrn_buffer_);
  }

 private:
  void LayerSetup(const mshadow::Tensor<xpu, 4, Dtype> &data,
                  const mshadow::Tensor<xpu, 4, Dtype> &out) {
    size_ = param_.nsize;
    CHECK_EQ(size_ % 2, 1) << "LRN only supports odd values for local size";

    alpha_ = param_.alpha;
    beta_ = param_.beta;
    k_ = param_.knorm;
    size_t dim = 4, sizes[4], strides[4];
    channels_ = data.shape_[1];
    height_ = data.shape_[2];
    width_ = data.shape_[3];
    num_ = data.shape_[0];
    sizes[0] = width_;
    sizes[1] = height_;
    sizes[2] = channels_;
    sizes[3] = num_;

    strides[0] = 1;
    strides[1] = sizes[0];
    strides[2] = sizes[0] * sizes[1];
    strides[3] = sizes[0] * sizes[1] * sizes[2];

    fwd_bottom_data_->name = "fwd_bottom_data_   @ " + getName();
    fwd_top_data_->name = "fwd_top_data_      @ " + getName();
    bwd_top_diff_->name = "bwd_top_diff_      @ " + getName();
    bwd_bottom_diff_->name = "bwd_bottom_diff_   @ " + getName();

    fwd_bottom_data_->create_user_layout(dim, sizes, strides);
    fwd_top_data_->create_user_layout(dim, sizes, strides);
    bwd_bottom_diff_->create_user_layout(dim, sizes, strides);
    bwd_top_diff_->create_user_layout(dim, sizes, strides);
  }

 public:
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1U);
    CHECK_EQ(out_data.size(), 2U);
    CHECK_EQ(param_.nsize % 2, 1U) << "LRN only supports odd values for local_size";
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, Dtype> data = mkl_experimental_direct_get<xpu, 4, Dtype>(
      in_data[lrn_enum::kData], s);
    Tensor<xpu, 4, Dtype> out = mkl_experimental_direct_get<xpu, 4, Dtype>(
      out_data[lrn_enum::kOut], s);
    if (!init_mkldnn_) {
      LayerSetup(data, out);
      init_mkldnn_ = true;
    }

    const void* bottom_data = NULL;
#if MKL_EXPERIMENTAL == 1
    bottom_data =
          reinterpret_cast<void*>(mkl_prv_data<Dtype>(in_data[lrn_enum::kData]));
#endif
#if MKL_EXPERIMENTAL == 1
    if (NULL != bottom_data) {
      if (lrnFwd == NULL) {
        std::shared_ptr<MKLMemHolder> bottom_data_mem =
          in_data[lrn_enum::kData].Mkl_mem_;
        std::shared_ptr<PrvMemDescr> bottom_prv_descriptor =
          bottom_data_mem->get_prv_descriptor();
        CHECK_EQ(bottom_prv_descriptor->get_descr_type(),
            PrvMemDescr::PRV_DESCR_MKL2017);
        std::shared_ptr<MKLData<Dtype> > mem_descr
          = std::static_pointer_cast<MKLData<Dtype>>(bottom_prv_descriptor);
        CHECK(mem_descr != nullptr);
        fwd_bottom_data_ = mem_descr;

        dnnError_t e;
        dnnLayout_t lrn_buffer_l = NULL;

        e = dnnLRNCreateForward<Dtype>(&lrnFwd, NULL, fwd_bottom_data_->layout_int,
                                       size_, alpha_, beta_, k_);
        CHECK_EQ(e, E_SUCCESS);

        fwd_top_data_->create_internal_layout(lrnFwd, dnnResourceDst);

        e = dnnLRNCreateBackward<Dtype>(&lrnBwd, NULL,
                                        fwd_bottom_data_->layout_int, fwd_bottom_data_->layout_int,
                                        size_, alpha_, beta_, k_);
        CHECK_EQ(e, E_SUCCESS);

        e = dnnLayoutCreateFromPrimitive<Dtype>(
              &lrn_buffer_l, lrnFwd, dnnResourceWorkspace);
        CHECK_EQ(e, E_SUCCESS);
        e = dnnAllocateBuffer<Dtype>(
              reinterpret_cast<void **>(&lrn_buffer_), lrn_buffer_l);
        CHECK_EQ(e, E_SUCCESS);
        dnnLayoutDelete<Dtype>(lrn_buffer_l);

        bwd_top_diff_->create_internal_layout(lrnBwd, dnnResourceDiffDst);
        bwd_bottom_diff_->create_internal_layout(lrnBwd, dnnResourceDiffSrc);
      }
    }
#endif
    if (bottom_data == NULL) {
      if (lrnFwd == NULL) {
        dnnError_t e;
        dnnLayout_t lrn_buffer_l = NULL;
        e = dnnLRNCreateForward<Dtype>(&lrnFwd, NULL, fwd_bottom_data_->layout_usr,
                                       size_, alpha_, beta_, k_);
        CHECK_EQ(e, E_SUCCESS);

        e = dnnLayoutCreateFromPrimitive<Dtype>(
              &lrn_buffer_l, lrnFwd, dnnResourceWorkspace);
        CHECK_EQ(e, E_SUCCESS);
        e = dnnAllocateBuffer<Dtype>(
              reinterpret_cast<void **>(&lrn_buffer_), lrn_buffer_l);
        CHECK_EQ(e, E_SUCCESS);
        dnnLayoutDelete<Dtype>(lrn_buffer_l);

        e = dnnLRNCreateBackward<Dtype>(&lrnBwd, NULL,
                                        fwd_bottom_data_->layout_usr, fwd_bottom_data_->layout_usr,
                                        size_, alpha_, beta_, k_);
        CHECK_EQ(e, E_SUCCESS);
      }
      bottom_data = data.dptr_;
    }

    dnnError_t e;
    void* lrn_res[dnnResourceNumber];
    lrn_res[dnnResourceSrc] = const_cast<void*>(bottom_data);

    lrn_res[dnnResourceDst] = fwd_top_data_->get_output_ptr(
      out.dptr_, fwd_top_data_, out_data[lrn_enum::kOut]);
    lrn_res[dnnResourceWorkspace] = lrn_buffer_;
    e = dnnExecute<Dtype>(lrnFwd, lrn_res);
    CHECK_EQ(e, E_SUCCESS);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1);
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 2);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, Dtype> grad = mkl_experimental_direct_get<xpu, 4, Dtype>(
      out_grad[lrn_enum::kOut], s);
    Tensor<xpu, 4, Dtype> data = mkl_experimental_direct_get<xpu, 4, Dtype>(
      in_data[lrn_enum::kData], s);
    Tensor<xpu, 4, Dtype> grad_in = mkl_experimental_direct_get<xpu, 4, Dtype>(
      in_grad[lrn_enum::kData], s);
    dnnError_t e;
    void* lrn_res[dnnResourceNumber];
    lrn_res[dnnResourceDiffDst] =
      bwd_top_diff_->get_converted_prv(grad.dptr_, true, out_grad[lrn_enum::kOut]);
    lrn_res[dnnResourceWorkspace] = lrn_buffer_;
    lrn_res[dnnResourceSrc] =
      fwd_bottom_data_->get_converted_prv(data.dptr_, false, in_data[lrn_enum::kData]);

    lrn_res[dnnResourceDiffSrc] = bwd_bottom_diff_->get_output_ptr(
      grad_in.dptr_, bwd_bottom_diff_, in_grad[lrn_enum::kData]);
    e = dnnExecute<Dtype>(lrnBwd, lrn_res);
    CHECK_EQ(e, E_SUCCESS);
  }

 private:
  LRNParam param_;
  int size_;
  int pre_pad_;
  Dtype alpha_;
  Dtype beta_;
  Dtype k_;
  int num_;
  int channels_;
  int height_;
  int width_;
  bool init_mkldnn_;

 private:
  dnnPrimitive_t lrnFwd, lrnBwd;
  std::shared_ptr<MKLData<Dtype> > fwd_top_data_;
  std::shared_ptr<MKLData<Dtype> > fwd_bottom_data_;

  std::shared_ptr<MKLData<Dtype> > bwd_top_diff_;
  std::shared_ptr<MKLData<Dtype> > bwd_bottom_diff_;

  Dtype *lrn_buffer_;
};  // class MKLLRNOp
#endif

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MKL_MKL_LRN_INL_H_