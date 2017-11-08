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
* \file mkldnn_relu-inl.h
* \brief
* \author young.jin.kim@intel.com
*         ashok.emani@intel.com
*         deepthi.karkada@intel.com
*         louis.feng@intel.com
*         adam.d.straw@intel.com
*
*******************************************************************************/
#ifndef MXNET_OPERATOR_MKL_MKLDNN_RELU_INL_H_
#define MXNET_OPERATOR_MKL_MKLDNN_RELU_INL_H_


#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../operator_common.h"
#include "./mkl_util-inl.h"

namespace mxnet {
namespace op {

template<typename xpu, typename Dtype>
class MKLDNNReluOp : public Operator, public MKLDNNLayer<Dtype> {
 public:
  std::string getName() {
    std::string name = "MKLDNNReluOp";
    return name;
  }
  MKLDNNReluOp() : MKLDNNLayer<Dtype>()
    , fwd_top_data(NULL), fwd_bottom_data(NULL), prv_mpd(NULL)
    , num_(0), width_(0), height_(0), channels_(0) {
    init_mkldnn_ = false;
  }
  ~MKLDNNReluOp() {
  }

 private:
  void LayerSetup(const mshadow::Tensor<xpu, 4, Dtype> &data) {
    this->width_ = data.shape_[3];
    this->height_ = data.shape_[2];
    this->channels_ = data.shape_[1];
    this->num_ = data.shape_[0];
  }
  void InitReLUFwd(const std::vector<TBlob> &in_data) {
    void * bottom_data = reinterpret_cast<void *>(mkl_prv_data<Dtype>(in_data[activation::kData]));
    std::shared_ptr<MKLDNNMemoryDescriptor<Dtype> > bottom_prv_descriptor
      = get_mkldnn_prv_descriptor<Dtype>(in_data[activation::kData]);
    std::shared_ptr<memory::desc> bottom_data_md, top_data_md;
    std::shared_ptr<memory::primitive_desc> usr_mpd(NULL);

    int32_t n = this->num_;
    int32_t iw = this->width_;
    int32_t ih = this->height_;
    int32_t ic = this->channels_;
    Dtype negative_slope = 0;
    mkldnn::engine cpu_engine = CpuEngine::Instance().get_engine();
    memory::data_type mpcsn = memory::data_type::f32;

    if (bottom_data != NULL) {
      bottom_data_md.reset(new memory::desc(bottom_prv_descriptor->prv_memory_pd()->desc()));
      usr_mpd = bottom_prv_descriptor->usr_memory_pd();
      prv_mpd = bottom_prv_descriptor->prv_memory_pd();
    } else {
      bottom_data_md.reset(new memory::desc({ { n, ic, ih, iw } }, mpcsn, memory::format::nchw));
      usr_mpd.reset(new memory::primitive_desc(*bottom_data_md, cpu_engine));
    }
    top_data_md = bottom_data_md;

    // ---- Initialize relu primitive descriptor -------------
    relu_forward::desc fwd_inference_desc(prop_kind::forward_scoring,
      *bottom_data_md, negative_slope);
    fwd_inference_pd.reset(new relu_forward::primitive_desc(fwd_inference_desc, cpu_engine));
    /* relu_forward::desc fwd_training_desc(prop_kind::forward_training, */
    /*   *bottom_data_md, negative_slope); */
    // relu_forward is being deprecated, use new eltwise_forward
    eltwise_forward::desc fwd_training_desc(prop_kind::forward_training, eltwise_relu, *bottom_data_md, negative_slope);
    fwd_training_pd.reset(new relu_forward::primitive_desc(fwd_training_desc, cpu_engine));
    fwd_bottom_data.reset(new MKLDNNData<Dtype>(usr_mpd, prv_mpd));
    fwd_bottom_data->name = "fwd_bottom_data   @ " + this->getName();
    fwd_top_data.reset(new MKLDNNData<Dtype>(usr_mpd, prv_mpd));
    fwd_top_data->name = "fwd_top_data   @ " + this->getName();
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
        Tensor<xpu, 4, Dtype> data;
        Tensor<xpu, 4, Dtype> out;
        if (in_data[activation::kData].ndim() == 2) {
          Shape<4> dshape = Shape4(in_data[activation::kData].shape_[0],
            in_data[activation::kData].shape_[1], 1, 1);
          data = mkl_experimental_direct_get_with_shape<xpu, 4, Dtype>(
            in_data[activation::kData], dshape, s);
          out = mkl_experimental_direct_get_with_shape<xpu, 4, Dtype>(
            out_data[activation::kOut], dshape, s);
        } else if (in_data[activation::kData].ndim() == 3) {
          Shape<4> dshape = Shape4(in_data[activation::kData].shape_[0],
            in_data[activation::kData].shape_[1],
            in_data[activation::kData].shape_[2], 1);
          data = mkl_experimental_direct_get_with_shape<xpu, 4, Dtype>(
            in_data[activation::kData], dshape, s);
          out = mkl_experimental_direct_get_with_shape<xpu, 4, Dtype>(
            out_data[activation::kOut], dshape, s);
        } else {
          data = mkl_experimental_direct_get<xpu, 4, Dtype>(in_data[activation::kData], s);
          out = mkl_experimental_direct_get<xpu, 4, Dtype>(out_data[activation::kOut], s);
        }

    if (!init_mkldnn_) {
      LayerSetup(data);
      InitReLUFwd(in_data);
      init_mkldnn_ = true;
      in_place_ = (data.dptr_ == out.dptr_);
      // ---- Initialize memory descriptors -------------

      input_primitive = fwd_bottom_data->get_converted_prv(data.dptr_,
        false, in_data[activation::kData]);
      output_memory = fwd_top_data->create_output_memory(
        out.dptr_, out_data[activation::kOut], fwd_top_data, in_place_);
      if (ctx.is_train) {
        reluFwd.reset(new relu_forward(*fwd_training_pd, *input_primitive, *output_memory));
      } else {
        reluFwd.reset(new relu_forward(*fwd_inference_pd, *input_primitive, *output_memory));
      }
    } else {
      fwd_bottom_data->sync_converted_prv(data.dptr_,
        false, in_data[activation::kData]);
        fwd_top_data->sync_output_memory(
          out_data[activation::kOut], fwd_top_data, in_place_);
    }
    reluFwd.submit();
  }

  void InitReLUBwd(const std::vector<TBlob> &out_grad, const std::vector<TBlob> &in_data) {
    int32_t n = this->num_;
    int32_t iw = this->width_;
    int32_t ih = this->height_;
    int32_t ic = this->channels_;
    Dtype negative_slope = 0;
    void * top_diff_data =
      const_cast<Dtype*>(mkl_prv_data<Dtype>(out_grad[activation::kOut]));
    bool top_diff_is_prv = (top_diff_data != NULL);
    mkldnn::engine cpu_engine = CpuEngine::Instance().get_engine();
    memory::data_type mpcsn = memory::data_type::f32;
    // ---- Initialize memory descriptors -------------
    std::shared_ptr<memory::desc> bottom_diff_md;
    std::shared_ptr<memory::desc> top_diff_md;
    std::shared_ptr<memory::desc> top_data_md;

    std::shared_ptr<memory::primitive_desc> usr_diff_mpd;
    std::shared_ptr<memory::primitive_desc> prv_diff_mpd;

    std::shared_ptr<memory::desc> default_md;
    default_md.reset(new memory::desc({ { n, ic, ih, iw } }, mpcsn, memory::format::nchw));
    if (top_diff_is_prv) {
      std::shared_ptr<MKLDNNMemoryDescriptor<Dtype> > mem_descr
        = get_mkldnn_prv_descriptor<Dtype>(out_grad[activation::kOut]);
      usr_diff_mpd = mem_descr->usr_memory_pd();
      prv_diff_mpd = mem_descr->prv_memory_pd();
    } else {
      if (prv_mpd != NULL) prv_diff_mpd = prv_mpd;
      usr_diff_mpd.reset(new memory::primitive_desc(*default_md, cpu_engine));
    }
    if (prv_diff_mpd != NULL)
      top_diff_md.reset(new memory::desc(prv_diff_mpd->desc()));
    else 
      top_diff_md.reset(new memory::desc(*default_md));
    top_data_md = top_diff_md;
    bottom_diff_md = top_diff_md;
    /* relu_backward::desc reluBwd_desc(*top_diff_md, *top_data_md, negative_slope); */
    // use eltwise instead of relu_backward
    eltwise_backward::desc reluBwd_desc(eltwise_relu, *top_diff_md, *top_data_md, negative_slope);
    bwd_pd.reset(new relu_backward::primitive_desc(reluBwd_desc, cpu_engine,
      *fwd_training_pd));
    bwd_top_diff.reset(new MKLDNNData<Dtype>(usr_diff_mpd, prv_diff_mpd));
    bwd_top_diff->name = "bwd_top_diff   @ " + this->getName();
    bwd_bottom_diff.reset(new MKLDNNData<Dtype>(usr_diff_mpd, prv_diff_mpd));
    bwd_bottom_diff->name = "bwd_bottom_diff   @ " + this->getName();
    bwd_bottom_data.reset(new MKLDNNData<Dtype>(usr_diff_mpd, prv_diff_mpd));
    bwd_bottom_data->name = "bwd_bottom_data   @ " + this->getName();
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
    CHECK(in_data.size() == 1 && in_grad.size() == 1);
    CHECK_EQ(req.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, Dtype> m_out_grad;
    Tensor<xpu, 4, Dtype> m_in_grad;
    Tensor<xpu, 4, Dtype> m_out_data;

    if (out_grad[activation::kOut].ndim() == 2) {
      Shape<4> dshape = Shape4(out_grad[activation::kOut].shape_[0],
        out_grad[activation::kOut].shape_[1], 1, 1);
      m_out_grad = mkl_experimental_direct_get_with_shape<xpu, 4, Dtype>(
        out_grad[activation::kOut], dshape, s);
      m_out_data = mkl_experimental_direct_get_with_shape<xpu, 4, Dtype>(
        out_data[activation::kOut], dshape, s);
      m_in_grad = mkl_experimental_direct_get_with_shape<xpu, 4, Dtype>(
        in_grad[activation::kData], dshape, s);
    } else if (out_grad[activation::kOut].ndim() == 3) {
      Shape<4> dshape = Shape4(out_grad[activation::kOut].shape_[0],
        out_grad[activation::kOut].shape_[1],
        out_grad[activation::kOut].shape_[2], 1);
      m_out_grad = mkl_experimental_direct_get_with_shape<xpu, 4, Dtype>(
        out_grad[activation::kOut], dshape, s);
      m_out_data = mkl_experimental_direct_get_with_shape<xpu, 4, Dtype>(
        out_data[activation::kOut], dshape, s);
      m_in_grad = mkl_experimental_direct_get_with_shape<xpu, 4, Dtype>(
        in_grad[activation::kData], dshape, s);
    } else {
      m_out_grad = mkl_experimental_direct_get<xpu, 4, Dtype>(out_grad[activation::kOut], s);
      m_out_data = mkl_experimental_direct_get<xpu, 4, Dtype>(out_data[activation::kOut], s);
      m_in_grad = mkl_experimental_direct_get<xpu, 4, Dtype>(in_grad[activation::kData], s);
    }
    in_place_b_ = (m_out_grad.dptr_ != m_in_grad.dptr_);
if (bwd_pd == nullptr) {
    InitReLUBwd(out_grad, in_data);
    // use the src memory from forward call
    /* src_memory = bwd_bottom_data->get_converted_prv(m_out_data.dptr_, false, */
    /*   out_data[activation::kOut]); */
    diff_dst_memory = bwd_top_diff->get_converted_prv(m_out_grad.dptr_,
      false, out_grad[activation::kOut]);
    diff_src_memory = bwd_bottom_diff->create_output_memory(m_in_grad.dptr_,
      in_grad[activation::kData], bwd_bottom_diff, in_place_b_);
    reluBwd.reset(new relu_backward(*bwd_pd, *input_primitive, *diff_dst_memory,
      *diff_src_memory));
    } else {
    // use the src memory from forward call
    /* bwd_bottom_data->sync_converted_prv(false, */
    /*   out_data[activation::kOut]); */
    bwd_top_diff->sync_converted_prv(m_out_grad.dptr_,
      false, out_grad[activation::kOut]);
    bwd_bottom_diff->sync_output_memory(
      in_grad[activation::kData], bwd_bottom_diff, in_place_b_);
    }
    reluBwd.submit();

}

 private:
  bool init_mkldnn_;
  bool in_place_;
  bool in_place_b_;

  std::shared_ptr<MKLDNNData<Dtype> > fwd_top_data, fwd_bottom_data;
  std::shared_ptr<MKLDNNData<Dtype> > bwd_bottom_data, bwd_top_diff;
  std::shared_ptr<MKLDNNData<Dtype> > bwd_bottom_diff;
  std::shared_ptr<relu_forward::primitive_desc> fwd_inference_pd;
  std::shared_ptr<relu_forward::primitive_desc> fwd_training_pd;
  std::shared_ptr<relu_backward::primitive_desc> bwd_pd;
  std::shared_ptr<memory::primitive_desc> prv_mpd;
  int32_t num_, width_, height_, channels_;
  std::shared_ptr<memory> input_primitive;
  std::shared_ptr<memory> output_memory;
  std::shared_ptr<memory> src_memory, diff_dst_memory, diff_src_memory;
  MKLDNNPrimitive<Dtype> reluFwd, reluBwd;
};  // class MKLDNNReluOp

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MKL_MKLDNN_RELU_INL_H_
