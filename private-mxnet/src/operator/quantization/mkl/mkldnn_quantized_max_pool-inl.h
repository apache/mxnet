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
*         deepthi.karkada@intel.com
*
*******************************************************************************/

#ifndef MXNET_OPERATOR_MKL_DNN_MKLDNN_QUANTIZED_MAX_POOL_INL_H_
#define MXNET_OPERATOR_MKL_DNN_MKLDNN_QUANTIZED_MAX_POOL_INL_H_
#include <vector>
#include <string>
#include <utility>


namespace mxnet {
namespace op {
template<typename SrcType , typename DstType>
class MKLDNNQuantPoolingOp : public Operator{
 public:
  std::string getName() {
     std::string name = "MKLDNNQuantPoolingOp";
     return name;
  }
  explicit MKLDNNQuantPoolingOp(QuantizedMaxPoolParam param) : /*MKLDNNLayer<Dtype>()
    , */num_(0), channels_(0), width_(0), height_(0), width_out_(0), height_out_(0)
    , kernel_w_(0), kernel_h_(0), stride_w_(0), stride_h_(0)
    , pad_t_(0), pad_b_(0), pad_l_(0), pad_r_(0) {
    this->param_ = param;
    this->init_mkldnn_ = false;
    pooling_algorithm_ = pooling_max;
  }
  virtual ~MKLDNNQuantPoolingOp() {
  }

 private:
  void LayerSetUp(const mshadow::Tensor<cpu, 4, SrcType> &data,
                  const mshadow::Tensor<cpu, 4, DstType> &out) {
    if(param_.layout == mshadow::kNCHW){
      channels_ = data.shape_[1];
      height_ = data.shape_[2];
      width_ = data.shape_[3];
    } else if(param_.layout == mshadow::kNHWC){
      channels_ = data.shape_[3];
      height_ = data.shape_[1];
      width_ = data.shape_[2];
    }

    num_ = data.shape_[0];
    kernel_h_ = param_.kernel[0];
    kernel_w_ = param_.kernel[1];
    
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
      CHECK_LT(pad_t_, kernel_h_);
      CHECK_LT(pad_l_, kernel_w_);
    }
    if(param_.layout == mshadow::kNCHW){
      height_out_ = out.shape_[2];
      width_out_ = out.shape_[3];
    } else if(param_.layout == mshadow::kNHWC){
      height_out_ = out.shape_[1];
      width_out_ = out.shape_[2];
    }
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
       (const_cast<SrcType*>(mkl_prv_data<SrcType>(in_data[pool_enum::kData])) != NULL);
     mkldnn::engine cpu_engine = CpuEngine::Instance().get_engine();
     //memory::data_type mpcsn = memory::data_type::f32;
     memory::dims bottom_tz = { n, c, ih, iw };
     memory::dims top_tz = { n, c, oh, ow };
     memory::format mfmt_nchw = memory::format::nchw;


     // ---- Initialize memory descriptors -------------
     typedef typename memory::primitive_desc MemPD;

     memory::format cmfmt = mfmt_nchw;
     if (bottom_data_is_prv) {
       std::shared_ptr<MKLDNNData<SrcType> > mem_descr
         = get_mkldnn_prv_descriptor<SrcType>(in_data[pool_enum::kData].Mkl_mem_);
       cmfmt = static_cast<memory::format>(mem_descr->prv_memory_pd()->desc().data.format);
     }
     std::shared_ptr<memory::desc> init_fwd_bottom_md(
       new memory::desc({ bottom_tz }, (mkldnn::memory::data_type)data_type_enum<SrcType>::type, cmfmt));
     std::shared_ptr<memory::desc> init_fwd_top_md(new memory::desc({ top_tz }, (mkldnn::memory::data_type)data_type_enum<DstType>::type, cmfmt));
     std::shared_ptr<MemPD> usr_bottom_data_mpd(new MemPD({ { bottom_tz }, (mkldnn::memory::data_type)data_type_enum<SrcType>::type, mfmt_nchw },
       cpu_engine));
     std::shared_ptr<MemPD> usr_top_data_mpd(
       new MemPD({ { top_tz }, (mkldnn::memory::data_type)data_type_enum<DstType>::type, mfmt_nchw }, cpu_engine));

     pooling_forward::desc poolingFwdInference_desc(prop_kind::forward_scoring,
        pooling_algorithm_, *init_fwd_bottom_md, *init_fwd_top_md
       , { sh, sw }, { kh, kw }, { pt, pl }, { pb, pr }, padding_kind::zero);
     /*pooling_forward::desc poolingFwdTraining_desc(prop_kind::forward_training
       , pooling_algorithm_, *init_fwd_bottom_md, *init_fwd_top_md
       , { sh, sw }, { kh, kw }, { pt, pl }, { pb, pr }, padding_kind::zero);*/
     poolingFwdInference_pd.reset(new pooling_forward::primitive_desc(
       poolingFwdInference_desc, cpu_engine));
     CHECK(poolingFwdInference_pd);
     /*poolingFwdTraining_pd.reset(new pooling_forward::primitive_desc(
       poolingFwdTraining_desc, cpu_engine));
     CHECK(poolingFwdTraining_pd);*/

     // ---- Initialize remaining memory descriptors -------------
     std::shared_ptr<MemPD> prv_fwd_bottom_data_mpd;
     std::shared_ptr<MemPD> prv_fwd_top_data_mpd;
     if (bottom_data_is_prv) {
       prv_fwd_bottom_data_mpd.reset(new MemPD(*init_fwd_bottom_md, cpu_engine));
       prv_fwd_top_data_mpd.reset(new MemPD(*init_fwd_top_md, cpu_engine));
     }

     fwd_bottom_data.reset(new MKLDNNData<SrcType>(usr_bottom_data_mpd, prv_fwd_bottom_data_mpd));
     fwd_bottom_data->name = "fwd_bottom_data   @ " + this->getName();

     fwd_top_data.reset(new MKLDNNData<DstType>(usr_top_data_mpd, prv_fwd_top_data_mpd));
     fwd_top_data->name = "fwd_top_data   @ " + this->getName();
     // ---- Initialize pooling primitive descriptor -------------
     /*if (pooling_algorithm_ != algorithm::pooling_avg) {
       indices_pd.reset(
         new memory::primitive_desc(poolingFwdTraining_pd->workspace_primitive_desc()));
       indices_memory.reset(new memory(*indices_pd));
     }*/
  }
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 3U);
    CHECK_EQ(out_data.size(), 3U);
    Stream<cpu> *s = ctx.get_stream<cpu>();
    if (param_.kernel.ndim() >= 3) {
      LOG(FATAL) << "Not implmented";
    }

    Tensor<cpu, 4, SrcType> data = mkl_experimental_direct_get<cpu, 4, SrcType>(
      in_data[pool_enum::kData], s);
    Tensor<cpu, 4, DstType> out = mkl_experimental_direct_get<cpu, 4, DstType>(
      out_data[pool_enum::kOut], s);
    if (!init_mkldnn_) {
      LayerSetUp(data, out);
      init_mkldnn_ = true;
    }
    if (poolingFwdInference_pd == NULL)
      InitPoolingFwd(in_data);
    // ---  init primitive and prv_memory descriptors ----------------------
    std::shared_ptr<memory> fwd_input_primitive, fwd_output_memory;
    fwd_input_primitive = fwd_bottom_data->get_converted_prv(data.dptr_, false,
      in_data[pool_enum::kData]);
    fwd_output_memory = fwd_top_data->create_output_memory(out.dptr_, out_data[pool_enum::kOut],
      fwd_top_data);
    MKLDNNPrimitive<SrcType> poolingFwd;
    if (ctx.is_train && pooling_algorithm_ != algorithm::pooling_avg) {
    /*poolingFwd.reset(new pooling_forward(*poolingFwdTraining_pd, *fwd_input_primitive,
        *fwd_output_memory, *indices_memory));*/
    LOG(FATAL) << "Quantized training is not implmented currently";
    } else {
      poolingFwd.reset(new pooling_forward(*poolingFwdInference_pd, *fwd_input_primitive,
        *fwd_output_memory));
    }
    poolingFwd.submit();
  }

 private:
  QuantizedMaxPoolParam param_;
  int32_t num_, channels_, width_, height_, width_out_, height_out_;
  int32_t kernel_w_, kernel_h_, stride_w_, stride_h_;
  int32_t  pad_t_, pad_b_, pad_l_, pad_r_;
  bool global_pooling_;
  std::shared_ptr<pooling_forward::primitive_desc> poolingFwdInference_pd;
  std::shared_ptr<pooling_forward::primitive_desc> poolingFwdTraining_pd;
  std::shared_ptr<pooling_backward::primitive_desc> poolingBwd_pd;

  std::shared_ptr<MKLDNNData<SrcType> > fwd_bottom_data, fwd_top_data,
    bwd_top_diff, bwd_bottom_diff;
  std::shared_ptr<memory::primitive_desc> indices_pd;
  std::shared_ptr<memory> indices_memory;
  bool init_mkldnn_;
  float alpha_;
  float beta_;
  algorithm pooling_algorithm_;
};  // class MKLDNNQuantPoolingOp
}   // namespace op
}   // namespace mxnet

#endif  // MXNET_OPERATOR_MKL_DNN_MKLDNN_QUANTIZED_MAX_POOL_INL_H_
