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
* \file mkl_relu-inl.h
* \brief
* \author young.jin.kim@intel.com
*         deepthi.karkada@intel.com
*
*******************************************************************************/
#ifndef MXNET_OPERATOR_MKL_DNN_MKLDNN_QUANTIZED_RELU_INL_H_
#define MXNET_OPERATOR_MKL_DNN_MKLDNN_QUANTIZED_RELU_INL_H_


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

template<typename Dtype>
class MKLDNNQuantReluOp : public Operator {
 public:
  std::string getName() {
    std::string name = "MKLDNNQuantReluOp";
    return name;
  }
  MKLDNNQuantReluOp() : 
    fwd_top_data(NULL), fwd_bottom_data(NULL)
    , num_(0), width_(0), height_(0), channels_(0) {
    init_mkldnn_ = false;
  }
  ~MKLDNNQuantReluOp() {
  }

 private:
  void LayerSetup(const mshadow::Tensor<cpu, 4, Dtype> &data) {
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
    std::shared_ptr<memory::primitive_desc> usr_mpd(NULL), prv_mpd(NULL);

    int32_t n = this->num_;
    int32_t iw = this->width_;
    int32_t ih = this->height_;
    int32_t ic = this->channels_;
    Dtype negative_slope = 0;
    mkldnn::engine cpu_engine = CpuEngine::Instance().get_engine();
    //memory::data_type mpcsn = memory::data_type::f32;
    memory::data_type mpcsn = (mkldnn::memory::data_type)data_type_enum<Dtype>::type;

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
    //CHECK_EQ(in_data.size(), 1);
    //CHECK_EQ(out_data.size(), 1);
    Stream<cpu> *s = ctx.get_stream<cpu>();
    Tensor<cpu, 4, Dtype> data;
    Tensor<cpu, 4, Dtype> out;
    if (in_data[activation::kData].ndim() == 2) {
      Shape<4> dshape = Shape4(in_data[activation::kData].shape_[0],
        in_data[activation::kData].shape_[1], 1, 1);
      data = mkl_experimental_direct_get_with_shape<cpu, 4, Dtype>(
        in_data[activation::kData], dshape, s);
      out = mkl_experimental_direct_get_with_shape<cpu, 4, Dtype>(
        out_data[activation::kOut], dshape, s);
    } else if (in_data[activation::kData].ndim() == 3) {
      Shape<4> dshape = Shape4(in_data[activation::kData].shape_[0],
        in_data[activation::kData].shape_[1],
        in_data[activation::kData].shape_[2], 1);
      data = mkl_experimental_direct_get_with_shape<cpu, 4, Dtype>(
        in_data[activation::kData], dshape, s);
      out = mkl_experimental_direct_get_with_shape<cpu, 4, Dtype>(
        out_data[activation::kOut], dshape, s);
    } else {
      data = mkl_experimental_direct_get<cpu, 4, Dtype>(in_data[activation::kData], s);
      out = mkl_experimental_direct_get<cpu, 4, Dtype>(out_data[activation::kOut], s);
    }

    if (!init_mkldnn_) {
      LayerSetup(data);
      InitReLUFwd(in_data);
      init_mkldnn_ = true;
    }
    // ---- Initialize memory descriptors -------------

    std::shared_ptr<memory> input_primitive;
    input_primitive = fwd_bottom_data->get_converted_prv(data.dptr_,
      false, in_data[activation::kData]);
    std::shared_ptr<memory> output_memory = fwd_top_data->create_output_memory(
      out.dptr_, out_data[activation::kOut], fwd_top_data, (data.dptr_ == out.dptr_));
    MKLDNNPrimitive<Dtype> reluFwd;
    if (ctx.is_train) {
      LOG(FATAL) << "Quantized training is not implemented yet";
    } else {
      reluFwd.reset(new relu_forward(*fwd_inference_pd, *input_primitive, *output_memory));
    }
    reluFwd.submit();
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    LOG(FATAL) << "Quantized relu backprop is not implementated yet";
  }

 private:
  bool init_mkldnn_;

  std::shared_ptr<MKLDNNData<Dtype> > fwd_top_data, fwd_bottom_data;
  std::shared_ptr<relu_forward::primitive_desc> fwd_inference_pd;
  int32_t num_, width_, height_, channels_;
};  // class MKLDNNQuantReluOp

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MKL_DNN_MKLDNN_QUANTIZED_RELU_INL_H_
