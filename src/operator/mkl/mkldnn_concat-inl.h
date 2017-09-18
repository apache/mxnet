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
* \file mkldnn_concat-inl.h
* \brief
* \author young.jin.kim@intel.com
*         ashok.emani@intel.com
*         deepthi.karkada@intel.com
*         louis.feng@intel.com
*         adam.d.straw@intel.com
*
*******************************************************************************/
#ifndef MXNET_OPERATOR_MKL_MKLDNN_CONCAT_INL_H_
#define MXNET_OPERATOR_MKL_MKLDNN_CONCAT_INL_H_

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
class MKLDNNConcatOp : public Operator, public MKLDNNLayer<Dtype> {
 public:
  static std::string getName() {
    std::string name = "MKLDNNConcatOp";
    return name;
  }
  explicit MKLDNNConcatOp(ConcatParam param) : MKLDNNLayer<Dtype>()
    , size_(param.num_args), dimension_(param.dim), split_channels_(param.num_args) {
    init_mkldnn_ = false;
  }
  virtual ~MKLDNNConcatOp() {
  }

 private:
  void LayerSetup(const std::vector<mshadow::Tensor<xpu, 4, Dtype> > &data,
                  size_t data_shape_size) {
    for (size_t i = 1; i < size_; ++i) {
      for (size_t j = 1; j < data_shape_size; ++j) {
        if (j == dimension_) continue;
        CHECK_EQ(data[0].shape_[j], data[i].shape_[j]);
      }
    }
    n_ = data[0].shape_[0];
    c_ = 0;
    h_ = data[0].shape_[2];
    w_ = data[0].shape_[3];
    for (size_t i = 0; i < size_; ++i) {
      CHECK_EQ((int)data_shape_size, data[i].shape_.kDimension);
      split_channels_[i] = data[i].shape_[dimension_];
      c_ += split_channels_[i];
    }
  }

  void InitConcatFwd(const OpContext &ctx, const std::vector<TBlob> &in_data,
                       const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    CHECK_EQ(static_cast<int>(in_data.size()), size_);
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    std::vector<Tensor<xpu, 4, Dtype> > data(size_);
    Tensor<xpu, 4, Dtype> out;
    if (in_data[0].ndim() == 2) {
      for (size_t i = 0; i < size_; ++i) {
        Shape<4> dshape = Shape4(in_data[i].shape_[0],
                                 in_data[i].shape_[1], 1, 1);
        data[i] = mkl_experimental_direct_get_with_shape<xpu, 4, Dtype>(
          in_data[i], dshape, s);
      }
      Shape<4> dshape = Shape4(out_data[concat_enum::kOut].shape_[0],
                               out_data[concat_enum::kOut].shape_[1], 1, 1);
      out = mkl_experimental_direct_get_with_shape<xpu, 4, Dtype>(
        out_data[concat_enum::kOut], dshape, s);
    } else if (in_data[0].ndim() == 3) {
      for (size_t i = 0; i < size_; ++i) {
        Shape<4> dshape = Shape4(in_data[i].shape_[0],
          in_data[i].shape_[1], in_data[i].shape_[2], 1);
        data[i] = mkl_experimental_direct_get_with_shape<xpu, 4, Dtype>(
          in_data[i], dshape, s);
      }
      Shape<4> dshape = Shape4(out_data[concat_enum::kOut].shape_[0],
        out_data[concat_enum::kOut].shape_[1],
        out_data[concat_enum::kOut].shape_[2], 1);
      out = mkl_experimental_direct_get_with_shape<xpu, 4, Dtype>(
        out_data[concat_enum::kOut], dshape, s);
    } else {
      for (size_t i = 0; i < size_; ++i) {
        data[i] = mkl_experimental_direct_get<xpu, 4, Dtype>(in_data[i], s);
      }
      out = mkl_experimental_direct_get<xpu, 4, Dtype>(out_data[concat_enum::kOut], s);
    }

    LayerSetup(data, 4);
    mkldnn::engine cpu_engine = CpuEngine::Instance().get_engine();
    memory::data_type mtype = memory::data_type::f32;
    memory::format mfmt_nchw = memory::format::nchw;
    memory::dims output_tz = {n_, c_, h_, w_};

    for (size_t i = 0; i < size_; ++i) {
      memory::format mfmt = mfmt_nchw;
      fwd_bottom_data_.push_back(std::shared_ptr<MKLDNNData<Dtype> >());
      memory::dims input_tz = {n_, (int32_t)split_channels_[i], h_, w_};

      std::shared_ptr<memory::primitive_desc> prv_src_mpd;
      std::shared_ptr<memory::primitive_desc> usr_src_mpd(
        new memory::primitive_desc({input_tz, mtype, mfmt_nchw}, cpu_engine));

      if (const_cast<Dtype*>(mkl_prv_data<Dtype>(in_data[i])) != NULL) {
        std::shared_ptr<MKLDNNMemoryDescriptor<Dtype> > mem_descr
          = get_mkldnn_prv_descriptor<Dtype>(in_data[i]);
        mfmt = static_cast<memory::format>(
              mem_descr->prv_memory_pd()->desc().data.format);
        prv_src_mpd.reset(new memory::primitive_desc(
                {input_tz, mtype, mfmt}, cpu_engine));
      }

      bottom_data_mpd.push_back(memory::primitive_desc(
          {input_tz, mtype, mfmt}, cpu_engine));

      fwd_bottom_data_[i].reset(new MKLDNNData<Dtype>(usr_src_mpd, prv_src_mpd));
    }

    std::shared_ptr<memory::primitive_desc> usr_dst_mpd(new memory::primitive_desc(
        {output_tz, mtype, mfmt_nchw}, cpu_engine));

    fwd_pd.reset(new concat::primitive_desc(static_cast<int>(dimension_), bottom_data_mpd));

    std::shared_ptr<memory::primitive_desc> prv_dst_mpd(new memory::primitive_desc(
        fwd_pd->dst_primitive_desc()));

    fwd_top_data_.reset(new MKLDNNData<Dtype>(usr_dst_mpd, prv_dst_mpd));

    for (size_t i = 0; i < size_; ++i) {
      inputs.push_back(fwd_bottom_data_[i]->get_converted_prv(data[i].dptr_, false, in_data[i]));
    }

    for (size_t i = 0; i < inputs.size(); i++) {
        inputs_at.push_back(*inputs[i]);
    }
    output_memory = fwd_top_data_->create_output_memory(
      out.dptr_, out_data[concat_enum::kOut], fwd_top_data_);
    concatFwd.reset(new mkldnn::concat(*fwd_pd, inputs_at, *output_memory));
  }

 public:
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
     if (!init_mkldnn_) {
        InitConcatFwd(ctx, in_data, out_data);
        init_mkldnn_ = true;
      } else {
        for (size_t i = 0; i < size_; ++i) {
          fwd_bottom_data_[i]->sync_converted_prv(false, in_data[i]);
        }
        fwd_top_data_->sync_output_memory(
          out_data[concat_enum::kOut], fwd_top_data_);
     }
     concatFwd.submit();
  }

  void InitConcatBwd(const std::vector<TBlob> &out_grad,
                  const std::vector<mshadow::Tensor<xpu, 4, Dtype> > &data,
                  const mshadow::Tensor<xpu, 4, Dtype> &out) {
    mkldnn::engine cpu_engine = CpuEngine::Instance().get_engine();
    memory::data_type mtype = memory::data_type::f32;
    memory::format mfmt_nchw = memory::format::nchw;
    memory::format diff_dst_mfmt = mfmt_nchw;
    memory::dims input_tz = {n_, c_, h_, w_};
    memory::dims offsets = {0, 0, 0, 0};

    std::shared_ptr<memory::primitive_desc> prv_diff_dst_mpd;
    std::shared_ptr<memory::primitive_desc> usr_diff_dst_mpd(
      new memory::primitive_desc({input_tz, mtype, mfmt_nchw},
        cpu_engine));

    bool top_diff_is_prv =
      (const_cast<Dtype*>(mkl_prv_data<Dtype>(out_grad[concat_enum::kOut])) != NULL);
    if (top_diff_is_prv) {
        std::shared_ptr<MKLDNNMemoryDescriptor<Dtype> > mem_descr
          = get_mkldnn_prv_descriptor<Dtype>(out_grad[concat_enum::kOut]);
        diff_dst_mfmt = static_cast<memory::format>(
            mem_descr->prv_memory_pd()->desc().data.format);
        prv_diff_dst_mpd.reset(new memory::primitive_desc(
              {input_tz, mtype, diff_dst_mfmt}, cpu_engine));
    }

    bwd_top_diff_.reset(new MKLDNNData<Dtype>(usr_diff_dst_mpd, prv_diff_dst_mpd));

    for (size_t i = 0; i < size_; ++i) {
      bwd_bottom_diff_.push_back(std::shared_ptr<MKLDNNData<Dtype> >());
      bwd_pd.push_back(std::shared_ptr<reorder::primitive_desc>());
      memory::dims dims = {n_, (int32_t)split_channels_[i], h_, w_};
      std::shared_ptr<memory::primitive_desc> usr_diff_src_mpd(
        new memory::primitive_desc({dims, mtype, mfmt_nchw},
            cpu_engine));
      std::shared_ptr<memory::primitive_desc> prv_diff_src_mpd(
        new memory::primitive_desc({dims, mtype, diff_dst_mfmt},
            cpu_engine));
      bwd_bottom_diff_[i].reset(new MKLDNNData<Dtype>(
            usr_diff_src_mpd, prv_diff_src_mpd));

      auto view_pd = top_diff_is_prv ?
        view::primitive_desc(*prv_diff_dst_mpd, dims, offsets) :
        view::primitive_desc(*usr_diff_dst_mpd, dims, offsets);
      auto view_dst_pd = view_pd.dst_primitive_desc();
      bwd_pd[i].reset(new reorder::primitive_desc(view_dst_pd, *prv_diff_src_mpd));
      offsets[dimension_] += split_channels_[i];
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    CHECK_EQ(out_grad.size(), 1);
    CHECK_EQ(in_grad.size(), static_cast<size_t>(size_));
    Stream<xpu> *s = ctx.get_stream<xpu>();
    std::vector<Tensor<xpu, 4, Dtype> > grad_in(size_);
    Tensor<xpu, 4, Dtype> grad;
    if (in_grad[0].ndim() == 2) {
      Shape<4> dshape = Shape4(out_grad[concat_enum::kOut].shape_[0],
        out_grad[concat_enum::kOut].shape_[1], 1, 1);
      grad = mkl_experimental_direct_get_with_shape<xpu, 4, Dtype>(
        out_grad[concat_enum::kOut], dshape, s);
      for (size_t i = 0; i < size_; ++i) {
        dshape = Shape4(in_grad[i].shape_[0],
          in_grad[i].shape_[1], 1, 1);
        grad_in[i] = mkl_experimental_direct_get_with_shape<xpu, 4, Dtype>(
          in_grad[i], dshape, s);
      }
    } else if (in_grad[0].ndim() == 3) {
      Shape<4> dshape = Shape4(out_grad[concat_enum::kOut].shape_[0],
        out_grad[concat_enum::kOut].shape_[1],
        out_grad[concat_enum::kOut].shape_[2], 1);
      grad = mkl_experimental_direct_get_with_shape<xpu, 4, Dtype>(
        out_grad[concat_enum::kOut], dshape, s);
      for (size_t i = 0; i < size_; ++i) {
        dshape = Shape4(in_grad[i].shape_[0],
          in_grad[i].shape_[1], in_grad[i].shape_[2], 1);
        grad_in[i] = mkl_experimental_direct_get_with_shape<xpu, 4, Dtype>(
          in_grad[i], dshape, s);
      }
    } else {
      grad = mkl_experimental_direct_get<xpu, 4, Dtype>(out_grad[concat_enum::kOut], s);
      for (size_t i = 0; i < size_; ++i) {
        grad_in[i] = mkl_experimental_direct_get<xpu, 4, Dtype>(in_grad[i], s);
      }
    }

    int need_bwd = 0;
    for (size_t n = 0; n < size_; n++) {
      need_bwd += req[n];
    }
    if (!need_bwd) {
      return;
    }

    if (bwd_pd.empty()) {
      InitConcatBwd(out_grad, grad_in, grad);
    }

    for (size_t i = 0; i < size_; ++i) {
     std::shared_ptr<memory> bwd_reorder_input_memory =
      bwd_top_diff_->get_converted_prv(grad.dptr_, true, out_grad[concat_enum::kOut]);
     std::shared_ptr<memory> bwd_reorder_output_memory =
      bwd_bottom_diff_[i]->create_output_memory(grad_in[i].dptr_, in_grad[i], bwd_bottom_diff_[i]);

     MKLDNNPrimitive<Dtype> concatBwd;
     concatBwd.reset(
        new reorder(*bwd_pd[i], *bwd_reorder_input_memory, *bwd_reorder_output_memory));
     concatBwd.submit();
    }
  }

 private:
  int32_t n_, c_, h_, w_;
  size_t size_;
  size_t dimension_;
  bool init_mkldnn_;
  std::vector<size_t> split_channels_;

  std::shared_ptr<MKLDNNData<Dtype> > fwd_top_data_;
  std::vector< std::shared_ptr<MKLDNNData<Dtype> > > fwd_bottom_data_;
  std::shared_ptr<MKLDNNData<Dtype> > bwd_top_diff_;
  std::vector< std::shared_ptr<MKLDNNData<Dtype> > > bwd_bottom_diff_;
  MKLDNNPrimitive<Dtype> concatFwd;
  std::vector<std::shared_ptr<memory>> inputs;
  std::vector<primitive::at> inputs_at;
  std::vector<memory::primitive_desc> bottom_data_mpd;
  std::shared_ptr<memory::desc> top_data_md;
  std::shared_ptr<memory> output_memory;
  std::shared_ptr<concat::primitive_desc> fwd_pd;
  std::vector<std::shared_ptr<mkldnn::reorder::primitive_desc>> bwd_pd;
};  // class MKLDNNConcatOp

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MKL_MKLDNN_CONCAT_INL_H_
