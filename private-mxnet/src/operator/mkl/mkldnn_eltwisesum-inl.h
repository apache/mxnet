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
* \file mkl_elementwise-inl.h
* \brief
* \author young.jin.kim@intel.com
*         deepthi.karkada@intel.com
*
*******************************************************************************/
#ifndef MXNET_OPERATOR_MKL_DNN_MKLDNN_ELEMENTWISE_SUM_INL_H_
#define MXNET_OPERATOR_MKL_DNN_MKLDNN_ELEMENTWISE_SUM_INL_H_

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
class MKLDNNEltwiseSumOp : public Operator {
 public:
  std::string getName() {
    std::string name = "MKLDNNEltwiseSumOp";
    return name;
  }
  MKLDNNEltwiseSumOp() : MKLDNNLayer<Dtype>()
    , fwd_top_data(NULL), fwd_bottom_data(NULL) {
    init_mkldnn_ = false;
  }
  ~MKLDNNEltwiseSumOp() {
  }
 private:
  void LayerSetup(const std::vector<mshadow::Tensor<cpu, 1, Dtype> > &data,
  size_t data_shape_size) {
    num_inputs_ = data.size();
    channels_ = data.shape_[1];
    height_ = data.shape_[2];
    width_ = data.shape_[3];
    num_ = data.shape_[0];
    
  }
 public:
  void InitEltwiseSumFwd(const std::vector<TBlob> &in_data) {
    int32_t n = this->num_;
    int32_t iw = this->width_;
    int32_t ih = this->height_;
    int32_t ic = this->channels_;

    std::vector<double> scale(num_inputs_, 1.0);

    mkldnn::engine cpu_engine = CpuEngine::Instance().get_engine();
    memory::data_type mpcsn = memory::data_type::f32;
    memory::format mfmt_nchw = memory::format::nchw;
    memory::dims bottom_tz = { n, ic, ih, iw };
    memory::dims top_tz = {n, ic, ih, iw };

    //----- Initialize memory descriptors --------------
    //shared_ptr<memory::desc> bottom_data_md, top_data_md;
    typedef typename memory::primitive_desc MemPD;
    std::vector<memory::primitive_desc> bottom_data_mpd, top_data_md;
    for(auto i = 0; i < num_inputs_; i++)
    {
      fwd_bottom_data.push_back(std::shared_ptr<MKLDNNData<Dtype> >());
      memory::format cmfmt = mfmt_nchw;

      std::shared_ptr<MemPD> prv_bottom_data_mpd;
      std::shared_ptr<MemPD> usr_bottom_data_mpd(
        new memory::primitive_desc({bottom_tz, mpcsn, mfmt_nchw}, cpu_engine));

      bool bottom_data_is_prv =
       (const_cast<Dtype*>(mkl_prv_data<Dtype>(in_data[i])) != NULL);
      if (bottom_data_is_prv) {
       std::shared_ptr<MKLDNNData<Dtype> > mem_descr
         = get_mkldnn_prv_descriptor<Dtype>(in_data[i].Mkl_mem_);
       cmfmt = static_cast<memory::format>(mem_descr->prv_memory_pd()->desc().data.format);
       prv_bottom_data_mpd.reset(new memory::primitive_desc({bottom_tz, mpcsn, mfmt_nchw}, cpu_engine));
      }
      bottom_data_mpd.push_back(memory::primitive_desc({bottom_tz, mpcsn, mfmt_nchw}, cpu_engine));
      fwd_bottom_data[i].reset(new MKLDNNData<Dtype>(
            usr_bottom_data_mpd, prv_bottom_data_mpd, in_data[i], this));
      //bottom_data_mpd.push_back((new memory::desc({ bottom_tz }, mpcsn, cmfmt)));
    }
    std::shared_ptr<memory::primitive_desc> usr_top_data_md(new memory::desc({ top_tz }, mpcsn, mfmt_nchw));
      
    eltwiseFwd_pd.reset(new sum::primitive_desc({{n, ic, ih, iw}, mpcsn, memory::format::any}, scale, bottom_data_mpd));
    CHECK(eltwiseFwd_pd);
    std::shared_ptr<memory::primitive_desc> prv_dst_mpd(new memory::primitive_desc(eltwiseFwd_pd->dst_primitive_desc()));

    fwd_top_data.reset(new MKLDNNData<Dtype>(usr_top_data_md, prv_dst_mpd));
  }

  virtual void Forward(const nnvm::NodeAttrs& attrs,
  const OpContext& ctx,
  const std::vector<TBlob>& in_data,
  const std::vector<OpReqType>& req,
  const std::vector<TBlob>& out_data) {
  using namespace mshadow;
  using namespace mshadow::expr;

  if (req[0] == kNullOp) return;
  CHECK_EQ(out_data.size(), 1U);
  size_t size = in_data.size();
  Stream<cpu> *s = ctx.get_stream<cpu>();
  Tensor<cpu, 1, Dtype> data;
  Tensor<cpu, 1, Dtype> out;

  for (size_t i = 0; i < size; ++i) {
    data[i]  = mkl_experimental_direct_get<cpu, 1, Dtype>(in_data[i], s);
  }
  out = mkl_experimental_direct_get_with_shape<cpu, 1, Dtype>(out_data[0], s);

  if (!init_mkldnn_) {
      LayerSetup(data, size);
      InitEltwiseSumFwd(in_data);
      init_mkldnn_ = true;
   }
   // ---  init primitive and prv_memory descriptors ----------------------
    std::vector<primitive::at> inp;
    std::shared_ptr<memory> fwd_output_memory;

    for(auto i = 0; i < num_inputs_; i++)
    {
      std::shared_ptr<memory> fwd_input_memory;
      fwd_input_memory = fwd_bottom_data[i]->get_converted_prv(data[i].dptr_, false,
      in_data[i]);
      inp.push_back(*fwd_input_memory);
    }
    fwd_output_memory = fwd_top_data->create_output_memory(out.dptr_, out_data[eltwise::kOut],
      fwd_top_data);
    MKLDNNPrimitive<Dtype> eltwiseFwd;
    eltwiseFwd.reset(new mkldnn::sum(*eltwiseFwd_pd, inp,
        *fwd_output_memory));
    eltwiseFwd.submit();

}

  private:
  bool init_mkldnn_;

  std::shared_ptr<MKLDNNData<Dtype> > fwd_top_data, fwd_bottom_data;
  std::shared_ptr<sum::primitive_desc> eltwiseFwd_pd;
  int32_t num_inputs_;
  int32_t num_, channels_, width_, height_;
};  // class MKLDNNEltwiseSumOp

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MKL_DNN_MKLDNN_RELU_INL_H_