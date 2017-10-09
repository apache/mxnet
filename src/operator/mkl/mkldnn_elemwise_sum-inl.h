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
* \file mkldnn_fully_connected-inl.h
* \brief
* \author young.jin.kim@intel.com
*         ashok.emani@intel.com
*         deepthi.karkada@intel.com
*         louis.feng@intel.com
*         adam.d.straw@intel.com
*
*
*******************************************************************************/

#pragma once

#include <dmlc/logging.h>
#include <cstring>
#include <vector>
#include <mkldnn_types.h>
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "./mkl_util-inl.h"

namespace mxnet {
namespace op {
  struct MKLDNNSum {
    template<typename DType>
    MSHADOW_XINLINE static DType sum(int i, const DType* a) {
      return a[i];
    }
    template<typename DType, typename... DTypes>
    MSHADOW_XINLINE static DType sum(int i, const DType* a, const DTypes... b) {
      return a[i] + sum(i, b...);
    }
    template<typename DType, typename... DTypes>
    MSHADOW_XINLINE static void Map(int i, DType* out, const OpReqType req, const DType* in0,
                                    const DTypes... ins) {
      KERNEL_ASSIGN(out[i], req, sum(i, in0, ins...));
    }
  };
  template<typename xpu, typename DType>
  void MKLDNNElementWiseSumCompute(const nnvm::NodeAttrs &attrs,
                                   const OpContext &ctx,
                                   const std::vector<TBlob> &in_data,
                                   const std::vector<OpReqType> &req,
                                   const std::vector<TBlob> &out_data) {
    using namespace mxnet_op;
#if 1
    {
      LOG(INFO) << __FUNCTION__;
      LOG(INFO) << "in_data.size: " << in_data.size() << " shape: " << in_data[0].shape_;
      int out_size = static_cast<int>((out_data[0].Size() + DataType<DType>::kLanes - 1)
                                      / DataType<DType>::kLanes);
      LOG(INFO) << "out_size: " << out_size << "shape: " << out_data[0].shape_;
      LOG(INFO) << "req[0]: " << req[0];
    }
#endif
#if 1
    {
      auto printTensor = [] (const std::string& name, const mshadow::Tensor<xpu, 1, DType>& t) {
        std::cout << "BEFORE " << name << " @" << t.dptr_ << " (" << t.size(0) << "): ";
        for (int i = 0; i < std::min(20, (int)t.size(0)); ++i) {
          std::cout << t[i] << " ";
        }
        std::cout << std::endl;
      };
      mshadow::Stream <xpu> *s = ctx.get_stream<xpu>();
      mshadow::Tensor<xpu, 1, DType> indata0 = in_data[0].FlatTo1D<xpu, DType>(s);
      printTensor("elemwise-sum indata.0", indata0);
      mshadow::Tensor<xpu, 1, DType> indata1 = in_data[1].FlatTo1D<xpu, DType>(s);
      printTensor("elemwise-sum indata.1", indata1);
      mshadow::Tensor<xpu, 1, DType> outdata0= out_data[0].FlatTo1D<xpu, DType>(s);
      printTensor("elemwise-sum outdata.0", outdata0);
    }
#endif
    if (req[0] == kNullOp) return;
    size_t size = in_data.size();
    Stream<xpu> *s = ctx.get_stream<xpu>();

#if 0
    DType * out_dptr = out_data[0].dptr<DType>();
    int out_size = static_cast<int>((out_data[0].Size() + DataType<DType>::kLanes - 1)
                                    / DataType<DType>::kLanes);
    switch (size) {
      case 2: {
        DType * in_0_dptr = in_data[0].dptr<DType>();
        DType * in_1_dptr = in_data[1].dptr<DType>();
        Kernel<MKLDNNSum, xpu>::Launch(s, out_size, out_dptr, req[0], in_0_dptr, in_1_dptr);
        break;
      }
      case 3: {
        DType * in_0_dptr = in_data[0].dptr<DType>();
        DType * in_1_dptr = in_data[1].dptr<DType>();
        DType * in_2_dptr = in_data[2].dptr<DType>();
        Kernel<MKLDNNSum, xpu>::Launch(s, out_size, out_dptr, req[0], in_0_dptr, in_1_dptr, in_2_dptr);
        break;
      }
      case 4: {
        DType * in_0_dptr = in_data[0].dptr<DType>();
        DType * in_1_dptr = in_data[1].dptr<DType>();
        DType * in_2_dptr = in_data[2].dptr<DType>();
        DType * in_3_dptr = in_data[3].dptr<DType>();
        Kernel<MKLDNNSum, xpu>::Launch(s, out_size, out_dptr, req[0], in_0_dptr, in_1_dptr, in_2_dptr,
                                 in_3_dptr);
        break;
      }
      default: {
        DType * in_0_dptr = in_data[0].dptr<DType>();
        Kernel<MKLDNNSum, xpu>::Launch(s, out_size, out_dptr, req[0], in_0_dptr);
        for (size_t i = 1; i < size; ++i) {
          DType * in_dptr = in_data[i].dptr<DType>();
          Kernel<MKLDNNSum, xpu>::Launch(s, out_size, out_dptr, req[0], out_dptr, in_dptr);
        }
        break;
      }
    }
#endif

#if 1
    using namespace mshadow;
    using namespace mshadow::expr;
    mkldnn::engine cpu_engine = mxnet::CpuEngine::Instance().get_engine();

    // get data shape info
    int32_t n = in_data[0].shape_[0];
    int32_t c = in_data[0].shape_[1];
    int32_t h = in_data[0].shape_[2];
    int32_t w = in_data[0].shape_[3];
    // setup output memory
    memory::desc default_desc = {{n, c, h, w}, memory::data_type::f32,
                                     memory::format::nchw};
//    memory::desc output_prv_desc = {{n, c, h, w}, memory::data_type::f32,
//                                    memory::format::nchw};
    // Inputs
    // get input memory descriptors
    std::vector<primitive::at> inputs;
    std::vector<memory::primitive_desc> input_usr_mpd;
    std::vector<memory::primitive_desc> input_prv_mpd;
    for (size_t i = 0; i < in_data.size(); ++i) {
      LOG(INFO) << "input " << i;
      void *input_ptr = mkl_prv_data<DType>(in_data[i]);
      if (input_ptr != nullptr) {
        LOG(INFO) << "input " << i << " is MKL";
        std::shared_ptr<MKLDNNData<DType>> input_dnn_data =
            get_mkldnn_prv_descriptor<DType>(in_data[i]);
        std::shared_ptr<memory> input_memory =
            input_dnn_data->get_converted_prv((float *) in_data[i].dptr_, false,
                                              in_data[i]);

        input_usr_mpd.push_back(*input_dnn_data->usr_memory_pd());
        input_prv_mpd.push_back(*input_dnn_data->prv_memory_pd());
        LOG(INFO) << "input usr format " << input_usr_mpd[i].desc().data.format;
        LOG(INFO) << "input prv format " << input_prv_mpd[i].desc().data.format;
        inputs.push_back(*input_memory);
      } else {
        LOG(INFO) << "input " << i << " is NOT MKL";

        std::shared_ptr<memory::primitive_desc> input_usr_mpd;
        input_usr_mpd.reset(new memory::primitive_desc(default_desc,
                                                       cpu_engine));

        std::shared_ptr<memory::primitive_desc> input_prv_mpd;
        input_prv_mpd.reset(new memory::primitive_desc(default_desc,
                                                       cpu_engine));

        std::shared_ptr<MKLDNNData<DType>> input_dnn_data;
        input_dnn_data.reset(new MKLDNNData<DType>(input_usr_mpd,
                                                   input_prv_mpd));

        // create output memory primitive
        std::shared_ptr<memory> input_memory =
            input_dnn_data->create_output_memory((float *) in_data[i].dptr_,
                                                 in_data[i],
                                                 input_dnn_data);
      }
    }

    std::shared_ptr<memory> output;
    memory::desc output_prv_desc(default_desc);
    void *output_ptr = mkl_prv_data<DType>(out_data[0]);
    if (output_ptr != nullptr) {
      LOG(INFO) << "output is MKL";
      std::shared_ptr<MKLDNNData<DType>> output_dnn_data =
          get_mkldnn_prv_descriptor<DType>(out_data[0]);
      std::shared_ptr<memory::primitive_desc>
          output_usr_mpd = output_dnn_data->usr_memory_pd();
      std::shared_ptr<memory::primitive_desc>
          output_prv_mpd = output_dnn_data->prv_memory_pd();
      LOG(INFO) << "output usr format " << output_usr_mpd->desc().data.format;
      LOG(INFO) << "output prv format " << output_prv_mpd->desc().data.format;

      output_prv_desc = output_prv_mpd->desc().data;
      output =
          output_dnn_data->get_converted_prv((float *) out_data[0].dptr_, false,
                                             out_data[0]);
    } else {
      LOG(INFO) << "output is NOT MKL";

      std::shared_ptr<memory::primitive_desc> output_usr_mpd;
      output_usr_mpd.reset(new memory::primitive_desc(default_desc,
                                                      cpu_engine));
      LOG(INFO) << "output usr done";
      std::shared_ptr<memory::primitive_desc> output_prv_mpd;
      output_prv_mpd.reset(new memory::primitive_desc(output_prv_desc,
                                                      cpu_engine));
      LOG(INFO) << "output prv done";
      std::shared_ptr<MKLDNNData<DType>> output_dnn_data;
      output_dnn_data.reset(new MKLDNNData<DType>(output_usr_mpd,
                                                  output_prv_mpd));
      LOG(INFO) << "output dnn data done";
      // create output memory primitive
      output =
          output_dnn_data->create_output_memory((float *) out_data[0].dptr_,
                                                out_data[0],
                                                output_dnn_data);
    }
//    LOG(INFO) << "output done";
    std::vector<double> scale(in_data.size(), 1.0);

    // sum primitive descriptor
    // need output memory::desc, scale per input, and memory primitive_desc for
    // inputs
    sum::primitive_desc sum_pd(output_prv_desc, scale, input_prv_mpd);

    MKLDNNPrimitive<DType> elemwise_sum;
    elemwise_sum.reset(new mkldnn::sum(sum_pd, inputs, *output));
//    LOG(INFO) << "before submit";
    elemwise_sum.submit();
//    LOG(INFO) << "after submit";
#endif
#if 1
    {
      auto printTensor = [] (const std::string& name, const mshadow::Tensor<xpu, 1, DType>& t) {
        std::cout << "AFTER " << name << " @" << t.dptr_ << " (" << t.size(0) << "): ";
        for (int i = 0; i < std::min(20, (int)t.size(0)); ++i) {
          std::cout << t[i] << " ";
        }
        std::cout << std::endl;
      };
      mshadow::Stream <xpu> *s = ctx.get_stream<xpu>();
      mshadow::Tensor<xpu, 1, DType> indata0 = in_data[0].FlatTo1D<xpu, DType>(s);
      printTensor("elemwise-sum indata.0", indata0);
      mshadow::Tensor<xpu, 1, DType> indata1 = in_data[1].FlatTo1D<xpu, DType>(s);
      printTensor("elemwise-sum indata.1", indata1);
      mshadow::Tensor<xpu, 1, DType> outdata0= out_data[0].FlatTo1D<xpu, DType>(s);
      printTensor("elemwise-sum outdata.0", outdata0);
    }
#endif

  }

}
}
