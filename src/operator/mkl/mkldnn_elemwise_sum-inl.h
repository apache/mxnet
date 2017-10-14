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
* \file mkldnn_elemwise_sum-inl.h
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
#include "mkl_util-inl.h"

namespace mxnet {
namespace op {

/**
 * Adds n input data element-wise and store output to a single output buffer.
 * @tparam xpu
 * @tparam DType
 * @param attrs
 * @param ctx
 * @param in_data
 * @param req
 * @param out_data
 */
template<typename xpu, typename DType>
void MKLDNNElementWiseSumCompute(const nnvm::NodeAttrs &attrs,
                                 const OpContext &ctx,
                                 const std::vector<TBlob> &in_data,
                                 const std::vector<OpReqType> &req,
                                 const std::vector<TBlob> &out_data) {
  using namespace mxnet_op;
  using namespace mshadow;
  using namespace mshadow::expr;

  if (req[0] == kNullOp) return;
  // expecting to sum at least two input buffers
  assert(in_data.size() >= 2);

  Stream<xpu> *s = ctx.get_stream<xpu>();
  mkldnn::engine cpu_engine = mxnet::CpuEngine::Instance().get_engine();

  // assuming all in_data TBlobs have same shape
  assert(in_data[0].shape_[0] == in_data[1].shape_[0]);
  assert(in_data[0].shape_[1] == in_data[1].shape_[1]);
  assert(in_data[0].shape_[2] == in_data[1].shape_[2]);
  assert(in_data[0].shape_[3] == in_data[1].shape_[3]);
  // get data shape info
  int32_t n = in_data[0].shape_[0];
  int32_t c = in_data[0].shape_[1];
  int32_t h = in_data[0].shape_[2];
  int32_t w = in_data[0].shape_[3];

  // if we are dealing with cpu only input/output data, we will use this
  // descriptor format as default
  memory::desc default_usr_desc =
      {{n, c, h, w}, memory::data_type::f32, memory::format::nchw};

  // start with the output data descriptor which will determine the layout for
  // the inputs. which may need to be converted.
  std::shared_ptr<memory> output;

  // this is needed for the sum primitive descriptor, initialize to be the
  // same as default usr descriptor.
  memory::desc output_prv_desc(default_usr_desc);

  std::shared_ptr<memory::primitive_desc> output_usr_mpd;
  std::shared_ptr<memory::primitive_desc> output_prv_mpd;
  // check if output data has a valid prv buffer set up
  // TODO lfeng: it's possible that mkl prv data exists but is not valid
  // (head_ != HEAD_AT_PRV), this should not happen in general and could mean
  // the previous MKLDNN operator has a bug causing the output data to have
  // head_ flag set to HEAD_AT_CPU. We should find a way to detect this case.
  void *output_ptr = mkl_prv_data<DType>(out_data[0]);
  if (output_ptr != nullptr) {
    // the output data has a valid prv buffer, we will use it directly
    std::shared_ptr<MKLDNNData<DType>>
        output_dnn_data = get_mkldnn_prv_descriptor<DType>(out_data[0]);
    // get memory primitive descriptor for usr and prv
    output_usr_mpd = output_dnn_data->usr_memory_pd();
    output_prv_mpd = output_dnn_data->prv_memory_pd();

    output_prv_desc = output_prv_mpd->desc().data;

    // use the output prv memory directly
    output = output_dnn_data->get_prv_memory();

  } else {
    // TODO lfeng: this should be rare and expensive, maybe output a warning?
    // if output data does not have a mkl prv buffer, we assume usr
    // layout, default is nchw
    output_usr_mpd.reset(new memory::primitive_desc(default_usr_desc,
                                                    cpu_engine));
    output_prv_mpd.reset(new memory::primitive_desc(output_prv_desc,
                                                    cpu_engine));
    std::shared_ptr<MKLDNNData<DType>> output_dnn_data;
    output_dnn_data.reset(new MKLDNNData<DType>(output_usr_mpd,
                                                output_prv_mpd));
    // create output memory primitive and update the out_data[0].Mkl_mem_ data
    // to use output_dnn_data, in this case it means cpu data buffer will be
    // used for output prv data (since their layouts are the same)
    output =
        output_dnn_data->create_output_memory(static_cast<DType *>(out_data[0].dptr_),
                                              out_data[0],
                                              output_dnn_data);
  }

  // Inputs - get input memory descriptors
  std::vector<primitive::at> inputs;
  // store an input memory primitive descriptor for each input data, this is
  // required for creating sum primitive descriptor
  std::vector<memory::primitive_desc> input_prv_mpd_array;
  std::shared_ptr<MKLDNNData<DType>> input_dnn_data;
  for (size_t i = 0; i < in_data.size(); ++i) {
    std::shared_ptr<memory::primitive_desc> input_usr_mpd;
    std::shared_ptr<memory::primitive_desc> input_prv_mpd;
    // checking if we have mkldnn prv data
    void *input_ptr = mkl_prv_data<DType>(in_data[i]);
    if (input_ptr != nullptr) {
      // input data has valid prv buffer
      input_dnn_data = get_mkldnn_prv_descriptor<DType>(in_data[i]);
      input_usr_mpd = input_dnn_data->usr_memory_pd();
      input_prv_mpd = input_dnn_data->prv_memory_pd();
      // check input prv descriptor match output prv descriptor
      if (input_prv_mpd != output_prv_mpd) {
        // TODO lfeng: this should be rare and expensive, maybe output a warning?
        // input and output prv layout are different, we don't want modify
        // the input data object prv buffer, so we need to create annew
        // MKLDNNData object and do a conversion and copy data to new memory
        // buffer, this is expensive.
        input_prv_mpd.reset(new memory::primitive_desc(output_prv_desc,
                                                       cpu_engine));
        input_dnn_data.reset(new MKLDNNData<DType>(input_usr_mpd,
                                                   input_prv_mpd));
      }
      input_prv_mpd_array.push_back(*input_prv_mpd);
    } else {
      // default usr descriptor
      input_usr_mpd.reset(new memory::primitive_desc(default_usr_desc,
                                                     cpu_engine));

      // for prv buffer, we want to match with the output prv desc
      input_prv_mpd.reset(new memory::primitive_desc(output_prv_desc,
                                                     cpu_engine));
      input_prv_mpd_array.push_back(*input_prv_mpd);

      input_dnn_data.reset(new MKLDNNData<DType>(input_usr_mpd, input_prv_mpd));
    }

    // this is where the magic happens. Depending on how the layouts are
    // configured, we should get a prv pointer with valid layout for the input.
    std::shared_ptr<memory> input_memory =
        input_dnn_data->get_converted_prv(static_cast<float *>(in_data[i].dptr_),
                                          false,
                                          in_data[i]);
    inputs.push_back(*input_memory);
  }

  // scaling factor for each input data
  std::vector<double> scale(in_data.size(), 1.0);

  // sum primitive descriptor
  // need output memory::desc, scale per input, and memory primitive_desc for
  // inputs
  sum::primitive_desc sum_pd(output_prv_desc, scale, input_prv_mpd_array);

  MKLDNNPrimitive<DType> elemwise_sum;
  elemwise_sum.reset(new mkldnn::sum(sum_pd, inputs, *output));
  elemwise_sum.submit();
}

/**
 * Intended for adding two input buffers element-wise and store in a single
 * output buffer.
 * @tparam xpu
 * @param attrs
 * @param ctx
 * @param in_data
 * @param req
 * @param out_data
 */
template<typename xpu>
void MKLDNNElementWiseAddCompute(const nnvm::NodeAttrs &attrs,
                                 const OpContext &ctx,
                                 const std::vector<TBlob> &in_data,
                                 const std::vector<OpReqType> &req,
                                 const std::vector<TBlob> &out_data) {


  if (req[0] == kNullOp) return;
  CHECK_EQ(in_data.size(), 2U);
  CHECK_EQ(out_data.size(), 1U);
  CHECK_EQ(out_data[0].type_flag_, mshadow::kFloat32) << "elemwise_add data"
      " type must be float";

  MKLDNNElementWiseSumCompute<cpu, float>(attrs, ctx, in_data, req, out_data);
}
}
}
