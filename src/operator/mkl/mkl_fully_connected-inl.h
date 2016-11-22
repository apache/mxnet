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
* \file mkl_fully_connected-inl.h
* \brief
* \author zhenlin.luo@intel.com
*          lingyan.guo@intel.com
*         
*
*******************************************************************************/
#ifndef MXNET_OPERATOR_MKL_MKL_FULLY_CONNECTED_INL_H_
#define MXNET_OPERATOR_MKL_MKL_FULLY_CONNECTED_INL_H_
#include <string>
#include <algorithm>
#include <vector>
#include "../activation-inl.h"
#include "./mkl_util-inl.h"

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
class MKLFullyConnectedOp : public Operator {
 public:
  explicit MKLFullyConnectedOp(FullyConnectedParam p):
    init_mkldnn_(false),
    fullyConnectedFwd(NULL),
    fullyConnectedBwdData(NULL),
    fullyConnectedBwdFilter(NULL),
    fullyConnectedBwdBias(NULL) {
    param_ = p;
    fwd_top_data = MKLData<DType>::create();
    fwd_bottom_data = MKLData<DType>::create();
    bwd_bottom_diff = MKLData<DType>::create();
    bwd_top_diff = MKLData<DType>::create();
  }

  ~MKLFullyConnectedOp() {
    dnnDelete<DType>(fullyConnectedFwd);
    dnnDelete<DType>(fullyConnectedBwdData);
    dnnDelete<DType>(fullyConnectedBwdFilter);
    dnnDelete<DType>(fullyConnectedBwdBias);
  }
  static std::string getName() {
    return "MKLFullyConnectedOp";
  }

 private:
    void LayerSetUp(const mshadow::Tensor<xpu, 4, DType> &data,
                   const mshadow::Tensor<xpu, 4, DType> &out) {
    size_t src_sizes[4];
    size_t dst_sizes[2];

    size_t dim = 4;
    int status;
    const size_t input_batch_size = data.size(0);
    const size_t input_channels = data.size(1);
    const size_t input_height = data.size(2);
    const size_t input_width = data.size(3);

    const size_t output_batch_size = out.size(0);
    const size_t output_channels = out.size(1);

    src_sizes[0] = input_width;
    src_sizes[1] = input_height;
    src_sizes[2] = input_channels;
    src_sizes[3] = input_batch_size;

    dst_sizes[0] = output_channels;
    dst_sizes[1] = output_batch_size;

    // Names are for debugging only
    fwd_bottom_data->name = "fwd_bottom_data   @ " + getName();
    fwd_top_data->name = "fwd_top_data      @ " + getName();
    bwd_bottom_diff->name = "bwd_bottom_diff   @ " + getName();
    bwd_top_diff->name = "bwd_top_diff      @ " + getName();

    dnnPrimitiveAttributes_t attributes = NULL;
    status = dnnPrimitiveAttributesCreate<DType>(&attributes);
    CHECK_EQ(status, 0);
    if (!param_.no_bias) {
      status = dnnInnerProductCreateForwardBias<DType>(&fullyConnectedFwd,
                  attributes,
                  dim,
                  src_sizes,
                  output_channels);
      CHECK_EQ(status, 0)
      << "Failed dnnInnerProductCreateForwardBias with status "
      << status << "\n";
    } else {
      status = dnnInnerProductCreateForward<DType>(&fullyConnectedFwd,
                attributes,
                dim,
                src_sizes,
                output_channels);
      CHECK_EQ(status, 0)
      << "Failed dnnInnerProductCreateForward with status "
      << status << "\n";
    }
    status = dnnInnerProductCreateBackwardData<DType>(&fullyConnectedBwdData,
            attributes,
            dim,
            src_sizes,
            output_channels);
    CHECK_EQ(status, 0)
      << "Failed dnnInnerProductCreateBackwardData with status "
      << status << "\n";
    status = dnnInnerProductCreateBackwardFilter<DType>(&fullyConnectedBwdFilter,
            attributes,
            dim,
            src_sizes,
            output_channels);
    CHECK_EQ(status, 0)
      << "Failed dnnInnerProductCreateBackwardFilter with status "
      << status << "\n";
    if (!param_.no_bias) {
      status = dnnInnerProductCreateBackwardBias<DType>(&fullyConnectedBwdBias,
                  attributes,
                  2,
                  dst_sizes);
      CHECK_EQ(status, 0) << "Backward Bias failed with status " << status;
    }
  }


  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;

    if (req[fullc::kOut] == kNullOp) return;
    CHECK_EQ(req[fullc::kOut], kWriteTo);
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1);
    int status;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const TShape& ishape = in_data[fullc::kData].shape_;
    const TShape& oshape = out_data[fullc::kOut].shape_;

    Tensor<xpu, 4, DType> data;
    Tensor<xpu, 4, DType> out;

    Shape4(in_data[fullc::kData].shape_[0], in_data[fullc::kData].shape_[1], 1, 1);

    Shape<4> dshape = Shape4(ishape[0], ishape.ProdShape(1, ishape.ndim()), 1, 1);
    Shape<4> odshape = Shape4(oshape[0], oshape.ProdShape(1, oshape.ndim()), 1, 1);

    data = in_data[fullc::kData].get_with_shape<xpu, 4, DType>(dshape, s);
    out = out_data[fullc::kOut].get_with_shape<xpu, 4, DType>(odshape, s);

    if (!init_mkldnn_) {
      LayerSetUp(data, out);
      init_mkldnn_ = true;
    }

    res_fullyConnected[dnnResourceSrc] =
      reinterpret_cast<void *>(in_data[fullc::kData].dptr_);
    res_fullyConnected[dnnResourceDst] =
      reinterpret_cast<void *>(out_data[fullc::kOut].dptr_);
    res_fullyConnected[dnnResourceFilter] =
      reinterpret_cast<void *>(in_data[fullc::kWeight].dptr_);
    if (!param_.no_bias) {
      res_fullyConnected[dnnResourceBias] = reinterpret_cast<void *>(in_data[fullc::kBias].dptr_);
    }

    status = dnnExecute<DType>(fullyConnectedFwd, res_fullyConnected);
    CHECK_EQ(status, 0) << "Forward FC failed with status " << status;
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;

    CHECK_EQ(out_grad.size(), 1);
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK(in_data.size() == expected && in_grad.size() == expected);
    CHECK_EQ(req.size(), expected);
    int status;
    res_fullyConnected[dnnResourceSrc] =
      reinterpret_cast<void *>(in_data[fullc::kData].dptr_);
    res_fullyConnected[dnnResourceFilter] =
      reinterpret_cast<void *>(in_data[fullc::kWeight].dptr_);

    res_fullyConnected[dnnResourceDiffDst] =
      reinterpret_cast<void *>(out_grad[fullc::kOut].dptr_);
    res_fullyConnected[dnnResourceDiffSrc] =
      reinterpret_cast<void *>(in_grad[fullc::kData].dptr_);
    res_fullyConnected[dnnResourceDiffFilter] =
      reinterpret_cast<void *>(in_grad[fullc::kWeight].dptr_);
    if (!param_.no_bias) {
      res_fullyConnected[dnnResourceDiffBias] =
        reinterpret_cast<void *>(in_grad[fullc::kBias].dptr_);
    }
    status = dnnExecute<DType>(fullyConnectedBwdFilter, res_fullyConnected);
    CHECK_EQ(status, 0) << "Backward FC Filter failed with status " << status;
    if (!param_.no_bias) {
      status = dnnExecute<DType>(fullyConnectedBwdBias, res_fullyConnected);
      CHECK_EQ(status, 0) << "Backward FC Bias failed with status " << status;
    }
    status = dnnExecute<DType>(fullyConnectedBwdData, res_fullyConnected);
    CHECK_EQ(status, 0) << "Backward FC Data failed with status " << status;
  }

 private:
  bool init_mkldnn_;
  dnnPrimitive_t fullyConnectedFwd;
  dnnPrimitive_t fullyConnectedBwdData;
  dnnPrimitive_t fullyConnectedBwdFilter;
  dnnPrimitive_t fullyConnectedBwdBias;
  std::shared_ptr<MKLData<DType>> fwd_top_data, fwd_bottom_data;
  std::shared_ptr<MKLData<DType>> bwd_bottom_diff, bwd_top_diff;
  FullyConnectedParam param_;
  void* res_fullyConnected[dnnResourceNumber];
};  // class MKLFullyConnectedOp
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_MKL_MKL_FULLY_CONNECTED_INL_H_
