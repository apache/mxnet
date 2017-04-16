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
  explicit MKLFullyConnectedOp(const FullyConnectedParam& p,
                               const std::vector<TShape>& in_shapes,
                               const std::vector<TShape>& out_shapes):
    param_(p) {
    LayerSetUp(in_shapes, out_shapes);
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
  void LayerSetUp(const std::vector<TShape>& in_shapes,
                  const std::vector<TShape>& out_shapes) {
    const TShape& ishape = in_shapes[fullc::kData];

    const size_t dim = 4;
    const size_t src_sizes[4] = {1, 1, ishape.ProdShape(1, ishape.ndim()), ishape[0]};
    const size_t dst_sizes[2] = {param_.num_hidden, ishape[0]};
    const size_t output_channels = param_.num_hidden;

    dnnPrimitiveAttributes_t attributes = NULL;
    MKLDNN_CALL(dnnPrimitiveAttributesCreate<DType>(&attributes));
    if (!param_.no_bias) {
      MKLDNN_CALL(dnnInnerProductCreateForwardBias<DType>(
            &fullyConnectedFwd,
            attributes,
            dim,
            src_sizes,
            output_channels));
    } else {
      MKLDNN_CALL(dnnInnerProductCreateForward<DType>(
            &fullyConnectedFwd,
            attributes,
            dim,
            src_sizes,
            output_channels));
    }
    MKLDNN_CALL(dnnInnerProductCreateBackwardData<DType>(
          &fullyConnectedBwdData,
          attributes,
          dim,
          src_sizes,
          output_channels));
    MKLDNN_CALL(dnnInnerProductCreateBackwardFilter<DType>(
          &fullyConnectedBwdFilter,
          attributes,
          dim,
          src_sizes,
          output_channels));
    if (!param_.no_bias) {
      MKLDNN_CALL(dnnInnerProductCreateBackwardBias<DType>(
            &fullyConnectedBwdBias,
            attributes,
            2,
            dst_sizes));
    }
    // TODO(minjie): Shouldn't `attributes` be destroyed?
  }


  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;

    void* res_fullyConnected[dnnResourceNumber];
    if (req[fullc::kOut] == kNullOp) return;
    CHECK_EQ(req[fullc::kOut], kWriteTo);
    CHECK_EQ(in_data.size(), param_.no_bias ? 2 : 3);
    CHECK_EQ(out_data.size(), 1);
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
    res_fullyConnected[dnnResourceSrc] =
      reinterpret_cast<void *>(in_data[fullc::kData].dptr_);
    res_fullyConnected[dnnResourceDst] =
      reinterpret_cast<void *>(out_data[fullc::kOut].dptr_);
    res_fullyConnected[dnnResourceFilter] =
      reinterpret_cast<void *>(in_data[fullc::kWeight].dptr_);
    if (!param_.no_bias) {
      res_fullyConnected[dnnResourceBias] = reinterpret_cast<void *>(in_data[fullc::kBias].dptr_);
    }

    MKLDNN_CALL(dnnExecute<DType>(fullyConnectedFwd, res_fullyConnected));
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

    void* res_fullyConnected[dnnResourceNumber];
    CHECK_EQ(out_grad.size(), 1);
    const size_t expected = param_.no_bias ? 2 : 3;
    CHECK(in_data.size() == expected && in_grad.size() == expected);
    CHECK_EQ(req.size(), expected);
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
    MKLDNN_CALL(dnnExecute<DType>(fullyConnectedBwdFilter, res_fullyConnected));
    if (!param_.no_bias) {
      MKLDNN_CALL(dnnExecute<DType>(fullyConnectedBwdBias, res_fullyConnected));
    }
    MKLDNN_CALL(dnnExecute<DType>(fullyConnectedBwdData, res_fullyConnected));
  }

 private:
  dnnPrimitive_t fullyConnectedFwd{nullptr};
  dnnPrimitive_t fullyConnectedBwdData{nullptr};
  dnnPrimitive_t fullyConnectedBwdFilter{nullptr};
  dnnPrimitive_t fullyConnectedBwdBias{nullptr};
  const FullyConnectedParam param_;
};  // class MKLFullyConnectedOp
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_MKL_MKL_FULLY_CONNECTED_INL_H_
