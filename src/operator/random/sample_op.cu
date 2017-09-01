/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file sample_op.cu
 * \brief GPU Implementation of sample op
 */
#include "./sample_op.h"

namespace mxnet {
namespace op {

// GPU versions of uniform and normal distribution.
template<>
void SampleUniformDnsImpl<gpu>(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx,
                               const OpReqType& req,
                               TBlob* output) {
  using namespace mxnet::op;
  using namespace mshadow::expr;
  typedef gpu xpu;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const SampleUniformParam& param = nnvm::get<SampleUniformParam>(attrs.parsed);
  mshadow::Random<xpu, float> *prnd = ctx.requested[0].get_random<xpu, float>(s);
  if (output->type_flag_ != mshadow::kFloat32) {
    MSHADOW_REAL_TYPE_SWITCH(output->type_flag_, DType, {
      // Not float32: use workspace and copy to output
      mshadow::Tensor<xpu, 2, DType> out = output->FlatTo2D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, float> workspace =
        ctx.requested[1].get_space_typed<xpu, 1, float>
        (mshadow::Shape1(out.shape_.Size()), s);
      prnd->SampleUniform(&workspace, param.low, param.high);
      out = reshape(tcast<DType>(workspace), mshadow::Shape2(out.shape_[0], out.shape_[1]));
    });
  } else {
    // float32: write directly into output
    mshadow::Tensor<xpu, 2, float> out = output->FlatTo2D<xpu, float>(s);
    prnd->SampleUniform(&out, param.low, param.high);
  }
}

template<>
void SampleUniform_<gpu>(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  TBlob out = outputs[0];
  SampleUniformDnsImpl<gpu>(attrs, ctx, req[0], &out);
}

template<>
void SampleNormalDnsImpl<gpu>(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const OpReqType& req,
                              TBlob* output) {
  using namespace mxnet::op;
  using namespace mshadow::expr;
  typedef gpu xpu;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const SampleNormalParam& param = nnvm::get<SampleNormalParam>(attrs.parsed);
  mshadow::Random<xpu, float> *prnd = ctx.requested[0].get_random<xpu, float>(s);
  if (output->type_flag_ != mshadow::kFloat32) {
    MSHADOW_REAL_TYPE_SWITCH(output->type_flag_, DType, {
      // Not float32: use workspace and copy to output
      mshadow::Tensor<xpu, 2, DType> out = output->FlatTo2D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, float> workspace =
        ctx.requested[1].get_space_typed<xpu, 1, float>
        (mshadow::Shape1(out.shape_.Size()), s);
      prnd->SampleGaussian(&workspace, param.loc, param.scale);
      out = reshape(tcast<DType>(workspace), mshadow::Shape2(out.shape_[0], out.shape_[1]));
    });
  } else {
    // float32: write directly into output
    mshadow::Tensor<xpu, 2, float> out = output->FlatTo2D<xpu, float>(s);
    prnd->SampleGaussian(&out, param.loc, param.scale);
  }
}

template<>
void SampleNormal_<gpu>(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  TBlob out = outputs[0];
  SampleNormalDnsImpl<gpu>(attrs, ctx, req[0], &out);
}

NNVM_REGISTER_OP(_random_uniform)
.set_attr<FCompute>("FCompute<gpu>", SampleUniform_<gpu>)
.set_attr<FComputeEx>("FComputeEx<gpu>", SampleUniformEx_<gpu>);

NNVM_REGISTER_OP(_random_normal)
.set_attr<FCompute>("FCompute<gpu>", SampleNormal_<gpu>)
.set_attr<FComputeEx>("FComputeEx<gpu>", SampleNormalEx_<gpu>);

}  // namespace op
}  // namespace mxnet
