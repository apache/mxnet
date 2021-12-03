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
 * \file quantize_asym-inl.h
 * \brief implementation of asymmetric quantize operation
 */
#ifndef MXNET_OPERATOR_QUANTIZATION_QUANTIZE_ASYM_INL_H_
#define MXNET_OPERATOR_QUANTIZATION_QUANTIZE_ASYM_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mshadow/tensor.h>
#include <mxnet/operator_util.h>
#include <vector>

#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../tensor/broadcast_reduce_op.h"
#include "./quantization_utils.h"

namespace mxnet {
namespace op {

struct QuantizeAsymParam : public dmlc::Parameter<QuantizeAsymParam> {
  dmlc::optional<float> min_calib_range;
  dmlc::optional<float> max_calib_range;

  DMLC_DECLARE_PARAMETER(QuantizeAsymParam) {
    DMLC_DECLARE_FIELD(min_calib_range)
        .set_default(dmlc::optional<float>())
        .describe(
            "The minimum scalar value in the form of float32. If "
            "present, it will be used to "
            "quantize the fp32 data.");
    DMLC_DECLARE_FIELD(max_calib_range)
        .set_default(dmlc::optional<float>())
        .describe(
            "The maximum scalar value in the form of float32. If "
            "present, it will be used to "
            "quantize the fp32 data.");
  }
};

// quantize float to uint8_t
struct quantize_asymmetric {
  template <typename DstDType, typename SrcDType>
  MSHADOW_XINLINE static void Map(int i,
                                  DstDType* out,
                                  float* oscale,
                                  float* oshift,
                                  const SrcDType* in,
                                  const float scale,
                                  const float shift) {
    out[i]  = static_cast<DstDType>(in[i] * scale + shift + 0.5);
    *oscale = scale;
    *oshift = shift;
  }
};

template <typename xpu>
class QuantizeAsymOp {
 public:
  explicit QuantizeAsymOp(const nnvm::NodeAttrs& attrs) : attrs_(attrs) {}

  void Forward(const OpContext& ctx,
               const std::vector<TBlob>& inputs,
               const std::vector<OpReqType>& req,
               const std::vector<TBlob>& outputs) {
    using namespace mshadow;
    using namespace mxnet_op;
    using mshadow::red::limits::MaxValue;
    using mshadow::red::limits::MinValue;

    CHECK_EQ(outputs[0].type_flag_, mshadow::kUint8)
        << "Asymmetric quantization only supports uint8 outputs.";
    mshadow::Stream<xpu>* s    = ctx.get_stream<xpu>();
    const int input_data_dtype = inputs[0].type_flag_;
    if (input_data_dtype == mshadow::kUint8) {
      *outputs[1].dptr<float>() = 1;
      *outputs[2].dptr<float>() = 0;
      UnaryOp::IdentityCompute<xpu>(attrs_, ctx, {inputs[0]}, req, outputs);
    } else if (input_data_dtype == mshadow::kInt8) {
      const float scale = 1;
      const float shift = 128;
      Kernel<quantize_asymmetric, xpu>::Launch(s,
                                               outputs[0].Size(),
                                               outputs[0].dptr<uint8_t>(),
                                               outputs[1].dptr<float>(),
                                               outputs[2].dptr<float>(),
                                               inputs[0].dptr<int8_t>(),
                                               scale,
                                               shift);
    } else if (input_data_dtype == mshadow::kFloat32) {
      const QuantizeAsymParam& param = nnvm::get<QuantizeAsymParam>(attrs_.parsed);
      if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
        const float scale =
            MaxValue<uint8_t>() / (param.max_calib_range.value() - param.min_calib_range.value());
        const float shift = MaxValue<uint8_t>() - param.max_calib_range.value() * scale;
        Kernel<quantize_asymmetric, xpu>::Launch(s,
                                                 outputs[0].Size(),
                                                 outputs[0].dptr<uint8_t>(),
                                                 outputs[1].dptr<float>(),
                                                 outputs[2].dptr<float>(),
                                                 inputs[0].dptr<float>(),
                                                 scale,
                                                 shift);
      } else {
        mxnet::TShape src_shape, dst_shape;
        const size_t float_bytes      = sizeof(float);
        const size_t temp_reduce_size = ConfigReduce<xpu, float>(
            s, inputs[0].shape_, mxnet::TShape(1, 1), &src_shape, &dst_shape);
        Tensor<xpu, 1, char> temp_space = ctx.requested[0].get_space_typed<xpu, 1, char>(
            Shape1(2 * float_bytes + temp_reduce_size), s);
        const int dev_id = ctx.run_ctx.ctx.dev_id;
        TBlob in_min_t(
            reinterpret_cast<float*>(temp_space.dptr_), Shape1(1), xpu::kDevMask, dev_id);
        TBlob in_max_t(
            reinterpret_cast<float*>(temp_space.dptr_) + 1, Shape1(1), xpu::kDevMask, dev_id);
        Tensor<xpu, 1, char> workspace(
            temp_space.dptr_ + 2 * float_bytes, Shape1(temp_reduce_size), s);
        broadcast::Reduce<red::minimum, 2, float, mshadow::op::identity>(
            s, in_min_t.reshape(dst_shape), kWriteTo, workspace, inputs[0].reshape(src_shape));
        broadcast::Reduce<red::maximum, 2, float, mshadow::op::identity>(
            s, in_max_t.reshape(dst_shape), kWriteTo, workspace, inputs[0].reshape(src_shape));
        const float scale =
            MaxValue<uint8_t>() / (*in_max_t.dptr<float>() - *in_min_t.dptr<float>());
        const float shift = MaxValue<uint8_t>() - *in_max_t.dptr<float>() * scale;
        Kernel<quantize_asymmetric, xpu>::Launch(s,
                                                 outputs[0].Size(),
                                                 outputs[0].dptr<uint8_t>(),
                                                 outputs[1].dptr<float>(),
                                                 outputs[2].dptr<float>(),
                                                 inputs[0].dptr<float>(),
                                                 scale,
                                                 shift);
      }
    } else {
      LOG(FATAL) << "Asymmetric quantizaiton only supports int8, uint8 and "
                    "float inputs";
    }
  }

 private:
  nnvm::NodeAttrs attrs_;
};

template <typename xpu>
void QuantizeAsymForward(const OpStatePtr& state_ptr,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  QuantizeAsymOp<xpu>& op = state_ptr.get_state<QuantizeAsymOp<xpu>>();
  op.Forward(ctx, inputs, req, outputs);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_QUANTIZATION_QUANTIZE_ASYM_INL_H_
