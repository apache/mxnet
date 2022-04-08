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
 * \file dnnl_quantized_elemwise_add.cc
 * \brief
 */

#if MXNET_USE_ONEDNN == 1
#include "operator/nn/dnnl/dnnl_base-inl.h"
#include "operator/nn/dnnl/dnnl_ops-inl.h"
#include "operator/quantization/quantization_utils.h"
#include "operator/quantization/quantized_elemwise_add-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(QuantizeElemwiseAddParam);

static inline float GetScale(const NDArray& data, float min, float max) {
  auto data_range = (data.dtype() == mshadow::kInt8) ? kInt8Range : kUint8Range;
  return data_range / MaxAbs(min, max);
}

class DNNLQuantizedElemwiseSumFwd {
 public:
  dnnl::sum::primitive_desc fwd_pd;

  DNNLQuantizedElemwiseSumFwd(const dnnl::memory::desc& output_md,
                              const std::vector<float>& scales,
                              const std::vector<dnnl::memory::desc>& inputs_md)
      : fwd_pd(output_md, scales, inputs_md, CpuEngine::Get()->get_engine()) {
    fwd_ = std::make_shared<dnnl::sum>(fwd_pd);
  }

  const dnnl::sum& GetFwd() const {
    return *fwd_;
  }

 private:
  std::shared_ptr<dnnl::sum> fwd_;
  std::shared_ptr<dnnl::memory> out_;
};

static DNNLQuantizedElemwiseSumFwd& GetQuantizedElemwiseSumForward(
    const dnnl::memory::desc& output_md,
    const std::vector<float>& scales,
    const std::vector<dnnl::memory::desc>& inputs_md) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<OpSignature, DNNLQuantizedElemwiseSumFwd, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<OpSignature, DNNLQuantizedElemwiseSumFwd, OpHash> fwds;
#endif
  OpSignature key;
  key.AddSign(output_md);
  key.AddSign(scales);
  key.AddSign(inputs_md);
  auto it = fwds.find(key);
  if (it == fwds.end()) {
    DNNLQuantizedElemwiseSumFwd fwd(output_md, scales, inputs_md);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

static void DNNLQuantizedElemwiseAddForward(const nnvm::NodeAttrs& attrs,
                                            const OpContext& ctx,
                                            const std::vector<NDArray>& inputs,
                                            const std::vector<OpReqType>& req,
                                            const std::vector<NDArray>& outputs) {
  const QuantizeElemwiseAddParam& params = nnvm::get<QuantizeElemwiseAddParam>(attrs.parsed);
  // A, B, A_min, A_max, B_min, B_max
  CHECK_EQ(inputs.size(), 6U) << "should be A, B, A_min, A_max, B_min, B_max";
  // C, C_min, C_max
  CHECK_EQ(outputs.size(), 3U) << "should be C, C_min, C_max";
  // Collect data min,max,absmax
  const float A_min    = inputs[q_elemwise_add::kAMin].data().dptr<float>()[0];
  const float B_min    = inputs[q_elemwise_add::kBMin].data().dptr<float>()[0];
  const float A_max    = inputs[q_elemwise_add::kAMax].data().dptr<float>()[0];
  const float B_max    = inputs[q_elemwise_add::kBMax].data().dptr<float>()[0];
  const float A_absmax = MaxAbs(A_min, A_max);
  const float B_absmax = MaxAbs(B_min, B_max);
  const bool is_A_int8 = (inputs[q_elemwise_add::kDataA].dtype() == mshadow::kInt8);
  const float A_range  = is_A_int8 ? kInt8Range : kUint8Range;
  const float A_scale  = GetScale(inputs[q_elemwise_add::kDataA], A_min, A_max);
  const float B_scale  = GetScale(inputs[q_elemwise_add::kDataB], B_min, B_max);
  auto A_mem           = inputs[q_elemwise_add::kDataA].GetDNNLData();
  auto B_mem           = inputs[q_elemwise_add::kDataB].GetDNNLData();
  dnnl::memory* rescaled_mem;              // rescaled_mem is for reorder dnnl memory
  double output_data_range = kInt32Range;  // output default set as int32
  if (outputs[q_elemwise_add::kOut].dtype() == mshadow::kInt8) {
    output_data_range = kInt8Range;
  } else if (outputs[q_elemwise_add::kOut].dtype() == mshadow::kUint8) {
    output_data_range = kUint8Range;
  }

  float output_min     = 0;
  float output_max     = 0;
  float output_scale   = 0;
  if (params.max_calib_range.has_value() && params.min_calib_range.has_value()) {
    output_min     = params.min_calib_range.value();
    output_max     = params.max_calib_range.value();
    output_scale   = output_data_range / MaxAbs(output_min, output_max);
  } else {
    output_max = A_absmax + B_absmax;
    output_min = -output_max;
  }
  // 2: scale 0 for input A, scale 1 for input B
  const int scales_num = 2;
  std::vector<float> scales(scales_num, 1);
  auto engine = CpuEngine::Get()->get_engine();
  if (inputs[q_elemwise_add::kDataA].dtype() != inputs[q_elemwise_add::kDataB].dtype()) {
    auto s8_desc           = (is_A_int8 == true) ? A_mem->get_desc() : B_mem->get_desc();
    rescaled_mem = TmpMemMgr::Get()->Alloc(s8_desc);
    float u8_reorder_scale = 0;
    if (params.max_calib_range.has_value() && params.min_calib_range.has_value()) {
      if (is_A_int8 == true) {
        u8_reorder_scale = output_scale / B_scale;
        scales[0]        = output_scale / A_scale;
      } else {
        u8_reorder_scale = output_scale / A_scale;
        scales[1]        = output_scale / B_scale;
      }
    } else {
      // x*A_absmax/A_range = y*(A_absmax+B_absmax)/output_range
      if (is_A_int8 == true) {
        u8_reorder_scale = B_absmax * output_data_range / (output_max * kUint8Range);
        scales[0]        = A_absmax * output_data_range / (output_max * A_range);
      } else {
        u8_reorder_scale = A_absmax * output_data_range / (output_max * A_range);
        scales[1]        = B_absmax * output_data_range / (output_max * kInt8Range);
      }
    }
    std::vector<float> reorder_scale = {u8_reorder_scale};
    dnnl::primitive_attr reorder_attr;
    reorder_attr.set_output_scales(0, reorder_scale);
    auto u8_mem = (is_A_int8 == true) ? B_mem : A_mem;
    const auto reorder_pd =
        dnnl::reorder::primitive_desc(engine, u8_mem->get_desc(), engine, s8_desc, reorder_attr);
    dnnl_args_map_t args({{DNNL_ARG_FROM, *u8_mem}, {DNNL_ARG_TO, *rescaled_mem}});
    DNNLStream::Get()->RegisterPrimArgs(dnnl::reorder(reorder_pd), args);

    if (is_A_int8 == true) {
      B_mem = rescaled_mem;
    } else {
      A_mem = rescaled_mem;
    }
  } else {
    // same data type and has same data range
    if (params.max_calib_range.has_value() && params.min_calib_range.has_value()) {
      scales[0] = output_scale / A_scale;
      scales[1] = output_scale / B_scale;
    } else {
      scales[0] = A_absmax * output_data_range / (output_max * A_range);
      scales[1] = B_absmax * output_data_range / (output_max * A_range);
    }
  }

  std::vector<dnnl::memory::desc> in_desc;
  in_desc.push_back(A_mem->get_desc());
  in_desc.push_back(B_mem->get_desc());

  auto output_md                   = outputs[q_elemwise_add::kOut].GetDNNLData()->get_desc();
  DNNLStream* stream               = DNNLStream::Get();
  DNNLQuantizedElemwiseSumFwd& fwd = GetQuantizedElemwiseSumForward(output_md, scales, in_desc);
  auto out_mem                     = CreateDNNLMem(outputs[q_elemwise_add::kOut],
                               fwd.fwd_pd.dst_desc(),
                               req[q_elemwise_add::kOut],
                               &inputs[q_elemwise_add::kDataA]);
  dnnl_args_map_t args({{DNNL_ARG_MULTIPLE_SRC, *A_mem},
                        {DNNL_ARG_MULTIPLE_SRC + 1, *B_mem},
                        {DNNL_ARG_DST, *out_mem.second}});
  stream->RegisterPrimArgs(fwd.GetFwd(), args);
  CommitOutput(outputs[q_elemwise_add::kOut], out_mem);
  stream->Submit();

  outputs[q_elemwise_add::kMin].data().dptr<float>()[0] = output_min;
  outputs[q_elemwise_add::kMax].data().dptr<float>()[0] = output_max;
}

inline static bool ElemwiseAddStorageType(const nnvm::NodeAttrs& attrs,
                                          const int dev_mask,
                                          DispatchMode* dispatch_mode,
                                          std::vector<int>* in_attrs,
                                          std::vector<int>* out_attrs) {
  // Check num of inputs: A, B, A_min, A_max, B_min, B_max
  CHECK_EQ(in_attrs->size(), 6U);
  // Check num of outputs: C, C_min, C_max
  CHECK_EQ(out_attrs->size(), 3U);

  return DNNLStorageType(attrs, dev_mask, true, dispatch_mode, in_attrs, out_attrs);
}

NNVM_REGISTER_OP(_contrib_quantized_elemwise_add)
    .set_attr<FInferStorageType>("FInferStorageType", ElemwiseAddStorageType)
    .set_attr<FComputeEx>("FComputeEx<cpu>", DNNLQuantizedElemwiseAddForward)
    .set_attr<bool>("TIsDNNL", true)
    .set_attr_parser(ParamParser<QuantizeElemwiseAddParam>)
    .add_arguments(QuantizeElemwiseAddParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
