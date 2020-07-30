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
 * Copyright (c) 2020 by Contributors
 * \file quantized_bert_ffn1_to_ffn2.cu
 * \brief
*/

#include <mxnet/base.h>
//#include <limits>

#include <mxnet/operator_util.h>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include "../../mxnet_op.h"
#include "../../mshadow_op.h"
#include "../../elemwise_op_common.h"

namespace mxnet {
namespace op {

using namespace mshadow;

using mshadow::red::limits::MaxValue;
using mshadow::red::limits::MinValue;

template<typename T>
MSHADOW_XINLINE int Sign(T val) {
  return (val > T(0)) - (val < T(0));
}

template<typename T>
MSHADOW_XINLINE T Abs(T a) {
#ifdef __CUDACC__
  return ::abs(a);
#else
  return std::abs(a);
#endif
}

template<typename T>
MSHADOW_XINLINE T Max(T a, T b) {
#ifdef __CUDACC__
  return ::max(a, b);
#else
  return std::max(a, b);
#endif
}

template<typename T>
MSHADOW_XINLINE T Min(T a, T b) {
#ifdef __CUDACC__
  return ::min(a, b);
#else
  return std::min(a, b);
#endif
}

template<typename T>
MSHADOW_XINLINE float MaxAbs(T a, T b) {
  return Max(Abs(static_cast<float>(a)), Abs(static_cast<float>(b)));
}

template<typename T>
MSHADOW_XINLINE float MinAbs(T a, T b) {
  return Min(Abs(static_cast<float>(a)), Abs(static_cast<float>(b)));
}

/*!
 * \brief Get the scaling factor for converting type T to float.
 */
template<typename T>
MSHADOW_XINLINE float FloatForOneQuantizedLevel(float range_min, float range_max, bool all_sign) {
  float range_data = MaxAbs(range_min, range_max);
  float range_T = all_sign ? MinAbs(MinValue<T>(), MaxValue<T>()) : MaxValue<T>();
  return range_data / range_T;
}

template <typename TA, typename TB, typename TC>
MSHADOW_XINLINE void QuantizationRangeForMultiplication(float min_a, float max_a, float min_b,
                                                        float max_b, float *min_c, float *max_c,
                                                        bool all_sign) {
  const float a_float_for_one_quant_level = FloatForOneQuantizedLevel<TA>(min_a, max_a, all_sign);
  const float b_float_for_one_quant_level = FloatForOneQuantizedLevel<TB>(min_b, max_b, all_sign);
  const float range_c =
      MinAbs(static_cast<int64_t>(MinValue<TC>()), static_cast<int64_t>(MaxValue<TC>()));
  const float c_float_for_one_quant_level =
      a_float_for_one_quant_level * b_float_for_one_quant_level;
  *max_c = c_float_for_one_quant_level * range_c;
  *min_c = -*max_c;
}

/**
 * Flatten additional dimensions after the first
 * @tparam xpu
 * @tparam DType
 * @param tblob
 * @param ctx
 * @return 2 Dimensional Tensor with upper shapes collapsed
 */
template<typename xpu, typename DType>
Tensor<xpu, 2, DType> FlattenAs2DTail(const TBlob& tblob, const OpContext& ctx) {
  const TShape& shape = tblob.shape_;
  Stream<xpu> *stream = ctx.get_stream<xpu>();
  return tblob.get_with_shape<xpu, 2, DType>(
      Shape2(shape[0], shape.ProdShape(1, shape.ndim())), stream);
}

/**
 * Flatten dimensions except last
 * @tparam xpu
 * @tparam DType
 * @param tblob
 * @param ctx
 * @return 2 Dimensional tensor with front shapes collapsed
 */
template<typename xpu, typename DType>
Tensor<xpu, 2, DType> FlattenAs2DHead(const TBlob& tblob, const OpContext& ctx) {
  const TShape& shape = tblob.shape_;
  Stream<xpu> *stream = ctx.get_stream<xpu>();
  return tblob.get_with_shape<xpu, 2, DType>(
      Shape2(shape.ProdShape(0, shape.ndim()-1), shape[shape.ndim()-1]), stream);
}

struct QuantizedBERTFFN1TOFFN2Param : public dmlc::Parameter<QuantizedBERTFFN1TOFFN2Param> {
  // for quantized FC of ffn1 (should be a biased FC)
  int num_hidden;
  bool flatten;

  // for quantization before ffn2
  dmlc::optional<float> min_calib_range;
  dmlc::optional<float> max_calib_range;

  DMLC_DECLARE_PARAMETER(QuantizedBERTFFN1TOFFN2Param) {
    DMLC_DECLARE_FIELD(num_hidden).set_lower_bound(1)
    .describe("Number of hidden nodes of the output.");
    DMLC_DECLARE_FIELD(flatten).set_default(true)
    .describe("Whether to collapse all but the first axis of the input data tensor."); 
    DMLC_DECLARE_FIELD(min_calib_range)
      .set_default(dmlc::optional<float>())
      .describe("The minimum scalar value in the form of float32. If present, it will be used to "
                "quantize the fp32 data into int8 or uint8.");
    DMLC_DECLARE_FIELD(max_calib_range)
      .set_default(dmlc::optional<float>())
      .describe("The maximum scalar value in the form of float32. If present, it will be used to "
                "quantize the fp32 data into int8 or uint8.");    
  }
  bool operator==(const QuantizedBERTFFN1TOFFN2Param& other) const {
    return this->num_hidden == other.num_hidden &&
           this->flatten == other.flatten &&
           this->min_calib_range == other.min_calib_range &&
           this->max_calib_range == other.max_calib_range;
  }
};

#if CUDA_VERSION >= 8000

struct PreparationSingleKernel {
  MSHADOW_XINLINE static void Map(int i,
                                  float *min_out,
                                  float *max_out,
                                  const float *min_data,
                                  const float *max_data,
                                  const float *min_weight,
                                  const float *max_weight,
                                  const float *min_bias,
                                  const float *max_bias,
                                  float *float_for_one_quant_tmp, 
                                  float *quantized_to_float_scale,
                                  const float imin_range,
                                  const float imax_range,
                                  float *quantization_from_float_scale) {
    typedef int32_t T1;
    typedef int8_t  T2;

    // use min/max values of weight and data to update the min/max values of output
      QuantizationRangeForMultiplication<int8_t, int8_t, int32_t>(
        min_data[i], max_data[i], min_weight[i], max_weight[i], min_out, max_out, true);
    
    // get float_for_one_out_quant / float_for_one_bias_quant outside QuantizedBiasAddKernel
      float float_for_one_quant_out = FloatForOneQuantizedLevel<T1>(*min_out, *max_out, true);
      float float_for_one_quant_bias = FloatForOneQuantizedLevel<T2>(*min_bias, *max_bias, true);

      // the tmp space to store float_for_one_quant is 32 bits (1 float numbers)
      *float_for_one_quant_tmp = float_for_one_quant_bias / float_for_one_quant_out;

    // get quantizedtofloat scale
      float quantized_range = MinAbs(MinValue<T1>(), MaxValue<T1>());
      float real_range = MaxAbs(*min_out, *max_out);
      float scale_toFP = real_range / quantized_range;

      // the tmp space to store quantized_to_float_scale is 32 bits (1 float numbers)
      *quantized_to_float_scale = scale_toFP;

    // get quantization_from_float_scale and set omin_range, omax_range for quantize_v2_zero_centered
      real_range = MaxAbs(imin_range, imax_range);
      float scale_fromFP = MinAbs(MinValue<T2>(), MaxValue<T2>()) / real_range;

      *min_out = -real_range;
      *max_out = real_range;

      // the tmp space to store quantization_from_float_scale is 32 bits (1 float numbers)
      *quantization_from_float_scale = scale_fromFP;

  }
};

#endif  // CUDA_VERSION >= 8000

#if defined(__CUDACC__)

const float SQRT_2 = 1.4142135623730950488016887242096f;
// compatible with mshadow_op.h version
template <typename DType>
__device__ inline DType gelu(const DType val) {
  return DType(0.5f * static_cast<float>(val) *
               (1.0f + erff(static_cast<float>(val) / SQRT_2))); //erff() for single precision
}

#define row_num_each_thread 8

// value + bias_value * (range1 / limit_range1) * (limit_range2 / range2)
__global__ void quantized_add_bias_redequantize_gelu_quantize_kernel(int32_t* outCUBLAS,
                                          int8_t* out,
                                          int8_t* bias,
                                          size_t bias_length,
                                          const float *float_for_one_quant_tmp,
                                          const float *quantized_to_float_scale,
                                          const float *quantization_from_float_scale,
                                          const float quantized_range) {

  int64_t* outCUBLASload = reinterpret_cast<int64_t*>(outCUBLAS);
  int16_t* outload = reinterpret_cast<int16_t*>(out);
  int16_t* biasload = reinterpret_cast<int16_t*>(bias);
   
  for (index_t i = threadIdx.x; i < bias_length / 2; i += blockDim.x){

    int16_t scratch_bias = *(biasload + i);
    int8_t* scratch_bias_aft_load = reinterpret_cast<int8_t*>(&scratch_bias);

  #pragma unroll
    for(int rw = 0; rw < row_num_each_thread; rw++){
      int idx = (blockIdx.x * row_num_each_thread + rw) * bias_length / 2 + i;

      int64_t scratch_outCUBLAS = *(outCUBLASload + idx);
      int32_t* scratch_outCUBLAS_aft_load = reinterpret_cast<int32_t*>(&scratch_outCUBLAS);

      int16_t scratch_out = *(outload + idx);
      int8_t* scratch_out_aft_load = reinterpret_cast<int8_t*>(&scratch_out);

      // add_bias and dequantize to float
      mshadow::half::half_t FCout = (scratch_outCUBLAS_aft_load[0] + scratch_bias_aft_load[0] * (*float_for_one_quant_tmp)) * 
                                      (*quantized_to_float_scale);
      // gelu
      mshadow::half::half_t geluout = gelu<mshadow::half::half_t>(FCout);
      // quantize it back to int8
      scratch_out_aft_load[0] = Sign(geluout) * fminf(fabsf(geluout) * (*quantization_from_float_scale) + 0.5f, quantized_range);


      // add_bias and dequantize to float
      FCout = (scratch_outCUBLAS_aft_load[1] + scratch_bias_aft_load[1] * (*float_for_one_quant_tmp)) * 
                                      (*quantized_to_float_scale);
      // gelu
      geluout = gelu<mshadow::half::half_t>(FCout);
      // quantize it back to int8
      scratch_out_aft_load[1] = Sign(geluout) * fminf(fabsf(geluout) * (*quantization_from_float_scale) + 0.5f, quantized_range);


      *(outload + idx) = scratch_out;
    }
  }
}

void FusedQuantizedAddBias_ReDequantize_GELU_Quantize(int8_t* bias,
                      Tensor<gpu, 2, int8_t> data,
                      int8_t* out,
                      Tensor<gpu, 2, int32_t> outTensorCUBLAS,
                      Stream<gpu>* s,
                      const float *float_for_one_quant_tmp,
                      const int bias_len,
                      const float *quantized_to_float_scale,
                      const float *quantization_from_float_scale) {
    
    int nthreads_quant_addbias = 256;

    if(bias_len % 512 == 0){
      nthreads_quant_addbias = 512;
    }else if(bias_len % 256 == 0){
      nthreads_quant_addbias = 256;
    }else if(bias_len % 128 == 0){
      nthreads_quant_addbias = 128;
    }else if(bias_len % 64 == 0){
      nthreads_quant_addbias = 64;
    }
    
    if(bias_len <= 32){
      nthreads_quant_addbias = 32;
    }

    float quantized_range = MinAbs(MaxValue<int8_t>(), MinValue<int8_t>());

    quantized_add_bias_redequantize_gelu_quantize_kernel<<<data.size(0) / row_num_each_thread, //row_num_each_thread = 8
                                  nthreads_quant_addbias,
                                  0,
                                  Stream<gpu>::GetStream(s)>>>(outTensorCUBLAS.dptr_,
                                                                out,
                                                                bias,
                                                                bias_len,
                                                                float_for_one_quant_tmp,
                                                                quantized_to_float_scale,
                                                                quantization_from_float_scale,
                                                                quantized_range);
}

#endif  // __CUDACC__

template<typename SrcType>
void QuantizedBERTFFN1TOFFN2ForwardGPU(const nnvm::NodeAttrs& attrs,
                                       const OpContext &ctx,
                                       const std::vector<TBlob> &inputs,
                                       const std::vector<OpReqType> &req,
                                       const std::vector<TBlob> &outputs) {
#if CUDA_VERSION >= 8000
  typedef int32_t CmpType;

  const QuantizedBERTFFN1TOFFN2Param& param = nnvm::get<QuantizedBERTFFN1TOFFN2Param>(attrs.parsed);
  using namespace mshadow;
  using namespace mxnet_op;
  size_t num_inputs = 3;
  CHECK_EQ(inputs.size(),  num_inputs * 3);
  CHECK_EQ(outputs.size(), 3U);
  Stream<gpu> *s = ctx.get_stream<gpu>();
  CHECK_EQ(s->blas_handle_ownership_, Stream<gpu>::OwnHandle);

  const TBlob& data   =  inputs[0];
  const TBlob& weight =  inputs[1];
  const TBlob& out    = outputs[0];

  mxnet::TShape dshape = data.shape_;
  mxnet::TShape wshape = weight.shape_;
  mxnet::TShape oshape = out.shape_;
  // (m, n) * (k, n).T = (m, k)
  // A * B.T = C

  // allocate workspace for storaging both outTensorCUBLAS and FloatForOneQuant, QuantizedToFloatScaleFactor and 
  //  QuantizedFromFloatScaleFactor
  size_t workspace_size = sizeof(int32_t) * out.Size() + sizeof(float) * 3;
  auto workspace = ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(workspace_size), s);
  char* ptr = workspace.dptr_;

  Tensor<gpu, 2, SrcType> dataTensor;

  Tensor<gpu, 2, int8_t> outTensor;
    if (!param.flatten) {
      dataTensor = FlattenAs2DHead<gpu, SrcType>(data, ctx);
      outTensor = FlattenAs2DHead<gpu, int8_t>(out, ctx);
    } else {
      dataTensor = FlattenAs2DTail<gpu, SrcType>(data, ctx);
      outTensor = FlattenAs2DTail<gpu, int8_t>(out, ctx);
    }

    // workspace: temporary storage for output tensor, which is in int32 type for storaging CUBLAS output
    Tensor<gpu, 2, int32_t> outTensorCUBLAS = Tensor<gpu, 2, int32_t>(reinterpret_cast<int32_t*>(ptr), outTensor.shape_, s);
    ptr += sizeof(int32_t) * out.Size();
    // workspace: FloatForOneQuant
    Tensor<gpu, 1, float> FloatForOneQuant = Tensor<gpu, 1, float>(reinterpret_cast<float*>(ptr), Shape1(1), s);
    ptr += sizeof(float);
    // workspace: QuantizedToFloatScaleFactor
    Tensor<gpu, 1, float> QuantizedToFloatScaleFactor = Tensor<gpu, 1, float>(reinterpret_cast<float*>(ptr), Shape1(1), s);
    ptr += sizeof(float);
    // workspace: QuantizedFromFloatScaleFactor
    Tensor<gpu, 1, float> QuantizedFromFloatScaleFactor = Tensor<gpu, 1, float>(reinterpret_cast<float*>(ptr), Shape1(1), s);

  Tensor<gpu, 2, SrcType> weightTensor = weight.get<gpu, 2, SrcType>(s);

  // row_C = col_C(T) = cublas(col_B * col_A(T)) = cublas(row_B(T), row_A)
  // row_C = col_C(T) = cublas(col_B(T) * col_A(T)) = cublas(row_B, row_A)
 
  // A->dataTensor, B->weightTensor, C->outTensor
  CmpType alpha = 1.0f;
  CmpType beta  = 0.0f;
  const cudaDataType src_type = mshadow::DataType<SrcType>::kCudaFlag;
  const cudaDataType dst_type = mshadow::DataType<int32_t>::kCudaFlag;
  const cudaDataType cmp_type = mshadow::DataType<CmpType>::kCudaFlag;
  CUBLAS_CALL(cublasGemmEx(s->blas_handle_,
                           CUBLAS_OP_T,
                           CUBLAS_OP_N,
                           outTensorCUBLAS.size(1),
                           outTensorCUBLAS.size(0),
                           weightTensor.size(1),
                           &alpha,
                           weightTensor.dptr_,
                           src_type,
                           weightTensor.stride_,
                           dataTensor.dptr_,
                           src_type,
                           dataTensor.stride_,
                           &beta,
                           outTensorCUBLAS.dptr_,
                           dst_type,
                           outTensorCUBLAS.stride_,
                           cmp_type,
                           CUBLAS_GEMM_DFALT));
   
  Kernel<PreparationSingleKernel, gpu>::Launch(s, 1,
      outputs[1].dptr<float>(), outputs[2].dptr<float>(),
      inputs[3].dptr<float>(),   inputs[4].dptr<float>(),
      inputs[5].dptr<float>(), inputs[6].dptr<float>(),
      inputs[7].dptr<float>(), inputs[8].dptr<float>(),
      FloatForOneQuant.dptr_,
      QuantizedToFloatScaleFactor.dptr_,
      param.min_calib_range.value(), param.max_calib_range.value(),
      QuantizedFromFloatScaleFactor.dptr_);

  const TBlob& bias = inputs[2];

  Tensor<gpu, 1, SrcType> biasTensor = bias.get_with_shape<gpu, 1, SrcType>(Shape1(wshape[0]), s);
  CHECK_EQ(biasTensor.shape_[0], wshape[0])
      << "Incomplete bias tensor detected: bias.data().shape[1] != weight.data().shape[0]."
         " This is not supported by FCForward. If bias is in row_sparse format, please"
         " make sure all row ids are present.";

  // with_bias case with float16 out
  // a kernel that fuse requantize, dequantize into add_bias
  FusedQuantizedAddBias_ReDequantize_GELU_Quantize(bias.dptr<SrcType>(), dataTensor,
                                          out.dptr<int8_t>(), outTensorCUBLAS, s,
                                          FloatForOneQuant.dptr_, biasTensor.shape_[0],
                                          QuantizedToFloatScaleFactor.dptr_,
                                          QuantizedFromFloatScaleFactor.dptr_);

#else
  LOG(FATAL) << "QuantizedBERTFFN1TOFFN2ForwardGPU only supports CUDA >= 8.0";
#endif  // CUDA_VERSION >= 8000
}

bool QuantizedBERTFFN1TOFFN2Shape(const nnvm::NodeAttrs& attrs,
                                  mxnet::ShapeVector *in_shape,
                                  mxnet::ShapeVector *out_shape) {
  const QuantizedBERTFFN1TOFFN2Param& param = nnvm::get<QuantizedBERTFFN1TOFFN2Param>(attrs.parsed);
  using namespace mshadow;
  uint32_t num_inputs = 3;
  CHECK_EQ(in_shape->size(), num_inputs * 3);
  CHECK_EQ(out_shape->size(), 3U);

  mxnet::TShape dshape = (*in_shape)[0];
  // require data ndim to be known
  if (!mxnet::ndim_is_known(dshape)) return false;

  index_t num_input;
  if (!param.flatten) {
    num_input = dshape[dshape.ndim() - 1];
  } else {
    num_input = dshape.ProdShape(1, dshape.ndim());
  }

  mxnet::TShape wshape = Shape2(param.num_hidden, num_input);
  SHAPE_ASSIGN_CHECK(*in_shape, 1, wshape);
    
  mxnet::TShape bshape = Shape1(param.num_hidden);
  SHAPE_ASSIGN_CHECK(*in_shape, 2, bshape);

  for (size_t i = num_inputs; i < 3 * num_inputs; ++i) {
    SHAPE_ASSIGN_CHECK(*in_shape, i, mxnet::TShape(1, 1));
  }

  if (!param.flatten) {
    mxnet::TShape result_shape(dshape);
    result_shape[dshape.ndim() - 1] = param.num_hidden;
    SHAPE_ASSIGN_CHECK(*out_shape, 0, result_shape);
  } else {
    SHAPE_ASSIGN_CHECK(*out_shape, 0, Shape2(dshape[0], param.num_hidden));
  }
  SHAPE_ASSIGN_CHECK(*out_shape, 1, mxnet::TShape(1, 1));
  SHAPE_ASSIGN_CHECK(*out_shape, 2, mxnet::TShape(1, 1));

  if ((*out_shape)[0].ndim() > 0) {
    dshape[0] = ((*out_shape)[0])[0];
    SHAPE_ASSIGN_CHECK(*in_shape, 0, dshape);
  }
  return true;
}

bool QuantizedBERTFFN1TOFFN2Type(const nnvm::NodeAttrs& attrs,
                                 std::vector<int> *in_type,
                                 std::vector<int> *out_type) {
  const QuantizedBERTFFN1TOFFN2Param& param = nnvm::get<QuantizedBERTFFN1TOFFN2Param>(attrs.parsed);
  uint32_t num_inputs = 3;
  CHECK_EQ(in_type->size(), num_inputs * 3);
  CHECK_EQ(out_type->size(), 3U);

  TYPE_ASSIGN_CHECK(*in_type, 0, mshadow::kInt8);
  for (size_t i = 1; i < num_inputs; ++i) {
    TYPE_ASSIGN_CHECK(*in_type, i, mshadow::kInt8);
  }
  for (size_t i = num_inputs; i < 3 * num_inputs; ++i) {
    TYPE_ASSIGN_CHECK(*in_type, i, mshadow::kFloat32);
  }

  TYPE_ASSIGN_CHECK(*out_type, 0, mshadow::kInt8);
  TYPE_ASSIGN_CHECK(*out_type, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_type, 2, mshadow::kFloat32);
  return true;
}

bool QuantizedBERTFFN1TOFFN2StorageType(const nnvm::NodeAttrs& attrs,
                                        const int dev_mask,
                                        DispatchMode* dispatch_mode,
                                        std::vector<int> *in_attrs,
                                        std::vector<int> *out_attrs) {
  const QuantizedBERTFFN1TOFFN2Param& param = nnvm::get<QuantizedBERTFFN1TOFFN2Param>(attrs.parsed);
  uint32_t num_inputs = 3;
  CHECK_EQ(in_attrs->size(), num_inputs * 3);
  CHECK_EQ(out_attrs->size(), 3U);

  *dispatch_mode = DispatchMode::kFCompute;

  for (auto &v : *out_attrs) {
    v = kDefaultStorage;
    if (common::stype_string(v).compare("unknown") == 0) {
      return false;
    }
  }

  for (auto &v : *in_attrs) {
    v = kDefaultStorage;
    if (common::stype_string(v).compare("unknown") == 0) {
      return false;
    }
  }
  return true;
}

DMLC_REGISTER_PARAMETER(QuantizedBERTFFN1TOFFN2Param);

NNVM_REGISTER_OP(_contrib_quantized_bert_ffn1_to_ffn2_fusion)
.describe(R"code(Quantized bert_ffn1_to_ffn2_fusion operator for input, weight and bias data type of int8,
and calculates the fullyconnected outputs, apply GELU operator and then quantize the output back to int8. 
For each argument, two more arguments of type float32 must be provided representing the thresholds of 
quantizing argument from data type float32 to int8. The final outputs contain the result in int8, and min
and max thresholds representing the threholds for quantizing the float output into int8.

.. Note::
    This operator only supports forward propogation. DO NOT use it in training.)code" ADD_FILELINE)
.set_num_inputs(
  [](const NodeAttrs& attrs) {
    const QuantizedBERTFFN1TOFFN2Param& param = nnvm::get<QuantizedBERTFFN1TOFFN2Param>(attrs.parsed);
    return 9;
  })
.set_num_outputs(3)
.set_attr_parser(ParamParser<QuantizedBERTFFN1TOFFN2Param>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    const QuantizedBERTFFN1TOFFN2Param& param = nnvm::get<QuantizedBERTFFN1TOFFN2Param>(attrs.parsed);
      return std::vector<std::string>{"data", "weight", "bias", "min_data", "max_data",
                                      "min_weight", "max_weight", "min_bias", "max_bias"};
  })
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"output", "min_output", "max_output"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", QuantizedBERTFFN1TOFFN2Shape)
.set_attr<nnvm::FInferType>("FInferType", QuantizedBERTFFN1TOFFN2Type)
.set_attr<FInferStorageType>("FInferStorageType", QuantizedBERTFFN1TOFFN2StorageType)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.add_argument("data", "NDArray-or-Symbol", "Input data.")
.add_argument("weight", "NDArray-or-Symbol", "weight.")
.add_argument("bias", "NDArray-or-Symbol", "bias.")
.add_argument("min_data", "NDArray-or-Symbol", "Minimum value of data.")
.add_argument("max_data", "NDArray-or-Symbol", "Maximum value of data.")
.add_argument("min_weight", "NDArray-or-Symbol", "Minimum value of weight.")
.add_argument("max_weight", "NDArray-or-Symbol", "Maximum value of weight.")
.add_argument("min_bias", "NDArray-or-Symbol", "Minimum value of bias.")
.add_argument("max_bias", "NDArray-or-Symbol", "Maximum value of bias.")
.add_arguments(QuantizedBERTFFN1TOFFN2Param::__FIELDS__())
.set_attr<FCompute>("FCompute<gpu>", QuantizedBERTFFN1TOFFN2ForwardGPU<int8_t>);

}  // namespace op
}  // namespace mxnet