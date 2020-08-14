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
 *  Copyright (c) 2018 by Contributors
 * \file quantize_v2.cu
 * \brief
 */
#include "./quantize_v2-inl.h"

namespace mxnet {
namespace op {

// get quantization scale factor and set omin_range, omax_range for quantize_v2_zero_centered_kernel
struct QuantScaleANDSetOutRange {
  MSHADOW_XINLINE static void Map(int i,
                                float *omin_range,
                                float *omax_range,
                                const float imin_range,
                                const float imax_range){

    float real_range = MaxAbs(imin_range, imax_range);

    *omin_range = -real_range;
    *omax_range = real_range;
  }
};

#if defined(__CUDACC__)

template <typename SrcDType, typename SrcLoadType, typename DstDType, typename DstLoadType>
__global__ void quantize_v2_zero_centered_kernel(DstDType *out,
                                                SrcDType *in,
                                                const float quantized_range,
                                                const float quantization_scale_tmp,
                                                const int elem_per_thread) {
    
    const index_t row = blockIdx.x;
    const int row_len = elem_per_thread * blockDim.x;

    SrcLoadType* inload = reinterpret_cast<SrcLoadType*>(in);
    DstLoadType* outload = reinterpret_cast<DstLoadType*>(out);

    const int load_ratio = sizeof(SrcLoadType) / sizeof(SrcDType);

    for (index_t i = threadIdx.x; i < row_len / load_ratio; i += blockDim.x){
      int idx = row * row_len / load_ratio + i;

      SrcLoadType scratch_in = *(inload + idx);
      SrcDType* scratch_in_aft_load = reinterpret_cast<SrcDType*>(&scratch_in);

      DstLoadType scratch_out = *(outload + idx);
      DstDType* scratch_out_aft_load = reinterpret_cast<DstDType*>(&scratch_out);

    #pragma unroll
      for(int j = 0; j < load_ratio; j++){
        scratch_out_aft_load[j] = Sign(scratch_in_aft_load[j]) * 
                                  fminf(fabsf(scratch_in_aft_load[j]) * quantization_scale_tmp + 0.5f, quantized_range);
      }

      *(outload + idx) = scratch_out;
    }
}

template<typename SrcDType, typename DstDType>
void QuantizeV2ZeroCenteredGPU(mshadow::Stream<gpu>* s,
                                DstDType *out,
                                SrcDType *in,
                                const float quantized_range,
                                const int num_elem,
                                const float quantization_scale_tmp) {

    int nthreads = 256;

    if(num_elem >= 512){
      nthreads = 512;
    }else if(num_elem >= 256){
      nthreads = 256;
    }else if(num_elem >= 128){
      nthreads = 128;
    }else if(num_elem >= 64){
      nthreads = 64;
    }else{
      nthreads = 32;
    }
    
    int elem_per_thread = 8;

    int Srcltype = mxnet::common::cuda::get_load_type(num_elem * sizeof(SrcDType));

    MXNET_LOAD_TYPE_SWITCH(Srcltype, SrcLoadType,{

      int load_ratio = sizeof(SrcLoadType) / sizeof(SrcDType);
  
      if(load_ratio == 4){
        quantize_v2_zero_centered_kernel<SrcDType, SrcLoadType, DstDType, int32_t>
                          <<< (num_elem + (elem_per_thread*nthreads) - 1) / (elem_per_thread*nthreads),
                          nthreads,
                          0,
                          mshadow::Stream<gpu>::GetStream(s)>>>(out,
                                                                in,
                                                                quantized_range,
                                                                quantization_scale_tmp,
                                                                elem_per_thread);
      }else if(load_ratio == 2){
        quantize_v2_zero_centered_kernel<SrcDType, SrcLoadType, DstDType, mshadow::half::half_t>
                          <<< (num_elem + (elem_per_thread*nthreads) - 1) / (elem_per_thread*nthreads),
                          nthreads,
                          0,
                          mshadow::Stream<gpu>::GetStream(s)>>>(out,
                                                                in,
                                                                quantized_range,
                                                                quantization_scale_tmp,
                                                                elem_per_thread);
      }else if(load_ratio == 1){
        quantize_v2_zero_centered_kernel<SrcDType, SrcLoadType, DstDType, int8_t>
                          <<< (num_elem + (elem_per_thread*nthreads) - 1) / (elem_per_thread*nthreads),
                          nthreads,
                          0,
                          mshadow::Stream<gpu>::GetStream(s)>>>(out,
                                                                in,
                                                                quantized_range,
                                                                quantization_scale_tmp,
                                                                elem_per_thread);
      }else{
        LOG(FATAL) << "Unsupported Load Type.";
      }

    });

}

#endif  // __CUDACC__

template<>
class QuantizeV2Operator<gpu> {
 public:
  explicit QuantizeV2Operator(const nnvm::NodeAttrs &attrs) : attrs_(attrs) {}

  void Forward(const OpContext &ctx, const std::vector<TBlob> &inputs,
               const std::vector<OpReqType> &req, const std::vector<TBlob> &outputs) {
    using namespace mshadow;
    using namespace mxnet_op;
    using mshadow::red::limits::MaxValue;
    using mshadow::red::limits::MinValue;
    Stream<gpu> *s = ctx.get_stream<gpu>();
    const QuantizeV2Param &param = nnvm::get<QuantizeV2Param>(attrs_.parsed);
    auto out_type = GetQuantizeOutputType(param);
    if (out_type == mshadow::kUint8) {
      LOG(FATAL) << "currently, uint8 quantization is only supported by CPU, "
                    "please switch to the context of CPU or int8 data type for GPU.";
    }
  
    if (inputs[0].type_flag_ == mshadow::kUint8 || inputs[0].type_flag_ == mshadow::kInt8) {
      if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
        *outputs[1].dptr<float>() = param.min_calib_range.value();
        *outputs[2].dptr<float>() = param.max_calib_range.value();
      } else {
        if (inputs[0].type_flag_ == mshadow::kUint8) {
          *outputs[1].dptr<float>() = 0;
          *outputs[2].dptr<float>() = 255;
        } else {
          *outputs[1].dptr<float>() = -127;
          *outputs[2].dptr<float>() = 127;
        }
      }
      UnaryOp::IdentityCompute<gpu>(attrs_, ctx, {inputs[0]}, req, outputs);
    } else if (inputs[0].type_flag_ == mshadow::kFloat32) {
      typedef float FloatDType;
      if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
        if (out_type == mshadow::kInt8) {  // zero-centered quantization

            Kernel<QuantScaleANDSetOutRange, gpu>::Launch(s, 1,
                outputs[1].dptr<float>(), outputs[2].dptr<float>(),
                param.min_calib_range.value(), param.max_calib_range.value());
            float real_range = MaxAbs(param.min_calib_range.value(), param.max_calib_range.value());
            float scale = MinAbs(MaxValue<int8_t>(), MinValue<int8_t>()) / real_range;
            QuantizeV2ZeroCenteredGPU<FloatDType, int8_t>(s,
                                    outputs[0].dptr<int8_t>(),
                                    inputs[0].dptr<FloatDType>(),
                                    MinAbs(MaxValue<int8_t>(), MinValue<int8_t>()),
                                    outputs[0].Size(),
                                    scale);
        } else {
          LOG(FATAL) << "quantize op on GPU only supports int8 as output type";
        }
      } else {  // model is not calibrated
        mxnet::TShape src_shape, dst_shape;
        const size_t actual_float_size = sizeof(float);
        const size_t temp_reduce_size = ConfigReduce<gpu, FloatDType>(
            s, inputs[0].shape_, mxnet::TShape(1, 1), &src_shape, &dst_shape);
        Tensor<gpu, 1, char> temp_space = ctx.requested[0].get_space_typed<gpu, 1, char>(
            Shape1(2 * actual_float_size + temp_reduce_size), s);
        const int dev_id = ctx.run_ctx.ctx.dev_id;
        TBlob in_min_t(reinterpret_cast<FloatDType *>(temp_space.dptr_), Shape1(1), gpu::kDevMask,
                       dev_id);
        TBlob in_max_t(reinterpret_cast<FloatDType *>(temp_space.dptr_) + 1, Shape1(1), gpu::kDevMask,
                       dev_id);
        Tensor<gpu, 1, char> workspace(temp_space.dptr_ + 2 * actual_float_size,
                                       Shape1(temp_reduce_size), s);
        broadcast::Reduce<red::minimum, 2, FloatDType, mshadow::op::identity>(
            s, in_min_t.reshape(dst_shape), kWriteTo, workspace, inputs[0].reshape(src_shape));
        broadcast::Reduce<red::maximum, 2, FloatDType, mshadow::op::identity>(
            s, in_max_t.reshape(dst_shape), kWriteTo, workspace, inputs[0].reshape(src_shape));
        if (out_type == mshadow::kInt8) {  // zero-centered quantization
          Kernel<quantize_v2_zero_centered, gpu>::Launch(
              s, outputs[0].Size(), outputs[0].dptr<int8_t>(), outputs[1].dptr<float>(),
              outputs[2].dptr<float>(), inputs[0].dptr<FloatDType>(), in_min_t.dptr<float>(),
              in_max_t.dptr<float>(), MinAbs(MaxValue<int8_t>(), MinValue<int8_t>()));
        } else {
          LOG(FATAL) << "quantize op on GPU only supports int8 as output type";
        }
      }
    } else if (inputs[0].type_flag_ == mshadow::kFloat16) {
      typedef mshadow::half::half_t FP16DType;
      if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
        if (out_type == mshadow::kInt8) {  // zero-centered quantization

            Kernel<QuantScaleANDSetOutRange, gpu>::Launch(s, 1,
                outputs[1].dptr<float>(), outputs[2].dptr<float>(),
                param.min_calib_range.value(), param.max_calib_range.value());
            float real_range = MaxAbs(param.min_calib_range.value(), param.max_calib_range.value());
            float scale = MinAbs(MaxValue<int8_t>(), MinValue<int8_t>()) / real_range;
            QuantizeV2ZeroCenteredGPU<FP16DType, int8_t>(s,
                                    outputs[0].dptr<int8_t>(),
                                    inputs[0].dptr<FP16DType>(),
                                    MinAbs(MaxValue<int8_t>(), MinValue<int8_t>()),
                                    outputs[0].Size(),
                                    scale);
        } else {
          LOG(FATAL) << "quantize op on GPU only supports int8 as output type";
        }
      }else{// model is not calibrated
        mxnet::TShape src_shape, dst_shape;
        const size_t actual_float16_size = sizeof(FP16DType);
        const size_t temp_reduce_size = ConfigReduce<gpu, FP16DType>(
            s, inputs[0].shape_, mxnet::TShape(1, 1), &src_shape, &dst_shape);
        Tensor<gpu, 1, char> temp_space = ctx.requested[0].get_space_typed<gpu, 1, char>(
            Shape1(2 * actual_float16_size + temp_reduce_size), s);
        const int dev_id = ctx.run_ctx.ctx.dev_id;
        TBlob in_min_t(reinterpret_cast<FP16DType *>(temp_space.dptr_), Shape1(1), gpu::kDevMask,
                       dev_id);
        TBlob in_max_t(reinterpret_cast<FP16DType *>(temp_space.dptr_) + 1, Shape1(1), gpu::kDevMask,
                       dev_id);
        Tensor<gpu, 1, char> workspace(temp_space.dptr_ + 2 * actual_float16_size,
                                       Shape1(temp_reduce_size), s);
        broadcast::Reduce<red::minimum, 2, FP16DType, mshadow::op::identity>(
            s, in_min_t.reshape(dst_shape), kWriteTo, workspace, inputs[0].reshape(src_shape));
        broadcast::Reduce<red::maximum, 2, FP16DType, mshadow::op::identity>(
            s, in_max_t.reshape(dst_shape), kWriteTo, workspace, inputs[0].reshape(src_shape));
        if (out_type == mshadow::kInt8) {  // zero-centered quantization
          Kernel<quantize_v2_zero_centered, gpu>::Launch(
              s, outputs[0].Size(), outputs[0].dptr<int8_t>(), outputs[1].dptr<float>(),
              outputs[2].dptr<float>(), inputs[0].dptr<FP16DType>(), in_min_t.dptr<FP16DType>(),
              in_max_t.dptr<FP16DType>(), MinAbs(MaxValue<int8_t>(), MinValue<int8_t>()));
        } else {
          LOG(FATAL) << "quantize op on GPU only supports int8 as output type";
        }
      }
    } else {
      LOG(FATAL) << "quantize op only supports int8, uint8, float32 and float16 as input type";
    }
  }

 private:
  nnvm::NodeAttrs attrs_;
};

NNVM_REGISTER_OP(_contrib_quantize_v2)
.set_attr<FStatefulCompute>("FStatefulCompute<gpu>", QuantizeV2Forward<gpu>);

}  // namespace op
}  // namespace mxnet
