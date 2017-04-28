/*!
 *  Copyright (c) 2017 by Contributors
 * \file quantize.cu
 * \brief
 */
#include <limits>
#include "./quantize_down_and_shrink_range-inl.h"
#include "./quantization_utils.h"
#include "../tensor/broadcast_reduce_op.h"

namespace mxnet {
namespace op {

template<typename Reducer, typename DType>
static void Reduce(const OpContext& ctx,
                   TBlob out, TBlob data,
                   int req_cnt) {
  mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
  TShape src_shape, dst_shape;
  BroadcastReduceShapeCompact(data.shape_, out.shape_, &src_shape, &dst_shape);
  constexpr int NDim = 2;
  CHECK_EQ(dst_shape.ndim(), NDim);
  CHECK_EQ(src_shape.ndim(), NDim);

  const TBlob in_data  = data.reshape(src_shape);
  const TBlob out_data =  out.reshape(dst_shape);

  size_t workspace_size = broadcast::ReduceWorkspaceSize<NDim, DType>(
    s, out_data, kWriteTo, in_data);
  mshadow::Tensor<gpu, 1, char> workspace =
    ctx.requested[req_cnt].get_space_typed<gpu, 1, char>(mshadow::Shape1(workspace_size), s);
  broadcast::Reduce<Reducer, NDim, DType, mshadow::op::identity>(
    s, out_data, kWriteTo, workspace, in_data);
}

void QuantizeDownAndShrinkRangeComputeGPU(
    const nnvm::NodeAttrs& attrs,
    const OpContext& ctx,
    const std::vector<TBlob>& inputs,
    const std::vector<OpReqType>& req,
    const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  typedef int32_t SrcDType;
  typedef int8_t  DstDType;
  Stream<gpu> *s = ctx.get_stream<gpu>();
  int req_cnt = 0;

  size_t space_size = 2 * sizeof(float) + 2 * sizeof(SrcDType);
  Tensor<gpu, 1, char> space =
    ctx.requested[req_cnt++].get_space_typed<gpu, 1, char>(Shape1(space_size), s);

  TBlob actual_min_quantized(
    reinterpret_cast<SrcDType*>(space.dptr_ + 8), Shape1(1), gpu::kDevMask);
  TBlob actual_max_quantized(
    reinterpret_cast<SrcDType*>(space.dptr_ + 8) + 1, Shape1(1), gpu::kDevMask);

  Reduce<red::minimum, SrcDType>(ctx, actual_min_quantized, inputs[0], req_cnt++);
  Reduce<red::maximum, SrcDType>(ctx, actual_max_quantized, inputs[0], req_cnt++);


  Tensor<gpu, 1, float> actual_min_float(
    reinterpret_cast<float*>(space.dptr_), Shape1(1), s);
  Tensor<gpu, 1, float> actual_max_float(
    reinterpret_cast<float*>(space.dptr_) + 1, Shape1(1), s);

  Kernel<QuantizedToFloatStruct, gpu>::Launch(s, 1,
      actual_min_float.dptr_, actual_min_quantized.dptr<SrcDType>(),
      inputs[1].dptr<float>(), inputs[2].dptr<float>());
  Kernel<QuantizedToFloatStruct, gpu>::Launch(s, 1,
      actual_max_float.dptr_, actual_max_quantized.dptr<SrcDType>(),
      inputs[1].dptr<float>(), inputs[2].dptr<float>());

  Kernel<RequantizeManyInNewRangeStruct, gpu>::Launch(s, inputs[0].Size(),
      outputs[0].dptr<DstDType>(), outputs[1].dptr<float>(), outputs[2].dptr<float>(),
      inputs[0].dptr<SrcDType>(), inputs[1].dptr<float>(), inputs[2].dptr<float>(),
      actual_min_float.dptr_, actual_max_float.dptr_);
}

NNVM_REGISTER_OP(quantize_down_and_shrink_range)
.set_attr<FCompute>("FCompute<gpu>", QuantizeDownAndShrinkRangeComputeGPU);

}  // namespace op
}  // namespace mxnet
