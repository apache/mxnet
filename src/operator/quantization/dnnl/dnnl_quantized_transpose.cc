
#if MXNET_USE_ONEDNN == 1
#include "../../nn/dnnl/dnnl_transpose-inl.h"
#include "../../tensor/matrix_op-inl.h"

namespace mxnet {
namespace op {

inline static bool QuantizedTransposeStorageType(const nnvm::NodeAttrs& attrs,
                                                 const int dev_mask,
                                                 DispatchMode* dispatch_mode,
                                                 std::vector<int>* in_attrs,
                                                 std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 3U);
  return DNNLStorageType(attrs, dev_mask, true, dispatch_mode, in_attrs, out_attrs);
}

static void DNNLQuantizedTransposeForward(const nnvm::NodeAttrs& attrs,
                                          const OpContext& ctx,
                                          const std::vector<NDArray>& inputs,
                                          const std::vector<OpReqType>& req,
                                          const std::vector<NDArray>& outputs) {
  CHECK(inputs[0].dtype() == mshadow::kUint8 || inputs[0].dtype() == mshadow::kInt8)
      << "dnnl_quantized_pooling op only supports uint8 and int8 as input type";
  if (req[0] == kNullOp) {
    return;
  }
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 3U);
  DNNLRun(DNNLTransposeForward<TransposeParam>, attrs, ctx, inputs[0], req[0], outputs[0]);
  outputs[1].data().dptr<float>()[0] = inputs[1].data().dptr<float>()[0];
  outputs[2].data().dptr<float>()[0] = inputs[2].data().dptr<float>()[0];
}

NNVM_REGISTER_OP(_contrib_quantized_transpose)
    .set_attr<FInferStorageType>("FInferStorageType", QuantizedTransposeStorageType)
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& n) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<FComputeEx>("FComputeEx<cpu>", DNNLQuantizedTransposeForward)
    .set_attr<bool>("TIsDNNL", true);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
