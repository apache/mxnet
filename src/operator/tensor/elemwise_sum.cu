/*!
 * Copyright (c) 2015 by Contributors
 * \file elemwise_sum.cu
 * \brief elementwise sum operator
*/
#include "./elemwise_sum.h"
#include "../../ndarray/ndarray_function.h"

namespace mxnet {
namespace op {

void ElementWiseSumComputeExGPU(const nnvm::NodeAttrs& attrs,
                                const OpContext& op_ctx,
                                const std::vector<NDArray>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<NDArray>& outputs) {
  CHECK(!inputs.empty());
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  if (req[0] == kNullOp) return;
  CHECK_EQ(req[0], kWriteTo) << "ElementWiseSumComputeExGPU only supports req = kWriteTo";
  if (inputs[0].storage_type() == kRowSparseStorage) {
    mshadow::Stream<gpu>* s = op_ctx.get_stream<gpu>();
    NDArray out_nd = outputs[0];
    mxnet::ndarray::ElementwiseSum<gpu>(s, op_ctx.requested[0], inputs, &out_nd);
  } else {
    LOG(FATAL) << "Not implemented: "
               << operator_string(attrs, op_ctx, inputs, req, outputs);
  }
}

NNVM_REGISTER_OP(add_n)
.set_attr<FCompute>("FCompute<gpu>", ElementWiseSumComputeWithHalf2<gpu>)
.set_attr<FComputeEx>("FComputeEx<gpu>", ElementWiseSumComputeExGPU);

}  // namespace op
}  // namespace mxnet
