#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/c_runtime_api.h>
#include <mxnet/base.h>
#include "../tvmop/tvm_op_module.h"
#include "../tensor/elemwise_binary_op.h"

namespace mxnet {
namespace op {

void TVMVectorAddCompute(const nnvm::NodeAttrs& attrs,
                         const mxnet::OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  tvm::runtime::TVMOpModule::Get()->Call("myadd", {inputs[0], inputs[1], outputs[0]});
}

}  // namespace op
}  // namespace mxnet

NNVM_REGISTER_OP(tvm_vector_add)
.set_num_inputs(2)
.set_num_outputs(1)
.add_argument("a", "NDArray-or-Symbol", "first input")
.add_argument("b", "NDArray-or-Symbol", "second input")
.set_attr<mxnet::FInferShape>("FInferShape", mxnet::op::ElemwiseShape<2, 1>)
.set_attr<nnvm::FInferType>("FInferType", mxnet::op::ElemwiseType<2, 1>)
.set_attr<mxnet::FCompute>("FCompute<cpu>", mxnet::op::TVMVectorAddCompute);
