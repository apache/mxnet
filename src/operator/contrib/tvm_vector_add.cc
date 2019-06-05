#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/c_runtime_api.h>
#include <mxnet/base.h>
#include "../tvmop/tvm_op_module.h"
#include "../tensor/elemwise_binary_op.h"

namespace mxnet {
namespace op {

template<const char* func>
void TVMVectorAddCompute(const nnvm::NodeAttrs& attrs,
                         const mxnet::OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  tvm::runtime::TVMOpModule::Get()->Call(func, ctx, {inputs[0], inputs[1], outputs[0]});
}

static constexpr char func_cpu[] = "vadd";
static constexpr char func_gpu[] = "cuda_vadd";
NNVM_REGISTER_OP(tvm_vector_add)
    .set_num_inputs(2)
    .set_num_outputs(1)
    .add_argument("a", "NDArray-or-Symbol", "first input")
    .add_argument("b", "NDArray-or-Symbol", "second input")
    .set_attr<mxnet::FInferShape>("FInferShape", mxnet::op::ElemwiseShape<2, 1>)
    .set_attr<nnvm::FInferType>("FInferType", mxnet::op::ElemwiseType<2, 1>)
    .set_attr<mxnet::FCompute>("FCompute<cpu>", mxnet::op::TVMVectorAddCompute<func_cpu>)
    .set_attr<mxnet::FCompute>("FCompute<gpu>", mxnet::op::TVMVectorAddCompute<func_gpu>);


inline bool BinaryBroadcastShape(const nnvm::NodeAttrs& attrs,
                                 mxnet::ShapeVector *in_attrs,
                                 mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  mxnet::TShape& lhs = (*in_attrs)[0];
  mxnet::TShape& rhs = (*in_attrs)[1];

  // avoid pre-mature shape inference.
  if (!mxnet::ndim_is_known(lhs) || !mxnet::ndim_is_known(rhs)) return false;

  if (lhs == rhs) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, lhs);
    return shape_is_known(lhs);
  }
  mxnet::TShape out(std::max(lhs.ndim(), rhs.ndim()), -1);
  const int bl = out.ndim() - lhs.ndim();
  const int br = out.ndim() - rhs.ndim();
  for (int i = 0; i < out.ndim(); ++i) {
    int l = 1, r = 1;
    if (i >= bl) l = lhs[i-bl];
    if (i >= br) r = rhs[i-br];
    if (!mxnet::dim_size_is_known(l) || !mxnet::dim_size_is_known(r)) continue;
    if (l != r) {
      // Make it compatible with NumPy.
      // For example, (2, 3) cannot broadcast to (2, 0, 3), but (1, 3) can broadcast to (2, 0, 3).
      CHECK(l == 1 || r == 1)
        << "operands could not be broadcast together with shapes " << lhs << " " << rhs;
      out[i] = (l == 1 ? r : l);
    } else {
      out[i] = l;
    }
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, out);
  return shape_is_known(lhs) && shape_is_known(rhs) && shape_is_known(out);
}

void TVMBcastAddCompute(const nnvm::NodeAttrs& attrs,
                        const mxnet::OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  tvm::runtime::TVMOpModule::Get()->Call("bcast_add", ctx, {inputs[0], inputs[1], outputs[0]});
}

NNVM_REGISTER_OP(tvm_bcast_add)
    .set_num_inputs(2)
    .set_num_outputs(1)
    .add_argument("a", "NDArray-or-Symbol", "first input")
    .add_argument("b", "NDArray-or-Symbol", "second input")
    .set_attr<mxnet::FInferShape>("FInferShape", BinaryBroadcastShape)
    .set_attr<nnvm::FInferType>("FInferType", mxnet::op::ElemwiseType<2, 1>)
    .set_attr<mxnet::FCompute>("FCompute<cpu>", mxnet::op::TVMBcastAddCompute);

}  // namespace op
}  // namespace mxnet
