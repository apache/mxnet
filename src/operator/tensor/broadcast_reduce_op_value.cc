/*!
 *  Copyright (c) 2016 by Contributors
 * \file broadcast_reduce_op.cc
 * \brief CPU Implementation of broadcast and reduce functions.
 */
#include "./broadcast_reduce_op.h"

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(ReduceAxesParam);
DMLC_REGISTER_PARAMETER(ReduceAxisParam);
DMLC_REGISTER_PARAMETER(BroadcastAxesParam);
DMLC_REGISTER_PARAMETER(BroadcastToParam);

inline std::string get_reduce_axes_description(const std::string& op_name, int line) {
  std::string doc = R"code(Compute the __op__ of array elements over given axes.

The argument ``axis`` specifies the axes to compute over:

- **()**: compute over all elements into a scalar array with shape ``(1,)``. This is
  the default option.
- **int**: compute over along a particular axis. If input has shape ``(n, m, k)``,
  use ``axis=0`` will result in an array with shape ``(m, k)``.
- **tuple of int**: compute over multiple axes. Again assume input shape ``(n, m,
  k)``, with ``axis=(0,2)`` we obtain a ``(m,)`` shape array.

If ``keepdims = 1``, then the result array will has the same number of dimensions
as the input, while the reduced axes will have size 1.


Defined in )code";
  doc += std::string(__FILE__) + std::string(":L") + std::to_string(line);
  size_t pos = 0;
  std::string holder("__op__");
  while ((pos = doc.find(holder, pos)) != std::string::npos) {
    doc.replace(pos, holder.length(), op_name);
    pos += op_name.length();
  }
  return doc;
}

MXNET_OPERATOR_REGISTER_REDUCE(sum)
.add_alias("sum_axis")
.describe(get_reduce_axes_description("sum", __LINE__))
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesCompute<cpu, mshadow::red::sum>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_sum"});

MXNET_OPERATOR_REGISTER_REDUCE_BACKWARD(_backward_sum)
.set_num_inputs(1)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesBackwardUseNone<cpu>);

MXNET_OPERATOR_REGISTER_REDUCE(mean)
.describe(get_reduce_axes_description("mean", __LINE__))
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesCompute<cpu, mshadow::red::sum, true>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_mean"});

MXNET_OPERATOR_REGISTER_REDUCE_BACKWARD(_backward_mean)
.set_num_inputs(1)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesBackwardUseNone<cpu, true>);

MXNET_OPERATOR_REGISTER_REDUCE(prod)
.describe(get_reduce_axes_description("product", __LINE__))
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesCompute< cpu, mshadow_op::product>)
.set_attr<nnvm::FGradient>("FGradient", ReduceGrad{ "_backward_prod" });

MXNET_OPERATOR_REGISTER_REDUCE_BACKWARD(_backward_prod)
.set_num_inputs(3)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesBackwardUseInOut< cpu, mshadow_op::rdiv>);

MXNET_OPERATOR_REGISTER_REDUCE(nansum)
.describe(R"code(Compute the sum of array elements over given axes with ``NaN`` ignored

Refer to ``sum`` for more details.

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesCompute<cpu, mshadow_op::nansum>)
.set_attr<nnvm::FGradient>("FGradient", ReduceGrad{ "_backward_nansum" });

MXNET_OPERATOR_REGISTER_REDUCE_BACKWARD(_backward_nansum)
.set_num_inputs(3)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesBackwardUseInOut<cpu, mshadow_op::nansum_grad>);

MXNET_OPERATOR_REGISTER_REDUCE(nanprod)
.describe(R"code(Compute the product of array elements over given axes with ``NaN`` ignored

Refer to ``prod`` for more details.

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesCompute<cpu, mshadow_op::nanprod>)
.set_attr<nnvm::FGradient>("FGradient", ReduceGrad{ "_backward_nanprod" });

MXNET_OPERATOR_REGISTER_REDUCE_BACKWARD(_backward_nanprod)
.set_num_inputs(3)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesBackwardUseInOut<cpu, mshadow_op::nanprod_grad>);

MXNET_OPERATOR_REGISTER_REDUCE(max)
.add_alias("max_axis")
.describe(get_reduce_axes_description("max", __LINE__))
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesCompute<cpu, mshadow::red::maximum>)
.set_attr<nnvm::FGradient>("FGradient", ReduceGrad{"_backward_max"});

MXNET_OPERATOR_REGISTER_REDUCE_BACKWARD(_backward_max)
.set_num_inputs(3)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesBackwardUseInOut<cpu, mshadow_op::eq>);

MXNET_OPERATOR_REGISTER_REDUCE(min)
.add_alias("min_axis")
.describe(get_reduce_axes_description("min", __LINE__))
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesCompute<cpu, mshadow::red::minimum>)
.set_attr<nnvm::FGradient>("FGradient", ReduceGrad{"_backward_min"});

MXNET_OPERATOR_REGISTER_REDUCE_BACKWARD(_backward_min)
.set_num_inputs(3)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesBackwardUseInOut<cpu, mshadow_op::eq>);

MXNET_OPERATOR_REGISTER_BROADCAST(broadcast_axis)
.add_alias("broadcast_axes")
.describe(R"code(Broadcast an array over particular axes.

Broadcasting is allowed on axes which size 1, such as from ``(2,1,3,1)`` to
``(2,8,3,9)``. Elemenets will be duplicated on the broadcasted axes.

For example::

   // given (1,2,1) shape x
   x = [[[ 1.],
         [ 2.]]]

   // broadcast on axis 2
   broadcast_axis(x, axis=2, size=3) = [[[ 1.,  1.,  1.],
                                         [ 2.,  2.,  2.]]]
   // broadcast on axes 0 and 2
   broadcast_axis(x, axis=(0,2), size=(2,3)) = [[[ 1.,  1.,  1.],
                                                 [ 2.,  2.,  2.]],
                                                [[ 1.,  1.,  1.],
                                                 [ 2.,  2.,  2.]]]
)code" ADD_FILELINE)
.set_attr_parser(ParamParser<BroadcastAxesParam>)
.add_arguments(BroadcastAxesParam::__FIELDS__())
.set_attr<nnvm::FInferShape>("FInferShape", BroadcastAxesShape)
.set_attr<FCompute>("FCompute<cpu>", BroadcastCompute<cpu>);

MXNET_OPERATOR_REGISTER_BROADCAST(broadcast_to)
.describe(R"code(Broadcast an array to a new shape.

Broadcasting is allowed on axes which size 1, such as from ``(2,1,3,1)`` to
``(2,8,3,9)``. Elemenets will be duplicated on the broadcasted axes.

For example::

   broadcast_to([[1,2,3]], shape=(2,3)) = [[ 1.,  2.,  3.],
                                           [ 1.,  2.,  3.]])

The dimensions that will not be changed can also use the special code ``0`` that
means copy the original value. So with ``shape=(2,0)`` we will obtain the same
results in the above example.

)code" ADD_FILELINE)
.set_attr_parser(ParamParser<BroadcastToParam>)
.add_arguments(BroadcastToParam::__FIELDS__())
.set_attr<nnvm::FInferShape>("FInferShape", BroadcastToShape)
.set_attr<FCompute>("FCompute<cpu>", BroadcastCompute<cpu>);

// backward op for broadcast.
NNVM_REGISTER_OP(_broadcast_backward)
.set_attr_parser(ParamParser<ReduceAxesParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesCompute<cpu, mshadow::red::sum>);

NNVM_REGISTER_OP(norm)
.describe(R"code(Compute the L2 norm.

Flatten then input array and then compute the l2 norm.

Examples::

  x = [[1, 2],
       [3, 4]]

  norm(x) = [5.47722578]

)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape",
  [](const nnvm::NodeAttrs& attrs,
     std::vector<TShape> *in_attrs,
     std::vector<TShape> *out_attrs) {
    CHECK_EQ(in_attrs->size(), 1U);
    CHECK_EQ(out_attrs->size(), 1U);
    if ((*in_attrs)[0].ndim() == 0) return false;
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::Shape1(1));
    return true;
  })
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", L2NormCompute<cpu>)
.add_argument("src", "ndarray-or-symbol", "Source input");

}  // namespace op
}  // namespace mxnet
