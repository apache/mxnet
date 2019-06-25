#include "./np_arctan2_op.h"

namespace mxnet {
namespace op {
NNVM_REGISTER_OP(_np_arctan2)
.describe(R"code(This operators implements the arctan2 function:
.. math::

    f(x1, x2) = arctan(x1/x2)

where :math:`x1` and `x2` are two input tensor and all oprations
in the function are element-wise.

Example:

  .. code-block:: python
     :emphasize-lines: 
     x1 = [1, -1]
     x2 = [1, 1]
     y = arctan2(x1, x2)
     y = [0.7853982， -0.7853982]


)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"x1", "x2"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", BinaryBroadcastShape)
.set_attr<nnvm::FInferType>("FInferType", Arctan2OpType)
.set_attr<FCompute>("FCompute<cpu>", Arctan2OpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_np_backward_arctan2"})
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.add_argument("x1", "NDArray-or-Symbol", "The input array")
.add_argument("x2", "NDArray-or-Symbol", "The input array");

NNVM_REGISTER_OP(_np_backward_arctan2)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", Arctan2OpBackward<cpu>);

}
}
