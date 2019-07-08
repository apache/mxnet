#include "./np_arctan2_op.h"
#include <mxnet/base.h>
#include "../mshadow_op.h"
#include "../operator_common.h"
#include "../tensor/elemwise_binary_op.h"
#include "../tensor/elemwise_binary_broadcast_op.h"

namespace mxnet {
namespace op {
inline bool Arctan2OpType(const nnvm::NodeAttrs& attrs,
                          std::vector<int>* in_attrs,
                          std::vector<int>* out_attrs){
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);

  TYPE_ASSIGN_CHECK(*in_attrs, 0, in_attrs->at(1));
  TYPE_ASSIGN_CHECK(*in_attrs, 1, in_attrs->at(0));
  //check if it is float16, float32 or float64
  if(in_attrs->at(0) >=0 && in_attrs->at(0) <= 2){
    TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  }else{
    //assign double to it
    TYPE_ASSIGN_CHECK(*out_attrs, 0, 1);
  }
  return out_attrs->at(0) != -1;
}

NNVM_REGISTER_OP(_npi_arctan2)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"x1", "x2"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", BinaryBroadcastShape)
.set_attr<nnvm::FInferType>("FInferType", Arctan2OpType)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, Arctan2OpForward>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_arctan2"})
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.add_argument("x1", "NDArray-or-Symbol", "The input array")
.add_argument("x2", "NDArray-or-Symbol", "The input array");

NNVM_REGISTER_OP(_backward_npi_arctan2)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastBackwardUseIn<cpu,Arctan2OpBackward>);

}
}
