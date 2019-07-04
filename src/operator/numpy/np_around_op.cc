#include "np_around_op.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(AroundParam);

NNVM_REGISTER_OP(_npi_around)
.set_attr_parser(ParamParser<AroundParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"x"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", AroundOpForward<cpu>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.add_argument("x", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(AroundParam::__FIELDS__());

}// namespace op
}// namespace mxnet