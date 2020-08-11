#include "min_ex-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(min_ex)
.describe("some description")
.set_num_inputs(0)
.set_num_outputs(0)
.set_attr<mxnet::FInferShape>("FInferShape", MinExOpShape)
.set_attr<nnvm::FInferType>("FInferType", MinExOpType)
.set_attr<FCompute>("FCompute<cpu>", MinExForward);

}  // namespace op                                                                                                                                                                     
}  // namespace mxnet      
