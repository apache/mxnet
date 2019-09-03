#include "./np_choice_op.h"
#include <algorithm>

namespace mxnet {
namespace op {

template<>
void _swap<cpu>(int64_t& a, int64_t& b) {
    std::swap(a, b);
}

template<>
void _sort<cpu>(float* key, int64_t* data, index_t length) {
  std::sort(data, data + length,
            [key](int64_t const& i, int64_t const& j) -> bool {
              return key[i] > key[j];
            });
}


DMLC_REGISTER_PARAMETER(NumpyChoiceParam);


NNVM_REGISTER_OP(_npi_choice)
.describe("random choice")
.set_num_inputs(
  [](const nnvm::NodeAttrs& attrs) {
    int num_input = 0;
    const NumpyChoiceParam& param = nnvm::get<NumpyChoiceParam>(attrs.parsed);
    if (param.weighted) num_input += 1;
    if (!param.a.has_value()) num_input += 1;
    return num_input;
  }
)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"input1", "input2"};
  })
.set_attr_parser(ParamParser<NumpyChoiceParam>)
.set_attr<mxnet::FInferShape>("FInferShape", NumpyChoiceOpShape)
.set_attr<nnvm::FInferType>("FInferType", NumpyChoiceOpType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const nnvm::NodeAttrs& attrs) {
      return std::vector<ResourceRequest>{
        ResourceRequest::kRandom, ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", NumpyChoiceForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("input1", "NDArray-or-Symbol", "Source input")
.add_argument("input2", "NDArray-or-Symbol", "Source input")
.add_arguments(NumpyChoiceParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet


