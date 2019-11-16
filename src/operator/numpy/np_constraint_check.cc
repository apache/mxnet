#include "./np_constraint_check.h"

namespace mxnet {
namespace op {

template<>
void GetReduceOutput<cpu>(mshadow::Stream<cpu> *s, const TBlob &output_blob, bool *red_output) {
  *red_output = static_cast<bool>(*output_blob.dptr<bool>());
}

inline bool ConstraintCheckShape(const nnvm::NodeAttrs& attrs,
                        std::vector<TShape>* in_attrs,
                        std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  if (!shape_is_known(in_attrs->at(0))) {
    return false;
  }
  // Only 1-D support is supported.
  CHECK_EQ(in_attrs->at(0).ndim(), 1U) << "Only 1-D input is supported.";
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape(0, -1))
  return true;
}

inline bool ConstraintCheckType(const nnvm::NodeAttrs& attrs,
                       std::vector<int>* in_attrs,
                       std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  CHECK(in_attrs->at(0) == mshadow::kBool);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kBool);
  return out_attrs->at(0) != -1 && in_attrs->at(0) != -1;
}

DMLC_REGISTER_PARAMETER(ConstraintCheckParam);

NNVM_REGISTER_OP(_npx_constraint_check)
.set_attr_parser(ParamParser<ConstraintCheckParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"input"};
  })
.set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs",
    [](const NodeAttrs& attrs) {
  return 0;
})
.set_attr<mxnet::FInferShape>("FInferShape", ConstraintCheckShape)
.set_attr<nnvm::FInferType>("FInferType", ConstraintCheckType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", ConstraintCheckForward<cpu>)
.add_argument("input", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(ConstraintCheckParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
