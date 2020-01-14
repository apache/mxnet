#include <mxnet/base.h>
#include "./beamsearch_set_finished-inl.h"
#include "../tensor/elemwise_unary_op.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(BeamsearchSetFinishedParam);

NNVM_REGISTER_OP(_contrib_beamsearch_set_finished)
.describe(R"code(Sets finished beams of the beam to a mask value (aside from the score index) and forces beams at max length to output the EOS sequence.

Returns an array of the same shape of the input data array and the same values except for the designated rows whose elements are to be masked or be replaced by beam scores/EOS probabilities.

Example::

    x = [[ -1.,  -2.,  -3.,  -4.],
         [ -5.,  -6.,  -7.,  -8.],
         [ -9., -10., -11., -12.],
         [-13., -14., -15., -16.]]

    scores = [[-17.],
              [-18.],
              [-19.],
              [-20.]]

    finished = [0, 1, 0, 1]

    over_max = [0, 0, 1, 1]

    beamsearch_set_finished(x, scores, finished, over_max, score_idx=0,
                            eos_idx=2, mask_val=-1e15) = [[  -1.,   -2.,   -3.,   -4.],
                                                          [ -18., -1e15, -1e15, -1e15],
                                                          [-1e15, -1e15,  -11., -1e15],
                                                          [ -20., -1e15, -1e15, -1e15]]

.. Note::
    This operator only supports forward propagation. DO NOT use it in training.

)code")
.set_attr_parser(ParamParser<BeamsearchSetFinishedParam>)
.set_num_inputs(4)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "scores", "finished", "over_max"}; })
.set_attr<nnvm::FListOutputNames>("FListOutputNames", [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"output"}; })
.set_attr<FInferShape>("FInferShape", BeamsearchSetFinishedShape)
.set_attr<nnvm::FInferType>("FInferType", BeamsearchSetFinishedType)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int>>{{0, 0}}; })
.set_attr<FCompute>("FCompute<cpu>", BeamsearchSetFinishedForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"beamsearch_noop_grad"})
.add_argument("data", "NDArray-or-Symbol", "Input distribution of tokens")
.add_argument("scores", "NDArray-or-Symbol", "Running scores for the sequences")
.add_argument("finished", "NDArray-or-Symbol", "Finished beams")
.add_argument("over_max", "NDArray-or-Symbol", "Beams at or exceeding maximum length")
.add_arguments(BeamsearchSetFinishedParam::__FIELDS__());

NNVM_REGISTER_OP(_contrib_beamsearch_noop_grad)
.set_num_inputs(1)
.set_num_outputs(4)
.set_attr<FCompute>("FCompute<cpu>", NoopGrad<cpu>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true);
}
}
