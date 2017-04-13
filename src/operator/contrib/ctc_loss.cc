/*!
 * Copyright (c) 2015 by Contributors
 * \file ctc_loss.cc
 * \brief
 * \author Sebastian Bodenstein
*/

#include "./ctc_loss-inl.h"
#include "./ctc_include/detail/cpu_ctc.h"

namespace mshadow {

template <typename DType>
ctcStatus_t compute_ctc_cost(const Tensor<cpu, 3, DType> activations,
                             DType *costs, DType *grads,
                             std::vector<int> &labels,
                             std::vector<int> &label_lengths,
                             std::vector<int> &input_lengths, void *workspace) {
  int minibatch = static_cast<int>(activations.size(1));
  int alphabet_size = static_cast<int>(activations.size(2));
  int blank_label = 0;
  CpuCTC<DType> ctc(alphabet_size, minibatch, workspace, blank_label);
  if (grads == NULL) {
    return ctc.score_forward(activations.dptr_, costs, labels.data(),
                             label_lengths.data(), input_lengths.data());
  } else {
    return ctc.cost_and_grad(activations.dptr_, grads, costs, labels.data(),
                             label_lengths.data(), input_lengths.data());
  }
};

} // namespace mshadow

namespace mxnet {
namespace op {
template <> Operator *CreateOp<cpu>(CTCLossParam param, int dtype) {
  return new CTCLossOp<cpu>(param);
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *CTCLossProp::CreateOperatorEx(Context ctx,
                                        std::vector<TShape> *in_shape,
                                        std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(CTCLossParam);

MXNET_REGISTER_OP_PROPERTY(CTCLoss, CTCLossProp)
    .describe(R"code(Compute the Connectionist Temporal Loss.
)code" ADD_FILELINE)
    .add_argument("data", "ndarray-or-symbol",
                  "A rank-3 tensor of the form "
                  "[max sequence length, batchsize, alphabet size].")
    .add_argument("labels", "ndarray-or-symbol", "A rank-2 tensor of the form "
                                                 "[max label num, batchsize].")
    .add_arguments(CTCLossParam::__FIELDS__());

NNVM_REGISTER_OP(CTCLoss).add_alias("ctc_loss");

} // namespace op
} // namespace mxnet
