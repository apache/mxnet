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
                             DType *costs, DType *grads, int *labels,
                             int *label_lengths, int *input_lengths,
                             void *workspace, int train) {
  int minibatch = static_cast<int>(activations.size(1));
  int alphabet_size = static_cast<int>(activations.size(2));
  int blank_label = 0;
  CpuCTC<DType> ctc(alphabet_size, minibatch, workspace, blank_label);
  if (train)
    return ctc.cost_and_grad(activations.dptr_, grads, costs, labels,
                             label_lengths, input_lengths);
  else
    return ctc.score_forward(activations.dptr_, costs, labels, label_lengths,
                             input_lengths);
}

}  // namespace mshadow

namespace mxnet {
namespace op {
template <>
Operator *CreateOp<cpu>(CTCLossParam param, int dtype) {
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

MXNET_REGISTER_OP_PROPERTY(_contrib_CTCLoss, CTCLossProp)
    .describe(R"code(Connectionist Temporal Classification Loss.

The shapes of the inputs and outputs:

- **data**: *(sequence_length, batch_size, alphabet_size + 1)*
- **label**: *(batch_size, label_sequence_length)*
- **out**: *(batch_size)*.

``label`` is a tensor of integers between 1 and *alphabet_size*. If a 
sequence of labels is shorter than *label_sequence_length*, use the special 
padding character 0 at the end of the sequence to conform it to the correct 
length. For example, if *label_sequence_length* = 4, and one has two sequences 
of labels [2, 1] and [3, 2, 2], the resulting ```label``` tensor should be 
padded to be::

  [[2, 1, 0, 0], [3, 2, 2, 0]]

The ``data`` tensor consists of sequences of activation vectors. The layer 
applies a softmax to each vector, which then becomes a vector of probabilities 
over the alphabet. Note that the 0th element of this vector is reserved for the 
special blank character.

See *Connectionist Temporal Classification: Labelling Unsegmented 
Sequence Data with Recurrent Neural Networks*, A. Graves *et al*. for more 
information.

)code" ADD_FILELINE)
    .add_argument("data", "NDArray-or-Symbol", "Input data to the ctc_loss op.")
    .add_argument("label", "NDArray-or-Symbol",
                  "Ground-truth labels for the loss.")
    .add_arguments(CTCLossParam::__FIELDS__());

NNVM_REGISTER_OP(_contrib_CTCLoss).add_alias("_contrib_ctc_loss");

}  // namespace op
}  // namespace mxnet
