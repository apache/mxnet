/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file ctc_loss.cc
 * \brief CPU Implementation of CTC Loss op
 */
#include "./ctc_loss-inl.h"
#include "../../../3rdparty/ctc_include/detail/cpu_ctc.h"

namespace mshadow {
template <typename DType>
ctcStatus_t compute_ctc_cost(const Tensor<cpu, 3, DType> activations,
                             DType *costs, DType *grads, int *labels,
                             int *label_lengths, int *data_lengths,
                             void *workspace, bool isTraining, int blank_label) {
  int minibatch = static_cast<int>(activations.size(1));
  int alphabet_size = static_cast<int>(activations.size(2));
  mxnet_warpctc::CpuCTC<DType> ctc(alphabet_size, minibatch, workspace, blank_label);
  if (isTraining) {
    return ctc.cost_and_grad(activations.dptr_, grads, costs, labels,
                             label_lengths, data_lengths);
  } else {
    return ctc.score_forward(activations.dptr_, costs, labels, label_lengths,
                             data_lengths);
  }
}
}  // namespace mshadow

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(CTCLossOpParam);

NNVM_REGISTER_OP(CTCLoss)
.add_alias("ctc_loss")
.add_alias("_contrib_CTCLoss")
.add_alias("_contrib_ctc_loss")
.describe(R"code(Connectionist Temporal Classification Loss.

.. note:: The existing alias ``contrib_CTCLoss`` is deprecated.

The shapes of the inputs and outputs:

- **data**: `(sequence_length, batch_size, alphabet_size)`
- **label**: `(batch_size, label_sequence_length)`
- **out**: `(batch_size)`

The `data` tensor consists of sequences of activation vectors (without applying softmax),
with i-th channel in the last dimension corresponding to i-th label
for i between 0 and alphabet_size-1 (i.e always 0-indexed).
Alphabet size should include one additional value reserved for blank label.
When `blank_label` is ``"first"``, the ``0``-th channel is be reserved for
activation of blank label, or otherwise if it is "last", ``(alphabet_size-1)``-th channel should be
reserved for blank label.

``label`` is an index matrix of integers. When `blank_label` is ``"first"``,
the value 0 is then reserved for blank label, and should not be passed in this matrix. Otherwise,
when `blank_label` is ``"last"``, the value `(alphabet_size-1)` is reserved for blank label.

If a sequence of labels is shorter than *label_sequence_length*, use the special
padding value at the end of the sequence to conform it to the correct
length. The padding value is `0` when `blank_label` is ``"first"``, and `-1` otherwise.

For example, suppose the vocabulary is `[a, b, c]`, and in one batch we have three sequences
'ba', 'cbb', and 'abac'. When `blank_label` is ``"first"``, we can index the labels as
`{'a': 1, 'b': 2, 'c': 3}`, and we reserve the 0-th channel for blank label in data tensor.
The resulting `label` tensor should be padded to be::

  [[2, 1, 0, 0], [3, 2, 2, 0], [1, 2, 1, 3]]

When `blank_label` is ``"last"``, we can index the labels as
`{'a': 0, 'b': 1, 'c': 2}`, and we reserve the channel index 3 for blank label in data tensor.
The resulting `label` tensor should be padded to be::

  [[1, 0, -1, -1], [2, 1, 1, -1], [0, 1, 0, 2]]

``out`` is a list of CTC loss values, one per example in the batch.

See *Connectionist Temporal Classification: Labelling Unsegmented
Sequence Data with Recurrent Neural Networks*, A. Graves *et al*. for more
information on the definition and the algorithm.

)code" ADD_FILELINE)
.set_attr_parser(ParamParser<CTCLossOpParam>)
.set_num_inputs(CTCLossOpNumInputs)
.set_num_outputs(2)
.set_attr<nnvm::FListInputNames>("FListInputNames", CTCLossOpListInputNames)
.set_attr<nnvm::FListOutputNames>("FListOutputNAmes",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"out", "grad"};
  })
.set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs",
  [](const NodeAttrs& attrs) {
    return 1;
  })
.set_attr<mxnet::FInferShape>("FInferShape", CTCLossOpShape)
.set_attr<nnvm::FInferType>("FInferType", CTCLossOpType)
.set_attr<FInferStorageType>("FInferStorageType", CTCLossOpStorageType)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs)
  { return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.set_attr<FCompute>("FCompute<cpu>", CTCLossOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_ctc_loss"})
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")
.add_argument("label", "NDArray-or-Symbol", "Ground-truth labels for the loss.")
.add_argument("data_lengths", "NDArray-or-Symbol",
              "Lengths of data for each of the samples. Only required "
              "when use_data_lengths is true.")
.add_argument("label_lengths", "NDArray-or-Symbol",
              "Lengths of labels for each of the samples. Only required "
              "when use_label_lengths is true.")
.add_arguments(CTCLossOpParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_ctc_loss)
.set_attr_parser(ParamParser<CTCLossOpParam>)
.set_num_inputs(4)
.set_num_outputs(CTCLossOpNumInputs)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", CTCLossOpBackward<cpu>);

}  // namespace op
}  // namespace mxnet
