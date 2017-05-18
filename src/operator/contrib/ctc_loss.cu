/*!
 * Copyright (c) 2015 by Contributors
 * \file ctc_loss.cu
 * \brief
 * \author Sebastian Bodenstein
*/
#include <algorithm>
#include "./ctc_loss-inl.h"
#include "./ctc_include/detail/gpu_ctc.h"

namespace mshadow {

template <typename DType>
ctcStatus_t compute_ctc_cost(const Tensor<gpu, 3, DType> activations,
                             DType *costs, DType *grads, int *labels,
                             int *label_lengths, int *input_lengths,
                             void *workspace, int train) {
  int minibatch = static_cast<int>(activations.size(1));
  int alphabet_size = static_cast<int>(activations.size(2));
  int blank_label = 0;
  GpuCTC<DType> ctc(alphabet_size, minibatch, workspace,
                    activations.stream_->stream_, blank_label);
  if (train)
    return ctc.cost_and_grad(activations.dptr_, grads, costs, labels,
                             label_lengths, input_lengths);
  else
    return ctc.score_forward(activations.dptr_, costs, labels,
                             label_lengths, input_lengths);
}

}  // namespace mshadow

////////////////////////////////////////////////////////////////////////////////

namespace mxnet {
namespace op {
template <>
Operator *CreateOp<gpu>(CTCLossParam param, int dtype) {
  return new CTCLossOp<gpu>(param);
}

}  // namespace op
}  // namespace mxnet
