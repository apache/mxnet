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
                             DType *costs, DType *grads,
                             std::vector<int> &labels,
                             std::vector<int> &label_lengths,
                             std::vector<int> &input_lengths, void *workspace) {
  int minibatch = static_cast<int>(activations.size(1));
  int alphabet_size = static_cast<int>(activations.size(2));
  int blank_label = 0;
  GpuCTC<DType> ctc(alphabet_size, minibatch, workspace,
                    activations.stream_->stream_, blank_label);
  if (grads == NULL) {
    return ctc.score_forward(activations.dptr_, costs, labels.data(),
                             label_lengths.data(), input_lengths.data());
  } else {
    return ctc.cost_and_grad(activations.dptr_, grads, costs, labels.data(),
                             label_lengths.data(), input_lengths.data());
  }
};

} // namespace mshadow

////////////////////////////////////////////////////////////////////////////////

namespace mxnet {
namespace op {
template <> Operator *CreateOp<gpu>(CTCLossParam param, int dtype) {
  return new CTCLossOp<gpu>(param);
}

} // namespace op
} // namespace mxnet
