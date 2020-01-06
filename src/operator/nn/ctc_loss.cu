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
 * Copyright (c) 2018 by Contributors
 * \file ctc_loss.cu
 * \brief GPU Implementation of ctc_loss op
 */

#include "./ctc_loss-inl.h"
#include "../../../3rdparty/ctc_include/detail/gpu_ctc.h"

namespace mshadow {

template <typename DType>
ctcStatus_t compute_ctc_cost(const Tensor<gpu, 3, DType> activations,
                             DType *costs, DType *grads, int *labels,
                             int *label_lengths, int *input_lengths,
                             void *workspace, int train, int blank_label) {
  int minibatch = static_cast<int>(activations.size(1));
  int alphabet_size = static_cast<int>(activations.size(2));
  mxnet_warpctc::GpuCTC<DType> ctc(alphabet_size, minibatch, workspace,
                    activations.stream_->stream_, blank_label);
  if (train)
    return ctc.cost_and_grad(activations.dptr_, grads, costs, labels,
                             label_lengths, input_lengths);
  else
    return ctc.score_forward(activations.dptr_, costs, labels,
                             label_lengths, input_lengths);
}
}  // namespace mshadow

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(CTCLoss)
.add_alias("ctc_loss")
.add_alias("_contrib_ctc_loss")
.add_alias("_contrib_CTCLoss")
.set_attr<FCompute>("FCompute<gpu>", CTCLossOpForward<gpu>);

NNVM_REGISTER_OP(_backward_ctc_loss)
.set_attr<FCompute>("FCompute<gpu>", CTCLossOpBackward<gpu>);

}  // namespace op
}  // namespace mxnet
