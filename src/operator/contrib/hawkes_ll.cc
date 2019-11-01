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
 * \file hawkes_ll.cc
 * \brief Log likelihood of a marked self-exciting Hawkes process
 * \author Caner Turkmen <turkmen.ac@gmail.com>
 */

#include "./hawkes_ll-inl.h"
#include "../tensor/init_op.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_contrib_hawkesll)
    .describe(R"code(Computes the log likelihood of a univariate Hawkes process.

The log likelihood is calculated on point process observations represented
as *ragged* matrices for *lags* (interarrival times w.r.t. the previous point),
and *marks* (identifiers for the process ID). Note that each mark is considered independent,
i.e., computes the joint likelihood of a set of Hawkes processes determined by the conditional intensity:

.. math::

  \lambda_k^*(t) = \lambda_k + \alpha_k \sum_{\{t_i < t, y_i = k\}} \beta_k \exp(-\beta_k (t - t_i))

where :math:`\lambda_k` specifies the background intensity ``lda``, :math:`\alpha_k` specifies the *branching ratio* or ``alpha``, and :math:`\beta_k` the delay density parameter ``beta``.

``lags`` and ``marks`` are two NDArrays of shape (N, T) and correspond to the representation of the point process observation, the first dimension corresponds to the batch index, and the second to the sequence. These are "left-aligned" *ragged* matrices (the first index of the second dimension is the beginning of every sequence. The length of each sequence is given by ``valid_length``, of shape (N,) where ``valid_length[i]`` corresponds to the number of valid points in ``lags[i, :]`` and ``marks[i, :]``.

``max_time`` is the length of the observation period of the point process. That is, specifying ``max_time[i] = 5`` computes the likelihood of the i-th sample as observed on the time interval :math:`(0, 5]`. Naturally, the sum of all valid ``lags[i, :valid_length[i]]`` must be less than or equal to 5.

The input ``state`` specifies the *memory* of the Hawkes process. Invoking the memoryless property of exponential decays, we compute the *memory* as

.. math::

    s_k(t) = \sum_{t_i < t} \exp(-\beta_k (t - t_i)).

The ``state`` to be provided is :math:`s_k(0)` and carries the added intensity due to past events before the current batch. :math:`s_k(T)` is returned from the function where :math:`T` is ``max_time[T]``.

Example::

  # define the Hawkes process parameters
  lda = nd.array([1.5, 2.0, 3.0]).tile((N, 1))
  alpha = nd.array([0.2, 0.3, 0.4])  # branching ratios should be < 1
  beta = nd.array([1.0, 2.0, 3.0])

  # the "data", or observations
  ia_times = nd.array([[6, 7, 8, 9], [1, 2, 3, 4], [3, 4, 5, 6], [8, 9, 10, 11]])
  marks = nd.zeros((N, T)).astype(np.int32)

  # starting "state" of the process
  states = nd.zeros((N, K))

  valid_length = nd.array([1, 2, 3, 4])  # number of valid points in each sequence
  max_time = nd.ones((N,)) * 100.0  # length of the observation period

  A = nd.contrib.hawkesll(
      lda, alpha, beta, states, ia_times, marks, valid_length, max_time
  )

References:

-  Bacry, E., Mastromatteo, I., & Muzy, J. F. (2015).
   Hawkes processes in finance. Market Microstructure and Liquidity
   , 1(01), 1550005.
)code" ADD_FILELINE)
    .set_num_inputs(8)
    .set_num_outputs(2)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
        [](const NodeAttrs& attrs) {
        return std::vector<std::string>{
          "lda", "alpha", "beta", "state", "lags",
          "marks", "valid_length", "max_time"
        };
    })
    .set_attr<nnvm::FListOutputNames>("FListOutputNames",
        [](const NodeAttrs& attrs) {
        return std::vector<std::string>{"output", "out_state"};
    })
    .set_attr<mxnet::FInferShape>("FInferShape", HawkesLLOpShape)
    .set_attr<nnvm::FInferType>("FInferType", HawkesLLOpType)
    .set_attr<FCompute>("FCompute<cpu>", HawkesLLForward<cpu>)
    .set_attr<nnvm::FGradient>(
      "FGradient", ElemwiseGradUseIn{"_contrib_backward_hawkesll"}
    )
    .set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
        return std::vector<ResourceRequest>{ResourceRequest::Type::kTempSpace};
    })
    .set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
    .add_argument(
      "lda", "NDArray-or-Symbol",
      "Shape (N, K) The intensity for each of the K processes, for each sample"
    )
    .add_argument(
      "alpha", "NDArray-or-Symbol",
      "Shape (K,) The infectivity factor (branching ratio) for each process"
    )
    .add_argument(
      "beta", "NDArray-or-Symbol",
      "Shape (K,) The decay parameter for each process"
    )
    .add_argument(
      "state", "NDArray-or-Symbol",
      "Shape (N, K) the Hawkes state for each process"
    )
    .add_argument(
      "lags", "NDArray-or-Symbol",
      "Shape (N, T) the interarrival times"
    )
    .add_argument(
      "marks", "NDArray-or-Symbol",
      "Shape (N, T) the marks (process ids)"
    )
    .add_argument(
      "valid_length", "NDArray-or-Symbol",
      "The number of valid points in the process"
    )
    .add_argument(
      "max_time", "NDArray-or-Symbol",
      "the length of the interval where the processes were sampled");

NNVM_REGISTER_OP(_contrib_backward_hawkesll)
    .set_num_inputs(10)
    .set_num_outputs(8)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_attr<FCompute>("FCompute<cpu>", HawkesLLBackward<cpu>)
    .set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
        return std::vector<ResourceRequest>{ResourceRequest::Type::kTempSpace};
    });
}  // namespace op
}  // namespace mxnet
