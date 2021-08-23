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
 * \file hawkes_ll-inl.h
 * \brief Log likelihood of a marked self-exciting Hawkes process
 * \author Caner Turkmen <turkmen.ac@gmail.com>
 */
#ifndef MXNET_OPERATOR_CONTRIB_HAWKES_LL_INL_H_
#define MXNET_OPERATOR_CONTRIB_HAWKES_LL_INL_H_

#include <mxnet/operator.h>
#include <vector>

#include "../operator_common.h"
#include "../mshadow_op.h"
#include "../mxnet_op.h"

namespace mxnet {
namespace op {

namespace hawkesll {
  enum HawkesLLOpInputs {kMu, kAlpha, kBeta, kState, kIATimes, kMarks,
                         kValidLength, kMaxTime};
  enum HawkesLLGradInputs {kOutGradLL, kOutGradStates, kGradMu, kGradAlpha,
                           kGradBeta, kGradState, kGradIATimes, kGradMarks,
                           kGradValidLength, kGradMaxTime};
  enum HawkesLLOpOutputs {kOutLL, kOutStates};
  enum HawkesLLOpResource {kTempSpace};
}  // namespace hawkesll

inline bool HawkesLLOpType(const nnvm::NodeAttrs& attrs,
                            std::vector<int>* in_attrs,
                            std::vector<int>* out_attrs) {
  // check dimensions of the type vectors
  CHECK_EQ(in_attrs->size(), 8U);
  CHECK_EQ(out_attrs->size(), 2U);

  TYPE_ASSIGN_CHECK(*out_attrs, hawkesll::kOutLL, in_attrs->at(0))
  TYPE_ASSIGN_CHECK(*out_attrs, hawkesll::kOutStates, in_attrs->at(0))

  for (index_t j = 0; j < 8; ++j) {
    if (j != hawkesll::kMarks) {
      TYPE_ASSIGN_CHECK(*in_attrs, j, out_attrs->at(0))
    }
  }
  TYPE_ASSIGN_CHECK(*in_attrs, hawkesll::kMarks, 4)  // int32

  return out_attrs->at(hawkesll::kOutLL) != -1;
}

inline bool HawkesLLOpShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape>* in_attrs,
                             std::vector<TShape>* out_attrs) {
  using namespace mshadow;
  int N, T, K;

  CHECK_EQ(in_attrs->size(), 8U);
  CHECK_EQ(out_attrs->size(), 2U);

  // check ndims
  CHECK_EQ(in_attrs->at(hawkesll::kMu).ndim(), 2);  // mu (N, K)
  CHECK_EQ(in_attrs->at(hawkesll::kAlpha).ndim(), 1);  // branching ratio (K,)
  CHECK_EQ(in_attrs->at(hawkesll::kBeta).ndim(), 1);  // decay exponent (K,)
  CHECK_EQ(in_attrs->at(hawkesll::kState).ndim(), 2);  // Hawkes states (N, K)
  CHECK_EQ(in_attrs->at(hawkesll::kIATimes).ndim(), 2);  // i.a. times  (N, T)
  CHECK_EQ(in_attrs->at(hawkesll::kMarks).ndim(), 2);  // marks (N, T)
  CHECK_EQ(in_attrs->at(hawkesll::kValidLength).ndim(), 1);  // valid len (N,)
  CHECK_EQ(in_attrs->at(hawkesll::kMaxTime).ndim(), 1);  // max_time (N,)

  N = in_attrs->at(hawkesll::kIATimes)[0];  // number of samples in batch
  T = in_attrs->at(hawkesll::kIATimes)[1];  // time length
  K = in_attrs->at(hawkesll::kMu)[1];  // number of marks

  // check inputs consistent
  CHECK_EQ(in_attrs->at(hawkesll::kMu)[0], N);
  CHECK_EQ(in_attrs->at(hawkesll::kMu)[1], K);
  CHECK_EQ(in_attrs->at(hawkesll::kAlpha)[0], K);
  CHECK_EQ(in_attrs->at(hawkesll::kBeta)[0], K);
  CHECK_EQ(in_attrs->at(hawkesll::kState)[0], N);
  CHECK_EQ(in_attrs->at(hawkesll::kState)[1], K);
  CHECK_EQ(in_attrs->at(hawkesll::kMarks)[0], N);
  CHECK_EQ(in_attrs->at(hawkesll::kMarks)[1], T);
  CHECK_EQ(in_attrs->at(hawkesll::kValidLength)[0], N);
  CHECK_EQ(in_attrs->at(hawkesll::kMaxTime)[0], N);

  // infer output type
  SHAPE_ASSIGN_CHECK(*out_attrs, hawkesll::kOutLL, Shape1(N))
  SHAPE_ASSIGN_CHECK(*out_attrs, hawkesll::kOutStates, Shape2(N, K))

  return out_attrs->at(hawkesll::kOutLL).ndim() != 0U &&
    out_attrs->at(hawkesll::kOutStates).Size() != 0U;
}

template<int req>
struct hawkesll_forward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i,
                                  DType* out_loglike,
                                  DType* out_state,
                                  const DType* mu,
                                  const DType* alpha,
                                  const DType* beta,
                                  DType* state,
                                  const DType* lags,
                                  const int32_t* marks,
                                  DType* valid_length,
                                  DType* max_time,
                                  int K,
                                  int T,
                                  DType* temp_register
                                  ) {
    int32_t ci;  // current mark
    DType ll = 0;  // log likelihood
    DType t = 0;  // current time
    DType d, ed, lda, comp;
    DType *last_ = &temp_register[i * K];

    const DType *mu_ = &mu[i * K];
    const DType *lag_ = &lags[i * T];
    const int32_t *mark_ = &marks[i * T];
    DType *state_ = &out_state[i * K];

    // iterate over points in sequence
    for (index_t j = 0; j < valid_length[i]; ++j) {
      ci = mark_[j];
      t += lag_[j];
      d = t - last_[ci];
      ed = expf(-beta[ci] * d);

      lda = mu_[ci] + alpha[ci] * beta[ci] * state_[ci] * ed;
      comp = mu_[ci] * d + alpha[ci] * state_[ci] * (1 - ed);

      ll += logf(lda) - comp;

      KERNEL_ASSIGN(state_[ci], req, 1 + (state_[ci] * ed))

      last_[ci] = t;
    }

    KERNEL_ASSIGN(out_loglike[i], req, ll)
  }
};

template<int req>
struct hawkesll_forward_compensator {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i,
                                  DType* rem_comp,
                                  DType* out_state,
                                  const DType* mu,
                                  const DType* alpha,
                                  const DType* beta,
                                  const DType* max_time,
                                  const int K,
                                  const DType* last_buffer
                                  ) {
    DType d, ed;
    int m = i % K;  // mark
    int j = i / K;  // particle

    // take care of the remaining compensators and state update
    d = max_time[j] - last_buffer[i];
    ed = expf(-beta[m] * d);

    // return the remaining compensator
    KERNEL_ASSIGN(rem_comp[i], req,
                  mu[i] * d + alpha[m] * out_state[i] * (1 - ed))

    // update the state
    KERNEL_ASSIGN(out_state[i], req, ed * out_state[i])
  }
};

template<typename xpu>
void HawkesLLForward(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;

  Stream<xpu> *s = ctx.get_stream<xpu>();

  CHECK_EQ(inputs.size(), 8U);
  CHECK_EQ(outputs.size(), 2U);

  const TBlob& out_loglike = outputs[hawkesll::kOutLL];
  const TBlob& out_state = outputs[hawkesll::kOutStates];

  int K = inputs[hawkesll::kMu].shape_[1];
  int N = inputs[hawkesll::kIATimes].shape_[0];
  int T = inputs[hawkesll::kIATimes].shape_[1];

  MSHADOW_TYPE_SWITCH(out_loglike.type_flag_, DType, {
    Tensor<xpu, 2, DType> temp_space = ctx.requested[hawkesll::kTempSpace]
                                          .get_space_typed<xpu, 2, DType>(
                                            Shape2(2*N, K),
                                            s);

    Tensor<xpu, 2, DType> last_buffer =
        Tensor<xpu, 2, DType>(&temp_space.dptr_[0], Shape2(N, K), s);
    Tensor<xpu, 2, DType> rem_comp =
        Tensor<xpu, 2, DType>(&temp_space.dptr_[N*K], Shape2(N, K), s);

    Tensor<xpu, 1, DType> out_loglike_ts =
        out_loglike.get_with_shape<xpu, 1, DType>(Shape1(N), s);

    last_buffer = DType(0.0);
    rem_comp = DType(0.0);

    Tensor<xpu, 2, DType> out_state_ts =
        out_state.get_with_shape<xpu, 2, DType>(Shape2(N, K), s);
    Tensor<xpu, 2, DType> in_state_ts =
        inputs[hawkesll::kState].get_with_shape<xpu, 2, DType>(Shape2(N, K), s);

    mshadow::Copy(out_state_ts, in_state_ts, s);

    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<hawkesll_forward<req_type>, xpu>::Launch(
        s, N,
        out_loglike.dptr<DType>(),
        out_state.dptr<DType>(),
        inputs[hawkesll::kMu].dptr<DType>(),  // mu
        inputs[hawkesll::kAlpha].dptr<DType>(),  // alpha
        inputs[hawkesll::kBeta].dptr<DType>(),  // beta
        inputs[hawkesll::kState].dptr<DType>(),  // states
        inputs[hawkesll::kIATimes].dptr<DType>(),  // interarrival times
        inputs[hawkesll::kMarks].dptr<int32_t>(),  // marks
        inputs[hawkesll::kValidLength].dptr<DType>(),  // valid_length
        inputs[hawkesll::kMaxTime].dptr<DType>(),  // max_time
        K,
        T,
        last_buffer.dptr_);
    });

    // in parallel, we take care of the remaining compensators
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<hawkesll_forward_compensator<req_type>, xpu>::Launch(
        s, N * K,
        rem_comp.dptr_,
        out_state.dptr<DType>(),
        inputs[hawkesll::kMu].dptr<DType>(),  // mu
        inputs[hawkesll::kAlpha].dptr<DType>(),  // alpha
        inputs[hawkesll::kBeta].dptr<DType>(),  // beta
        inputs[hawkesll::kMaxTime].dptr<DType>(),  // max_time
        K,
        last_buffer.dptr_);
    });
    out_loglike_ts -= mshadow::expr::sumall_except_dim<0>(rem_comp);
  })
}

template<int req>
struct hawkesll_backward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i,  // indexes the sample (particle)
                                  DType* mu_gbfr,
                                  DType* alpha_gbfr,
                                  DType* beta_gbfr,  // (N, K)
                                  const DType* mu,   // (N, K)
                                  const DType* alpha,   // (K,)
                                  const DType* beta,    // (K,)
                                  const DType* lags,    // (N, T)
                                  const int32_t* marks,  // (N, T)
                                  const DType* valid_length,  // (N,)
                                  const DType* max_time,  // (N,)
                                  const int K,
                                  const int T,
                                  DType* last_buffer,
                                  DType* phi_buffer,
                                  DType* phig_buffer
                                  ) {
    int32_t ci;
    int32_t part_ix_K = i*K, part_ix_T = i*T;

    DType t = 0, d, lda, ed;
    DType* last_ = &last_buffer[part_ix_K];
    DType* state_ = &phi_buffer[part_ix_K];
    DType* dstate_ = &phig_buffer[part_ix_K];

    DType* mug_ = &mu_gbfr[part_ix_K];
    DType* alphag_ = &alpha_gbfr[part_ix_K];
    DType* betag_ = &beta_gbfr[part_ix_K];

    const DType* lag_ = &lags[part_ix_T];
    const int32_t* mark_ = &marks[part_ix_T];

    // iterate over points
    for (index_t j = 0; j < valid_length[i]; ++j){
      ci = mark_[j];
      t += lag_[j];
      d = t - last_[ci];
      ed = expf(-beta[ci] * d);

      lda = mu[part_ix_K + ci] + alpha[ci] * beta[ci] * state_[ci] * ed;

      KERNEL_ASSIGN(mug_[ci], req, mug_[ci] + (1 / lda) - d)
      KERNEL_ASSIGN(alphag_[ci], req,
                    (
                      alphag_[ci]
                      + (beta[ci] * state_[ci] * ed) / lda
                      - state_[ci] * (1 - ed)
                    )
      )
      KERNEL_ASSIGN(betag_[ci], req,
                    betag_[ci]
                    + alpha[ci] * ed
                    * (state_[ci] * (1 - beta[ci] * d) + beta[ci] * dstate_[ci])
                    / lda
                    - alpha[ci]
                    * (dstate_[ci] * (1 - ed) + state_[ci] * d * ed)
      )

      KERNEL_ASSIGN(dstate_[ci], req, ed * (-d * state_[ci] + dstate_[ci]))
      KERNEL_ASSIGN(state_[ci], req, 1 + (state_[ci] * ed))

      last_[ci] = t;
    }
  }
};


template<int req>
struct hawkesll_backward_compensator {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i,
                                  DType* mu_gbfr,
                                  DType* alpha_gbfr,
                                  DType* beta_gbfr,  // (N, K)
                                  DType* out_grad,  // read this  (N,)
                                  const DType* mu,  // (N, K)
                                  const DType* alpha,   // (K,)
                                  const DType* beta,    // (K,)
                                  const DType* max_time,  // (N,)
                                  const int K,
                                  DType* last_buffer,
                                  DType* phi_buffer,
                                  DType* phig_buffer
                                  ) {
    DType d, ed;
    int m = i % K;  // mark
    int j = i / K;  // particle
    int32_t part_ix_K = j*K;
    DType* mug_ = &mu_gbfr[part_ix_K];
    DType* alphag_ = &alpha_gbfr[part_ix_K];
    DType* betag_ = &beta_gbfr[part_ix_K];

    // take care of the remaining compensators and state update
    d = max_time[j] - last_buffer[i];
    ed = expf(-beta[m] * d);

    // take care of the gradients of the remaining compensator
    KERNEL_ASSIGN(mug_[m], req, mug_[m] - d)
    KERNEL_ASSIGN(alphag_[m], req,
                  alphag_[m] - phi_buffer[i] * (1 - ed)
    )
    KERNEL_ASSIGN(betag_[m], req,
                  betag_[m] - alpha[m] * (
                    phig_buffer[i] * (1 - ed)
                    + phi_buffer[i] * d * ed
                  )
    )

    // // correct the gradients with respect to output gradients
    KERNEL_ASSIGN(mug_[m], req, out_grad[j] * mug_[m])
    KERNEL_ASSIGN(alphag_[m], req, out_grad[j] * alphag_[m])
    KERNEL_ASSIGN(betag_[m], req, out_grad[j] * betag_[m])
  }
};

template<typename xpu>
void HawkesLLBackward(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 10U);
  CHECK_EQ(outputs.size(), 8U);
  CHECK_EQ(req.size(), 8U);

  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();

  int K = inputs[hawkesll::kGradMu].shape_[1];  // mu data
  int N = inputs[hawkesll::kGradIATimes].shape_[0];
  int T = inputs[hawkesll::kGradIATimes].shape_[1];

  CHECK_EQ(inputs[hawkesll::kOutGradLL].shape_[0], N);  // grad of out 0 (LL)
  CHECK_EQ(inputs[hawkesll::kOutGradStates].shape_[0], N);  // grad out 1-states
  CHECK_EQ(inputs[hawkesll::kOutGradStates].shape_[1], K);

  // sufficient statistics are not differentiated w.r.t.
  CHECK_EQ(req[hawkesll::kIATimes], OpReqType::kNullOp);
  CHECK_EQ(req[hawkesll::kMarks], OpReqType::kNullOp);
  CHECK_EQ(req[hawkesll::kValidLength], OpReqType::kNullOp);
  CHECK_EQ(req[hawkesll::kMaxTime], OpReqType::kNullOp);

  const TBlob& out_grad = inputs[hawkesll::kOutGradLL];

  using namespace mshadow;
  using namespace mxnet_op;
  MSHADOW_TYPE_SWITCH(out_grad.type_flag_, DType, {
    // allocate gradient buffers
    Tensor<xpu, 2, DType> bfr =
        ctx.requested[hawkesll::kTempSpace]
           .get_space_typed<xpu, 2, DType>(Shape2(6*N, K), s);

    Tensor<xpu, 2, DType> alpha_gbfr =
        Tensor<xpu, 2, DType>(&bfr.dptr_[N*K], Shape2(N, K), s);
    Tensor<xpu, 2, DType> beta_gbfr =
        Tensor<xpu, 2, DType>(&bfr.dptr_[2*N*K], Shape2(N, K), s);
    Tensor<xpu, 2, DType> last_buffer =
        Tensor<xpu, 2, DType>(&bfr.dptr_[3*N*K], Shape2(N, K), s);
    Tensor<xpu, 2, DType> phig_buffer =
        Tensor<xpu, 2, DType>(&bfr.dptr_[4*N*K], Shape2(N, K), s);
    Tensor<xpu, 2, DType> phi_buffer =
        Tensor<xpu, 2, DType>(&bfr.dptr_[5*N*K], Shape2(N, K), s);

    alpha_gbfr = DType(0.0);
    beta_gbfr = DType(0.0);
    last_buffer = DType(0.0);
    phig_buffer = DType(0.0);

    mshadow::Copy(phi_buffer,
                  inputs[hawkesll::kGradState]
                    .get_with_shape<xpu, 2, DType>(Shape2(N, K), s),
                  s);

    // get the gradient to be output
    Tensor<xpu, 2, DType> in_grad_mu =
        outputs[hawkesll::kMu].get_with_shape<xpu, 2, DType>(Shape2(N, K), s);
    Tensor<xpu, 1, DType> in_grad_alpha =
        outputs[hawkesll::kAlpha].get_with_shape<xpu, 1, DType>(Shape1(K), s);
    Tensor<xpu, 1, DType> in_grad_beta =
        outputs[hawkesll::kBeta].get_with_shape<xpu, 1, DType>(Shape1(K), s);

    in_grad_mu = DType(0.0);

    MXNET_ASSIGN_REQ_SWITCH(req[hawkesll::kMu], req_type, {
      Kernel<hawkesll_backward<req_type>, xpu>::Launch(
        s,
        N,
        in_grad_mu.dptr_, alpha_gbfr.dptr_, beta_gbfr.dptr_,  // gradients
        inputs[hawkesll::kGradMu].dptr<DType>(),  // mu data
        inputs[hawkesll::kGradAlpha].dptr<DType>(),  // alpha data
        inputs[hawkesll::kGradBeta].dptr<DType>(),  // beta data
        inputs[hawkesll::kGradIATimes].dptr<DType>(),  // lags data
        inputs[hawkesll::kGradMarks].dptr<int32_t>(),  // marks data
        inputs[hawkesll::kGradValidLength].dptr<DType>(),  // valid_length data
        inputs[hawkesll::kGradMaxTime].dptr<DType>(),  // max_time data
        K,
        T,
        last_buffer.dptr_,  // buffer to keep timestamp of last item
        phi_buffer.dptr_,  // "states"
        phig_buffer.dptr_);  // derivatives of "states"
    });

    MXNET_ASSIGN_REQ_SWITCH(req[hawkesll::kMu], req_type, {
      Kernel<hawkesll_backward_compensator<req_type>, xpu>::Launch(
        s,
        N * K,
        in_grad_mu.dptr_, alpha_gbfr.dptr_, beta_gbfr.dptr_,  // gradients
        out_grad.dptr<DType>(),
        inputs[hawkesll::kGradMu].dptr<DType>(),  // mu data
        inputs[hawkesll::kGradAlpha].dptr<DType>(),  // alpha data
        inputs[hawkesll::kGradBeta].dptr<DType>(),  // beta data
        inputs[hawkesll::kGradMaxTime].dptr<DType>(),  // max_time data
        K,
        last_buffer.dptr_,  // buffer to keep timestamp of last item
        phi_buffer.dptr_,  // "states"
        phig_buffer.dptr_);  // derivatives of "states"
    });

    // reduce the gradients
    Assign(in_grad_alpha, req[hawkesll::kAlpha],
           mshadow::expr::sumall_except_dim<1>(alpha_gbfr)
           )

    Assign(in_grad_beta, req[hawkesll::kBeta],
           mshadow::expr::sumall_except_dim<1>(beta_gbfr)
           )
  })
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_HAWKES_LL_INL_H_
