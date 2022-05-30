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
 * \file dnnl_pow_mul_scalar-inl.h
 */

#ifndef MXNET_OPERATOR_NN_DNNL_DNNL_POW_MUL_SCALAR_INL_H_
#define MXNET_OPERATOR_NN_DNNL_DNNL_POW_MUL_SCALAR_INL_H_

#if MXNET_USE_ONEDNN == 1

#include <vector>

#include "operator/tensor/elemwise_binary_scalar_op.h"

namespace mxnet {
namespace op {

struct DNNLPowMulScalarParam : public dmlc::Parameter<DNNLPowMulScalarParam> {
  float exponent;
  float multiplier;
  bool exp_is_int;
  bool mul_is_int;

  DMLC_DECLARE_PARAMETER(DNNLPowMulScalarParam) {
    DMLC_DECLARE_FIELD(exponent).describe("Exponent for power operation.").set_default(1);
    DMLC_DECLARE_FIELD(multiplier).describe("Multiplier for multiply operation.").set_default(1);
    DMLC_DECLARE_FIELD(exp_is_int)
        .describe("Indicate whether exponent is int type.")
        .set_default(true);
    DMLC_DECLARE_FIELD(mul_is_int)
        .describe("Indicate whether multiplier is int type.")
        .set_default(true);
  }

  bool operator==(const DNNLPowMulScalarParam& other) const {
    return this->exponent == other.exponent && this->multiplier == other.multiplier &&
           this->exp_is_int == other.exp_is_int && this->mul_is_int == other.mul_is_int;
  }
};

using eltwise_fwd_t    = dnnl::eltwise_forward;
using eltwise_fwd_pd_t = dnnl::eltwise_forward::primitive_desc;

typedef ParamOpSign<DNNLPowMulScalarParam> DNNLPowMulScalarSignature;

class DNNLPowMulScalarFwd {
 public:
  static DNNLPowMulScalarFwd& GetCached(const DNNLPowMulScalarParam& param,
                                        const NDArray& input,
                                        const NDArray& output);

  DNNLPowMulScalarFwd(const DNNLPowMulScalarParam& param, const NDArray& input);

  void Execute(const NDArray& input, const OpReqType& req, const NDArray& output);

 private:
  std::shared_ptr<eltwise_fwd_t> fwd;
  std::shared_ptr<eltwise_fwd_pd_t> fwd_pd;
};

template <bool subgraph>
inline void DNNLPowMulScalarForward(const nnvm::NodeAttrs& attrs,
                                    const OpContext& ctx,
                                    const std::vector<NDArray>& inputs,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<NDArray>& outputs) {
  DNNLPowMulScalarParam param;
  if (subgraph) {
    param = nnvm::get<DNNLPowMulScalarParam>(attrs.parsed);
  } else {
    param.multiplier = 1;
    param.exponent   = nnvm::get<NumpyBinaryScalarParam>(attrs.parsed).scalar;
  }
  DNNLPowMulScalarFwd& fwd = DNNLPowMulScalarFwd::GetCached(param, inputs[0], outputs[0]);
  fwd.Execute(inputs[0], req[0], outputs[0]);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_NN_DNNL_DNNL_POW_MUL_SCALAR_INL_H_
