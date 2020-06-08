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
 *  Copyright (c) 2019 by Contributors
 * \file np_bicount_op.cc
 * \brief numpy compatible bincount operator CPU registration
 */

#include "./np_bincount_op-inl.h"

namespace mxnet {
namespace op {

void BinNumberCount(const NDArray& data, const int& minlength,
                    const NDArray& out, const size_t& N) {
  int bin = minlength;
  MSHADOW_TYPE_SWITCH(data.dtype(), DType, {
    DType* data_ptr = data.data().dptr<DType>();
    for (size_t i = 0; i < N; i++) {
      CHECK_GE(data_ptr[i], 0) << "input should be nonnegative number";
      if (data_ptr[i] + 1 > bin) {
        bin = data_ptr[i] + 1;
      }
    }
  });  // bin number = max(max(data) + 1, minlength)
  mxnet::TShape s(1, bin);
  const_cast<NDArray &>(out).Init(s);  // set the output shape forcefully
}

template<typename DType, typename OType>
void BincountCpuWeights(const DType* data, const OType* weights,
                        OType* out, const size_t& data_n) {
  for (size_t i = 0; i < data_n; i++) {
    int target = data[i];
    out[target] += weights[i];
  }
}

template<typename DType, typename OType>
void BincountCpu(const DType* data, OType* out, const size_t& data_n) {
  for (size_t i = 0; i < data_n; i++) {
    int target = data[i];
    out[target] += 1;
  }
}

template<>
void NumpyBincountForwardImpl<cpu>(const OpContext &ctx,
                                   const NDArray &data,
                                   const NDArray &weights,
                                   const NDArray &out,
                                   const size_t &data_n,
                                   const int &minlength) {
  using namespace mxnet_op;
  BinNumberCount(data, minlength, out, data_n);
  mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
  MSHADOW_TYPE_SWITCH(data.dtype(), DType, {
      MSHADOW_TYPE_SWITCH(weights.dtype(), OType, {
        size_t out_size = out.shape()[0];
        Kernel<set_zero, cpu>::Launch(s, out_size, out.data().dptr<OType>());
        BincountCpuWeights(data.data().dptr<DType>(), weights.data().dptr<OType>(),
                          out.data().dptr<OType>(), data_n);
      });
    });
}

template<>
void NumpyBincountForwardImpl<cpu>(const OpContext &ctx,
                                   const NDArray &data,
                                   const NDArray &out,
                                   const size_t &data_n,
                                   const int &minlength) {
  using namespace mxnet_op;
  BinNumberCount(data, minlength, out, data_n);
  mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
  MSHADOW_TYPE_SWITCH(data.dtype(), DType, {
      MSHADOW_TYPE_SWITCH(out.dtype(), OType, {
        size_t out_size = out.shape()[0];
        Kernel<set_zero, cpu>::Launch(s, out_size, out.data().dptr<OType>());
        BincountCpu(data.data().dptr<DType>(), out.data().dptr<OType>(), data_n);
      });
    });
}

DMLC_REGISTER_PARAMETER(NumpyBincountParam);

NNVM_REGISTER_OP(_npi_bincount)
.set_attr_parser(ParamParser<NumpyBincountParam>)
.set_num_inputs([](const NodeAttrs& attrs) {
  const NumpyBincountParam& params =
    nnvm::get<NumpyBincountParam>(attrs.parsed);
  return params.has_weights? 2 : 1;
  })
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    const NumpyBincountParam& params =
      nnvm::get<NumpyBincountParam>(attrs.parsed);
    return params.has_weights ?
           std::vector<std::string>{"data", "weights"} :
           std::vector<std::string>{"data"};
  })
.set_attr<FResourceRequest>("FResourceRequest",
[](const NodeAttrs& attrs) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<nnvm::FInferType>("FInferType", NumpyBincountType)
.set_attr<FInferStorageType>("FInferStorageType", NumpyBincountStorageType)
.set_attr<FComputeEx>("FComputeEx<cpu>", NumpyBincountForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("data", "NDArray-or-Symbol", "Data")
.add_argument("weights", "NDArray-or-Symbol", "Weights")
.add_arguments(NumpyBincountParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
