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
 * Copyright (c) 2020 by Contributors
 * \file quantized_bert_ffn1_to_ffn2-inl.h
 * \brief implementation of quantized bert_ffn1_to_ffn2 fused operator
*/
#ifndef MXNET_OPERATOR_QUANTIZED_BERT_FFN1_TO_FFN2_INL_H_
#define MXNET_OPERATOR_QUANTIZED_BERT_FFN1_TO_FFN2_INL_H_

namespace mxnet {
namespace op {

struct QuantizedBERTFFN1TOFFN2Param : public dmlc::Parameter<QuantizedBERTFFN1TOFFN2Param> {
  // for quantized FC of ffn1 (should be a biased FC)
  int num_hidden;
  bool flatten;

  // for quantization before ffn2
  dmlc::optional<float> min_calib_range;
  dmlc::optional<float> max_calib_range;

  DMLC_DECLARE_PARAMETER(QuantizedBERTFFN1TOFFN2Param) {
    DMLC_DECLARE_FIELD(num_hidden).set_lower_bound(1)
    .describe("Number of hidden nodes of the output.");
    DMLC_DECLARE_FIELD(flatten).set_default(true)
    .describe("Whether to collapse all but the first axis of the input data tensor."); 
    DMLC_DECLARE_FIELD(min_calib_range)
      .set_default(dmlc::optional<float>())
      .describe("The minimum scalar value in the form of float32. If present, it will be used to "
                "quantize the fp32 data into int8 or uint8.");
    DMLC_DECLARE_FIELD(max_calib_range)
      .set_default(dmlc::optional<float>())
      .describe("The maximum scalar value in the form of float32. If present, it will be used to "
                "quantize the fp32 data into int8 or uint8.");    
  }
  bool operator==(const QuantizedBERTFFN1TOFFN2Param& other) const {
    return this->num_hidden == other.num_hidden &&
           this->flatten == other.flatten &&
           this->min_calib_range == other.min_calib_range &&
           this->max_calib_range == other.max_calib_range;
  }
};

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_QUANTIZED_BERT_FFN1_TO_FFN2_INL_H_
