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
 * \file gradient_compression.h
 * \brief Gradient compression for kvstore
 * \author Rahul Huilgol
 */

#ifndef MXNET_KVSTORE_GRADIENT_COMPRESSION_H_
#define MXNET_KVSTORE_GRADIENT_COMPRESSION_H_
#include <dmlc/parameter.h>
#include <string>
#include <utility>
#include <vector>
#include "mxnet/ndarray.h"

namespace mxnet {
namespace kvstore {

enum class CompressionType {
  kNone, kTwoBit
};

struct GradientCompressionParam : public dmlc::Parameter<GradientCompressionParam> {
  std::string type;
  float threshold;
  DMLC_DECLARE_PARAMETER(GradientCompressionParam) {
    DMLC_DECLARE_FIELD(type)
      .describe("Type of gradient compression to use, like `2bit` for example");
    DMLC_DECLARE_FIELD(threshold).set_default(0.5)
      .describe("Threshold to use for 2bit gradient compression");
  }
};

class GradientCompression {
 public:
  GradientCompression();

  virtual ~GradientCompression() {}

  /*!
   * \brief sets parameters for gradient compression
   * \param kwargs a vector of pair of strings. A pair represents key and value
   * of the parameter. Will be parsed by GradientCompressionParam
   */
  void SetParams(const std::vector<std::pair<std::string, std::string> >& kwargs);

  /*!
   * \brief returns type of compression if any
   */
  CompressionType get_type();

  /*!
   * \brief returns as string the enum value of compression type
   */
  std::string get_type_str();

  /*!
   * \brief sets two bit gradient compression
   * \param threshold float value used for thresholding gradients
   */
  void SetTwoBitCompression(const float threshold);

  /*!
   * \brief encodes parameters of gc into a string
   */
  std::string EncodeParams();

  /*!
   * \brief decodes parameters of gc from a string and assigns them to member variables
   */
  void DecodeParams(const std::string &s);

  /*!
   * \brief returns compression factor, which is the factor by which size of gradient
   * reduces when using a particular type of compression
   */
  int GetCompressionFactor();

  /*!
   * \brief returns the size of compressed gradients given an original sized gradient array
   */
  int64_t GetCompressedSize(const int64_t original_size);

  /*!
  * \brief Issues quantize operation to be scheduled by the engine
  * Compresses `from` into `to` and accumulates the quantization error
  * into 'residual', using the quantization of type `type_`
  * \param from the ndarray containing original data to be quantized
  * \param to the target ndarray which contains quantized data
  * \param residual the ndarray which accumulates quantization error
  * \param priority Priority of the action.
  */
  void Quantize(const mxnet::NDArray &from, mxnet::NDArray *to,
                mxnet::NDArray *residual, const int priority);

  /*!
  * \brief Issues dequantize operation to be scheduled by the engine
  * Decompresses `from` into `to` using current parameters of `type` and `threshold`
  * \param from the ndarray containing quantized data
  * \param to the target ndarray which contains final dequantized data
  * \param priority Priority of the action.
  */
  void Dequantize(const mxnet::NDArray &from, mxnet::NDArray *to, const int priority);

 private:
  /*!
   * \brief denotes the type of gradient compression which has been set
   */
  CompressionType type_;

  /*!
   * \brief denotes threshold used for quantization and dequantization
   * Must be a positive value. All positive gradients will be thresholded to `threshold_` and
   * all negative gradients will be thresholded to -1*`threshold_`
   */
  float threshold_ = 0;
};
}  // namespace kvstore
}  // namespace mxnet
#endif  // MXNET_KVSTORE_GRADIENT_COMPRESSION_H_
