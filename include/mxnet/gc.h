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
 * \file gc.h
 * \brief Gradient compression for kvstore
 * \author Rahul Huilgol
 */

#ifndef MXNET_GC_H
#define MXNET_GC_H
#include <string>
#include"./ndarray.h"

namespace mxnet {
namespace kvstore {

enum CompressionType {
  GC_NONE, GC_TWO_BIT
};

class Gc {
 public:
  Gc();

  virtual ~Gc() {}

  /*!
   * \brief sets parameters for gradient compression
   * \param compression_type str representing types like 2bit
   * \param threshold float value used for thresholding gradients
   */
  void SetParams(const std::string &compression_type, const float threshold);

  /*!
   * \brief sets gradient compression to given mode
   * Active mode is when gradients are compressed
   * Compression is in inactive mode during init of parameters
   */
  void set_active(bool active);

  /*!
   * \brief returns boolean whether or not gc is in active mode
   */
  bool is_active();

  /*!
   * \brief returns type of compression if any
   */
  CompressionType get_type();

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
   * \brief denotes whether gradient compression is active
   * Value starts with false because we don't want initialization of parameters to be compressed.
   * That would lead to bad convergence results. Currently after initialization, gc becomes active.
   */
  bool active_;

  /*!
   * \brief denotes threshold used for quantization and dequantization
   * Must be a positive value. All positive gradients will be thresholded to `threshold_` and
   * all negative gradients will be thresholded to -1*`threshold_`
   */
  float threshold_ = 0;
};
}  // namespace kvstore
}  // namespace mxnet
#endif  // MXNET_GC_H
