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

#ifndef MXNET_KVSTORE_GC_H
#define MXNET_KVSTORE_GC_H
#include <string>
#include <sstream>
#include <vector>
#include"./ndarray.h"
#include "./base.h"
#include <mshadow/tensor.h>
namespace mxnet {
  namespace kvstore {

    enum CompressionType {
      GC_NONE, GC_TWO_BIT
    };

    class Gc {
    public:
      Gc();

      virtual ~Gc() {}

      void SetParams(const std::string &compression_type, const float threshold);

      void set_active();

      bool get_active();

      bool get_active_type();

      void SetTwoBitCompression(const float threshold);

      std::string EncodeParams();

      void DecodeParams(const std::string &s);

      int GetCompressionFactor();

      int64_t GetCompressedSize(const int64_t original_size);

      void Quantize(const mxnet::NDArray &from, mxnet::NDArray *to,
                    mxnet::NDArray *residual, const int priority);

      void Dequantize(const mxnet::NDArray &from, mxnet::NDArray *to, const int priority);

    private:
      CompressionType type_;

      bool active_;

      float threshold_ = 0;

    };

    void Quantize2BitImpl(mshadow::Stream<mshadow::cpu> *s, const std::vector<mxnet::TBlob> &inputs,
                          const float threshold);
    void Quantize2BitImpl(mshadow::Stream<mshadow::gpu> *s, const std::vector<mxnet::TBlob> &inputs,
                          const float threshold);
    void Dequantize2BitImpl(mshadow::Stream<mshadow::cpu> *s, const std::vector<mxnet::TBlob> &inputs,
                            const float threshold);
    void Dequantize2BitImpl(mshadow::Stream<mshadow::gpu> *s, const std::vector<mxnet::TBlob> &inputs,
                            const float threshold);
  }
}

#endif //MXNET_KVSTORE_GC_H
