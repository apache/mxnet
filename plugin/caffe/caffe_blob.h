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
 * Copyright (c) 2016 by Contributors
 * \file caffe_blob.h
 * \brief conversion between tensor and caffeBlob
 * \author Haoran Wang
*/
#ifndef PLUGIN_CAFFE_CAFFE_BLOB_H_
#define PLUGIN_CAFFE_CAFFE_BLOB_H_

#include <mxnet/tensor_blob.h>
#include <vector>
#include <caffe/blob.hpp>
#include <caffe/layer.hpp>

namespace mxnet {
namespace op {

namespace caffe {

// Declare Memory Type for Caffe blob
enum caffeMemoryTypes {Data, Grad, Non};

TShape Vector2TShape(const std::vector<int> &vec_int);
std::vector<int> TShape2Vector(const TShape &tshape);

// implementation of tensor to blob, called by TensorToBlob
template<typename Device, typename Dtype>
void SetDataGradToBlob(caffeMemoryTypes memType,
                       typename std::vector< ::caffe::Blob<Dtype>*>::iterator blob,
                       typename std::vector<TBlob>::const_iterator itr);

/**
 * \brief The interface to convert mxnet's tensor to caffe's blob
 * \brief called in caffe_operator_inl.h
 */
template<typename Device, typename Dtype>
void TBlob2CaffeBlob(caffeMemoryTypes memType,
                     typename std::vector< ::caffe::Blob<Dtype>*>::iterator blob,
                     typename std::vector<TBlob>::const_iterator tblob,
                     int n = 1) {
  for (int i = 0; i < n; ++i, ++blob, ++tblob) {
    (*blob)->Reshape(TShape2Vector((*tblob).shape_));
    SetDataGradToBlob<Device, Dtype>(memType, blob, tblob);
  }
}

template<typename Dtype>
void SetOpBlobs(::caffe::Layer<Dtype> *caffeOp,
                const std::vector< ::caffe::Blob<Dtype>*>& weights) {
  CHECK_EQ(caffeOp->blobs().size(), weights.size());
  for (int i = 0; i < weights.size(); ++i)
    caffeOp->blobs()[i].reset(weights[i]);
}

/**!
 * \brief Workaround for missing functions in ::caffe::Blob
 * \warning Do not add or override any virtual functions in this class
 * @tparam Dtype
 */
template<class Dtype>
class CaffeBlobFriend : public ::caffe::Blob<Dtype> {
 public:
  inline void set_cpu_diff(Dtype* diff) {
    CHECK(diff);
    this->diff_->set_cpu_data(diff);
  }

  inline void set_gpu_diff(Dtype* diff) {
    CHECK(diff);
    this->diff_->set_gpu_data(diff);
  }
};

#define MXCAFFEBLOB(__object$, __type$) \
  (static_cast<mxnet::op::caffe::CaffeBlobFriend<__type$> *>(__object$))

/**!
 * \brief Workaround for missing functions in ::caffe::Layer
 * \warning Do not add or override any virtual functions in this class
 * @tparam Dtype
 */
template <typename Dtype>
class CaffeLayerFriend : public ::caffe::Layer<Dtype> {
  explicit CaffeLayerFriend(const ::caffe::LayerParameter& param) = delete;
 public:
  inline void SetPhase(::caffe::Phase p) {
    this->phase_ = p;
  }
};

#define MXCAFFELAYER(__object$, __type$) \
  (static_cast<mxnet::op::caffe::CaffeLayerFriend<__type$> *>(__object$))

}  // namespace caffe
}  // namespace op
}  // namespace mxnet

#endif  // PLUGIN_CAFFE_CAFFE_BLOB_H_
