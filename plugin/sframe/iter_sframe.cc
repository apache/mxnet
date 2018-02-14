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
 * Copyright (c) 2015 by Contributors
 * \file iter_sframe_image.cc
 * \brief
 * \author Bing Xu
*/

#include <mxnet/io.h>
#include <dmlc/base.h>
#include <dmlc/io.h>
#include <dmlc/omp.h>
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <string>
#include <memory>
#include <unity/lib/image_util.hpp>
#include <unity/lib/gl_sframe.hpp>
#include <unity/lib/gl_sarray.hpp>
#include "../../src/io/inst_vector.h"
#include "../../src/io/image_recordio.h"
#include "../../src/io/image_augmenter.h"
#include "../../src/io/iter_prefetcher.h"
#include "../../src/io/iter_normalize.h"
#include "../../src/io/iter_batchloader.h"

namespace mxnet {
namespace io {

struct SFrameParam : public dmlc::Parameter<SFrameParam> {
  /*! \brief sframe path */
  std::string path_sframe;
  std::string data_field;
  std::string label_field;
  TShape data_shape;
  TShape label_shape;
  DMLC_DECLARE_PARAMETER(SFrameParam) {
    DMLC_DECLARE_FIELD(path_sframe).set_default("")
    .describe("Dataset Param: path to image dataset sframe");
    DMLC_DECLARE_FIELD(data_field).set_default("data")
    .describe("Dataset Param: data column in sframe");
    DMLC_DECLARE_FIELD(label_field).set_default("label")
    .describe("Dataset Param: label column in sframe");
    DMLC_DECLARE_FIELD(data_shape)
    .describe("Dataset Param: input data instance shape");
    DMLC_DECLARE_FIELD(label_shape)
    .describe("Dataset Param: input label instance shape");
  }
};  // struct SFrameImageParam

class SFrameIterBase : public IIterator<DataInst> {
 public:
  SFrameIterBase() {}

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.InitAllowUnknown(kwargs);
    sframe_ = graphlab::gl_sframe(param_.path_sframe)[{param_.data_field, param_.label_field}];
    range_it_.reset(new graphlab::gl_sframe_range(sframe_.range_iterator()));
    this->BeforeFirst();
  }

  virtual ~SFrameIterBase() {}

  virtual void BeforeFirst() {
    idx_ = 0;
    *range_it_ = sframe_.range_iterator();
    current_it_ = range_it_->begin();
  }

  virtual const DataInst &Value(void) const {
    return out_;
  }

  virtual bool Next() = 0;

 protected:
  /*! \brief index of instance */
  index_t idx_;
  /*! \brief output of sframe iterator */
  DataInst out_;
  /*! \brief temp space */
  InstVector tmp_;
  /*! \brief sframe iter parameter */
  SFrameParam param_;
  /*! \brief sframe object*/
  graphlab::gl_sframe sframe_;
  /*! \brief sframe range iterator */
  std::unique_ptr<graphlab::gl_sframe_range> range_it_;
  /*! \brief current iterator in range iterator */
  graphlab::gl_sframe_range::iterator current_it_;

 protected:
  /*! \brief copy data */
  template<int dim>
  void Copy_(mshadow::Tensor<cpu, dim> tensor, const graphlab::flex_vec &vec) {
    CHECK_EQ(tensor.shape_.Size(), vec.size());
    CHECK_EQ(tensor.CheckContiguous(), true);
    mshadow::Tensor<cpu, 1> flatten(tensor.dptr_, mshadow::Shape1(tensor.shape_.Size()));
    for (index_t i = 0; i < vec.size(); ++i) {
      flatten[i] = static_cast<float>(vec[i]);
    }
  }
};  // class SFrameIterBase

class SFrameImageIter : public SFrameIterBase {
 public:
  SFrameImageIter() :
    augmenter_(new ImageAugmenter()), prnd_(new common::RANDOM_ENGINE(8964)) {}

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    Parent::Init(kwargs);
    augmenter_->Init(kwargs);
    CHECK_EQ(Parent::param_.data_shape.ndim(), 3)
      << "Image shpae must be (channel, height, width)";
  }

  bool Next(void) override {
    if (Parent::current_it_ == Parent::range_it_->end()) {
      return false;
    }
    graphlab::image_type gl_img = (*Parent::current_it_)[0];
    graphlab::flex_vec gl_label = (*Parent::current_it_)[1];
    // TODO(bing): check not decoded
    // TODO(bing): check img shape
    CHECK_EQ(gl_label.size(), Parent::param_.label_shape.Size()) << "Label shape does not match";
    const unsigned char *raw_data = gl_img.get_image_data();
    cv::Mat res;
    cv::Mat buf(1, gl_img.m_image_data_size, CV_8U, const_cast<unsigned char*>(raw_data));
    res = cv::imdecode(buf, -1);
    res = augmenter_->Process(res, prnd_.get());
    const int n_channels = res.channels();
    if (!tmp_.Size()) {
      tmp_.Push(Parent::idx_++,
                Parent::param_.data_shape.get<3>(),
                Parent::param_.label_shape.get<1>());
    }
    mshadow::Tensor<cpu, 3> data = Parent::tmp_.data().Back();
    std::vector<int> swap_indices;
    if (n_channels == 1) swap_indices = {0};
    if (n_channels == 3) swap_indices = {2, 1, 0};
    for (int i = 0; i < res.rows; ++i) {
      uchar* im_data = res.ptr<uchar>(i);
      for (int j = 0; j < res.cols; ++j) {
        for (int k = 0; k < n_channels; ++k) {
          data[k][i][j] = im_data[swap_indices[k]];
        }
        im_data += n_channels;
      }
    }
    mshadow::Tensor<cpu, 1> label = Parent::tmp_.label().Back();
    Parent::Copy_<1>(label, gl_label);
    res.release();
    out_ = Parent::tmp_[0];
    ++current_it_;
    return true;
  }

 private:
  /*! \brief parent type */
  typedef SFrameIterBase Parent;
  /*! \brief image augmenter */
  std::unique_ptr<ImageAugmenter> augmenter_;
  /*! \brief randim generator*/
  std::unique_ptr<common::RANDOM_ENGINE> prnd_;
};  // class SFrameImageIter

class SFrameDataIter : public SFrameIterBase {
 public:
  bool Next() override {
    if (Parent::current_it_ == Parent::range_it_->end()) {
      return false;
    }
    graphlab::flex_vec gl_data = (*Parent::current_it_)[0];
    graphlab::flex_vec gl_label = (*Parent::current_it_)[1];
    CHECK_EQ(gl_data.size(), Parent::param_.data_shape.Size()) << "Data shape does not match";
    CHECK_EQ(gl_label.size(), Parent::param_.label_shape.Size()) << "Label shape does not match";
    if (!Parent::tmp_.Size()) {
        Parent::tmp_.Push(Parent::idx_++,
                  Parent::param_.data_shape.get<3>(),
                  Parent::param_.label_shape.get<1>());
    }
    mshadow::Tensor<cpu, 3> data = Parent::tmp_.data().Back();
    Parent::Copy_<3>(data, gl_data);
    mshadow::Tensor<cpu, 1> label = Parent::tmp_.label().Back();
    Parent::Copy_<1>(label, gl_label);
    out_ = Parent::tmp_[0];
    ++current_it_;
    return true;
  }

 private:
  /*! \brief parent type */
  typedef SFrameIterBase Parent;
};  // class SFrameDataIter

DMLC_REGISTER_PARAMETER(SFrameParam);

MXNET_REGISTER_IO_ITER(SFrameImageIter)
.describe("Naive SFrame image iterator prototype")
.add_arguments(SFrameParam::__FIELDS__())
.add_arguments(BatchParam::__FIELDS__())
.add_arguments(PrefetcherParam::__FIELDS__())
.add_arguments(ImageAugmentParam::__FIELDS__())
.add_arguments(ImageNormalizeParam::__FIELDS__())
.set_body([]() {
    return new PrefetcherIter(
        new BatchLoader(
            new ImageNormalizeIter(
              new SFrameImageIter())));
    });

MXNET_REGISTER_IO_ITER(SFrameDataIter)
.describe("Naive SFrame data iterator prototype")
.add_arguments(SFrameParam::__FIELDS__())
.add_arguments(BatchParam::__FIELDS__())
.add_arguments(PrefetcherParam::__FIELDS__())
.set_body([]() {
    return new PrefetcherIter(
        new BatchLoader(
              new SFrameDataIter()));
    });


}  // namespace io
}  // namespace mxnet

