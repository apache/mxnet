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
 *  Copyright (c) 2015 by Contributors
 * \file iter_normalize.h
 * \brief Iterator that subtracts mean and do a few augmentations.
 */
#ifndef MXNET_IO_ITER_NORMALIZE_H_
#define MXNET_IO_ITER_NORMALIZE_H_

#include <mxnet/base.h>
#include <mxnet/io.h>
#include <mxnet/ndarray.h>
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <dmlc/timer.h>
#include <mshadow/tensor.h>
#include <utility>
#include <string>
#include <vector>
#include "../common/utils.h"
#include "./image_iter_common.h"

namespace mxnet {
namespace io {

/*!
 * \brief Iterator that normalize a image.
 *  It also applies a few augmention before normalization.
 */
class ImageNormalizeIter : public IIterator<DataInst> {
 public:
  explicit ImageNormalizeIter(IIterator<DataInst> *base)
      : base_(base), meanfile_ready_(false) {
  }

  virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    param_.InitAllowUnknown(kwargs);
    base_->Init(kwargs);
    rnd_.seed(kRandMagic + param_.seed);
    outimg_.set_pad(false);
    meanimg_.set_pad(false);
    if (param_.mean_img.length() != 0) {
      std::unique_ptr<dmlc::Stream> fi(
          dmlc::Stream::Create(param_.mean_img.c_str(), "r", true));
      if (fi.get() == nullptr) {
        this->CreateMeanImg();
      } else {
        fi.reset(nullptr);
        if (param_.verbose) {
          LOG(INFO) << "Load mean image from " << param_.mean_img;
        }
        // use python compatible ndarray store format
        std::vector<NDArray> data;
        std::vector<std::string> keys;
        {
          std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(param_.mean_img.c_str(), "r"));
          NDArray::Load(fi.get(), &data, &keys);
        }
        CHECK_EQ(data.size(), 1U)
            << "Invalid mean image file format";
        data[0].WaitToRead();
        mshadow::Tensor<cpu, 3> src = data[0].data().get<cpu, 3, real_t>();
        meanimg_.Resize(src.shape_);
        mshadow::Copy(meanimg_, src);
        meanfile_ready_ = true;
      }
    }
  }

  virtual void BeforeFirst(void) {
    base_->BeforeFirst();
  }

  virtual const DataInst& Value(void) const {
    return out_;
  }

  virtual bool Next(void) {
    if (!this->Next_()) return false;
    return true;
  }

 private:
  /*! \brief base iterator */
  std::unique_ptr<IIterator<DataInst> > base_;
  /*! whether mean image is ready */
  bool meanfile_ready_;
  /*! \brief output data */
  DataInst out_;
  // normalize parameter.
  ImageNormalizeParam param_;
  /*! \brief mean image, if needed */
  mshadow::TensorContainer<cpu, 3> meanimg_;
  /*! \brief temp space for output image */
  mshadow::TensorContainer<cpu, 3> outimg_;
  /*! \brief random numeber engine */
  common::RANDOM_ENGINE rnd_;
  // random magic number of this iterator
  static const int kRandMagic = 0;

  /*! \brief internal next function, inlined for fater processing. */
  inline bool Next_(void) {
    if (!base_->Next()) return false;
    const DataInst &src = base_->Value();
    this->SetOutImg(src);
    out_.data.resize(2);
    out_.data[0] = outimg_;
    out_.data[1] = src.data[1];
    out_.index = src.index;
    out_.extra_data = src.extra_data;
    return true;
  }
  /*!
   * \brief Set the output image, after augmentation and normalization.
   * \param src The source image.
   */
  inline void SetOutImg(const DataInst &src) {
    using namespace mshadow::expr;  // NOLINT(*)

    std::uniform_real_distribution<float> rand_uniform(0, 1);
    std::bernoulli_distribution coin_flip(0.5);
    mshadow::Tensor<cpu, 3> data = src.data[0].get<cpu, 3, real_t>();

    outimg_.Resize(data.shape_);
    float contrast =
        rand_uniform(rnd_) * param_.max_random_contrast * 2 - param_.max_random_contrast + 1;
    float illumination =
        rand_uniform(rnd_) * param_.max_random_illumination * 2 - param_.max_random_illumination;
    bool flip = (param_.rand_mirror && coin_flip(rnd_)) || param_.mirror;

    // one-liner channel-wise normalization
    switch (data.shape_[0]) {
      case 4:
        if (meanfile_ready_ && flip) {
          outimg_[3] = mirror((data[3] - meanimg_[3]) * contrast + illumination)
            * param_.scale / param_.std_a;
        } else if (meanfile_ready_ && (!flip)) {
          outimg_[3] = ((data[3] - meanimg_[3]) * contrast + illumination)
            * param_.scale / param_.std_a;
        } else if (!meanfile_ready_ && flip) {
          outimg_[3] = mirror((data[3] - param_.mean_a) * contrast + illumination)
            * param_.scale / param_.std_a;
        } else {
          outimg_[3] = ((data[3] - param_.mean_a) * contrast + illumination)
            * param_.scale / param_.std_a;
        }
      case 3:
        if (meanfile_ready_ && flip) {
          outimg_[2] = mirror((data[2] - meanimg_[2]) * contrast + illumination)
            * param_.scale / param_.std_b;
        } else if (meanfile_ready_ && (!flip)) {
          outimg_[2] = ((data[2] - meanimg_[2]) * contrast + illumination)
            * param_.scale / param_.std_b;
        } else if (!meanfile_ready_ && flip) {
          outimg_[2] = mirror((data[2] - param_.mean_b) * contrast + illumination)
            * param_.scale / param_.std_b;
        } else {
          outimg_[2] = ((data[2] - param_.mean_b) * contrast + illumination)
            * param_.scale / param_.std_b;
        }
      case 2:
        if (meanfile_ready_ && flip) {
          outimg_[1] = mirror((data[1] - meanimg_[1]) * contrast + illumination)
            * param_.scale / param_.std_g;
        } else if (meanfile_ready_ && (!flip)) {
          outimg_[1] = ((data[1] - meanimg_[1]) * contrast + illumination)
            * param_.scale / param_.std_g;
        } else if (!meanfile_ready_ && flip) {
          outimg_[1] = mirror((data[1] - param_.mean_g) * contrast + illumination)
            * param_.scale / param_.std_g;
        } else {
          outimg_[1] = ((data[1] - param_.mean_g) * contrast + illumination)
            * param_.scale / param_.std_g;
        }
      case 1:
        if (meanfile_ready_ && flip) {
          outimg_[0] = mirror((data[0] - meanimg_[0]) * contrast + illumination)
            * param_.scale / param_.std_r;
        } else if (meanfile_ready_ && (!flip)) {
          outimg_[0] = ((data[0] - meanimg_[0]) * contrast + illumination)
            * param_.scale / param_.std_r;
        } else if (!meanfile_ready_ && flip) {
          outimg_[0] = mirror((data[0] - param_.mean_r) * contrast + illumination)
            * param_.scale / param_.std_r;
        } else {
          outimg_[0] = ((data[0] - param_.mean_r) * contrast + illumination)
            * param_.scale / param_.std_r;
        }
        break;
      default:
        LOG(FATAL) << "Expected image channels range 1-4, got " << data.shape_[0];
    }
  }

  // creat mean image.
  inline void CreateMeanImg(void) {
    if (param_.verbose) {
      LOG(INFO) << "Cannot find " << param_.mean_img
                << ": create mean image, this will take some time...";
    }
    double start = dmlc::GetTime();
    size_t imcnt = 1;  // NOLINT(*)
    CHECK(this->Next_()) << "input iterator failed.";
    meanimg_.Resize(outimg_.shape_);
    mshadow::Copy(meanimg_, outimg_);
    while (this->Next_()) {
      meanimg_ += outimg_;
      imcnt += 1;
      double elapsed = dmlc::GetTime() - start;
      if (imcnt % 10000L == 0 && param_.verbose) {
        LOG(INFO) << imcnt << " images processed, " << elapsed << " sec elapsed";
      }
    }
    meanimg_ *= (1.0f / imcnt);
    // save as mxnet python compatible format.
    TBlob tmp = meanimg_;
    {
      std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(param_.mean_img.c_str(), "w"));
      NDArray::Save(fo.get(),
                    {NDArray(tmp, 0)},
                    {"mean_img"});
    }
    if (param_.verbose) {
      LOG(INFO) << "Save mean image to " << param_.mean_img << "..";
    }
    meanfile_ready_ = true;
    this->BeforeFirst();
  }
};

/*!
 * \brief Iterator that normalize a image.
 *  It also applies a few augmention before normalization.
 */
class ImageDetNormalizeIter : public IIterator<DataInst> {
 public:
  explicit ImageDetNormalizeIter(IIterator<DataInst> *base)
      : base_(base), meanfile_ready_(false) {
  }

  virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    param_.InitAllowUnknown(kwargs);
    base_->Init(kwargs);
    rnd_.seed(kRandMagic + param_.seed);
    outimg_.set_pad(false);
    meanimg_.set_pad(false);
    if (param_.mean_img.length() != 0) {
      std::unique_ptr<dmlc::Stream> fi(
          dmlc::Stream::Create(param_.mean_img.c_str(), "r", true));
      if (fi.get() == nullptr) {
        this->CreateMeanImg();
      } else {
        fi.reset(nullptr);
        if (param_.verbose) {
          LOG(INFO) << "Load mean image from " << param_.mean_img;
        }
        // use python compatible ndarray store format
        std::vector<NDArray> data;
        std::vector<std::string> keys;
        {
          std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(param_.mean_img.c_str(), "r"));
          NDArray::Load(fi.get(), &data, &keys);
        }
        CHECK_EQ(data.size(), 1)
            << "Invalid mean image file format";
        data[0].WaitToRead();
        mshadow::Tensor<cpu, 3> src = data[0].data().get<cpu, 3, real_t>();
        meanimg_.Resize(src.shape_);
        mshadow::Copy(meanimg_, src);
        meanfile_ready_ = true;
      }
    }
  }

  virtual void BeforeFirst(void) {
    base_->BeforeFirst();
  }

  virtual const DataInst& Value(void) const {
    return out_;
  }

  virtual bool Next(void) {
    if (!this->Next_()) return false;
    return true;
  }

 private:
  /*! \brief base iterator */
  std::unique_ptr<IIterator<DataInst> > base_;
  // whether mean image is ready.
  bool meanfile_ready_;
  /*! \brief output data */
  DataInst out_;
  // normalize parameter.
  ImageDetNormalizeParam param_;
  /*! \brief mean image, if needed */
  mshadow::TensorContainer<cpu, 3> meanimg_;
  /*! \brief temp space for output image */
  mshadow::TensorContainer<cpu, 3> outimg_;
  /*! \brief random numeber engine */
  common::RANDOM_ENGINE rnd_;
  // random magic number of this iterator
  static const int kRandMagic = 0;

  /*! \brief internal next function, inlined for fater processing. */
  inline bool Next_(void) {
    if (!base_->Next()) return false;
    const DataInst &src = base_->Value();
    this->SetOutImg(src);
    out_.data.resize(2);
    out_.data[0] = outimg_;
    out_.data[1] = src.data[1];
    out_.index = src.index;
    out_.extra_data = src.extra_data;
    return true;
  }
  /*!
   * \brief Set the output image, after augmentation and normalization.
   * \param src The source image.
   */
  inline void SetOutImg(const DataInst &src) {
    using namespace mshadow::expr;  // NOLINT(*)
    mshadow::Tensor<cpu, 3> data = src.data[0].get<cpu, 3, real_t>();

    outimg_.Resize(data.shape_);

    if (param_.mean_r > 0.0f || param_.mean_g > 0.0f ||
        param_.mean_b > 0.0f || param_.mean_a > 0.0f) {
      // subtract mean per channel
      data[0] -= param_.mean_r;
      if (data.shape_[0] >= 3) {
        data[1] -= param_.mean_g;
        data[2] -= param_.mean_b;
      }
      if (data.shape_[0] == 4) {
        data[3] -= param_.mean_a;
      }
    } else if (!meanfile_ready_ || param_.mean_img.length() == 0) {
      // do not subtract anything
    } else {
      CHECK(meanfile_ready_);
      data -= meanimg_;
    }

    // std
    if (param_.std_r > 0.0f) {
      data[0] /= param_.std_r;
    }
    if (data.shape_[0] >= 3 && param_.std_g > 0.0f) {
      data[1] /= param_.std_g;
    }
    if (data.shape_[0] >= 3 && param_.std_b > 0.0f) {
      data[2] /= param_.std_b;
    }
    if (data.shape_[0] == 4 && param_.std_a > 0.0f) {
      data[3] /= param_.std_a;
    }
    outimg_ = data * param_.scale;
  }

  // creat mean image.
  inline void CreateMeanImg(void) {
    if (param_.verbose) {
      LOG(INFO) << "Cannot find " << param_.mean_img
                << ": create mean image, this will take some time...";
    }
    double start = dmlc::GetTime();
    size_t imcnt = 1;  // NOLINT(*)
    CHECK(this->Next_()) << "input iterator failed.";
    meanimg_.Resize(outimg_.shape_);
    mshadow::Copy(meanimg_, outimg_);
    while (this->Next_()) {
      meanimg_ += outimg_;
      imcnt += 1;
      double elapsed = dmlc::GetTime() - start;
      if (imcnt % 10000L == 0 && param_.verbose) {
        LOG(INFO) << imcnt << " images processed, " << elapsed << " sec elapsed";
      }
    }
    meanimg_ *= (1.0f / imcnt);
    // save as mxnet python compatible format.
    TBlob tmp = meanimg_;
    {
      std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(param_.mean_img.c_str(), "w"));
      NDArray::Save(fo.get(),
                    {NDArray(tmp, 0)},
                    {"mean_img"});
    }
    if (param_.verbose) {
      LOG(INFO) << "Save mean image to " << param_.mean_img << "..";
    }
    meanfile_ready_ = true;
    this->BeforeFirst();
  }
};
}  // namespace io
}  // namespace mxnet
#endif  // MXNET_IO_ITER_NORMALIZE_H_
