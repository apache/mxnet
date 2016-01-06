/*!
 *  Copyright (c) 2015 by Contributors
 * \file iter_normalize.h
 * \brief Iterator that substracts mean and do a few augmentations.
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

namespace mxnet {
namespace io {

// normalize parameters
struct ImageNormalizeParam :  public dmlc::Parameter<ImageNormalizeParam> {
  /*! \brief random seed */
  int seed;
  /*! \brief whether to mirror the image */
  bool mirror;
  /*! \brief whether to perform rand mirror the image */
  bool rand_mirror;
  /*! \brief mean file string */
  std::string mean_img;
  /*! \brief mean value for r channel */
  float mean_r;
  /*! \brief mean value for g channel */
  float mean_g;
  /*! \brief mean value for b channel */
  float mean_b;
  /*! \brief mean value for alpha channel */
  float mean_a;
  /*! \brief scale on color space */
  float scale;
  /*! \brief maximum ratio of contrast variation */
  float max_random_contrast;
  /*! \brief maximum value of illumination variation */
  float max_random_illumination;
  /*! \brief silent */
  bool verbose;
  // declare parameters
  DMLC_DECLARE_PARAMETER(ImageNormalizeParam) {
    DMLC_DECLARE_FIELD(seed).set_default(0)
        .describe("Augmentation Param: Random Seed.");
    DMLC_DECLARE_FIELD(mirror).set_default(false)
        .describe("Augmentation Param: Whether to mirror the image.");
    DMLC_DECLARE_FIELD(rand_mirror).set_default(false)
        .describe("Augmentation Param: Whether to mirror the image randomly.");
    DMLC_DECLARE_FIELD(mean_img).set_default("")
        .describe("Augmentation Param: Mean Image to be subtracted.");
    DMLC_DECLARE_FIELD(mean_r).set_default(0.0f)
        .describe("Augmentation Param: Mean value on R channel.");
    DMLC_DECLARE_FIELD(mean_g).set_default(0.0f)
        .describe("Augmentation Param: Mean value on G channel.");
    DMLC_DECLARE_FIELD(mean_b).set_default(0.0f)
        .describe("Augmentation Param: Mean value on B channel.");
    DMLC_DECLARE_FIELD(mean_a).set_default(0.0f)
        .describe("Augmentation Param: Mean value on Alpha channel.");
    DMLC_DECLARE_FIELD(scale).set_default(1.0f)
        .describe("Augmentation Param: Scale in color space.");
    DMLC_DECLARE_FIELD(max_random_contrast).set_default(0.0f)
        .describe("Augmentation Param: Maximum ratio of contrast variation.");
    DMLC_DECLARE_FIELD(max_random_illumination).set_default(0.0f)
        .describe("Augmentation Param: Maximum value of illumination variation.");
    DMLC_DECLARE_FIELD(verbose).set_default(true)
        .describe("Augmentation Param: Whether to print augmentor info.");
  }
};

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

    if (param_.mean_r > 0.0f || param_.mean_g > 0.0f ||
        param_.mean_b > 0.0f || param_.mean_a > 0.0f) {
      // substract mean per channel
      data[0] -= param_.mean_r;
      if (data.shape_[0] >= 3) {
        data[1] -= param_.mean_g;
        data[2] -= param_.mean_b;
      }
      if (data.shape_[0] == 4) {
        data[3] -= param_.mean_a;
      }
      if ((param_.rand_mirror && coin_flip(rnd_)) || param_.mirror) {
        outimg_ = mirror(data * contrast + illumination) * param_.scale;
      } else {
        outimg_ = (data * contrast + illumination) * param_.scale;
      }
    } else if (!meanfile_ready_ || param_.mean_img.length() == 0) {
      // do not substract anything
      if ((param_.rand_mirror && coin_flip(rnd_)) || param_.mirror) {
        outimg_ = mirror(data) * param_.scale;
      } else {
        outimg_ = F<mshadow::op::identity>(data) * param_.scale;
      }
    } else {
      CHECK(meanfile_ready_);
      if ((param_.rand_mirror && coin_flip(rnd_)) || param_.mirror) {
        outimg_ = mirror((data - meanimg_) * contrast + illumination) * param_.scale;
      } else {
        outimg_ = ((data - meanimg_) * contrast + illumination) * param_.scale;
      }
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
}  // namespace io
}  // namespace mxnet
#endif  // MXNET_IO_ITER_NORMALIZE_H_
