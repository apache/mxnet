/*!
 * Copyright (c) 2015 by Contributors
 * \file iter_mnist.cc
 * \brief register mnist iterator
*/
#include <mxnet/io.h>
#include <mxnet/base.h>
#include <dmlc/io.h>
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <string>
#include <vector>
#include <utility>
#include <map>
#include "./iter_prefetcher.h"
#include "../common/utils.h"

namespace mxnet {
namespace io {
// Define mnist io parameters
struct MNISTParam : public dmlc::Parameter<MNISTParam> {
  /*! \brief path */
  std::string image, label;
  /*! \brief whether to do shuffle */
  bool shuffle;
  /*! \brief whether to print info */
  bool silent;
  /*! \brief batch size */
  int batch_size;
  /*! \brief data mode */
  bool flat;
  /*! \brief random seed */
  int seed;
  /*! \brief partition the data into multiple parts */
  int num_parts;
  /*! \brief the index of the part will read*/
  int part_index;
  // declare parameters
  DMLC_DECLARE_PARAMETER(MNISTParam) {
    DMLC_DECLARE_FIELD(image).set_default("./train-images-idx3-ubyte")
        .describe("Dataset Param: Mnist image path.");
    DMLC_DECLARE_FIELD(label).set_default("./train-labels-idx1-ubyte")
        .describe("Dataset Param: Mnist label path.");
    DMLC_DECLARE_FIELD(batch_size).set_lower_bound(1).set_default(128)
        .describe("Batch Param: Batch Size.");
    DMLC_DECLARE_FIELD(shuffle).set_default(true)
        .describe("Augmentation Param: Whether to shuffle data.");
    DMLC_DECLARE_FIELD(flat).set_default(false)
        .describe("Augmentation Param: Whether to flat the data into 1D.");
    DMLC_DECLARE_FIELD(seed).set_default(0)
        .describe("Augmentation Param: Random Seed.");
    DMLC_DECLARE_FIELD(silent).set_default(false)
        .describe("Auxiliary Param: Whether to print out data info.");
    DMLC_DECLARE_FIELD(num_parts).set_default(1)
        .describe("partition the data into multiple parts");
    DMLC_DECLARE_FIELD(part_index).set_default(0)
        .describe("the index of the part will read");
  }
};

class MNISTIter: public IIterator<TBlobBatch> {
 public:
  MNISTIter(void) : loc_(0), inst_offset_(0) {
    img_.dptr_ = NULL;
    out_.data.resize(2);
  }
  virtual ~MNISTIter(void) {
    if (img_.dptr_ != NULL) delete []img_.dptr_;
  }
  // intialize iterator loads data in
  virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    std::map<std::string, std::string> kmap(kwargs.begin(), kwargs.end());
    param_.InitAllowUnknown(kmap);
    this->LoadImage();
    this->LoadLabel();
    if (param_.flat) {
      batch_data_.shape_ = mshadow::Shape4(param_.batch_size, 1, 1, img_.size(1) * img_.size(2));
    } else {
      batch_data_.shape_ = mshadow::Shape4(param_.batch_size, 1, img_.size(1), img_.size(2));
    }
    out_.data.clear();
    batch_label_.shape_ = mshadow::Shape2(param_.batch_size, 1);
    batch_label_.stride_ = 1;
    batch_data_.stride_ = batch_data_.size(3);
    out_.batch_size = param_.batch_size;
    if (param_.shuffle) this->Shuffle();
    if (param_.silent == 0) {
      mshadow::TShape s;
      s = batch_data_.shape_;
      if (param_.flat) {
        LOG(INFO) << "MNISTIter: load " << (unsigned)img_.size(0) << " images, shuffle="
            << param_.shuffle << ", shape=" << s.FlatTo2D();
      } else {
        LOG(INFO) << "MNISTIter: load " << (unsigned)img_.size(0) << " images, shuffle="
            << param_.shuffle << ", shape=" << s;
      }
    }
  }
  virtual void BeforeFirst(void) {
    this->loc_ = 0;
  }
  virtual bool Next(void) {
    if (loc_ + param_.batch_size <= img_.size(0)) {
      batch_data_.dptr_ = img_[loc_].dptr_;
      batch_label_.dptr_ = &labels_[loc_];
      out_.data.clear();
      if (param_.flat) {
          out_.data.push_back(TBlob(batch_data_.FlatTo2D()));
      } else {
          out_.data.push_back(TBlob(batch_data_));
      }
      out_.data.push_back(TBlob(batch_label_));
      loc_ += param_.batch_size;
      return true;
    } else {
      return false;
    }
  }
  virtual const TBlobBatch &Value(void) const {
    return out_;
  }

 private:
  inline void GetPart(int count, int* start, int *end) {
    CHECK_GE(param_.part_index, 0);
    CHECK_GT(param_.num_parts, 0);
    CHECK_GT(param_.num_parts, param_.part_index);

    *start = static_cast<int>(
        static_cast<double>(count) / param_.num_parts * param_.part_index);
    *end = static_cast<int>(
        static_cast<double>(count) / param_.num_parts * (param_.part_index+1));
  }

  inline void LoadImage(void) {
    dmlc::SeekStream* stdimg
        = dmlc::SeekStream::CreateForRead(param_.image.c_str());
    ReadInt(stdimg);
    int image_count = ReadInt(stdimg);
    int image_rows  = ReadInt(stdimg);
    int image_cols  = ReadInt(stdimg);

    int start, end;
    GetPart(image_count, &start, &end);
    image_count = end - start;
    if (start > 0) {
      stdimg->Seek(stdimg->Tell() + start * image_rows * image_cols);
    }

    img_.shape_ = mshadow::Shape3(image_count, image_rows, image_cols);
    img_.stride_ = img_.size(2);

    // allocate continuous memory
    img_.dptr_ = new float[img_.MSize()];
    for (int i = 0; i < image_count; ++i) {
      for (int j = 0; j < image_rows; ++j) {
        for (int k = 0; k < image_cols; ++k) {
          unsigned char ch;
          CHECK(stdimg->Read(&ch, sizeof(ch) != 0));
          img_[i][j][k] = ch;
        }
      }
    }
    // normalize to 0-1
    img_ *= 1.0f / 256.0f;
    delete stdimg;
  }
  inline void LoadLabel(void) {
    dmlc::SeekStream* stdlabel
        = dmlc::SeekStream::CreateForRead(param_.label.c_str());
    ReadInt(stdlabel);
    int labels_count = ReadInt(stdlabel);

    int start, end;
    GetPart(labels_count, &start, &end);
    labels_count = end - start;
    if (start > 0) {
      stdlabel->Seek(stdlabel->Tell() + start);
    }

    labels_.resize(labels_count);
    for (int i = 0; i < labels_count; ++i) {
      unsigned char ch;
      CHECK(stdlabel->Read(&ch, sizeof(ch) != 0));
      labels_[i] = ch;
      inst_.push_back((unsigned)i + inst_offset_);
    }
    delete stdlabel;
  }
  inline void Shuffle(void) {
    std::shuffle(inst_.begin(), inst_.end(), common::RANDOM_ENGINE(kRandMagic + param_.seed));
    std::vector<float> tmplabel(labels_.size());
    mshadow::TensorContainer<cpu, 3> tmpimg(img_.shape_);
    for (size_t i = 0; i < inst_.size(); ++i) {
      unsigned ridx = inst_[i] - inst_offset_;
      mshadow::Copy(tmpimg[i], img_[ridx]);
      tmplabel[i] = labels_[ridx];
    }
    // copy back
    mshadow::Copy(img_, tmpimg);
    labels_ = tmplabel;
  }

 private:
  inline static int ReadInt(dmlc::Stream *fi) {
    unsigned char buf[4];
    CHECK(fi->Read(buf, sizeof(buf)) == sizeof(buf))
        << "invalid mnist format";
#ifdef _MSC_VER
    return (buf[0] << 24 | buf[1] << 16 | buf[2] << 8 | buf[3]);
#else
    return reinterpret_cast<int>(buf[0] << 24 | buf[1] << 16 | buf[2] << 8 | buf[3]);
#endif
  }

 private:
  /*! \brief MNIST iter params */
  MNISTParam param_;
  /*! \brief output */
  TBlobBatch out_;
  /*! \brief current location */
  index_t loc_;
  /*! \brief image content */
  mshadow::Tensor<cpu, 3> img_;
  /*! \brief label content */
  std::vector<float> labels_;
  /*! \brief batch data tensor */
  mshadow::Tensor<cpu, 4> batch_data_;
  /*! \brief batch label tensor  */
  mshadow::Tensor<cpu, 2> batch_label_;
  /*! \brief instance index offset */
  unsigned inst_offset_;
  /*! \brief instance index */
  std::vector<unsigned> inst_;
  // magic number to setup randomness
  static const int kRandMagic = 0;
};  // class MNISTIter

DMLC_REGISTER_PARAMETER(MNISTParam);

MXNET_REGISTER_IO_ITER(MNISTIter)
.describe(R"code(Iterating on the MNIST dataset.

One can download the dataset from http://yann.lecun.com/exdb/mnist/

)code" ADD_FILELINE)
.add_arguments(MNISTParam::__FIELDS__())
.add_arguments(PrefetcherParam::__FIELDS__())
.set_body([]() {
    return new PrefetcherIter(new MNISTIter());
  });

}  // namespace io
}  // namespace mxnet
