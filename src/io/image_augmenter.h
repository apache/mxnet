/*!
 * \file image_augmenter_opencv.hpp
 * \brief threaded version of page iterator
 * \author Naiyan Wang, Tianqi Chen
 */
#ifndef MXNET_IO_IMAGE_AUGMENTER_H_
#define MXNET_IO_IMAGE_AUGMENTER_H_

#include <opencv2/opencv.hpp>
#include "../common/utils.h"

namespace mxnet {
namespace io {
/*! \brief image augmentation parameters*/
struct ImageAugmentParam : public dmlc::Parameter<ImageAugmentParam> {
  /*! \brief whether we do random cropping */
  bool rand_crop_;
  /*! \brief whether we do nonrandom croping */
  int crop_y_start_;
  /*! \brief whether we do nonrandom croping */
  int crop_x_start_;
  /*! \brief [-max_rotate_angle, max_rotate_angle] */
  int max_rotate_angle_;
  /*! \brief max aspect ratio */
  float max_aspect_ratio_;
  /*! \brief random shear the image [-max_shear_ratio, max_shear_ratio] */
  float max_shear_ratio_;
  /*! \brief max crop size */
  int max_crop_size_;
  /*! \brief min crop size */
  int min_crop_size_;
  /*! \brief max scale ratio */
  float max_random_scale_;
  /*! \brief min scale_ratio */
  float min_random_scale_;
  /*! \brief min image size */
  float min_img_size_;
  /*! \brief max image size */
  float max_img_size_;
  /*! \brief whether to mirror the image */
  bool mirror_;
  /*! \brief rotate angle */
  int rotate_;
  /*! \brief filled color while padding */
  int fill_value_;
  // declare parameters
  // TODO: didn't understand the range for some params
  DMLC_DECLARE_PARAMETER(ImageAugmentParam) {
    DMLC_DECLARE_FIELD(rand_crop_).set_default(true)
        .describe("Whether we de random cropping");
    DMLC_DECLARE_FIELD(crop_y_start_).set_default(-1)
        .describe("Where to nonrandom crop on y");
    DMLC_DECLARE_FIELD(crop_x_start_).set_default(-1)
        .describe("Where to nonrandom crop on x");
    DMLC_DECLARE_FIELD(max_rotate_angle_).set_default(0.0f)
        .describe("Rotate can be [-max_rotate_angle, max_rotate_angle]");
    DMLC_DECLARE_FIELD(max_aspect_ratio_).set_default(0.0f)
        .describe("Max aspect ratio");
    DMLC_DECLARE_FIELD(max_shear_ratio_).set_default(0.0f)
        .describe("Shear rotate can be made between [-max_shear_ratio_, max_shear_ratio_]");
    DMLC_DECLARE_FIELD(max_crop_size_).set_default(-1)
        .describe("Maximum crop size");
    DMLC_DECLARE_FIELD(min_crop_size_).set_default(-1)
        .describe("Minimum crop size");
    DMLC_DECLARE_FIELD(max_random_scale_).set_default(1.0f)
        .describe("Maxmum scale ratio");
    DMLC_DECLARE_FIELD(min_random_scale_).set_default(1.0f)
        .describe("Minimum scale ratio");       
    DMLC_DECLARE_FIELD(max_img_size_).set_default(1e10f)
        .describe("Maxmum image size");
    DMLC_DECLARE_FIELD(min_img_size_).set_default(0.0f)
        .describe("Minimum image size");
    DMLC_DECLARE_FIELD(mirror_).set_default(false)
        .describe("Whether to mirror the image");
    DMLC_DECLARE_FIELD(rotate_).set_default(-1.0f)
        .describe("Rotate angle");
    DMLC_DECLARE_FIELD(fill_value_).set_default(255)
        .describe("Filled value while padding");
  }
};

/*! \brief helper class to do image augmentation */
class ImageAugmenter {
 public:
  // contructor
  ImageAugmenter(void)
      : tmpres(false),
        rotateM(2, 3, CV_32F) {
  }
  virtual ~ImageAugmenter() {
  }
  // TODO: Hack the shape and rotate list, didn't use param
  virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    std::vector<std::pair<std::string, std::string> > kwargs_left;
    kwargs_left = param_.InitAllowUnknown(kwargs);
    for (size_t i = 0; i < kwargs_left.size(); i++) {
        if (!strcmp(kwargs_left[i].first.c_str(), "input_shape")) {
          CHECK(sscanf(kwargs_left[i].second.c_str(), "%u,%u,%u", &shape_[0], &shape_[1], &shape_[2]) == 3)
                       << "input_shape must be three consecutive integers without space example: 1,1,200 ";
        }
        if (!strcmp(kwargs_left[i].first.c_str(), "rotate_list")) {
          const char* val = kwargs_left[i].second.c_str();
          const char *end = val + strlen(val);
          char buf[128];
          while (val < end) {
            sscanf(val, "%[^,]", buf);
            val += strlen(buf) + 1;
            rotate_list_.push_back(atoi(buf));
          }
        }
    }
  }
  /*!
   * \brief augment src image, store result into dst
   *   this function is not thread safe, and will only be called by one thread
   *   however, it will tries to re-use memory space as much as possible
   * \param src the source image
   * \param source of random number
   * \param dst the pointer to the place where we want to store the result
   */
  virtual cv::Mat Process(const cv::Mat &src,
                          common::RANDOM_ENGINE *prnd) {
    // shear
    float s = NextDouble(prnd) * param_.max_shear_ratio_ * 2 - param_.max_shear_ratio_;
    // rotate
    int angle = NextUInt32(param_.max_rotate_angle_ * 2, prnd) - param_.max_rotate_angle_;
    if (param_.rotate_ > 0) angle = param_.rotate_;
    if (rotate_list_.size() > 0) {
      angle = rotate_list_[NextUInt32(rotate_list_.size() - 1, prnd)];
    }
    float a = cos(angle / 180.0 * M_PI);
    float b = sin(angle / 180.0 * M_PI);
    // scale
    float scale = NextDouble(prnd) * (param_.max_random_scale_ - param_.min_random_scale_) + param_.min_random_scale_;
    // aspect ratio
    float ratio = NextDouble(prnd) * param_.max_aspect_ratio_ * 2 - param_.max_aspect_ratio_ + 1;
    float hs = 2 * scale / (1 + ratio);
    float ws = ratio * hs;
    // new width and height
    float new_width = std::max(param_.min_img_size_, std::min(param_.max_img_size_, scale * src.cols));
    float new_height = std::max(param_.min_img_size_, std::min(param_.max_img_size_, scale * src.rows));
    //printf("%f %f %f %f %f %f %f %f %f\n", s, a, b, scale, ratio, hs, ws, new_width, new_height);
    cv::Mat M(2, 3, CV_32F);
    M.at<float>(0, 0) = hs * a - s * b * ws;
    M.at<float>(1, 0) = -b * ws;
    M.at<float>(0, 1) = hs * b + s * a * ws;
    M.at<float>(1, 1) = a * ws;
    float ori_center_width = M.at<float>(0, 0) * src.cols + M.at<float>(0, 1) * src.rows;
    float ori_center_height = M.at<float>(1, 0) * src.cols + M.at<float>(1, 1) * src.rows;
    M.at<float>(0, 2) = (new_width - ori_center_width) / 2;
    M.at<float>(1, 2) = (new_height - ori_center_height) / 2;
    cv::warpAffine(src, temp, M, cv::Size(new_width, new_height),
                     cv::INTER_LINEAR,
                     cv::BORDER_CONSTANT,
                     cv::Scalar(param_.fill_value_, param_.fill_value_, param_.fill_value_));
    cv::Mat res = temp;
    if (param_.max_crop_size_ != -1 || param_.min_crop_size_ != -1){
      CHECK(res.cols >= param_.max_crop_size_ && res.rows >= param_.max_crop_size_&& param_.max_crop_size_ >= param_.min_crop_size_)
          << "input image size smaller than max_crop_size";
      mshadow::index_t rand_crop_size = NextUInt32(param_.max_crop_size_- param_.min_crop_size_+1, prnd)+ param_.min_crop_size_;
      mshadow::index_t y = res.rows - rand_crop_size;
      mshadow::index_t x = res.cols - rand_crop_size;
      if (param_.rand_crop_ != 0) {
        y = NextUInt32(y + 1, prnd);
        x = NextUInt32(x + 1, prnd);
      }
      else {
        y /= 2; x /= 2;
      }
      cv::Rect roi(x, y, rand_crop_size, rand_crop_size);
      cv::resize(res(roi), res, cv::Size(shape_[1], shape_[2]));
    }
    else{
      CHECK(static_cast<mshadow::index_t>(res.cols) >= shape_[1] && static_cast<mshadow::index_t>(res.rows) >= shape_[2]) 
          << "input image size smaller than input shape";
      mshadow::index_t y = res.rows - shape_[2];
      mshadow::index_t x = res.cols - shape_[1];
      if (param_.rand_crop_ != 0) {
        y = NextUInt32(y + 1, prnd);
        x = NextUInt32(x + 1, prnd);
      }
      else {
        y /= 2; x /= 2;
      }
      cv::Rect roi(x, y, shape_[1], shape_[2]);
      res = res(roi);
    }
    return res;
  }
  /*!
   * \brief augment src image, store result into dst
   *   this function is not thread safe, and will only be called by one thread
   *   however, it will tries to re-use memory space as much as possible
   * \param src the source image
   * \param source of random number
   * \param dst the pointer to the place where we want to store the result
   */
  virtual mshadow::Tensor<cpu, 3> Process(mshadow::Tensor<cpu, 3> data,
                                          common::RANDOM_ENGINE *prnd) {
    if (!NeedProcess()) return data;
    cv::Mat res(data.size(1), data.size(2), CV_8UC3);
    for (index_t i = 0; i < data.size(1); ++i) {
      for (index_t j = 0; j < data.size(2); ++j) {
        res.at<cv::Vec3b>(i, j)[0] = data[2][i][j];
        res.at<cv::Vec3b>(i, j)[1] = data[1][i][j];
        res.at<cv::Vec3b>(i, j)[2] = data[0][i][j];
      }
    }
    res = this->Process(res, prnd);
    tmpres.Resize(mshadow::Shape3(3, res.rows, res.cols));
    for (index_t i = 0; i < tmpres.size(1); ++i) {
      for (index_t j = 0; j < tmpres.size(2); ++j) {
        cv::Vec3b bgr = res.at<cv::Vec3b>(i, j);
        tmpres[0][i][j] = bgr[2];
        tmpres[1][i][j] = bgr[1];
        tmpres[2][i][j] = bgr[0];
      }
    }
    return tmpres;
  }

  virtual void Process(unsigned char *dptr, size_t sz,
                       mshadow::TensorContainer<cpu, 3> *p_data,
                       common::RANDOM_ENGINE *prnd) {
    cv::Mat buf(1, sz, CV_8U, dptr);
    cv::Mat res = cv::imdecode(buf, 1);
    res = this->Process(res, prnd);
    p_data->Resize(mshadow::Shape3(3, res.rows, res.cols));
    for (index_t i = 0; i < p_data->size(1); ++i) {
      for (index_t j = 0; j < p_data->size(2); ++j) {
        cv::Vec3b bgr = res.at<cv::Vec3b>(i, j);
        (*p_data)[0][i][j] = bgr[2];
        (*p_data)[1][i][j] = bgr[1];
        (*p_data)[2][i][j] = bgr[0];
      }
    }
    res.release();
  }

 private:
  // whether skip processing
  inline bool NeedProcess(void) const {
    if (param_.max_rotate_angle_ > 0 || param_.max_shear_ratio_ > 0.0f
        || param_.rotate_ > 0 || rotate_list_.size() > 0) return true;
    if (param_.min_crop_size_ > 0 && param_.max_crop_size_ > 0) return true;
    return false;
  }
  // temp input space
  mshadow::TensorContainer<cpu, 3> tmpres;
  // temporal space
  cv::Mat temp0, temp, temp2;
  // rotation param
  cv::Mat rotateM;
  // parameters
  ImageAugmentParam param_;
  /*! \brief input shape */
  mshadow::Shape<4> shape_;
  /*! \brief list of possible rotate angle */
  std::vector<int> rotate_list_;
};
}  // namespace io
}  // namespace cxxnet
#endif
