/*!
 *  Copyright (c) 2017 by Contributors
 * \file export.h
 * \brief Export module that takes charge of code generation and document
 *  Generation for functions exported from R-side
 */

#include <cctype>
#include <cstring>
#include <string>
#include <vector>
#include <iomanip>
#include <sstream>
#include <random>
#include "dmlc/base.h"
#include "dmlc/io.h"
#include "dmlc/timer.h"
#include "dmlc/logging.h"
#include "dmlc/recordio.h"
#include <opencv2/opencv.hpp>
#include "image_recordio.h"
#include "base.h"
#include "im2rec.h"

namespace mxnet {
namespace R {

int GetInterMethod(int inter_method, int old_width, int old_height,
                   int new_width, int new_height, std::mt19937& prnd) {  // NOLINT(*)
  if (inter_method == 9) {
    if (new_width > old_width && new_height > old_height) {
      return 2;  // CV_INTER_CUBIC for enlarge
    } else if (new_width <old_width && new_height < old_height) {
      return 3;  // CV_INTER_AREA for shrink
    } else {
      return 1;  // CV_INTER_LINEAR for others
    }
  } else if (inter_method == 10) {
    std::uniform_int_distribution<size_t> rand_uniform_int(0, 4);
    return rand_uniform_int(prnd);
  } else {
    return inter_method;
  }
}

IM2REC* IM2REC::Get() {
  static IM2REC inst;
  return &inst;
}

void IM2REC::InitRcppModule() {
  using namespace Rcpp;  // NOLINT(*)
  IM2REC::Get()->scope_ = ::getCurrentScope();
  function("mx.internal.im2rec", &IM2REC::im2rec,
           Rcpp::List::create(_["image_lst"],
                              _["root"],
                              _["output_rec"],
                              _["label_width"],
                              _["pack_label"],
                              _["new_size"],
                              _["nsplit"],
                              _["partid"],
                              _["center_crop"],
                              _["quality"],
                              _["color_mode"],
                              _["unchanged"],
                              _["inter_method"],
                              _["encoding"]),
           "");
}

void IM2REC::im2rec(const std::string & image_lst, const std::string & root,
                    const std::string & output_rec,
                    int label_width, int pack_label, int new_size, int nsplit,
                    int partid, int center_crop, int quality,
                    int color_mode, int unchanged,
                    int inter_method, std::string encoding) {
  // Check parameters ranges
  if (color_mode != -1 && color_mode != 0 && color_mode != 1) {
    Rcpp::stop("Color mode must be -1, 0 or 1.");
  }
  if (encoding != std::string(".jpg") && encoding != std::string(".png")) {
    Rcpp::stop("Encoding mode must be .jpg or .png.");
  }
  if (label_width <= 1 && pack_label) {
    Rcpp::stop("pack_label can only be used when label_width > 1");
  }
  if (new_size > 0) {
    LOG(INFO) << "New Image Size: Short Edge " << new_size;
  } else {
    LOG(INFO) << "Keep origin image size";
  }
  if (center_crop) {
    LOG(INFO) << "Center cropping to square";
  }
  if (color_mode == 0) {
    LOG(INFO) << "Use gray images";
  }
  if (color_mode == -1) {
    LOG(INFO) << "Keep original color mode";
  }
  LOG(INFO) << "Encoding is " << encoding;

  if (encoding == std::string(".png") && quality > 9) {
    quality = 3;
  }
  if (inter_method != 1) {
    switch (inter_method) {
      case 0:
        LOG(INFO) << "Use inter_method CV_INTER_NN";
        break;
      case 2:
        LOG(INFO) << "Use inter_method CV_INTER_CUBIC";
        break;
      case 3:
        LOG(INFO) << "Use inter_method CV_INTER_AREA";
        break;
      case 4:
        LOG(INFO) << "Use inter_method CV_INTER_LANCZOS4";
        break;
      case 9:
        LOG(INFO) << "Use inter_method mod auto(cubic for enlarge, area for shrink)";
        break;
      case 10:
        LOG(INFO) << "Use inter_method mod rand(nn/bilinear/cubic/area/lanczos4)";
        break;
    }
  }
  std::random_device rd;
  std::mt19937 prnd(rd());
  using namespace dmlc;
  static const size_t kBufferSize = 1 << 20UL;
  mxnet::io::ImageRecordIO rec;
  size_t imcnt = 0;
  double tstart = dmlc::GetTime();
  dmlc::InputSplit *flist =
      dmlc::InputSplit::Create(image_lst.c_str(), partid, nsplit, "text");
  std::ostringstream os;
  if (nsplit == 1) {
    os << output_rec;
  } else {
    os << output_rec << ".part" << std::setw(3) << std::setfill('0') << partid;
  }
  LOG(INFO) << "Write to output: " << os.str();
  dmlc::Stream *fo = dmlc::Stream::Create(os.str().c_str(), "w");
  LOG(INFO) << "Output: " << os.str();
  dmlc::RecordIOWriter writer(fo);
  std::string fname, path, blob;
  std::vector<unsigned char> decode_buf;
  std::vector<unsigned char> encode_buf;
  std::vector<int> encode_params;
  if (encoding == std::string(".png")) {
    encode_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    encode_params.push_back(quality);
    LOG(INFO) << "PNG encoding compression: " << quality;
  } else {
    encode_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    encode_params.push_back(quality);
    LOG(INFO) << "JPEG encoding quality: " << quality;
  }
  dmlc::InputSplit::Blob line;
  std::vector<float> label_buf(label_width, 0.f);

  while (flist->NextRecord(&line)) {
    std::string sline(static_cast<char*>(line.dptr), line.size);
    std::istringstream is(sline);
    if (!(is >> rec.header.image_id[0] >> rec.header.label)) continue;
    label_buf[0] = rec.header.label;
    for (int k = 1; k < label_width; ++k) {
      RCHECK(is >> label_buf[k])
          << "Invalid ImageList, did you provide the correct label_width?";
    }
    if (pack_label) rec.header.flag = label_width;
    rec.SaveHeader(&blob);
    if (pack_label) {
      size_t bsize = blob.size();
      blob.resize(bsize + label_buf.size()*sizeof(float));
      memcpy(BeginPtr(blob) + bsize,
             BeginPtr(label_buf), label_buf.size()*sizeof(float));
    }
    RCHECK(std::getline(is, fname));
    // eliminate invalid chars in the end
    while (fname.length() != 0 &&
           (isspace(*fname.rbegin()) || !isprint(*fname.rbegin()))) {
      fname.resize(fname.length() - 1);
    }
    // eliminate invalid chars in beginning.
    const char *p = fname.c_str();
    while (isspace(*p)) ++p;
    path = root + p;
    // use "r" is equal to rb in dmlc::Stream
    dmlc::Stream *fi = dmlc::Stream::Create(path.c_str(), "r");
    decode_buf.clear();
    size_t imsize = 0;
    while (true) {
      decode_buf.resize(imsize + kBufferSize);
      size_t nread = fi->Read(BeginPtr(decode_buf) + imsize, kBufferSize);
      imsize += nread;
      decode_buf.resize(imsize);
      if (nread != kBufferSize) break;
    }
    delete fi;


    if (unchanged != 1) {
      cv::Mat img = cv::imdecode(decode_buf, color_mode);
      RCHECK(img.data != NULL) << "OpenCV decode fail:" << path;
      cv::Mat res = img;
      if (new_size > 0) {
        if (center_crop) {
          if (img.rows > img.cols) {
            int margin = (img.rows - img.cols)/2;
            img = img(cv::Range(margin, margin+img.cols), cv::Range(0, img.cols));
          } else {
            int margin = (img.cols - img.rows)/2;
            img = img(cv::Range(0, img.rows), cv::Range(margin, margin + img.rows));
          }
        }
        int interpolation_method = 1;
        if (img.rows > img.cols) {
          if (img.cols != new_size) {
            interpolation_method = GetInterMethod(inter_method, img.cols, img.rows,
                                                  new_size,
                                                  img.rows * new_size / img.cols, prnd);
            cv::resize(img, res, cv::Size(new_size,
                                          img.rows * new_size / img.cols),
                       0, 0, interpolation_method);
          } else {
            res = img.clone();
          }
        } else {
          if (img.rows != new_size) {
            interpolation_method = GetInterMethod(inter_method, img.cols,
                                                  img.rows, new_size * img.cols / img.rows,
                                                  new_size, prnd);
            cv::resize(img, res, cv::Size(new_size * img.cols / img.rows,
                                          new_size), 0, 0, interpolation_method);
          } else {
            res = img.clone();
          }
        }
      }
      encode_buf.clear();
      RCHECK(cv::imencode(encoding, res, encode_buf, encode_params));

      // write buffer
      size_t bsize = blob.size();
      blob.resize(bsize + encode_buf.size());
      memcpy(BeginPtr(blob) + bsize,
             BeginPtr(encode_buf), encode_buf.size());
    } else {
      size_t bsize = blob.size();
      blob.resize(bsize + decode_buf.size());
      memcpy(BeginPtr(blob) + bsize,
             BeginPtr(decode_buf), decode_buf.size());
    }
    writer.WriteRecord(BeginPtr(blob), blob.size());
    // write header
    ++imcnt;
    if (imcnt % 1000 == 0) {
      LOG(INFO) << imcnt << " images processed, " << GetTime() - tstart << " sec elapsed";
    }
  }
  LOG(INFO) << "Total: " << imcnt << " images processed, " << GetTime() - tstart << " sec elapsed";
  delete fo;
  delete flist;
}
}  // namespace R
}  // namespace mxnet
