/*!
 *  Copyright (c) 2017 by Contributors
 * \file export.h
 * \brief Export module that takes charge of code generation and document
 *  Generation for functions exported from R-side
 */

#ifndef MXNET_RCPP_IM2REC_H_
#define MXNET_RCPP_IM2REC_H_

#include <Rcpp.h>
#include <string>

namespace mxnet {
namespace R {

class IM2REC {
 public:
  /*!
   * \brief Export the generated file into path.
   * \param path The path to be exported.
   */
  static void im2rec(const std::string & image_lst, const std::string & root,
                     const std::string & output_rec,
                     int label_width = 1, int pack_label = 0, int new_size = -1, int nsplit = 1,
                     int partid = 0, int center_crop = 0, int quality = 95,
                     int color_mode = 1, int unchanged = 0,
                     int inter_method = 1, std::string encoding = ".jpg");
  // intialize the Rcpp module
  static void InitRcppModule();

 public:
  // get the singleton of exporter
  static IM2REC* Get();
  /*! \brief The scope of current module to export */
  Rcpp::Module* scope_;
};

}  // namespace R
}  // namespace mxnet

#endif  // MXNET_RCPP_IM2REC_H_
