/*!
 * Copyright (c) 2015 by Contributors
 * \file param.h
 * \brief Common operator parameters
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_PARAM_H_
#define MXNET_OPERATOR_PARAM_H_

#include <cstring>

namespace mxnet {
namespace op {
/*! \brief possible parameter for each operator */
struct Param {
  /*! \brief number of hidden layers */
  int num_hidden;
  /*! \brief number of output channel */
  int num_channel;
  /*! \brief number of parallel group */
  int num_group;
  /*! \brief kernel height */
  int kernel_y;
  /*! \brief kernel width */
  int kernel_x;
  /*! \brief stride in y dimension*/
  int stride_y;
  /*! \brief stride in x dimension */
  int stride_x;
  /*! \brief padding in y dimension */
  int pad_y;
  /*! \brief padding in x dimension */
  int pad_x;
  /*! \brief whether not include bias term */
  int no_bias;
  /*! \brief maximum temp_col_size allowed in each layer */
  int temp_col_max;
  /*! \brief reserved fields, for future compatibility */
  int reserved[64];

  // constructor
  Param() {
    memset(this, 0, sizeof(Param));
  }

  inline void SetParam(const char *name, const char* val) {
    if (!strcmp(name, "num_hidden")) num_hidden = atoi(val);
    if (!strcmp(name, "num_channel")) num_channel = atoi(val);
    if (!strcmp(name, "num_group")) num_group = atoi(val);
    if (!strcmp(name, "kernel_size")) {
      kernel_y = kernel_x = atoi(val);
    }
    if (!strcmp(name, "kernel_height")) kernel_y = atoi(val);
    if (!strcmp(name, "kernel_width")) kernel_x = atoi(val);
    if (!strcmp(name, "stride")) {
      stride_y = stride_x = atoi(val);
    }
    if (!strcmp(name, "stride_y")) stride_y = atoi(val);
    if (!strcmp(name, "stride_x")) stride_x = atoi(val);

    if (!strcmp(name, "pad")) {
      pad_y = pad_x  = atoi(val);
    }
    if (!strcmp(name, "pad_y")) pad_y = atoi(val);
    if (!strcmp(name, "pad_x")) pad_x = atoi(val);
    if (!strcmp(name, "no_bias")) no_bias = atoi(val);
    if (!strcmp(name, "temp_col_max")) temp_col_max = atoi(val) << 18;
  }
};  // struct Param
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_PARAM_H_
