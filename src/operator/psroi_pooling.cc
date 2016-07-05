/*!
 * Copyright (c) 2016 by Contributors
 * \file psroi_pooling.cc
 * \brief psroi pooling operator
 * \author Yi Li, Tairui Chen
*/
#include "./psroi_pooling-inl.h"
#include <mshadow/base.h>
#include <mshadow/tensor.h>
#include <mshadow/packet-inl.h>
#include <mshadow/dot_engine-inl.h>
#include <cassert>

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace mshadow {
template<typename Dtype>
inline void PSROIPoolForward(const Tensor<cpu, 4, Dtype> &out,
                           const Tensor<cpu, 4, Dtype> &data,
                           const Tensor<cpu, 2, Dtype> &bbox,
                           const Tensor<cpu, 4, Dtype> &mapping_channel,
                           const float spatial_scale_,
                           const int output_dim_, 
                           const int group_size_) {
  // NOT_IMPLEMENTED;
  return;
}

template<typename Dtype>
inline void PSROIPoolBackward(const Tensor<cpu, 4, Dtype> &in_grad,
                            const Tensor<cpu, 4, Dtype> &out_grad,
                            const Tensor<cpu, 2, Dtype> &bbox,
                            const Tensor<cpu, 4, Dtype> &mapping_channel,
                            const float spatial_scale_,
                            const int output_dim_) {
  // NOT_IMPLEMENTED;
  return;
}
}  // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator* CreateOp<cpu>(PSROIPoolingParam param) {
  return new PSROIPoolingOp<cpu>(param);
}

// DO_BIND_DISPATCH comes from static_operator_common.h
Operator* PSROIPoolingProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(PSROIPoolingParam);

MXNET_REGISTER_OP_PROPERTY(PSROIPooling, PSROIPoolingProp)
.describe("Performs region-of-interest pooling on inputs. Resize bounding box coordinates by "
"spatial_scale and crop input feature maps accordingly. The cropped feature maps are pooled "
"by max pooling to a fixed size output indicated by pooled_size. batch_size will change to "
"the number of region bounding boxes after PSROIPooling")
.add_argument("data", "Symbol", "Input data to the pooling operator, a 4D Feature maps")
.add_argument("rois", "Symbol", "Bounding box coordinates, a 2D array of "
"[[batch_index, x1, y1, x2, y2]]. (x1, y1) and (x2, y2) are top left and down right corners "
"of designated region of interest. batch_index indicates the index of corresponding image "
"in the input data")
.add_arguments(PSROIPoolingParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
