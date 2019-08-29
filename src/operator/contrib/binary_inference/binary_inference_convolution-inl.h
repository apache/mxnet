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
 * Copyright (c) 2018 by Contributors
 * \file binary_inference_convolution-inl.h
 * \brief
 * \ref: https://arxiv.org/abs/1705.09864
 * \author HPI-DeepLearning
*/
#ifndef MXNET_OPERATOR_CONTRIB_BINARY_INFERENCE_CONVOLUTION_INL_H_
#define MXNET_OPERATOR_CONTRIB_BINARY_INFERENCE_CONVOLUTION_INL_H_

#include <mxnet/io.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <dmlc/logging.h>
#include <dmlc/optional.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../../operator_common.h"
#include "../../nn/im2col.h"
#include "./xnor.h"


namespace mxnet {
    namespace op {

        namespace binary_inference_conv {
            enum BinaryInferenceConvolutionOpInputs {kData, kWeight, kBias};
            enum BinaryInferenceConvolutionOpOutputs {kOut};
            enum BinaryInferenceConvolutionOpResource {kTempSpace};            
        }

        struct BinaryInferenceConvolutionParam : public dmlc::Parameter<BinaryInferenceConvolutionParam> {
            mxnet::TShape kernel;
            mxnet::TShape stride;
            mxnet::TShape dilate;
            mxnet::TShape pad;
            uint32_t num_filter;
            uint32_t num_group;
            uint64_t workspace;
            bool no_bias;
            dmlc::optional<int> layout;
            DMLC_DECLARE_PARAMETER(BinaryInferenceConvolutionParam) {
              DMLC_DECLARE_FIELD(kernel).describe("convolution kernel size: (h, w) or (d, h, w)");
              DMLC_DECLARE_FIELD(stride).set_default(mxnet::TShape(0, -1))
                      .describe("convolution stride: (h, w) or (d, h, w)");
              DMLC_DECLARE_FIELD(dilate).set_default(mxnet::TShape(0, -1))
                      .describe("convolution dilate: (h, w) or (d, h, w)");
              DMLC_DECLARE_FIELD(pad).set_default(mxnet::TShape(0, -1))
                      .describe("pad for convolution: (h, w) or (d, h, w)");
              DMLC_DECLARE_FIELD(num_filter).set_range(1, 100000)
                      .describe("convolution filter(channel) number");
              DMLC_DECLARE_FIELD(num_group).set_default(1)
                      .describe("Number of group partitions.");
              DMLC_DECLARE_FIELD(workspace).set_default(1024).set_range(0, 8192)
                      .describe("Maximum temporary workspace allowed for convolution (MB).");
              DMLC_DECLARE_FIELD(no_bias).set_default(true)
                      .describe("Whether to disable bias parameter.");
              DMLC_DECLARE_FIELD(layout)
                      .add_enum("NCW", mshadow::kNCW)
                      .add_enum("NCHW", mshadow::kNCHW)
                      .add_enum("NCDHW", mshadow::kNCDHW)
                      .add_enum("NHWC", mshadow::kNHWC)
                      .add_enum("NDHWC", mshadow::kNDHWC)
                      .set_default(dmlc::optional<int>())
                      .describe("Set layout for input, output and weight. Empty for\n    "
                                        "default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.");
            }
            // Adjusts kernel size for effects of dilation in the dimension `dim`.
            index_t DilatedKernelSize(int dim) const {
              return 1 + (kernel[dim] - 1) * dilate[dim];
            }
        };

        template<typename xpu, typename DType>
        class BinaryInferenceConvolutionOp : public Operator {
        public:
            explicit BinaryInferenceConvolutionOp(BinaryInferenceConvolutionParam p) {
              this->param_ = p;
              // convert MBytes first to Bytes and then to elements.
              param_.workspace = (param_.workspace << 20) / sizeof(DType);
              CHECK(param_.layout.value() == mshadow::kNCW ||
                    param_.layout.value() == mshadow::kNCHW ||
                    param_.layout.value() == mshadow::kNCDHW)
                << "Only support NCW, NCHW and NCDHW layout";
            }

            virtual void Forward(const OpContext &ctx,
                                 const std::vector<TBlob> &in_data,
                                 const std::vector<OpReqType> &req,
                                 const std::vector<TBlob> &out_data,
                                 const std::vector<TBlob> &aux_args) {
              using namespace mshadow;
              using namespace mshadow::expr;
              CHECK_EQ(req[binary_inference_conv::kOut], kWriteTo);
              size_t expected = param_.no_bias ? 2 : 3;
              CHECK_EQ(in_data.size(), expected);
              CHECK_EQ(out_data.size(), 1U);
              CHECK_EQ(req[binary_inference_conv::kOut], kWriteTo);
              LayerSetUp(in_data[binary_inference_conv::kData].shape_, out_data[binary_inference_conv::kOut].shape_);
              Stream<xpu>* s = ctx.get_stream<xpu>();
              // allocate workspace for col_buffer
              Tensor<xpu, 1, DType> workspace = ctx.requested[binary_inference_conv::kTempSpace]
                      .get_space_typed<xpu, 1, DType>(Shape1(col_buffer_size_), s);
              // calculate the shape of col_buffer
              mxnet::TShape col_buffer_shape(num_spatial_axes_ + 1, -1);
              col_buffer_shape[0] = conv_in_channels_ * param_.kernel.Size();
              for (index_t i = 1; i < col_buffer_shape.ndim(); ++i) {
                col_buffer_shape[i] = out_data[0].shape_[i+1];
              }
              // create a column buffer using workspace and col_buffer_shape
              TBlob col_buffer(workspace.dptr_, col_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);

              // initialize weight and col_buffer 3D tensors for using gemm
              index_t M = conv_out_channels_ / group_; // number of output channels (num_filter) per group
              index_t N = conv_out_spatial_dim_;       // number of pixels of output images per channel
              index_t K = kernel_dim_;                 // number of input channels per group * kernel size
              Tensor<xpu, 3, DType> weight_3d;
              mxnet::op::xnor::BINARY_WORD* wmat_binarized = NULL;
              // binarized weigths
              wmat_binarized = (mxnet::op::xnor::BINARY_WORD*) in_data[binary_inference_conv::kWeight].dptr_;
              
              Tensor<xpu, 3, DType> col_buffer_3d = col_buffer.get_with_shape<xpu, 3, DType>(
                      Shape3(group_, K, N), s);
              Tensor<xpu, 4, DType> output_4d = out_data[binary_inference_conv::kOut].get_with_shape<xpu, 4, DType>(
                      Shape4(num_, group_, M, N), s);

              // check input dim
              CHECK_EQ(in_data[binary_inference_conv::kData].shape_[1] % mxnet::op::xnor::BITS_PER_BINARY_WORD, 0)
                << "input channel currently have to be multiple of " << mxnet::op::xnor::BITS_PER_BINARY_WORD << " but are: " << in_data[binary_inference_conv::kData].shape_[1];


              for (index_t n = 0; n < num_; ++n) {
                // transform image to col_buffer in order to use gemm
                im2col(s, in_data[binary_inference_conv::kData].dptr<DType>()+n*input_dim_, in_data[binary_inference_conv::kData].shape_,
                       col_buffer.shape_, param_.kernel, param_.pad, param_.stride, param_.dilate,
                       col_buffer.dptr<DType>());
                Tensor<xpu, 3, DType> output_3d = output_4d[n];
                
                for (index_t g = 0; g < group_; ++g) {                  
                  Tensor<xpu, 1, DType> binary_inputs_workspace =
                          ctx.requested[binary_inference_conv::kTempSpace].get_space_typed<xpu, 1, DType>(
                                  Shape1(N * K / (sizeof(DType) * CHAR_BIT)), s);

                  Tensor<xpu, 2, DType> temp_dst_gid = output_3d[g];                  
                  CHECK(g == 0) << "groups not yet supported for pre-binarized weights";
                  
                  //====== testing code =======//
                  // using ns = std::chrono::nanoseconds;
                  // using get_time = std::chrono::steady_clock;
                  // auto start = std::chrono::high_resolution_clock::now();
                  
                  BinaryConvolutionForward(M, N, K,
                                      wmat_binarized,
                                      binary_inputs_workspace,
                                      col_buffer_3d[g],
                                      temp_dst_gid);   
                  
                  // auto finish = std::chrono::high_resolution_clock::now();
                  // std::chrono::duration<double> elapsed = finish - start;
                  // std::cout << "Binary Conv Elapsed time: " << elapsed.count() << " s\n";           
                }
              }
            }

            virtual void Backward(const OpContext &ctx,
                                  const std::vector<TBlob>& out_grad,
                                  const std::vector<TBlob>& in_data,
                                  const std::vector<TBlob>& out_data,
                                  const std::vector<OpReqType>& req,
                                  const std::vector<TBlob>& in_grad,
                                  const std::vector<TBlob>& aux_args) {
              // nothing to do in backward pass
            }

        private:
            void LayerSetUp(const mxnet::TShape& ishape, const mxnet::TShape& oshape) {
              channel_axis_ = 1;  // hard code channel axis
              const index_t first_spatial_axis = channel_axis_ + 1;
              const index_t num_axes = param_.kernel.ndim() + 2;
              num_spatial_axes_ = num_axes - first_spatial_axis;
              is_1x1_ = true;
              for (index_t i = 0; i < param_.kernel.ndim(); ++i) {
                is_1x1_ &= param_.kernel[i] == 1 && param_.stride[i] == 1 && param_.pad[i] == 0;
                if (!is_1x1_) break;
              }

              // batch size
              num_ = ishape[0];
              // number of input channels
              channels_ = ishape[1];
              group_ = param_.num_group;
              conv_out_channels_ = param_.num_filter;
              conv_in_channels_ = channels_;
              bias_term_ = !param_.no_bias;
              kernel_dim_ = conv_in_channels_ / group_ * param_.kernel.Size();
              weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
              conv_out_spatial_dim_ = oshape.ProdShape(2, oshape.ndim());
              col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
              output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
              // size of the column buffer used for storing im2col-ed pixels
              col_buffer_size_ = kernel_dim_ * group_ * conv_out_spatial_dim_;
              // input/output image size (#channels * height * width)
              input_dim_ = ishape.ProdShape(1, ishape.ndim());
              output_dim_ = oshape.ProdShape(1, oshape.ndim());
              num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
              num_kernels_col2im_ = input_dim_;
            }

        private:
            BinaryInferenceConvolutionParam param_;
            index_t channel_axis_;  // channel axis of the input
            index_t channels_;  // number of channels of input image
            index_t num_spatial_axes_;  // number of spatial axes
            index_t num_;  // batch size
            index_t group_;  // number of groups
            index_t conv_out_channels_;  // number of output channels (num_filter)
            index_t conv_out_spatial_dim_;  // number of pixels of output images per channel
            index_t conv_in_channels_;  // number of input channels
            index_t kernel_dim_;  // number of input channels per group * kernel size
            index_t weight_offset_;  // number of output channels per group * kernel_dim_
            index_t col_offset_;
            index_t output_offset_;
            index_t col_buffer_size_;
            index_t input_dim_;
            index_t output_dim_;
            index_t num_kernels_im2col_;
            index_t num_kernels_col2im_;
            bool bias_term_;  // has bias term?
            bool is_1x1_;
        };  // class ConvolutionOp

        template<typename xpu>
        Operator* CreateOp(BinaryInferenceConvolutionParam param, int dtype,
                           mxnet::ShapeVector *in_shape,
                           mxnet::ShapeVector *out_shape,
                           Context ctx);

#if DMLC_USE_CXX11
        class BinaryInferenceConvolutionProp : public OperatorProperty {
        public:
            std::vector<std::string> ListArguments() const override {
              if (!param_.no_bias) {
                return {"data", "weight", "bias"};
              } else {
                return {"data", "weight"};
              }
            }

            void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
              using namespace mshadow;
              param_.Init(kwargs);
              if (param_.kernel.ndim() == 1) {
                param_.layout = param_.layout? param_.layout.value() : mshadow::kNCW;
                if (param_.stride.ndim() == 0) param_.stride = Shape1(1);
                if (param_.dilate.ndim() == 0) param_.dilate = Shape1(1);
                if (param_.pad.ndim() == 0) param_.pad = Shape1(0);
              } else if (param_.kernel.ndim() == 2) {
                param_.layout = param_.layout ? param_.layout.value() : mshadow::kNCHW;
                if (param_.stride.ndim() == 0) param_.stride = Shape2(1, 1);
                if (param_.dilate.ndim() == 0) param_.dilate = Shape2(1, 1);
                if (param_.pad.ndim() == 0) param_.pad = Shape2(0, 0);
              } else {
                CHECK_EQ(param_.kernel.ndim(), 3U) << param_.kernel.ndim() << "D convolution not supported";
                param_.layout = param_.layout ? param_.layout.value(): mshadow::kNCDHW;
                if (param_.stride.ndim() == 0) param_.stride = Shape3(1, 1, 1);
                if (param_.dilate.ndim() == 0) param_.dilate = Shape3(1, 1, 1);
                if (param_.pad.ndim() == 0) param_.pad = Shape3(0, 0, 0);
              }
            }

            std::map<std::string, std::string> GetParams() const override {
              return param_.__DICT__();
            }

            bool InferShape(mxnet::ShapeVector *in_shape,
                            mxnet::ShapeVector *out_shape,
                            mxnet::ShapeVector *aux_shape) const override {
              using namespace mshadow;
              if (!param_.no_bias) {
                LOG(WARNING) << "convolution with bias untested //mf";
                CHECK_EQ(in_shape->size(), 3U) << "Input:[data, weight, bias]";
              } else {
                CHECK_EQ(in_shape->size(), 2U) << "Input:[data, weight]";
              }
              // CHECK_EQ(out_shape->size(), 1) << "Output: [output]";
              out_shape->resize(1, mxnet::TShape());
              const mxnet::TShape &dshp = (*in_shape)[binary_inference_conv::kData];
              if (dshp.ndim() ==  0) return false;

              if (param_.kernel.ndim() != 2) {
                LOG(FATAL) << "Unknown convolution type (only 2d binary convolution supported)";
                return false;
              } else {
                // 2d conv
                CHECK_EQ(dshp.ndim(), 4U) \
              << "Input data should be 4D in batch-num_filter-y-x";
                Shape<4> dshape = ConvertLayout(dshp.get<4>(), param_.layout.value(), kNCHW);

                // defines shape of binary weights
                CHECK_EQ(param_.num_group, 1) << "groups not (yet?) supported for pre-binarized weights";                
                // this is the old 1-D version of binarized weights, will be removed in the final version
                // Shape<1> wshape = Shape1(dshape[1] * param_.num_filter * param_.kernel[0] * param_.kernel[1] / mxnet::op::xnor::BITS_PER_BINARY_WORD);
                // SHAPE_ASSIGN_CHECK(*in_shape, binary_inference_conv::kWeight, wshape);                
                Shape<4> wshape = Shape4(param_.num_filter / param_.num_group,
                                         dshape[1] / param_.num_group / mxnet::op::xnor::BITS_PER_BINARY_WORD,
                                         param_.kernel[0], param_.kernel[1]);
                wshape = ConvertLayout(wshape, kNCHW, param_.layout.value());
                wshape[0] *= param_.num_group;
                SHAPE_ASSIGN_CHECK(*in_shape, binary_inference_conv::kWeight, wshape);

                if (!param_.no_bias) {
                  SHAPE_ASSIGN_CHECK(*in_shape, binary_inference_conv::kBias, Shape1(param_.num_filter));
                }

                const index_t dilated_ksize_y = param_.DilatedKernelSize(0);
                const index_t dilated_ksize_x = param_.DilatedKernelSize(1);
                CHECK_EQ(dshape[1] % param_.num_group, 0U) \
          << "input num_filter must divide group size";
                CHECK_EQ(param_.num_filter % param_.num_group, 0U) \
          << "output num_filter must divide group size";
                CHECK_GT(param_.kernel.Size(), 0U) \
          << "incorrect kernel size: " << param_.kernel;
                CHECK_GT(param_.stride.Size(), 0U) \
          << "incorrect stride size: " << param_.stride;
                CHECK_GT(param_.dilate.Size(), 0U) \
          << "incorrect dilate size: " << param_.dilate;
                Shape<4> oshape;
                oshape[0] = dshape[0];
                oshape[1] = param_.num_filter;
                oshape[2] = dshape[2] ?
                            (AddPad(dshape[2], param_.pad[0]) - dilated_ksize_y) / param_.stride[0] + 1 : 0;
                oshape[3] = dshape[3] ?
                            (AddPad(dshape[3], param_.pad[1]) - dilated_ksize_x) / param_.stride[1] + 1 : 0;
                SHAPE_ASSIGN_CHECK(*out_shape, 0, ConvertLayout(oshape, kNCHW, param_.layout.value()));
                // Perform incomplete shape inference. Fill in the missing values in data shape.
                // 1) We can always fill in the batch_size.
                // 2) We can back-calculate the input height/width if the corresponding stride is 1.
                oshape = ConvertLayout((*out_shape)[0].get<4>(), param_.layout.value(), kNCHW);
                dshape[0] = oshape[0];
                if (oshape[2] && param_.stride[0] == 1) {
                  dshape[2] = oshape[2] + dilated_ksize_y - 1 - 2 * param_.pad[0];
                }
                if (oshape[3] && param_.stride[1] == 1) {
                  dshape[3] = oshape[3] + dilated_ksize_x - 1 - 2 * param_.pad[1];
                }
                SHAPE_ASSIGN_CHECK(*in_shape, binary_inference_conv::kData,
                                   ConvertLayout(dshape, kNCHW, param_.layout.value()));
                // Check whether the kernel sizes are valid
                if (dshape[2] != 0) {
                  CHECK_LE(dilated_ksize_y, AddPad(dshape[2], param_.pad[0])) << "kernel size exceed input";
                }
                if (dshape[3] != 0) {
                  CHECK_LE(dilated_ksize_x, AddPad(dshape[3], param_.pad[1])) << "kernel size exceed input";
                }
                return true;
              }
            }

            bool InferType(std::vector<int> *in_type,
                           std::vector<int> *out_type,
                           std::vector<int> *aux_type) const override {
              CHECK_GE(in_type->size(), 1U);
              int dtype = (*in_type)[0];
              CHECK_NE(dtype, -1) << "First input must have specified type";
              for (index_t i = 0; i < in_type->size(); ++i) {
                if ((*in_type)[i] == -1) {
                  (*in_type)[i] = dtype;
                } else {
                  if (i == binary_inference_conv::kWeight) {
                    continue;
                  }
                  CHECK_EQ((*in_type)[i], dtype) << "This layer requires uniform type. "
                                                 << "Expected " << dtype << " v.s. given "
                                                 << (*in_type)[i] << " at " << ListArguments()[i];
                }
              }
              
              (*in_type)[binary_inference_conv::kWeight] = mxnet::op::xnor::corresponding_dtype();
              
              out_type->clear();
              out_type->push_back(dtype);
              return true;
            }

            OperatorProperty* Copy() const override {
              auto ptr = new BinaryInferenceConvolutionProp();
              ptr->param_ = param_;
              return ptr;
            }

            std::string TypeString() const override {
              return "BinaryInferenceConvolution";
            }

            std::vector<int> DeclareBackwardDependency(
                    const std::vector<int> &out_grad,
                    const std::vector<int> &in_data,
                    const std::vector<int> &out_data) const override {
              return {out_grad[binary_inference_conv::kOut], in_data[binary_inference_conv::kData], in_data[binary_inference_conv::kWeight]};
            }

            std::vector<ResourceRequest> ForwardResource(
                    const mxnet::ShapeVector &in_shape) const override {
              return {ResourceRequest::kTempSpace};
            }

            std::vector<ResourceRequest> BackwardResource(
                    const mxnet::ShapeVector &in_shape) const override {
              return {ResourceRequest::kTempSpace};
            }

            Operator* CreateOperator(Context ctx) const override {
              LOG(FATAL) << "Not Implemented.";
              return NULL;
            }

            Operator* CreateOperatorEx(Context ctx, mxnet::ShapeVector *in_shape,
                                       std::vector<int> *in_type) const override;

        private:
            // Adds symmetric padding to a data input (in one dimension)
            index_t AddPad(index_t dsize, index_t pad) const {
              return dsize + 2 * pad;
            }

            BinaryInferenceConvolutionParam param_;
        };  // class BinaryInferenceConvolutionProp
#endif  // DMLC_USE_CXX11
    }  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_BINARY_INFERENCE_CONVOLUTION_INL_H_
