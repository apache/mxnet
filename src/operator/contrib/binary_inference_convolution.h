/*!
 * Copyright (c) 2017 by Contributors
 * \file qconvolution-inl.h
 * \brief
 * \author HPI-DeepLearning, Bing Xu, Jun Wu
*/
#ifndef MXNET_OPERATOR_QCONVOLUTION_INL_H_
#define MXNET_OPERATOR_QCONVOLUTION_INL_H_

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
#include "../../src/operator/operator_common.h"
#include "../../src/operator/nn/im2col.h"
#include "./xnor_cpu.h"
#include "./q_helper.h"


namespace mxnet {
    namespace op {

        namespace qconv {
            enum ConvolutionOpInputs {kData, kWeight, kBias};
            enum ConvolutionOpOutputs {kOut};
            enum ConvolutionOpResource {kTempSpace};
            enum ConvolutionOpCudnnTune {kOff, kLimited, kFastest};
            enum ScalingMode {scaling_none, scaling_forward, scaling_backward};
            enum GradientUpdateMode {bb, bf, ff};
        }

        struct QConvolutionParam : public dmlc::Parameter<QConvolutionParam> {
            TShape kernel;
            TShape stride;
            TShape dilate;
            TShape pad;
            uint32_t num_filter;
            uint32_t num_group;
            uint64_t workspace;
            bool no_bias;
            dmlc::optional<int> cudnn_tune;
            bool cudnn_off;
            dmlc::optional<int> layout;
            // mf quantization and binarization variables
            uint32_t act_bit;
            uint32_t weight_bit;
            bool binarized_weights_only;
            dmlc::optional<int> scaling_mode;   
            dmlc::optional<int> gradient_update_mode;
            DMLC_DECLARE_PARAMETER(QConvolutionParam) {
              DMLC_DECLARE_FIELD(kernel).describe("convolution kernel size: (h, w) or (d, h, w)");
              DMLC_DECLARE_FIELD(stride).set_default(TShape())
                      .describe("convolution stride: (h, w) or (d, h, w)");
              DMLC_DECLARE_FIELD(dilate).set_default(TShape())
                      .describe("convolution dilate: (h, w) or (d, h, w)");
              DMLC_DECLARE_FIELD(pad).set_default(TShape())
                      .describe("pad for convolution: (h, w) or (d, h, w)");
              DMLC_DECLARE_FIELD(num_filter).set_range(1, 100000)
                      .describe("convolution filter(channel) number");
              DMLC_DECLARE_FIELD(num_group).set_default(1)
                      .describe("Number of group partitions.");
              DMLC_DECLARE_FIELD(workspace).set_default(1024).set_range(0, 8192)
                      .describe("Maximum temporary workspace allowed for convolution (MB).");
              DMLC_DECLARE_FIELD(no_bias).set_default(true)
                      .describe("Whether to disable bias parameter.");
              DMLC_DECLARE_FIELD(cudnn_tune)
                      .add_enum("off", qconv::kOff)
                      .add_enum("limited_workspace", qconv::kLimited)
                      .add_enum("fastest", qconv::kFastest)
                      .set_default(dmlc::optional<int>())
                      .describe("Whether to pick convolution algo by running performance test.");
              DMLC_DECLARE_FIELD(cudnn_off).set_default(false)
                      .describe("Turn off cudnn for this layer.");
              DMLC_DECLARE_FIELD(layout)
                      .add_enum("NCW", mshadow::kNCW)
                      .add_enum("NCHW", mshadow::kNCHW)
                      .add_enum("NCDHW", mshadow::kNCDHW)
                      .add_enum("NHWC", mshadow::kNHWC)
                      .add_enum("NDHWC", mshadow::kNDHWC)
                      .set_default(dmlc::optional<int>())
                      .describe("Set layout for input, output and weight. Empty for\n    "
                                        "default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.");
              DMLC_DECLARE_FIELD(act_bit).set_default(1).set_range(1, 32)
                      .describe("Number of bits to quantize activations (inputs) to.");
              DMLC_DECLARE_FIELD(binarized_weights_only).set_default(false)
                      .describe("Params file contains only binarized weights. Set automatically by model converter.");
              DMLC_DECLARE_FIELD(weight_bit).set_default(1).set_range(1, 32)
                      .describe("Number of bits to quantize weights to.");
              DMLC_DECLARE_FIELD(scaling_mode)
                      .add_enum("none", qconv::scaling_none)
                      .add_enum("forward", qconv::scaling_forward)
                      .add_enum("backward", qconv::scaling_backward)
                      .set_default(dmlc::optional<int>(0))
                      .describe("Set whether or how to apply scaling factor to the conv output.\n"
                                "none: no scaling process;\n"
                                "forward: apply scaling scalar after standard forward and backward operations;\n"
                                "backward: only apply scaling scalar in backward pass");
              DMLC_DECLARE_FIELD(gradient_update_mode)
                      .add_enum("bb", qconv::bb)
                      .add_enum("bf", qconv::bf)
                      .add_enum("ff", qconv::ff)
                      .set_default(dmlc::optional<int>(0))
                      .describe("Set the mode of gradient calculation and update.\n"
                                "bb: calculate gradients on binary/quantized weights, update binary/quantized weights; \n"
                                "bf: calculate gradients on binary/quantized weights, update full-precision weights; \n"
                                "ff: calculate gradients on full-precision weights, update full-precision weights; \n"
                                "For disambiguation: we always use binary/quantized weights for forward calculation.");
            }
            // Adjusts kernel size for effects of dilation in the dimension `dim`.
            index_t DilatedKernelSize(int dim) const {
              return 1 + (kernel[dim] - 1) * dilate[dim];
            }
        };

        template<typename xpu, typename DType>
        class QConvolutionOp : public Operator {
        public:
            explicit QConvolutionOp(QConvolutionParam p) {
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
              CHECK_EQ(req[qconv::kOut], kWriteTo);
              size_t expected = param_.no_bias ? 2 : 3;
              CHECK_EQ(in_data.size(), expected);
              CHECK_EQ(out_data.size(), 1U);
              CHECK_EQ(req[qconv::kOut], kWriteTo);
              LayerSetUp(in_data[qconv::kData].shape_, out_data[qconv::kOut].shape_);
              Stream<xpu>* s = ctx.get_stream<xpu>();
              // allocate workspace for col_buffer
              Tensor<xpu, 1, DType> workspace = ctx.requested[qconv::kTempSpace]
                      .get_space_typed<xpu, 1, DType>(Shape1(col_buffer_size_), s);
              // calculate the shape of col_buffer
              TShape col_buffer_shape(num_spatial_axes_ + 1);
              col_buffer_shape[0] = conv_in_channels_ * param_.kernel.Size();
              for (index_t i = 1; i < col_buffer_shape.ndim(); ++i) {
                col_buffer_shape[i] = out_data[0].shape_[i+1];
              }
              // create a column buffer using workspace and col_buffer_shape
              TBlob col_buffer(workspace.dptr_, col_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);

              // initialize weight and col_buffer 3D tensors for using gemm
              index_t M = conv_out_channels_ / group_;
              index_t N = conv_out_spatial_dim_;
              index_t K = kernel_dim_;
              Tensor<xpu, 3, DType> weight_3d;
              mxnet::op::xnor_cpu::BINARY_WORD* wmat_binarized = NULL;
              if (param_.binarized_weights_only) {
                wmat_binarized = (mxnet::op::xnor_cpu::BINARY_WORD*) in_data[qconv::kWeight].dptr_;
              } else {
                weight_3d = in_data[qconv::kWeight].get_with_shape<xpu, 3, DType>(
                        Shape3(group_, M, K), s);
              }
              Tensor<xpu, 3, DType> col_buffer_3d = col_buffer.get_with_shape<xpu, 3, DType>(
                      Shape3(group_, K, N), s);
              Tensor<xpu, 4, DType> output_4d = out_data[qconv::kOut].get_with_shape<xpu, 4, DType>(
                      Shape4(num_, group_, M, N), s);

              // xnor related check
              CHECK_EQ(in_data[qconv::kData].shape_[1] % mxnet::op::xnor_cpu::BITS_PER_BINARY_WORD, 0)
                << "input channel currently have to be multiple of " << mxnet::op::xnor_cpu::BITS_PER_BINARY_WORD << " but are: " << in_data[qconv::kData].shape_[1];

              //============================================//
              //calc the scaling scalar for 1-bit mode.
              //Note that should on original weights and activations              
              DType scaling_scalar_w;
              if(this->param_.act_bit == 1 && this->param_.weight_bit == 1 
                 && param_.scaling_mode.value() == qconv::scaling_forward){
                //calc scaling scalar of original weights
                Tensor<xpu, 1, DType> w = in_data[qconv::kWeight].FlatTo1D<xpu, DType>(s);
                //Tensor<xpu, 1, DType> inputd = in_data[qconv::kData].FlatTo1D<xpu, DType>(s);
                scaling_scalar_w = q_helper::get_scaling_scalar(w);           
              }
              //============================================//  
              //============================================//
              //            WEIGHTS quantization            //
              // for training mode,                         //
              // we apply quantization function on weights. //
              //                                            //
              Tensor<xpu, 1, DType> w1d, w1d_copy;

              if(this->param_.weight_bit < 32 
                  && (ctx.is_train
                    || (!ctx.is_train 
                        && std::is_same<xpu, gpu>::value)
                    || (!ctx.is_train 
                        && std::is_same<xpu, cpu>::value 
                        && (this->param_.act_bit != 1 || this->param_.weight_bit != 1) 
                      )  
                    )
              ){
                // mf quantize weights
                w1d = in_data[qconv::kWeight].FlatTo1D<xpu, DType>(s);
                if (this->param_.gradient_update_mode.value() != qconv::bb){
                  w1d_copy = mshadow::NewTensor<xpu>(w1d.shape_, DType(1.0), true, w1d.stream_);
                  mshadow::Copy(w1d_copy, w1d, w1d.stream_);
                }
                q_helper::quantize_weights(w1d, this->param_.weight_bit);
                // /mf quantize weights
              }
              //                                            //
              //============================================//

              for (index_t n = 0; n < num_; ++n) {
                // transform image to col_buffer in order to use gemm
                im2col(s, in_data[qconv::kData].dptr<DType>()+n*input_dim_, in_data[qconv::kData].shape_,
                       col_buffer.shape_, param_.kernel, param_.pad, param_.stride, param_.dilate,
                       col_buffer.dptr<DType>());
                Tensor<xpu, 3, DType> output_3d = output_4d[n];

                //============================================//
                //             INPUT quantization             //
                // for training or prediction in gpu mode,    //
                // we apply quantization function on input    //
                // This process should be after padding elemt //
                // since the padding elements are all "0"     //
                //                                            //
                if(this->param_.act_bit < 32 
                  && (ctx.is_train
                    || (!ctx.is_train 
                        && std::is_same<xpu, gpu>::value)
                    || (!ctx.is_train 
                        && std::is_same<xpu, cpu>::value 
                        && (this->param_.act_bit != 1 || this->param_.weight_bit != 1) 
                        )  
                    )
                ){
                  q_helper::quantize_activations(col_buffer_3d, this->param_.act_bit);
                }
                //                                            //
                //============================================//
                for (index_t g = 0; g < group_; ++g) {


                  //==================================================================//
                  // For the training in order to make the training easier and faster,//
                  // we binarize the input and weights of Qconv layer to +1 and -1,   //
                  // still apply the standard dot() operator to generate the gemm     //
                  // result. But for 1-bit prediction by using CPU we then apply      //
                  //   xnor+_popc                                                     //
                  // to generate the same result as the dot() function.               //
                  // this means that for the prediction phase in 1-bit, the           //
                  //   QConvolutionForward(...)                                       //
                  // should produce the exactly same result as the dot(bina(..))method//
                  //                                                                  //
                  if(!ctx.is_train 
                      && std::is_same<xpu, cpu>::value 
                      && this->param_.act_bit == 1 
                      && this->param_.weight_bit == 1){

                    // @todo: watch out, we get 32bit float space here and later possibly cast it into 64bit space
                    Tensor<xpu, 1, DType> binary_inputs_workspace =
                            ctx.requested[qconv::kTempSpace].get_space_typed<xpu, 1, DType>(
                                    Shape1(N * K / (sizeof(DType) * CHAR_BIT)), s);
                    Tensor<xpu, 2, DType> temp_dst_gid = output_3d[g];
                    if (param_.binarized_weights_only) {
                      CHECK(g == 0) << "groups not yet supported for pre-binarized weights";
                      QConvolutionForward(M, N, K,
                                          wmat_binarized,
                                          binary_inputs_workspace,
                                          col_buffer_3d[g],
                                          temp_dst_gid);
                    } else {
                      QConvolutionForward(M, N, K,
                                          weight_3d[g],
                                          binary_inputs_workspace,
                                          col_buffer_3d[g],
                                          temp_dst_gid);
                    }
                  }else{ // for training phase...
                    ASSIGN_DISPATCH(output_3d[g], req[qconv::kOut], dot(weight_3d[g], col_buffer_3d[g]));
                    //this converting is just for mimicing 1-bit xnor-popc operations
                    if(this->param_.act_bit == 1 && this->param_.weight_bit == 1)
                      output_3d[g] = (ScalarExp<DType>(weight_3d[g].size(1)) + output_3d[g]) / scalar(DType(2.0));
                  }
                  //                                                                  //
                  //==================================================================//
                }
              }

              if (bias_term_) {
                Tensor<xpu, 1, DType> bias = in_data[qconv::kBias].get<xpu, 1, DType>(s);
                Tensor<xpu, 3, DType> output_3d = out_data[qconv::kOut].get_with_shape<xpu, 3, DType>(
                        Shape3(num_, conv_out_channels_, conv_out_spatial_dim_), s);
                // has bias term, broadcast it to the same shape of output_3d in channel dim
                output_3d += mshadow::expr::broadcast<1>(bias, output_3d.shape_);
              }
              //============================================//
              //calc the scaling scalar
              if(this->param_.act_bit == 1 && this->param_.weight_bit == 1 
                 && param_.scaling_mode.value() == qconv::scaling_forward){
                Tensor<xpu, 4, DType> o4d = out_data[qconv::kOut].get<xpu, 4, DType>(s);
                q_helper::tensor_mul_scalar(o4d, scaling_scalar_w);                            
              }
              //============================================//
              //============================================//
              //copy back the original weights
              if(this->param_.gradient_update_mode.value() != qconv::bb){
                mshadow::Copy(w1d, w1d_copy, w1d_copy.stream_);
                mshadow::FreeSpace(&w1d_copy);
              }
              //============================================//
            }

            virtual void Backward(const OpContext &ctx,
                                  const std::vector<TBlob>& out_grad,
                                  const std::vector<TBlob>& in_data,
                                  const std::vector<TBlob>& out_data,
                                  const std::vector<OpReqType>& req,
                                  const std::vector<TBlob>& in_grad,
                                  const std::vector<TBlob>& aux_args) {
              using namespace mshadow;
              using namespace mshadow::expr;
              CHECK_EQ(out_grad.size(), 1U);
              size_t expected = param_.no_bias == 0 ? 3 : 2;
              CHECK(in_data.size() == expected && in_grad.size() == expected);
              CHECK_EQ(req.size(), expected);
              CHECK_EQ(in_data[qconv::kWeight].CheckContiguous(), true);
              LayerSetUp(in_grad[qconv::kData].shape_, out_grad[qconv::kOut].shape_);
              Stream<xpu> *s = ctx.get_stream<xpu>();

              // allocate workspace for col_buffer
              Tensor<xpu, 1, DType> workspace = ctx.requested[qconv::kTempSpace]
                      .get_space_typed<xpu, 1, DType>(Shape1(col_buffer_size_), s);
              // calculate the shape of col_buffer
              TShape col_buffer_shape(num_spatial_axes_ + 1);
              col_buffer_shape[0] = conv_in_channels_ * param_.kernel.Size();
              for (index_t i = 1; i < col_buffer_shape.ndim(); ++i) {
                col_buffer_shape[i] = out_grad[qconv::kData].shape_[i+1];
              }
              // create a column buffer using workspace and col_buffer_shape
              TBlob col_buffer(workspace.dptr_, col_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);

              // initialize weight and col_buffer 3D tensors for using gemm
              // For computing dLoss/d(in_data[kData])
              index_t M = kernel_dim_;
              index_t N = conv_out_spatial_dim_;
              index_t K = conv_out_channels_ / group_;
              Tensor<xpu, 3, DType> weight_3d = in_data[qconv::kWeight].get_with_shape<xpu, 3, DType>(
                      Shape3(group_, K, M), s);
              Tensor<xpu, 4, DType> out_grad_4d = out_grad[qconv::kOut].get_with_shape<xpu, 4, DType>(
                      Shape4(num_, group_, K, N), s);
              Tensor<xpu, 3, DType> col_buffer_3d = col_buffer.get_with_shape<xpu, 3, DType>(
                      Shape3(group_, M, N), s);
              // For computing dLoss/dWeight
              Tensor<xpu, 3, DType> dweight_3d = in_grad[qconv::kWeight].get_with_shape<xpu, 3, DType>(
                      Shape3(group_, K, M), s);

              //============================================//
              //calc the scaling scalar for 1-bit mode.   
              //Note: gradient_update_mode "ff" don't need a scaling process, since
              //full precision weights in use.
              DType scaling_scalar_w;
              Tensor<xpu, 1, DType> w1d = in_data[qconv::kWeight].FlatTo1D<xpu, DType>(s);
              if(this->param_.act_bit == 1 
                  && this->param_.weight_bit == 1
                  && this->param_.scaling_mode.value() != qconv::scaling_none
                  && this->param_.gradient_update_mode.value() != qconv::ff){
                scaling_scalar_w = q_helper::get_scaling_scalar(w1d);
              }
              //============================================//  

              //========================================//
              // calculate gradients for binarized      //
              // or quantized weights, then later apply //
              // to original weights                    //
              // save here once, copy back later        //
              Tensor<xpu, 1, DType> w1d_copy;
              //use binary/quantized weights for gradient calc,
              if (this->param_.gradient_update_mode.value() == qconv::bf){  
                w1d_copy = mshadow::NewTensor<xpu>(w1d.shape_, DType(1.0), true, w1d.stream_);
                mshadow::Copy(w1d_copy, w1d, w1d.stream_);
                q_helper::quantize_weights(w1d, this->param_.weight_bit);
              }
              //                                        //
              //========================================//   

              //============================================//
              //calc the scaling scalar for 1-bit mode.        
               if(this->param_.act_bit == 1 
                && this->param_.weight_bit == 1
                && this->param_.scaling_mode.value() != qconv::scaling_none
                && this->param_.gradient_update_mode.value() != qconv::ff)
              {    
                q_helper::tensor_mul_scalar(w1d, scaling_scalar_w);              
              }
              //============================================//             

              for (index_t n = 0; n < num_; ++n) {
                Tensor<xpu, 3, DType> out_grad_3d = out_grad_4d[n];
                // gradient w.r.t. input data
                for (index_t g = 0; g < group_; ++g) {
                  col_buffer_3d[g] = dot(weight_3d[g].T(), out_grad_3d[g]);
                }
                col2im(s, col_buffer.dptr<DType>(), in_grad[qconv::kData].shape_, col_buffer.shape_,
                       param_.kernel, param_.pad, param_.stride, param_.dilate,
                       in_grad[qconv::kData].dptr<DType>()+n*input_dim_, req[qconv::kData]);

                // gradient w.r.t. weight, dWeight should accumulate across the batch and group
                im2col(s, in_data[qconv::kData].dptr<DType>()+n*input_dim_, in_data[qconv::kData].shape_,
                       col_buffer.shape_, param_.kernel, param_.pad, param_.stride, param_.dilate,
                       col_buffer.dptr<DType>());
                for (index_t g = 0; g < group_; ++g) {
                  if (0 == n) {
                    ASSIGN_DISPATCH(dweight_3d[g], req[qconv::kWeight],
                                    dot(out_grad_3d[g], col_buffer_3d[g].T()));
                  } else {
                    dweight_3d[g] += dot(out_grad_3d[g], col_buffer_3d[g].T());
                  }
                }
              }

              // gradient w.r.t bias
              if (bias_term_) {
                Tensor<xpu, 1, DType> dbias = in_grad[qconv::kBias].get<xpu, 1, DType>(s);
                Tensor<xpu, 3, DType> dout = out_grad[qconv::kOut].get_with_shape<xpu, 3, DType>(
                        Shape3(num_, conv_out_channels_, conv_out_spatial_dim_), s);
                ASSIGN_DISPATCH(dbias, req[qconv::kBias], sumall_except_dim<1>(dout));
              }

              //========================================//
              // gradient calculation done, swap back   //
              // weights and also free space            //
              if (param_.gradient_update_mode.value() == qconv::bf){   
                mshadow::Copy(w1d, w1d_copy, w1d_copy.stream_);
                mshadow::FreeSpace(&w1d_copy);
              }
              //                                        //
              //========================================//
            }

        private:
            void LayerSetUp(const TShape& ishape, const TShape& oshape) {
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
            QConvolutionParam param_;
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
        Operator* CreateOp(QConvolutionParam param, int dtype,
                           std::vector<TShape> *in_shape,
                           std::vector<TShape> *out_shape,
                           Context ctx);

#if DMLC_USE_CXX11
        class QConvolutionProp : public OperatorProperty {
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

            bool InferShape(std::vector<TShape> *in_shape,
                            std::vector<TShape> *out_shape,
                            std::vector<TShape> *aux_shape) const override {
              using namespace mshadow;
              if (!param_.no_bias) {
                LOG(WARNING) << "convolution with bias untested //mf";
                CHECK_EQ(in_shape->size(), 3U) << "Input:[data, weight, bias]";
              } else {
                CHECK_EQ(in_shape->size(), 2U) << "Input:[data, weight]";
              }
              // CHECK_EQ(out_shape->size(), 1) << "Output: [output]";
              out_shape->resize(1, TShape());
              const TShape &dshp = (*in_shape)[qconv::kData];
              if (dshp.ndim() ==  0) return false;

              if (param_.kernel.ndim() != 2) {
                LOG(FATAL) << "Unknown convolution type (only 2d binary convolution supported)";
                return false;
              } else {
                // 2d conv
                CHECK_EQ(dshp.ndim(), 4U) \
          << "Input data should be 4D in batch-num_filter-y-x";
                Shape<4> dshape = ConvertLayout(dshp.get<4>(), param_.layout.value(), kNCHW);

                if (param_.binarized_weights_only) {
                  CHECK_EQ(param_.num_group, 1) << "groups not (yet?) supported for pre-binarized weights";
                  Shape<1> wshape = Shape1(dshape[1] * param_.num_filter * param_.kernel[0] * param_.kernel[1] / mxnet::op::xnor_cpu::BITS_PER_BINARY_WORD);
                  SHAPE_ASSIGN_CHECK(*in_shape, qconv::kWeight, wshape);
                } else {
                  Shape<4> wshape = Shape4(param_.num_filter / param_.num_group,
                                           dshape[1] / param_.num_group,
                                           param_.kernel[0], param_.kernel[1]);
                  wshape = ConvertLayout(wshape, kNCHW, param_.layout.value());
                  wshape[0] *= param_.num_group;
                  SHAPE_ASSIGN_CHECK(*in_shape, qconv::kWeight, wshape);
                }

                if (!param_.no_bias) {
                  SHAPE_ASSIGN_CHECK(*in_shape, qconv::kBias, Shape1(param_.num_filter));
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
                SHAPE_ASSIGN_CHECK(*in_shape, qconv::kData,
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
                  if (param_.binarized_weights_only &&
                      (i == qconv::kWeight)) {
                    continue;
                  }
                  CHECK_EQ((*in_type)[i], dtype) << "This layer requires uniform type. "
                                                 << "Expected " << dtype << " v.s. given "
                                                 << (*in_type)[i] << " at " << ListArguments()[i];
                }
              }
              if (param_.binarized_weights_only) {
                (*in_type)[qconv::kWeight] = mxnet::op::xnor_cpu::corresponding_dtype();
              }
              out_type->clear();
              out_type->push_back(dtype);
              return true;
            }

            OperatorProperty* Copy() const override {
              auto ptr = new QConvolutionProp();
              ptr->param_ = param_;
              return ptr;
            }

            std::string TypeString() const override {
              return "QConvolution";
            }

            std::vector<int> DeclareBackwardDependency(
                    const std::vector<int> &out_grad,
                    const std::vector<int> &in_data,
                    const std::vector<int> &out_data) const override {
              return {out_grad[qconv::kOut], in_data[qconv::kData], in_data[qconv::kWeight]};
            }

            std::vector<ResourceRequest> ForwardResource(
                    const std::vector<TShape> &in_shape) const override {
              return {ResourceRequest::kTempSpace};
            }

            std::vector<ResourceRequest> BackwardResource(
                    const std::vector<TShape> &in_shape) const override {
              return {ResourceRequest::kTempSpace};
            }

            Operator* CreateOperator(Context ctx) const override {
              LOG(FATAL) << "Not Implemented.";
              return NULL;
            }

            Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                       std::vector<int> *in_type) const override;

        private:
            // Adds symmetric padding to a data input (in one dimension)
            index_t AddPad(index_t dsize, index_t pad) const {
              return dsize + 2 * pad;
            }

            QConvolutionParam param_;
        };  // class QConvolutionProp
#endif  // DMLC_USE_CXX11
    }  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_QCONVOLUTION_INL_H_
