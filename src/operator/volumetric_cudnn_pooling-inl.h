/*!
 * Copyright (c) 2015 by Contributors
 * \file cudnn_pooling-inl.h
 * \brief
 * \author Bing Xu
*/

#ifndef MXNET_OPERATOR_CUDNN_VOLUMETRIC_POOLING_INL_H_
#define MXNET_OPERATOR_CUDNN_VOLUMETRIC_POOLING_INL_H_

#include <algorithm>
#include <vector>
#include "./volumetric_pooling-inl.h"

namespace mxnet {
    namespace op {

        class CuDNNVolumetricPoolingOp : public Operator {
        public:
            explicit CuDNNVolumetricPoolingOp(VolumetricPoolingParam p) {
                param_ = p;
                init_cudnn_ = false;
                // TODO(xxx): fp16
                dtype_ = CUDNN_DATA_FLOAT;
                switch (param_.pool_type) {
                    case pool_enum::kMaxPooling:
                        mode_ = CUDNN_POOLING_MAX;
                        break;
                    case pool_enum::kAvgPooling:
                        mode_ = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
                        break;
                    default:
                        LOG(FATAL) << "Not implmented";
                }
            }

            ~CuDNNVolumetricPoolingOp() {
                CHECK_EQ(cudnnDestroyTensorDescriptor(in_desc_), CUDNN_STATUS_SUCCESS);
                CHECK_EQ(cudnnDestroyTensorDescriptor(out_desc_), CUDNN_STATUS_SUCCESS);
                CHECK_EQ(cudnnDestroyPoolingDescriptor(pooling_desc_), CUDNN_STATUS_SUCCESS);
            }

            virtual void Forward(const OpContext &ctx,
                                 const std::vector <TBlob> &in_data,
                                 const std::vector <OpReqType> &req,
                                 const std::vector <TBlob> &out_data,
                                 const std::vector <TBlob> &aux_args) {
                using namespace mshadow;
                using namespace mshadow::expr;
                CHECK_EQ(in_data.size(), 1);
                CHECK_EQ(out_data.size(), 1);
                Stream <gpu> *s = ctx.get_stream<gpu>();
                Tensor <gpu, 5> data = in_data[pool_enum::kData].get<gpu, 5, real_t>(s);
                Tensor <gpu, 5> out = out_data[pool_enum::kOut].get<gpu, 5, real_t>(s);
                CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
                if (!init_cudnn_) {
                    this->Init(s, in_data, out_data);
                }
                float alpha = 1.0f;
                float beta = 0.0f;
                CHECK_EQ(data.CheckContiguous(), true);
                CHECK_EQ(out.CheckContiguous(), true);
                CHECK_EQ(cudnnPoolingForward(s->dnn_handle_,
                                             pooling_desc_,
                                             &alpha,
                                             in_desc_,
                                             data.dptr_,
                                             &beta,
                                             out_desc_,
                                             out.dptr_), CUDNN_STATUS_SUCCESS);
            }

            virtual void Backward(const OpContext &ctx,
                                  const std::vector <TBlob> &out_grad,
                                  const std::vector <TBlob> &in_data,
                                  const std::vector <TBlob> &out_data,
                                  const std::vector <OpReqType> &req,
                                  const std::vector <TBlob> &in_grad,
                                  const std::vector <TBlob> &aux_args) {
                using namespace mshadow;
                using namespace mshadow::expr;
                CHECK_EQ(out_grad.size(), 1);
                CHECK_EQ(in_data.size(), 1);
                CHECK_EQ(out_data.size(), 1);
                CHECK_EQ(req.size(), 1);
                CHECK_EQ(in_grad.size(), 1);

                Stream <gpu> *s = ctx.get_stream<gpu>();
                Tensor <gpu, 5> m_out_grad = out_grad[pool_enum::kOut].get<gpu, 5, real_t>(s);
                Tensor <gpu, 5> m_in_data = in_data[pool_enum::kData].get<gpu, 5, real_t>(s);
                Tensor <gpu, 5> m_out_data = out_data[pool_enum::kOut].get<gpu, 5, real_t>(s);
                Tensor <gpu, 5> m_in_grad = in_grad[pool_enum::kData].get<gpu, 5, real_t>(s);
                CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
                float alpha = 1.0f;
                float beta = 0.0f;
                CHECK_EQ(cudnnPoolingBackward(s->dnn_handle_,
                                              pooling_desc_,
                                              &alpha,
                                              out_desc_,
                                              m_out_data.dptr_,
                                              out_desc_,
                                              m_out_grad.dptr_,
                                              in_desc_,
                                              m_in_data.dptr_,
                                              &beta,
                                              in_desc_,
                                              m_in_grad.dptr_), CUDNN_STATUS_SUCCESS);
            }

        private:
            inline void Init(mshadow::Stream <gpu> *s,
                             const std::vector <TBlob> &in_data,
                             const std::vector <TBlob> &out_data) {
                using namespace mshadow;
                CHECK_EQ(in_data.size(), 1);
                CHECK_EQ(out_data.size(), 1);
                if (!init_cudnn_) {
                    init_cudnn_ = true;
                    Tensor <gpu, 5> data = in_data[pool_enum::kData].get<gpu, 5, real_t>(s);
                    Tensor <gpu, 5> out = out_data[pool_enum::kOut].get<gpu, 5, real_t>(s);
                    CHECK_EQ(cudnnCreatePoolingDescriptor(&pooling_desc_), CUDNN_STATUS_SUCCESS);
                    CHECK_EQ(cudnnCreateTensorDescriptor(&in_desc_), CUDNN_STATUS_SUCCESS);
                    CHECK_EQ(cudnnCreateTensorDescriptor(&out_desc_), CUDNN_STATUS_SUCCESS);

                    int inDimA[] = {(int) data.shape_[0], (int) data.shape_[1], (int) data.shape_[2], (int) data.shape_[3], (int) data.shape_[4]};
                    int inStrideA[] = {
                            (int) (data.shape_[1] * data.shape_[2] * data.shape_[3] * data.shape_[4]),
                            (int) (data.shape_[2] * data.shape_[3] * data.shape_[4]),
                            (int) (data.shape_[3] * data.shape_[4]),
                            (int) data.shape_[4],
                            1
                    };
                    CHECK_EQ(cudnnSetTensorNdDescriptor(in_desc_, dtype_, 5, inDimA, inStrideA), CUDNN_STATUS_SUCCESS);

                    int outDimA[] = {(int) out.shape_[0], (int) out.shape_[1], (int) out.shape_[2], (int) out.shape_[3], (int) out.shape_[4]};
                    int outStrideA[] = {
                            (int) (out.shape_[1] * out.shape_[2] * out.shape_[3] * out.shape_[4]),
                            (int) (out.shape_[2] * out.shape_[3] * out.shape_[4]),
                            (int) (out.shape_[3] * out.shape_[4]),
                            (int) out.shape_[4],
                            1
                    };
                    CHECK_EQ(cudnnSetTensorNdDescriptor(out_desc_, dtype_, 5, outDimA, outStrideA), CUDNN_STATUS_SUCCESS);

                    int windowDimA[] = {(int) param_.kernel[0], (int) param_.kernel[1], (int) param_.kernel[2]};
                    int paddingA[] = {(int) param_.pad[0], (int) param_.pad[1], (int) param_.pad[2]};
                    int strideA[] = {(int) param_.stride[0], (int) param_.stride[1], (int) param_.stride[2]};
                    CHECK_EQ(cudnnSetPoolingNdDescriptor(pooling_desc_,
                                                         mode_,
                                                         3,
                                                         windowDimA,
                                                         paddingA,
                                                         strideA), CUDNN_STATUS_SUCCESS);
                }
            }

            bool init_cudnn_;
            cudnnDataType_t dtype_;
            cudnnHandle_t handle_;
            cudnnPoolingMode_t mode_;
            cudnnTensorDescriptor_t in_desc_;
            cudnnTensorDescriptor_t out_desc_;
            cudnnPoolingDescriptor_t pooling_desc_;
            VolumetricPoolingParam param_;
        };  // class CuDNNPoolingOp
    }  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CUDNN_POOLING_INL_H_

