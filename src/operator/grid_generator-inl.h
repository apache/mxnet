/*!
 * Copyright (c) 2017 by Contributors
 * \file grid_generator-inl.h
 * \brief 
 * The operator generate sampling grid
 * \author Xu Dong
*/
#ifndef MXNET_OPERATOR_GRID_GENERATOR_INL_H_
#define MXNET_OPERATOR_GRID_GENERATOR_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <vector>
#include <map>
#include <utility>
#include <string>
#include "./operator_common.h"

namespace mxnet {
namespace op {

namespace grid {
enum GridGeneratorOpInputs {kData};
enum GridGeneratorOpOutputs {kOut, kGridDst, kTmp};
enum GridGeneratorOpResource {kTempSpace};
enum GridGeneratorTransformType {kAffine, kWarp};
}

struct GridGeneratorParam : public dmlc::Parameter<GridGeneratorParam> {
  int transform_type;
  TShape target_shape;
  DMLC_DECLARE_PARAMETER(GridGeneratorParam) {
    int shape[] = {0, 0};
    DMLC_DECLARE_FIELD(transform_type)
    .add_enum("affine", grid::kAffine)
    .add_enum("warp", grid::kWarp)
    .describe("transformation type\n    "
              "if transformation type is affine, data is affine matrix : (batch, 6)\n    "
              "if transformation type is warp, data is optical flow : (batch, 2, h, w)");
    DMLC_DECLARE_FIELD(target_shape).set_default(TShape(shape, shape + 2))
    .describe("if transformation type is affine, the operator need a target_shape : (H, W)\n    "
              "if transofrmation type is warp, the operator will ignore target_shape");
  }
};

template<typename xpu, typename DType>
class GridGeneratorOp : public Operator {
 public:
  explicit GridGeneratorOp(GridGeneratorParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 3);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    switch (param_.transform_type) {
      case grid::kAffine: {
        // if transform_type is affine, data is affine matrix, input shape : (batch, 2, 3)
        Tensor<xpu, 4, DType> out = out_data[grid::kOut].get<xpu, 4, DType>(s);
        Tensor<xpu, 2, DType> grid_dst = out_data[grid::kGridDst].get<xpu, 2, DType>(s);
        Shape<3> data_shape = Shape3(out.size(0), 2, 3);
        Tensor<xpu, 3, DType> data = in_data[grid::kData]
          .get_with_shape<xpu, 3, DType>(data_shape, s);
        Tensor<cpu, 2, DType> workspace =
              ctx.requested[grid::kTempSpace].get_host_space_typed<2, DType>(
              grid_dst.shape_);
        Tensor<xpu, 3, DType> tmp = out_data[grid::kTmp].get<xpu, 3, DType>(s);
        for (index_t i = 1; i <= workspace.size(1); i++) {
          // grid dst coordinate is (x, y, 1)
          workspace[0][i-1] = -1.0 + (i-1) % param_.target_shape[1] * 2.0 /
                              (param_.target_shape[1] - 1);
          workspace[1][i-1] = -1.0 + (i-1) / param_.target_shape[1] * 2.0 /
                              (param_.target_shape[0] - 1);
          workspace[2][i-1] = 1.0;
        }
        // create a sampling grid, tmp : (batch, 2, H x W)
        Copy(grid_dst, workspace, grid_dst.stream_);
        for (index_t batch = 0; batch < data.size(0); batch++) {
              tmp[batch] = dot(data[batch], grid_dst);
        }
        // out : (batch, 2, H, W)
        out = reshape(tmp, out.shape_);
        break;
      }
      // Warping transformation
      case grid::kWarp: {
        // if transform_type is warp, data is optical flow, input shape : (batch, 2, height, width)
        // grid_src = grid_dst + optical flow
        Tensor<xpu, 4, DType> data = in_data[grid::kData].get<xpu, 4, DType>(s);
        Tensor<xpu, 4, DType> out = out_data[grid::kOut].get<xpu, 4, DType>(s);
        // grid_dst : (2, H, W) similar to the results of numpy.meshgrid
        Tensor<xpu, 3, DType> grid_dst = out_data[grid::kGridDst].get<xpu, 3, DType>(s);
        Tensor<xpu, 1, DType> tmp = out_data[grid::kTmp].get<xpu, 1, DType>(s);
        Tensor<xpu, 1, DType> workspace =
              ctx.requested[grid::kTempSpace]
                .get_space_typed<xpu, 1, DType>(Shape1(data.size(3)), s);
        workspace = range<DType>(0, data.size(3));
        grid_dst[0] = repmat(workspace, data.size(2));
        tmp = range<DType>(0, data.size(2), 1, data.size(3));
        grid_dst[1] = reshape(tmp, Shape2(data.size(2), data.size(3)));
        for (index_t batch = 0; batch < data.size(0); batch++) {
          // bilinear sampling op assume that grid has been nomalized
          out[batch][0] = (data[batch][0] + grid_dst[0]) /
            scalar<DType>((data.size(3) - 1) / 2.0) - scalar<DType>(1);
          out[batch][1] = (data[batch][1] + grid_dst[1]) /
            scalar<DType>((data.size(2) - 1) / 2.0) - scalar<DType>(1);
        }
        break;
      }
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 3);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    switch (param_.transform_type) {
      case grid::kAffine: {
        Tensor<xpu, 4, DType> grad = out_grad[grid::kOut].get<xpu, 4, DType>(s);
        Tensor<xpu, 2, DType> grid_dst = out_data[grid::kGridDst].get<xpu, 2, DType>(s);
        Shape<3> data_shape = Shape3(grad.size(0), 2, 3);
        Tensor<xpu, 3, DType> gdata = in_grad[grid::kData]
        .get_with_shape<xpu, 3, DType>(data_shape, s);
        Tensor<xpu, 3, DType> tmp = out_data[grid::kTmp].get<xpu, 3, DType>(s);
        tmp = reshape(grad, tmp.shape_);
        for (index_t batch = 0; batch < gdata.size(0); batch++) {
            // tmp : (2, H X W)  grid_dst.T : (H x W, 3)
            gdata[batch] = dot(tmp[batch], grid_dst.T());
        }
        break;
      }
      case grid::kWarp: {
        Tensor<xpu, 4, DType> grad = out_grad[grid::kOut].get<xpu, 4, DType>(s);
        Tensor<xpu, 4, DType> gdata = in_grad[grid::kData].get<xpu, 4, DType>(s);
        for (index_t batch = 0; batch < gdata.size(0); batch++) {
            gdata[batch][0] = grad[batch][0] / scalar<DType>((gdata.size(3)-1) / 2.0);
            gdata[batch][1] = grad[batch][1] / scalar<DType>((gdata.size(2)-1) / 2.0);
        }
        break;
      }
    }
  }

 private:
  GridGeneratorParam param_;
};  // class GridGeneratorOp

template<typename xpu>
Operator* CreateOp(GridGeneratorParam param, int dtype);

#if DMLC_USE_CXX11
class GridGeneratorProp : public OperatorProperty {
 public:
  int NumVisibleOutputs() const override {
    return 1;
  }

  int NumOutputs() const override {
    return 3;
  }

  std::vector<std::string> ListArguments() const override {
    return {"data"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "grid_dst", "tmp"};
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 1) << "Input:[data]";
    const TShape &lshape = (*in_shape)[grid::kData];
    if (lshape.ndim() ==  0) return false;
    out_shape->clear();
    switch (param_.transform_type) {
      case grid::kAffine: {
        CHECK_EQ(lshape.ndim(), 2) \
          << "if transform_type is affine, data is affine matrix"
          "affine matrix should be 2D in batch-num_hidden";
        CHECK_EQ(lshape[1], 6) << "incorrect data shape[1], should be 6";
        CHECK_GT(param_.target_shape[0], 0) \
            << "incorrect target_shape: " << param_.target_shape[0];
        CHECK_GT(param_.target_shape[1], 0) \
            << "incorrect target_shape: " << param_.target_shape[1];
        out_shape->push_back(Shape4(lshape[0], 2, param_.target_shape[0], param_.target_shape[1]));
        out_shape->push_back(Shape2(3, param_.target_shape[0] * param_.target_shape[1]));
        out_shape->push_back(Shape3(lshape[0], 2, param_.target_shape[0] * param_.target_shape[1]));
        break;
      }
      case grid::kWarp: {
        CHECK_EQ(lshape.ndim(), 4) \
          << "if transform_type is warp, data is optical flow"
             "optical flow should be 4D in batch-num_hidden-y-x";
        CHECK_EQ(lshape[1], 2) << "incorrect data shape[1], should be 2";
        out_shape->push_back(lshape);
        out_shape->push_back(Shape3(2, lshape[2], lshape[3]));
        out_shape->push_back(Shape1(lshape[2]*lshape[3]));
        break;
      }
    }
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                   std::vector<int> *out_type,
                   std::vector<int> *aux_type) const override {
      int dtype = -1;
      for (size_t i = 0; i < in_type->size(); ++i) {
        if (dtype == -1) {
          dtype = in_type->at(i);
        } else {
          CHECK(in_type->at(i) == dtype ||
                in_type->at(i) == -1) <<
                "Non-uniform data type in GridGenerator";
        }
      }
      if (dtype == -1) {
        LOG(FATAL) << "Not enough information to infer type in GridGenerator.";
        return false;
      }
      size_t nin = this->ListArguments().size();
      in_type->clear();
      for (size_t i = 0; i < nin; ++i) in_type->push_back(dtype);
      size_t naux = this->ListAuxiliaryStates().size();
      aux_type->clear();
      for (size_t i = 0; i < naux; ++i) aux_type->push_back(dtype);
      size_t nout = this->ListOutputs().size();
      out_type->clear();
      for (size_t i = 0; i < nout; ++i) out_type->push_back(dtype);
      return true;
    }

  OperatorProperty* Copy() const override {
    auto ptr = new GridGeneratorProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "GridGenerator";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    switch (param_.transform_type) {
      case grid::kAffine: {
        return {out_grad[grid::kOut],
                out_data[grid::kGridDst],
                out_data[grid::kTmp]};
      }
      case grid::kWarp: {
        return {out_grad[grid::kOut]};
      }
    }
    return {};
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    switch (param_.transform_type) {
      case grid::kAffine: {
        return {ResourceRequest::kTempSpace};
      }
      case grid::kWarp: {
        return {ResourceRequest::kTempSpace};
      }
    }
    return {};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  GridGeneratorParam param_;
};  // class GridGeneratorProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_GRID_GENERATOR_INL_H_
