/*!
 * Copyright (c) 2017 Microsoft
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file channel_operator-inl.h
 * \brief 
 * \author Haozhi Qi, Yi Li, Guodong Zhang, Jifeng Dai
*/
#ifndef MXNET_OPERATOR_CONTRIB_CHANNEL_OPERATOR_INL_H_
#define MXNET_OPERATOR_CONTRIB_CHANNEL_OPERATOR_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "../operator_common.h"

namespace mxnet {
  namespace op {
    // Declare enumeration of input order to make code more intuitive.
    // // These enums are only visible within this header
    namespace channel_operator {
      enum ChannelOperatorOpInputs { kData, kPick };
      enum ChannelOperatorOpOutputs { kOut, kTemp };
      enum ChannelOperatorOpType { kGroupMax, kGroupPick, kGroupSoftmax };
      enum ChannelOperatorOpPickType { kLabel, kScore };
    }  // activation

    struct ChannelOperatorParam : public dmlc::Parameter<ChannelOperatorParam> {
      // use int for enumeration
      int op_type;
      int group;
      int pick_type;
      DMLC_DECLARE_PARAMETER(ChannelOperatorParam) {
        DMLC_DECLARE_FIELD(op_type)
          .add_enum("Group_Max", channel_operator::kGroupMax)
          .add_enum("Group_Pick", channel_operator::kGroupPick)
          .add_enum("Group_Softmax", channel_operator::kGroupSoftmax)
          .describe("Channel operator to be applied.");
        DMLC_DECLARE_FIELD(group).describe("group size");
        DMLC_DECLARE_FIELD(pick_type)
          .add_enum("Label_Pick", channel_operator::kLabel)
          .add_enum("Score_Pick", channel_operator::kScore)
          .set_default(channel_operator::kLabel)
          .describe("pick type");
      }
    };

    /**
    * \brief This is the implementation of channel operator.
    * \tparam xpu The device that the op will be executed on.
    */
    template<typename xpu, typename DType>
    class ChannelOperatorOp : public Operator {
    public:
      explicit ChannelOperatorOp(ChannelOperatorParam p) {
        this->param_ = p;
      }

      virtual void Forward(const OpContext &ctx,
        const std::vector<TBlob> &in_data,
        const std::vector<OpReqType> &req,
        const std::vector<TBlob> &out_data,
        const std::vector<TBlob> &aux_args) {
        using namespace mshadow;
        using namespace mshadow::expr;
        size_t in_expected;
        size_t out_expected;
        if (param_.op_type == channel_operator::kGroupMax) {
          in_expected = 1;
          out_expected = 2;
        }
        else if (param_.op_type == channel_operator::kGroupSoftmax) {
          in_expected = 1;
          out_expected = 1;
        }
        else if (param_.op_type == channel_operator::kGroupPick) {
          in_expected = 2;
          if (param_.pick_type == channel_operator::kLabel)
            out_expected = 1;
          else
            out_expected = 2;
        }
        else {
          LOG(FATAL) << "No that operation type.";
        }
        CHECK_EQ(in_data.size(), in_expected);
        CHECK_EQ(out_data.size(), out_expected);
        Stream<xpu> *s = ctx.get_stream<xpu>();
        if (param_.op_type == channel_operator::kGroupSoftmax) {
          int total_size = in_data[channel_operator::kData].Size();
          int batch_size = in_data[channel_operator::kData].shape_[0];
          int channel_num = in_data[channel_operator::kData].shape_[1];
          int rest_size = total_size / (batch_size * channel_num);
          const Shape<3> data_shape = Shape3(batch_size*param_.group, channel_num / param_.group, rest_size);

          Tensor<xpu, 3, DType> data = in_data[channel_operator::kData].get_with_shape<xpu, 3, DType>(data_shape, s);
          Tensor<xpu, 3, DType> out = out_data[channel_operator::kOut].get_with_shape<xpu, 3, DType>(data_shape, s);
          Softmax(out, data);
        }
        else if (param_.op_type == channel_operator::kGroupMax) {
          Tensor<xpu, 4, DType> data = in_data[channel_operator::kData].get<xpu, 4, DType>(s);
          Tensor<xpu, 4, DType> out = out_data[channel_operator::kOut].get<xpu, 4, DType>(s);
          Tensor<xpu, 4, DType> max_idx = out_data[channel_operator::kTemp].get<xpu, 4, DType>(s);
          CHECK_EQ(data.CheckContiguous(), true);
          CHECK_EQ(out.CheckContiguous(), true);
          CHECK_EQ(max_idx.CheckContiguous(), true);

          GroupMaxForward(out, data, max_idx, param_.group);
        }
        else if (param_.op_type == channel_operator::kGroupPick) {
          Tensor<xpu, 4, DType> data = in_data[channel_operator::kData].get<xpu, 4, DType>(s);
          Tensor<xpu, 4, DType> out = out_data[channel_operator::kOut].get<xpu, 4, DType>(s);
          Tensor<xpu, 4, DType> pick_idx = in_data[channel_operator::kPick].get<xpu, 4, DType>(s);
          CHECK_EQ(data.CheckContiguous(), true);
          CHECK_EQ(out.CheckContiguous(), true);
          CHECK_EQ(pick_idx.CheckContiguous(), true);

          if (param_.pick_type == channel_operator::kScore) {
            Tensor<xpu, 4, DType> argmax_data = out_data[channel_operator::kTemp].get<xpu, 4, DType>(s);
            GetMaxIdx(pick_idx, argmax_data, param_.group);
            GroupPickForward(out, data, argmax_data, param_.group);
          }
          else {
            GroupPickForward(out, data, pick_idx, param_.group);
          }
        }
        else {
          LOG(FATAL) << "No that operation type.";
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

        size_t in_expected;
        size_t out_expected;
        if (param_.op_type == channel_operator::kGroupMax) {
          in_expected = 1;
          out_expected = 2;
        }
        else if (param_.op_type == channel_operator::kGroupSoftmax) {
          in_expected = 1;
          out_expected = 1;
        }
        else if (param_.op_type == channel_operator::kGroupPick) {
          in_expected = 2;
          if (param_.pick_type == channel_operator::kLabel)
            out_expected = 1;
          else
            out_expected = 2;
        }
        else {
          LOG(FATAL) << "No that operation type.";
        }
        CHECK_EQ(in_data.size(), in_expected);
        CHECK_EQ(out_data.size(), out_expected);
        Stream<xpu> *s = ctx.get_stream<xpu>();

        if (param_.op_type == channel_operator::kGroupMax) {
          Tensor<xpu, 4, DType> grad_out = out_grad[channel_operator::kOut].get<xpu, 4, DType>(s);
          Tensor<xpu, 4, DType> max_idx = out_data[channel_operator::kTemp].get<xpu, 4, DType>(s);
          Tensor<xpu, 4, DType> grad_in = in_grad[channel_operator::kData].get<xpu, 4, DType>(s);

          CHECK_EQ(grad_out.CheckContiguous(), true);
          CHECK_EQ(max_idx.CheckContiguous(), true);
          CHECK_EQ(grad_in.CheckContiguous(), true);

          Assign(grad_in, req[channel_operator::kData], 0);
          GroupMaxBackwardAcc(grad_in, grad_out, max_idx, param_.group);
        }
        else if (param_.op_type == channel_operator::kGroupSoftmax) {
          LOG(FATAL) << "Not Implemented.";
        }
        else if (param_.op_type == channel_operator::kGroupPick) {
          Tensor<xpu, 4, DType> grad_out = out_grad[channel_operator::kOut].get<xpu, 4, DType>(s);
          Tensor<xpu, 4, DType> pick_idx = in_data[channel_operator::kPick].get<xpu, 4, DType>(s);
          Tensor<xpu, 4, DType> grad_in = in_grad[channel_operator::kData].get<xpu, 4, DType>(s);
          Tensor<xpu, 4, DType> pick_diff = in_grad[channel_operator::kPick].get<xpu, 4, DType>(s);

          CHECK_EQ(grad_out.CheckContiguous(), true);
          CHECK_EQ(pick_idx.CheckContiguous(), true);
          CHECK_EQ(grad_in.CheckContiguous(), true);

          Assign(grad_in, req[channel_operator::kData], 0);
          Assign(pick_diff, req[channel_operator::kPick], 0);
          if (param_.pick_type == channel_operator::kScore) {
            LOG(FATAL) << "Not Implemented.";
          }
          else {
            GroupPickBackwardAcc(grad_in, grad_out, pick_idx, param_.group);
          }

        }
        else {
          LOG(FATAL) << "No that operation type.";
        }

      }
    private:
      ChannelOperatorParam param_;
    };  // class ChannelOperatorOp

        // Decalre Factory function, used for dispatch specialization
    template<typename xpu>
    Operator* CreateOp(ChannelOperatorParam type, int dtype);

#if DMLC_USE_CXX11
    class ChannelOperatorProp : public OperatorProperty {
    public:
      std::vector<std::string> ListArguments() const override {
        if (param_.op_type == channel_operator::kGroupMax ||
          param_.op_type == channel_operator::kGroupSoftmax) {
          return{ "data" };
        }
        else if (param_.op_type == channel_operator::kGroupPick) {
          return{ "data", "pick_idx" };
        }
        else {
          LOG(FATAL) << "No that operation type.";
          return{};
        }
      }

      std::vector<std::string> ListOutputs() const override {
        if (param_.op_type == channel_operator::kGroupSoftmax) {
          return{ "output" };
        }
        else if (param_.op_type == channel_operator::kGroupPick) {
          if (param_.pick_type == channel_operator::kLabel)
            return{ "output" };
          else
            return{ "output", "argmax_data" };
        }
        else if (param_.op_type == channel_operator::kGroupMax) {
          return{ "output", "max_idx" };
        }
        else {
          LOG(FATAL) << "No that operation type.";
          return{};
        }
      }

      int NumOutputs() const override {
        if (param_.op_type == channel_operator::kGroupSoftmax) {
          return 1;
        }
        else if (param_.op_type == channel_operator::kGroupPick) {
          if (param_.pick_type == channel_operator::kLabel)
            return 1;
          else
            return 2;
        }
        else if (param_.op_type == channel_operator::kGroupMax) {
          return 2;
        }
        else {
          LOG(FATAL) << "No that operation type.";
          return NULL;
        }
      }

      int NumVisibleOutputs() const override {
        return 1;
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


        // data: [batch_size, c, h, w]
        TShape dshape = in_shape->at(channel_operator::kData);
        CHECK_EQ(dshape.ndim(), 4) << "data should be a 4D tensor";

        if (param_.op_type == channel_operator::kGroupMax) {
          CHECK_EQ(in_shape->size(), 1) << "Input:[data]";
          // out: [num_rois, group, h, w]
          out_shape->clear();
          out_shape->push_back(
            Shape4(dshape[0], param_.group, dshape[2], dshape[3]));
          out_shape->push_back(
            Shape4(dshape[0], param_.group, dshape[2], dshape[3]));
          return true;
        }
        else if (param_.op_type == channel_operator::kGroupSoftmax) {
          CHECK_EQ(in_shape->size(), 1) << "Input:[data]";
          // out: [num_rois, c, h, w]
          out_shape->clear();
          out_shape->push_back(
            Shape4(dshape[0], dshape[1], dshape[2], dshape[3]));
          return true;
        }
        else if (param_.op_type == channel_operator::kGroupPick) {
          CHECK_EQ(in_shape->size(), 2) << "Input:[data, pick_idx]";
          // out: [num_rois, c/group, h, w]
          out_shape->clear();
          out_shape->push_back(
            Shape4(dshape[0], dshape[1] / param_.group, dshape[2], dshape[3]));
          if (param_.pick_type == channel_operator::kScore) {
            out_shape->push_back(
              Shape4(dshape[0], 1, 1, 1));
          }
          return true;
        }
        else {
          LOG(FATAL) << "No that operation type.";
          return false;
        }

      }

      bool InferType(std::vector<int> *in_type,
        std::vector<int> *out_type,
        std::vector<int> *aux_type) const override {
        int dtype = (*in_type)[0];
        CHECK_NE(dtype, -1) << "Input must have specified type";

        if (param_.op_type == channel_operator::kGroupMax) {
          CHECK_EQ(in_type->size(), 1);
          out_type->clear();
          out_type->push_back(dtype);
          out_type->push_back(dtype);
          return true;
        }
        else if (param_.op_type == channel_operator::kGroupSoftmax) {
          CHECK_EQ(in_type->size(), 1);
          out_type->clear();
          out_type->push_back(dtype);
          return true;
        }
        else if (param_.op_type == channel_operator::kGroupPick) {
          CHECK_EQ(in_type->size(), 2);
          out_type->clear();
          out_type->push_back(dtype);
          if (param_.pick_type == channel_operator::kScore) {
            out_type->push_back(dtype);
          }
          return true;
        }
        else {
          LOG(FATAL) << "No that operation type.";
          return false;
        }

      }

      OperatorProperty* Copy() const override {
        ChannelOperatorProp* channel_operator_sym = new ChannelOperatorProp();
        channel_operator_sym->param_ = this->param_;
        return channel_operator_sym;
      }

      std::string TypeString() const override {
        return "_contrib_ChannelOperator";
      }

      // decalre dependency and inplace optimization options
      std::vector<int> DeclareBackwardDependency(
        const std::vector<int> &out_grad,
        const std::vector<int> &in_data,
        const std::vector<int> &out_data) const override {
        if (param_.op_type == channel_operator::kGroupMax) {
          return{ out_grad[channel_operator::kOut],
            out_data[channel_operator::kTemp] };
        }
        else if (param_.op_type == channel_operator::kGroupSoftmax) {
          return{ out_grad[channel_operator::kOut],
            out_data[channel_operator::kOut] };
        }
        else if (param_.op_type == channel_operator::kGroupPick) {
          return{ out_grad[channel_operator::kOut],
            in_data[channel_operator::kPick] };
        }
        else {
          LOG(FATAL) << "No that operation type.";
          return{};
        }
      }


      Operator* CreateOperator(Context ctx) const override {
        LOG(FATAL) << "Not Implemented.";
        return NULL;
      }

      Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
        std::vector<int> *in_type) const override;


    private:
      ChannelOperatorParam param_;
    };  // class PSROIPoolingAlignProp
#endif
  }  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_CHANNEL_OPERATOR_INL_H_