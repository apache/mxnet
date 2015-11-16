/*!
 * Copyright (c) 2015 by Contributors
 * \file anchor_multibbs_regressioncost-inl.h
 * \brief
 * \author Ming Zhang
*/
#ifndef MXNET_OPERATOR_ANCHOR_MULTIBBS_REGRESSIONCOST_INL_H_
#define MXNET_OPERATOR_ANCHOR_MULTIBBS_REGRESSIONCOST_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"

#define MIN_NUM 1e-37f

namespace mxnet {
namespace op {

namespace anchor_multibbs_regcost_enum {
enum AnchorRegCostOpInputs {kData, kLabel, kCoordLabel, kBBsLabel, kAnchorInfoLabel};
enum AnchorRegCostOpOutputs {kOut};
enum AnchorRegCostOpResource {kTempSpace};
};


struct AnchorMultiBBsRegCostParam : public dmlc::Parameter<AnchorMultiBBsRegCostParam> {
  // use int for enumeration
  uint32_t anchornum;
  DMLC_DECLARE_PARAMETER(AnchorMultiBBsRegCostParam) {
    DMLC_DECLARE_FIELD(anchornum)
    .set_default(0)
    .describe("The Anchor Number.");
  }
};


template<typename xpu>
class AnchorMultiBBsRegCostOp : public Operator {
 public:
  explicit AnchorMultiBBsRegCostOp(AnchorMultiBBsRegCostParam p) {
    CHECK_NE(p.anchornum, 0) << "anchornum can not be equal 0.";
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    
    index_t anchornum = param_.anchornum;
    
    TBlob data_in = in_data[anchor_multibbs_regcost_enum::kData];
    TBlob label = in_data[anchor_multibbs_regcost_enum::kLabel];
    TBlob coordlabel = in_data[anchor_multibbs_regcost_enum::kCoordLabel];
    TBlob bbslabel = in_data[anchor_multibbs_regcost_enum::kBBsLabel];
    TBlob infolabel = in_data[anchor_multibbs_regcost_enum::kAnchorInfoLabel];
    TBlob data_out = out_data[anchor_multibbs_regcost_enum::kOut];

    TShape shape_in = data_in.shape_;

    Tensor<xpu, 4> tdata_in = data_in.get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> tlabel = label.get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> tcoordlabel = coordlabel.get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> tbbslabel = bbslabel.get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> tinfolabel = infolabel.get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> tdata_out = data_out.get<xpu, 4, real_t>(s);

#if 1
    for (index_t bi = 0; bi < shape_in[0]; bi++) {
      Tensor<xpu, 3> coords = tcoordlabel[bi];
      for (index_t ai = 0; ai < anchornum; ai++) {
        Tensor<xpu, 2> onelabel = tlabel[bi][ai];
        Tensor<xpu, 3> onebb = tbbslabel[bi].Slice(ai * 4, (ai + 1) * 4);
        Tensor<xpu, 3> onedatas = tdata_in[bi].Slice(ai * 4, (ai + 1) * 4);
        Tensor<xpu, 3> oneouts = tdata_out[bi].Slice(ai * 4, (ai + 1) * 4);
        Tensor<xpu, 3> oneinfo = tinfolabel[ai];

        for (index_t di = 0; di < 2; di++) {
          Tensor<xpu, 2> onedata = onedatas[di];
          Tensor<xpu, 2> onecoord = coords[di];
          oneouts[di] = F<mshadow_op::smooth_l1>(onedata -
                       ((onebb[di] - onecoord) / oneinfo[di])) * onelabel;
       //   printf("yx %f, %f\n", onebb[di][45/2][70/2], oneinfo[di][45/2][70/2]);
        }
        Tensor<xpu, 3> onedata2 = onedatas.Slice(2, 4);
        Tensor<xpu, 3> partbb = onebb.Slice(2, 4);
        for (index_t di = 0; di < 2; di++) {
          oneouts[di + 2] = F<mshadow_op::smooth_l1>(onedata2[di] -
                        (F<mshadow_op::log>(partbb[di] / oneinfo[di] + MIN_NUM))) * onelabel;
       //   printf("hw %f, %f\n", partbb[di][45/2][70/2], oneinfo[di][45/2][70/2]);
        }
      }
    }
#endif    
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
    Stream<xpu> *s = ctx.get_stream<xpu>();

    index_t anchornum = param_.anchornum;

    TBlob data_in = in_data[anchor_multibbs_regcost_enum::kData];
    TBlob label = in_data[anchor_multibbs_regcost_enum::kLabel];
    TBlob coordlabel = in_data[anchor_multibbs_regcost_enum::kCoordLabel];
    TBlob bbslabel = in_data[anchor_multibbs_regcost_enum::kBBsLabel];
    TBlob infolabel = in_data[anchor_multibbs_regcost_enum::kAnchorInfoLabel];
    TBlob grad_in = in_grad[anchor_multibbs_regcost_enum::kOut];

    TShape shape_in = data_in.shape_;

    Tensor<xpu, 4> tdata_in = data_in.get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> tlabel = label.get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> tcoordlabel = coordlabel.get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> tbbslabel = bbslabel.get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> tinfolabel = infolabel.get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> tgrad_in = grad_in.get<xpu, 4, real_t>(s);
#if 1
//#define DIM1 21
//#define DIM2 32
    for (index_t bi = 0; bi < shape_in[0]; bi++) {
      Tensor<xpu, 3> coords = tcoordlabel[bi];
      for (index_t ai = 0; ai < anchornum; ai++) {
        Tensor<xpu, 2> onelabel = tlabel[bi][ai];
        Tensor<xpu, 3> onebb = tbbslabel[bi].Slice(ai * 4, (ai + 1) * 4);
        Tensor<xpu, 3> onedatas = tdata_in[bi].Slice(ai * 4, (ai + 1) * 4);
        Tensor<xpu, 3> onegrads = tgrad_in[bi].Slice(ai * 4, (ai + 1) * 4);
        Tensor<xpu, 3> oneinfo = tinfolabel[ai];
        for (index_t di = 0; di < 2; di++) {
          Tensor<xpu, 2> onedata = onedatas[di];
          Tensor<xpu, 2> onecoord = coords[di];
          onegrads[di] = F<mshadow_op::smooth_l1_grad>(onedata -
                       ((onebb[di] - onecoord) / oneinfo[di])) * onelabel;
//          printf("yx g:%f, o:%f, yx:%f, ayx:%f, ahw:%f, l:%f\n", onegrads[di][DIM1][DIM2], onedata[DIM1][DIM2], onebb[di][DIM1][DIM2], onecoord[DIM1][DIM2], oneinfo[di][DIM1][DIM2], onelabel[DIM1][DIM2]);
        }
        Tensor<xpu, 3> onedata2 = onedatas.Slice(2, 4);
        Tensor<xpu, 3> partbb = onebb.Slice(2, 4);
        for (index_t di = 0; di < 2; di++) {
          onegrads[di + 2] = F<mshadow_op::smooth_l1_grad>(onedata2[di] -
                        (F<mshadow_op::log>(partbb[di] / oneinfo[di] + MIN_NUM))) * onelabel;
//          printf("hw g:%f, o:%f, hw:%f, ahw:%f, l:%f\n", onegrads[di + 2][DIM1][DIM2], onedata2[di][DIM1][DIM2], partbb[di][DIM1][DIM2], oneinfo[di][DIM1][DIM2], onelabel[DIM1][DIM2]);
        }
      }
    }
#endif
#if 0
    tgrad_in = tdata_out + 600.0f;
    TShape gshape = in_grad[anchor_regcost_enum::kOut].shape_;
    if (gshape[1] == 4 && gshape[2] == 43) {
    std::cout << "in_grad => " << gshape[0] << ", " << gshape[1] << ", " << gshape[2] << ", " << gshape[3] << " ==> \n";
    Tensor<xpu, 3> gradone = tgrad_in[0];
    for (index_t ddi = 0; ddi < gshape[1]; ddi++) {
    std::cout << "ddi:" << ddi << "--->";
    for (index_t ri = 0; ri < gshape[2]; ri++) {
      for (index_t ci = 0; ci < gshape[3]; ci++) {
        real_t tmpval = gradone[ddi][ri][ci];
        if (1 || fabs(tmpval) > 0.f) {
          std::cout << tmpval << ", ";
        }
      }
    }
    std::cout << "\n";
    }
    }
#endif


  }

  AnchorMultiBBsRegCostParam param_;
};


template<typename xpu>
Operator* CreateOp(AnchorMultiBBsRegCostParam param);


#if DMLC_USE_CXX11
class AnchorMultiBBsRegCostProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "label", "coordlabel", "allbbslabel", "infolabel"};
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
    CHECK_EQ(in_shape->size(), 5);

    TShape &datashape = (*in_shape)[anchor_multibbs_regcost_enum::kData];
    TShape &labelshape = (*in_shape)[anchor_multibbs_regcost_enum::kLabel];
    TShape &coordshape = (*in_shape)[anchor_multibbs_regcost_enum::kCoordLabel];
    TShape &bbshape = (*in_shape)[anchor_multibbs_regcost_enum::kBBsLabel];
    TShape &infoshape = (*in_shape)[anchor_multibbs_regcost_enum::kAnchorInfoLabel];

    labelshape = Shape4(datashape[0], param_.anchornum, datashape[2], datashape[3]);
    coordshape = Shape4(datashape[0], 2, datashape[2], datashape[3]);
    bbshape = Shape4(datashape[0], param_.anchornum*4, datashape[2], datashape[3]);
    infoshape = Shape4(param_.anchornum, 2, datashape[2], datashape[3]);

    out_shape->clear();
    out_shape->push_back(datashape);

    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new AnchorMultiBBsRegCostProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "AnchorMultiBBsRegCost";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {//out_data[anchor_regcost_enum::kOut],
            in_data[anchor_multibbs_regcost_enum::kData],
            in_data[anchor_multibbs_regcost_enum::kLabel],
            in_data[anchor_multibbs_regcost_enum::kCoordLabel],
            in_data[anchor_multibbs_regcost_enum::kBBsLabel],
            in_data[anchor_multibbs_regcost_enum::kAnchorInfoLabel],
            };
  };


  Operator* CreateOperator(Context ctx) const override;

 private:
  AnchorMultiBBsRegCostParam param_;
};  // class AnchorMultiBBsRegCostProp
#endif  // DMLC_USE_CXX11


}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_ANCHOR_MULTIBBS_REGRESSIONCOST_INL_H_


