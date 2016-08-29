/*!
 * Copyright (c) 2016 by Contributors
 * \file postprocess_rpn.cc
 * \brief post process of rpn operator
 * \author Ming Zhang
*/
#include "./postprocess_rpn-inl.h"
#include "./mshadow_op.h"

namespace mshadow {

inline void PostProcessRPNForward(const Tensor<cpu, 4> &datacls_in,
                           const Tensor<cpu, 4> &datareg_in,
                           const Tensor<cpu, 2> &anchorinfo_in,
                           const Tensor<cpu, 1> &otherinfo_in,
                           Tensor<cpu, 3> &bb_out) {
  
  CHECK_EQ(datacls_in.size(0), datareg_in.size(0));
  
  float clsthreshold = otherinfo_in[0];
  int originalH = otherinfo_in[1];
  int originalW = otherinfo_in[2];
  int dwBatchNum = datacls_in.size(0);
  int dwAnchorNum = anchorinfo_in.size(0);
  int bb_maxnum_per_batch = bb_out.size(1);
  
  int dwFeatH = datacls_in.size(2);
  int dwFeatW = datacls_in.size(3);
  int dwBBMemLen = bb_out.MemSize<0>();
  memset(bb_out.dptr_, 0, dwBBMemLen * sizeof(float));
  
  for (int bi = 0; bi < dwBatchNum; bi++) {
    const Tensor<cpu, 3> &datacls_onebatch = datacls_in[bi];
    const Tensor<cpu, 3> &datareg_onebatch = datareg_in[bi];
    Tensor<cpu, 2> bb_onebatch = bb_out[bi];
    int bb_num_now = 0;
    for (int ai = 0; ai < dwAnchorNum; ai++) {
      const Tensor<cpu, 1> &anchor_one = anchorinfo_in[ai];
      const Tensor<cpu, 2> &datacls_onea = datacls_onebatch[ai];
      const Tensor<cpu, 3> &datareg_onea = datareg_onebatch.Slice(ai*4, (ai+1)*4);
      float anchor_h = anchor_one[0];
      float anchor_w = anchor_one[1];
      for (int ri = 0; ri < dwFeatH; ri++) {
        const Tensor<cpu, 1> &datacls_row = datacls_onea[ri];
        const Tensor<cpu, 1> &regy_row = datareg_onea[0][ri];
        const Tensor<cpu, 1> &regx_row = datareg_onea[1][ri];
        const Tensor<cpu, 1> &regh_row = datareg_onea[2][ri];
        const Tensor<cpu, 1> &regw_row = datareg_onea[3][ri];
        for (int ci = 0; ci < dwFeatW; ci++) {
          if (datacls_row[ci] > clsthreshold && bb_num_now < bb_maxnum_per_batch) {
            Tensor<cpu, 1> bbnow = bb_onebatch[bb_num_now];
            bbnow[0] = regy_row[ci] * anchor_h + ((float)(ri) * originalH) / dwFeatH;
            bbnow[1] = regx_row[ci] * anchor_w + ((float)(ci) * originalW) / dwFeatW;
            bbnow[2] = expf(regh_row[ci]) * anchor_h;
            bbnow[3] = expf(regw_row[ci]) * anchor_w;
            bb_num_now++;
          }
        }
      }
    }
  }
}

} // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(PostProcessRPNParam param) {
  return new PostProcessRPNOp<cpu>(param);
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *PostProcessRPNProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(PostProcessRPNParam);

MXNET_REGISTER_OP_PROPERTY(PostProcessRPN, PostProcessRPNProp)
.describe("Post Process of RPN, ouput bounding boxes.")
.add_argument("datacls", "Symbol", "Input datacls layer of rpn to function.")
.add_argument("datareg", "Symbol", "Input datareg layer of rpn to function.")
.add_argument("anchorinfo", "Symbol", "Input anchor info of rpn to function.")
.add_arguments(PostProcessRPNParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
