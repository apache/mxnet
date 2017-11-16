/*!
 * Copyright (c) 2017 Microsoft
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file multi_proposal.cc
 * \brief
 * \author Xizhou Zhu
*/

#include "./multi_proposal-inl.h"


namespace mxnet {
namespace op {

template<typename xpu>
class MultiProposalOp : public Operator{
 public:
  explicit MultiProposalOp(MultiProposalParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    LOG(FATAL) << "not implemented";
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    LOG(FATAL) << "not implemented";
  }

 private:
  MultiProposalParam param_;
};  // class MultiProposalOp

template<>
Operator *CreateOp<cpu>(MultiProposalParam param) {
  return new MultiProposalOp<cpu>(param);
}

Operator* MultiProposalProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(MultiProposalParam);

MXNET_REGISTER_OP_PROPERTY(_contrib_MultiProposal, MultiProposalProp)
.describe("Generate region proposals via RPN")
.add_argument("cls_score", "NDArray-or-Symbol", "Score of how likely proposal is object.")
.add_argument("bbox_pred", "NDArray-or-Symbol", "BBox Predicted deltas from anchors for proposals")
.add_argument("im_info", "NDArray-or-Symbol", "Image size and scale.")
.add_arguments(MultiProposalParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
