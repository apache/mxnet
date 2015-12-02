/*!
 * \file random-inl.h
 * \brief Symbolic layer for random number generation.
 * \author Sebastian Nowozin
*/

#ifndef MXNET_OPERATOR_RANDOM_INL_H_
#define MXNET_OPERATOR_RANDOM_INL_H_

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

#include <iostream>

namespace random_enum {
enum RandomOpOutputs { kOut };
enum RandomOpType { kUniform, kGaussian };
enum RandomOpForwardResource { kRandom };
}  // namespace random_enum

namespace mxnet {
namespace op {

struct RandomParam : public dmlc::Parameter<RandomParam> {
  int num_output;
  int random_type;
  DMLC_DECLARE_PARAMETER(RandomParam) {
    DMLC_DECLARE_FIELD(num_output).set_range(1, 1 << 30)
    .describe("Number of output elements");
    DMLC_DECLARE_FIELD(random_type)
    .add_enum("uniform", random_enum::kUniform)
    .add_enum("gaussian", random_enum::kGaussian)
    .describe("Type of random numbers to generate");
  }
};  // struct RandomParam

template<typename xpu>
class RandomOp : public Operator {
 public:
  explicit RandomOp(RandomParam param) {
    this->random_type_ = param.random_type;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_data.size(), 1);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> out = out_data[random_enum::kOut].FlatTo2D<xpu, real_t>(s);

    Random<xpu> *prnd = ctx.requested[random_enum::kRandom].get_random<xpu>(s);
    if (random_type_ == random_enum::kUniform) {
       out = prnd->uniform(out.shape_);
    } else if (random_type_ == random_enum::kGaussian) {
       out = prnd->gaussian(out.shape_);
    }
	// Need Assign?
    //Assign(out, req[random::kOut], data * mask);
    //Assign(out, req[dropout::kOut], F<mshadow_op::identity>(data));
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    // No inputs and thus no gradient
  }

 private:
  int random_type_;
};  // class RandomOp


template<typename xpu>
Operator *CreateOp(RandomParam param);

#if DMLC_USE_CXX11
class RandomProp : public OperatorProperty {
 public:
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
    // XXX: how to do automatic shape inference here?
    //
    // Ideally we would like to avoid specifying the num_output value
    // altogether and instead fill an arbitrary sized array.
    // For example, the Random symbol may be useful to implement variational
    // Bayes parameter learning and any parametrized layer knows the size of
    // its parameter array.  It seems current shape inference is purely
    // feedforward which does not allow us to "backpropagate" shape
    // information from output.
    //
    // I would like to be able to write:
    // if (out_shape->size() == 0) {
    //   return false;
    // } else {
    //   // Nothing to do.
    //   return true;
    // }
    //
    // By making multiple passes over the symbolic graph all dependencies
    // could be resolved if the next layer consuming the random output can
    // provide shape information about its input.

    SHAPE_ASSIGN_CHECK(*out_shape, random_enum::kOut, Shape1(param_.num_output));

    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new RandomProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "Random";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return { };
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return { };
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return { };
  }

  std::vector<ResourceRequest> ForwardResource(
    const std::vector<TShape> &in_shape) const override {
    return { ResourceRequest::kRandom };
  }

  std::vector<std::string> ListArguments() const override {
    return { };
  }

  int NumOutputs() const override {
    return 1;
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output"};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  RandomParam param_;
};  // class RandomProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_RANDOM_INL_H_
