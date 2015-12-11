/*!
 * Copyright (c) 2015 by Contributors
 */

#include <iostream>
#include <utility>
#include <map>
#include <string>
#include <vector>

#include "mxnet/ndarray.h"
#include "mxnet/base.h"
#include "mxnet/operator.h"
#include "mxnet/symbolic.h"

using namespace std;

#if MSHADOW_USE_CUDA
#define DEV_CTX \
  (mxnet::Context::Create(mxnet::Context::kGPU, 0))  // use no.0 gpu
#else
#define DEV_CTX (mxnet::Context::Create(mxnet::Context::kCPU, 0))  // use cpu
#endif

/*
 * In this example,
 * we make by hand some data in 10 classes with some pattern
 * and try to use MLP to recognize the pattern.
 */

class MLP {
 public:
  mxnet::Symbol LeakyReLULayer(mxnet::Symbol input, std::string name = "relu") {
    mxnet::OperatorProperty* leaky_relu_op =
        mxnet::OperatorProperty::Create("LeakyReLU");

    std::vector<std::pair<std::string, std::string> > relu_config;
    relu_config.push_back(
        std::make_pair("act_type", "leaky"));  // rrelu leaky prelu elu
    relu_config.push_back(std::make_pair("slope", "0.25"));
    relu_config.push_back(std::make_pair("lower_bound", "0.125"));
    relu_config.push_back(std::make_pair("upper_bound", "0.334"));
    leaky_relu_op->Init(relu_config);
    std::vector<mxnet::Symbol> sym_vec;

    sym_vec.push_back(input);
    mxnet::Symbol leaky_relu =
        mxnet::Symbol::Create(leaky_relu_op)(sym_vec, name);
    return leaky_relu;
  }
  mxnet::Symbol FullyConnectedLayer(mxnet::Symbol input,
                                    std::string num_hidden = "28",
                                    std::string name = "fc") {
    mxnet::OperatorProperty* fully_connected_op =
        mxnet::OperatorProperty::Create("FullyConnected");

    std::vector<std::pair<std::string, std::string> > fc_config;
    fc_config.push_back(std::make_pair("num_hidden", num_hidden));
    fc_config.push_back(std::make_pair("no_bias", "false"));
    fully_connected_op->Init(fc_config);

    std::vector<mxnet::Symbol> sym_vec;
    sym_vec.push_back(input);
    mxnet::Symbol fc = mxnet::Symbol::Create(fully_connected_op)(sym_vec, name);

    return fc;
  }
  mxnet::Symbol SoftmaxLayer(mxnet::Symbol input,
                             std::string name = "softmax") {
    mxnet::OperatorProperty* softmax_output_op =
        mxnet::OperatorProperty::Create("SoftmaxOutput");

    std::vector<std::pair<std::string, std::string> > config;
    softmax_output_op->Init(config);

    std::vector<mxnet::Symbol> sym_vec;
    sym_vec.push_back(input);
    mxnet::Symbol softmax =
        mxnet::Symbol::Create(softmax_output_op)(sym_vec, name);

    return softmax;
  }

  void OutputAccuracy(mxnet::real_t* pred, mxnet::real_t* target) {
    int right = 0;
    for (int i = 0; i < 128; ++i) {
      float mx_p = pred[i * 10 + 0];
      float p_y = 0;
      for (int j = 0; j < 10; ++j) {
        if (pred[i * 10 + j] > mx_p) {
          mx_p = pred[i * 10 + j];
          p_y = j;
        }
      }
      if (p_y == target[i]) right++;
    }
    cout << "Accuracy: " << right / 128.0 << endl;
  }

  void Train() {
    // setup sym network
    mxnet::Symbol sym_x = mxnet::Symbol::CreateVariable("X");
    mxnet::Symbol sym_fc_1 = FullyConnectedLayer(sym_x, "512", "fc1");
    mxnet::Symbol sym_act_1 = LeakyReLULayer(sym_fc_1, "act_1");
    mxnet::Symbol sym_fc_2 = FullyConnectedLayer(sym_act_1, "10", "fc2");
    mxnet::Symbol sym_act_2 = LeakyReLULayer(sym_fc_2, "act_2");
    mxnet::Symbol sym_out = SoftmaxLayer(sym_act_2, "softmax");

    // prepare train data
    mxnet::Context ctx_cpu = mxnet::Context::Create(mxnet::Context::kCPU, 1);
    mxnet::Context ctx_dev = DEV_CTX;  // use gpu if possible

    mxnet::NDArray array_x(mshadow::Shape2(128, 28), ctx_dev, false);
    mxnet::NDArray array_y(mshadow::Shape1(128), ctx_dev, false);

    mxnet::real_t* aptr_x = new mxnet::real_t[128 * 28];
    mxnet::real_t* aptr_y = new mxnet::real_t[128];

    // we make the data by hand, in 10 classes, with some pattern
    for (int i = 0; i < 128; i++) {
      for (int j = 0; j < 28; j++) {
        aptr_x[i * 28 + j] = i % 10 * 1.0f;
      }
      aptr_y[i] = i % 10;
    }
    array_x.SyncCopyFromCPU(aptr_x, 128 * 28);
    array_x.WaitToRead();
    array_y.SyncCopyFromCPU(aptr_y, 128);
    array_y.WaitToRead();

    // init the parameters
    mxnet::NDArray array_w_1(mshadow::Shape2(512, 28), ctx_dev, false);
    mxnet::NDArray array_b_1(mshadow::Shape1(512), ctx_dev, false);
    mxnet::NDArray array_w_2(mshadow::Shape2(10, 512), ctx_dev, false);
    mxnet::NDArray array_b_2(mshadow::Shape1(10), ctx_dev, false);

    // the parameters should be initialized in some kind of distribution,
    // so it learns fast
    // but here just give a const value by hand
    array_w_1 = 0.5f;
    array_b_1 = 0.0f;
    array_w_2 = 0.5f;
    array_b_2 = 0.0f;

    // the grads
    mxnet::NDArray array_w_1_g(mshadow::Shape2(512, 28), ctx_dev, false);
    mxnet::NDArray array_b_1_g(mshadow::Shape1(512), ctx_dev, false);
    mxnet::NDArray array_w_2_g(mshadow::Shape2(10, 512), ctx_dev, false);
    mxnet::NDArray array_b_2_g(mshadow::Shape1(10), ctx_dev, false);

    // Bind the symolic network with the ndarray
    std::map<std::string, mxnet::Context> g2c;
    // all the input args
    std::vector<mxnet::NDArray> in_args;
    in_args.push_back(array_x);
    in_args.push_back(array_w_1);
    in_args.push_back(array_b_1);
    in_args.push_back(array_w_2);
    in_args.push_back(array_b_2);
    in_args.push_back(array_y);
    // all the grads
    std::vector<mxnet::NDArray> arg_grad_store;
    arg_grad_store.push_back(
        mxnet::NDArray());  // we don't need the grad of the input
    arg_grad_store.push_back(array_w_1_g);
    arg_grad_store.push_back(array_b_1_g);
    arg_grad_store.push_back(array_w_2_g);
    arg_grad_store.push_back(array_b_2_g);
    arg_grad_store.push_back(
        mxnet::NDArray());  // neither do we need the grad of the loss
    // how to handle the grad
    std::vector<mxnet::OpReqType> grad_req_type;
    grad_req_type.push_back(mxnet::kNullOp);
    grad_req_type.push_back(mxnet::kWriteTo);
    grad_req_type.push_back(mxnet::kWriteTo);
    grad_req_type.push_back(mxnet::kWriteTo);
    grad_req_type.push_back(mxnet::kWriteTo);
    grad_req_type.push_back(mxnet::kNullOp);
    std::vector<mxnet::NDArray> aux_states;

    cout << "make the Executor" << endl;
    mxnet::Executor* exe =
        mxnet::Executor::Bind(sym_out, ctx_dev, g2c, in_args, arg_grad_store,
                              grad_req_type, aux_states);

    cout << "Training" << endl;
    int max_iters = 20000;
    mxnet::real_t learning_rate = 0.0001;
    for (int iter = 0; iter < max_iters; ++iter) {
      exe->Forward(true);

      if (iter % 100 == 0) {
        cout << "epoch " << iter << endl;
        const std::vector<mxnet::NDArray>& out = exe->outputs();
        mxnet::NDArray c_cpu = out[0].Copy(ctx_cpu);
        c_cpu.WaitToRead();
        mxnet::real_t* cptr = static_cast<mxnet::real_t*>(c_cpu.data().dptr_);
        OutputAccuracy(cptr, aptr_y);
      }

      // update the parameters
      exe->Backward(std::vector<mxnet::NDArray>());
      for (int i = 1; i < 5; ++i) {
        in_args[i] -= arg_grad_store[i] * learning_rate;
      }
    }

    delete[] aptr_x;
    delete[] aptr_y;
  }
};

int main(int argc, char** argv) {
  MLP mlp;
  mlp.Train();
  return 0;
}

