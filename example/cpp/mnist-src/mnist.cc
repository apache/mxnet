/*!
 *  Copyright (c) 2015 by Contributors
 */
#include <fstream>
#include "cpp_net.hpp"

/*
 * in this example, we use the data from Kaggle mnist match
 * get the data from:
 * https://www.kaggle.com/c/digit-recognizer
 *
 */

using namespace mxnet;
using namespace std;

class MnistCppNet : public mxnet::CppNet {
 public:
  void LenetRun() {
    /*
     * LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick Haffner.
     * "Gradient-based learning applied to document recognition."
     * Proceedings of the IEEE (1998)
     * */

    /*define the symbolic net*/
    Symbol data = Symbol::CreateVariable("data");
    Symbol conv1 = OperatorSymbol("Convolution", data, "conv1",
        "kernel", mshadow::Shape2(5, 5),
        "num_filter", 20);
    Symbol tanh1 = OperatorSymbol("Activation", conv1, "tanh1",
        "act_type", "tanh");
    Symbol pool1 = OperatorSymbol("Pooling", tanh1, "pool1",
        "pool_type", "max",
        "kernel", mshadow::Shape2(2, 2),
        "stride", mshadow::Shape2(2, 2));
    Symbol conv2 = OperatorSymbol("Convolution", pool1, "conv2",
        "kernel", mshadow::Shape2(5, 5),
        "num_filter", 50);
    Symbol tanh2 = OperatorSymbol("Activation", conv2, "tanh2",
        "act_type", "tanh");
    Symbol pool2 = OperatorSymbol("Pooling", tanh2, "pool2",
        "pool_type", "max",
        "kernel", mshadow::Shape2(2, 2),
        "stride", mshadow::Shape2(2, 2));
    Symbol flatten = OperatorSymbol("Flatten", pool2, "flatten");
    Symbol fc1 = OperatorSymbol("FullyConnected", flatten, "fc1",
        "num_hidden", 500);
    Symbol tanh3 = OperatorSymbol("Activation", fc1, "tanh3",
        "act_type", "tanh");
    Symbol fc2 = OperatorSymbol("FullyConnected", tanh3, "fc2",
        "num_hidden", 10);
    Symbol lenet = OperatorSymbol("SoftmaxOutput", fc2, "softmax");

    /*setup basic configs*/
    int val_fold = 1;
    int W = 28;
    int H = 28;
    int batch_size = 42;
    int max_epoch = 10;
    float learning_rate = 1e-4;

    /*init some of the args*/
    ArgsMap args_map;
    args_map["data"] =
        mxnet::NDArray(mshadow::Shape4(batch_size, 1, W, H), ctx_dev, false);
    /*
     * we can also feed in some of the args other than the input all by
     * ourselves,
     * fc2-weight , fc1-bias for example:
     * */
    args_map["fc1_weight"] =
        mxnet::NDArray(mshadow::Shape2(500, 4 * 4 * 50), ctx_dev, false);
    mxnet::SampleGaussian(0, 1, &args_map["fc1_weight"]);
    args_map["fc2_bias"] = mxnet::NDArray(mshadow::Shape1(10), ctx_dev, false);
    args_map["fc2_bias"] = 0;
    InitArgArrays(lenet, args_map);
    InitOptimizer("ccsgd", "momentum", 0.9, "wd", 1e-4, "rescale_grad", 1.0,
                  "clip_gradient", 10);

    /*prepare the data*/
    vector<float> data_vec, label_vec;
    size_t data_count = GetData(&data_vec, &label_vec);
    const float *dptr = data_vec.data();
    const float *lptr = label_vec.data();
    NDArray data_array = NDArray(mshadow::Shape4(data_count, 1, W, H), ctx_cpu,
                                 false);  // store in main memory, and copy to
    // device memory while training
    NDArray label_array =
        NDArray(mshadow::Shape1(data_count), ctx_cpu,
                false);  // it's also ok if just store them all in device memory
    data_array.SyncCopyFromCPU(dptr, data_count * W * H);
    label_array.SyncCopyFromCPU(lptr, data_count);
    data_array.WaitToRead();
    label_array.WaitToRead();

    Train(data_array, label_array, max_epoch, val_fold, learning_rate);
  }

  size_t GetData(vector<float> *data, vector<float> *label) {
    const char *train_data_path = "./train.csv";
    ifstream inf(train_data_path);
    string line;
    inf >> line;  // ignore the header
    size_t _N = 0;
    while (inf >> line) {
      for (auto &c : line) c = (c == ',') ? ' ' : c;
      stringstream ss;
      ss << line;
      float _data;
      ss >> _data;
      label->push_back(_data);
      while (ss >> _data) data->push_back(_data / 256.0);
      _N++;
    }
    inf.close();
    return _N;
  }

  void TrainingCallBack(int iter, const mxnet::Executor *executor) {
    LG << "Iter " << iter << ", accuracy: " << ValAccuracy();
    /*
     * do something every epoch,
     * such as train-data shuffle, train-data augumentation , save the model ,
     * change the learning_rate etc.
     * */
  }

  explicit MnistCppNet(bool use_gpu = false, int dev_id = 0)
      : mxnet::CppNet(use_gpu, dev_id) {}
};

int main(int argc, char const *argv[]) {
  MnistCppNet mnist(true, 2);
  mnist.LenetRun();
  return 0;
}
