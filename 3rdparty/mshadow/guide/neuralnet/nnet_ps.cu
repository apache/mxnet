// this implements a simple two layer Multi-GPU neural net
// this implementation uses mshadow-ps to get gradient aggregation
// between cards
// this code is modified from nnet.cu
#include <vector>
#include <cmath>
#include <omp.h>
// header file to use mshadow
#include <mshadow/tensor.h>
#include <mshadow-ps/mshadow_ps.h>
// helper function to load mnist dataset
#include "./util.h"
// this namespace contains all data structures, functions
using namespace mshadow;
// this namespace contains all operator overloads
using namespace mshadow::expr;

// define sigmoid operation
struct sigmoid {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return 1.0f / (1.0f + expf(-a));
  }
};

/*! \brief interface for nnet, interfacd allows use to use GPU/CPU implementation in a unified way */
class INNet{
 public:
  virtual void Forward(const Tensor<cpu, 2, real_t>& inbatch,
                       Tensor<cpu, 2, real_t> &oubatch) = 0;
  virtual void Backprop(const Tensor<cpu, 2, real_t>& gradout) = 0;
  virtual ~INNet() {}
};

/*!
 * \brief simple two layer neural net
 *        this implementation is device invariant
 */
template<typename xpu>
class NNet : public INNet {
 public:
  // initialize the network
  NNet(int batch_size, int num_in, int num_hidden, int num_out,
       int devid, mshadow::ps::ISharedModel<xpu, real_t> *ps)
      : rnd(0), devid(devid), ps(ps) {
    mshadow::SetDevice<xpu>(devid);
    stream = mshadow::NewStream<xpu>();
    // set the computing streams
    ninput.set_stream(stream);
    nhidden.set_stream(stream);
    nhiddenbak.set_stream(stream);
    nout.set_stream(stream);
    hbias.set_stream(stream);
    obias.set_stream(stream);
    g_hbias.set_stream(stream);
    g_obias.set_stream(stream);
    Wi2h.set_stream(stream);
    Wh2o.set_stream(stream);
    g_Wi2h.set_stream(stream);
    g_Wh2o.set_stream(stream);
    rnd.set_stream(stream);
    // setup nodes
    ninput.Resize(Shape2(batch_size, num_in));
    nhidden.Resize(Shape2(batch_size, num_hidden));
    nhiddenbak.Resize(nhidden.shape_);
    nout.Resize(Shape2(batch_size, num_out));
    // setup bias
    hbias.Resize(Shape1(num_hidden)); g_hbias.Resize(hbias.shape_);
    obias.Resize(Shape1(num_out)); g_obias.Resize(obias.shape_);
    hbias = 0.0f; obias = 0.0f;
    // setup weights
    Wi2h.Resize(Shape2(num_in, num_hidden));  g_Wi2h.Resize(Wi2h.shape_);
    Wh2o.Resize(Shape2(num_hidden, num_out)); g_Wh2o.Resize(Wh2o.shape_);
    rnd.SampleGaussian(&Wi2h, 0, 0.01f);
    rnd.SampleGaussian(&Wh2o, 0, 0.01f);
    // initialize the key
    ps->InitKey(Wi2h.shape_, 0, devid);
    ps->InitKey(hbias.shape_, 1, devid);
    ps->InitKey(Wh2o.shape_, 2, devid);
    ps->InitKey(obias.shape_, 3, devid);
  }
  virtual ~NNet() {
    mshadow::SetDevice<xpu>(devid);
    mshadow::DeleteStream(stream);
  }
  // forward propagation
  virtual void Forward(const Tensor<cpu, 2, real_t> &inbatch,
                       Tensor<cpu, 2, real_t> &oubatch) {
    // size is same conventsion as numpy
    index_t batch_size = inbatch.size(0);
    // copy data to input layer
    Copy(ninput, inbatch, stream);
    // wait the last pull requst on layer to complete
    ps->PullWait(0, devid);
    // first layer, fullc
    nhidden = dot(ninput, Wi2h);
    // wait the pull request on hbias to complete
    ps->PullWait(1, devid);
    nhidden+= repmat(hbias, batch_size);
    // activation, sigmloid, backup activation in nhidden
    nhidden = F<sigmoid>(nhidden);
    Copy(nhiddenbak, nhidden, stream);
    // second layer fullc
    ps->PullWait(2, devid);
    nout = dot(nhiddenbak, Wh2o);
    ps->PullWait(3, devid);
    nout += repmat(obias, batch_size);
    // softmax calculation
    Softmax(nout, nout);
    // copy result out
    Copy(oubatch, nout, stream);
    // Copy with stream is non-blocking, use wait to wait until copy finishes
    stream->Wait();
  }
  // back propagation
  virtual void Backprop(const Tensor<cpu, 2, real_t> &gradout) {
    // copy gradient to output layer
    Copy(nout, gradout, stream);
    // calc grad of layer 2
    g_obias = sum_rows(nout);
    // sync proc defines the synchronization step
    this->SyncProc(obias, g_obias, 3);
    // update second layer weights
    g_Wh2o = dot(nhiddenbak.T(), nout);
    // backprop to layer 1
    nhiddenbak = dot(nout, Wh2o.T());
    this->SyncProc(Wh2o, g_Wh2o, 2);
    // calculate gradient of sigmoid layer
    nhidden = nhidden * (1.0f-nhidden) * nhiddenbak;
    // calc grad of layer 1
    g_hbias = sum_rows(nhidden);
    this->SyncProc(hbias, g_hbias, 1);
    g_Wi2h = dot(ninput.T(), nhidden);
    this->SyncProc(Wi2h, g_Wi2h, 0);
  }
  // synchronization function
  template<int dim>
  inline void SyncProc(mshadow::Tensor<xpu, dim> weight,
                       mshadow::Tensor<xpu, dim> grad,
                       int data_key) {
    // wait till last computation finishes
    stream->Wait();
    ps->Push(grad, data_key, devid, -data_key);
    ps->PullReq(grad, data_key, devid, -data_key,
                UpdateEntry::ApplyUpdate,
                new UpdateEntry(weight.FlatTo2D(), grad.FlatTo2D(), dim == 1));
  }
  // data structure defined to help using callback function
  struct UpdateEntry {
    mshadow::Tensor<xpu, 2> weight;
    mshadow::Tensor<xpu, 2> grad;
    bool is_bias;
    // constructor
    UpdateEntry(mshadow::Tensor<xpu, 2> weight,
                mshadow::Tensor<xpu, 2> grad,
                bool is_bias)
        : weight(weight), grad(grad),
          is_bias(is_bias) {}
    inline void Update(mshadow::Stream<xpu> *stream) {
      weight.set_stream(stream);
      const float wd = 0.00001;
      const float eta = 0.8;
      if (!is_bias) {
        weight -= eta * (wd * weight + grad);
      } else {
        weight -= eta * grad;
      }
    }
    // callback function to apply update
    inline static void ApplyUpdate(mshadow::Stream<xpu> *stream, void *arg) {
      UpdateEntry *e = static_cast<UpdateEntry*>(arg);
      e->Update(stream);
      delete e;
    }
  };

 private:
  // computing stream
  mshadow::Stream<xpu> *stream;
  // device id
  int devid;
  // parameter server interface
  mshadow::ps::ISharedModel<xpu, real_t> *ps;
  // random seed generator
  Random<xpu, real_t> rnd;
  // nodes in neural net
  TensorContainer<xpu, 2, real_t> ninput, nhidden, nhiddenbak, nout;
  // hidden bias, gradient
  TensorContainer<xpu, 1, real_t> hbias, obias, g_hbias, g_obias;
  // weight gradient
  TensorContainer<xpu, 2, real_t> Wi2h, Wh2o, g_Wi2h, g_Wh2o;
};

// helper function to get the max inde
inline int MaxIndex(Tensor<cpu, 1, real_t> pred) {
  int maxidx = 0;
  for(index_t i = 1; i < pred.size(0); ++i) {
    if(pred[i] > pred[maxidx]) maxidx = (int)i;
  }
  return maxidx;
}

namespace mshadow {
namespace ps {
// model updater is used when update is happening on server side
// if we only use parameter server for sum aggregation
// this is not needed, but we must declare this function to return NULL
template<>
IModelUpdater<float> *CreateModelUpdater(void) {
  return NULL;
}
}
}

template<typename xpu>
inline int Run(int argc, char *argv[]) {
  srand(0);
  // settings
  int batch_size = 100;
  int num_in = 28 * 28;
  int num_hidden = 100;
  int num_out = 10;
  int ndev = argc - 2;
  if (batch_size % ndev != 0) {
    fprintf(stderr, "choose number of devices ndev such that 100 MOD ndev == 0\n");
    return 0;
  }
  // choose which version to use
  std::vector<int> devs;
  for (int i = 2; i < argc; ++i) {
    devs.push_back(atoi(argv[i]));
  }
  mshadow::ps::ISharedModel<xpu, real_t>
      *ps = mshadow::ps::CreateSharedModel<xpu, real_t>("local");
  ps->Init(devs);

  std::vector<INNet *> nets(ndev);
  for (int i = 0; i < ndev; ++i) {
    mshadow::InitTensorEngine<xpu>(devs[i]);
    nets[i] = new NNet<xpu>(batch_size / ndev, num_in, num_hidden, num_out, devs[i], ps);
  }

  // label
  std::vector<int> ytrain, ytest;
  // data
  TensorContainer<cpu,2> xtrain, xtest;
  LoadMNIST("train-images-idx3-ubyte", "train-labels-idx1-ubyte", ytrain, xtrain, true);
  LoadMNIST("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", ytest, xtest, false);
  int num_iter = 20;

  for (int i = 0; i < num_iter; ++ i) {
    // mini-batch per device
    int step = batch_size / ndev;
    // running parallel threads
    #pragma omp parallel num_threads(ndev)
    {
      // temp output layer
      TensorContainer<cpu, 2, real_t> pred;
      pred.Resize(Shape2(step, num_out));
      int tid = omp_get_thread_num();
      mshadow::SetDevice<xpu>(devs[tid]);
      for (index_t j = 0; j + batch_size <= xtrain.size(0); j += batch_size) {
        nets[tid]->Forward(xtrain.Slice(j + tid * step, j + (tid + 1) * step), pred);
        // set gradient into pred
        for (int k = 0; k < step; ++ k) {
          pred[k][ytrain[j + tid * step + k]] -= 1.0f;
        }
        // scale gradient by batchs zie
        pred *= 1.0f / batch_size;
        // run backprop
        nets[tid]->Backprop(pred);
      }
    }
    // evaluation
    long nerr = 0;
    #pragma omp parallel num_threads(ndev) reduction(+:nerr)
    {
      // temp output layer
      TensorContainer<cpu, 2, real_t> pred;
      pred.Resize(Shape2(step, num_out));
      int tid = omp_get_thread_num();
      mshadow::SetDevice<xpu>(devs[tid]);
      for (index_t j = 0; j + batch_size <= xtest.size(0); j += batch_size) {
        nets[tid]->Forward(xtest.Slice(j + tid * step, j + (tid + 1) * step), pred);
        for (int k = 0; k < step; ++ k) {
          nerr += MaxIndex(pred[k]) != ytest[j + tid * step + k];
        }
      }
    }
    printf("round %d: test-err=%f\n", i, (float)nerr/xtest.size(0));
  }

  for(int i = 0; i < ndev; ++i) {
    mshadow::SetDevice<xpu>(devs[i]);
    delete nets[i];
    ShutdownTensorEngine<xpu>();
  }
  return 0;
}
int main(int argc, char *argv[]) {
  if (argc < 3) {
    printf("Usage: <device> devicelist\n"\
           "\tExample1: ./nnet_ps cpu 1 2 3\n"\
           "\tExample2: ./nnet_ps gpu 0 1\n");
    return 0;
  }
  if (!strcmp(argv[1], "cpu")) {
    Run<mshadow::cpu>(argc, argv);
  } else {
    Run<mshadow::gpu>(argc, argv);
  }
  return 0;
}
