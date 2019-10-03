// this implements a simple convolution neural net: conv-maxpool-fullc
#include <vector>
// header file to use mshadow
#include "mshadow/tensor.h"
// helper function to load mnist dataset
#include "util.h"
// this namespace contains all data structures, functions
using namespace mshadow;
// this namespace contains all operator overloads
using namespace mshadow::expr;

// define operations
struct relu{
  MSHADOW_XINLINE static real_t Map(real_t a) {
    using namespace std;
    return max(a, 0.0f);
  }
};
struct relu_grad {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return a > 0.0f ? 1.0f : 0.0f;
  }
};

/*! \brief interface for nnet, interfacd allows use to use GPU/CPU implementation in a unified way */
class INNet{
 public:
  virtual void Forward(const Tensor<cpu, 4, real_t>& inbatch, Tensor<cpu, 2, real_t> &oubatch) = 0;
  virtual void Backprop(const Tensor<cpu, 2, real_t>& gradout) = 0;
  virtual void Update(void) = 0;
  virtual ~INNet() {}
};

/*!
 * \brief simple two layer conv-net conv-pool-flat-fullc
 *        this implementation is device invariant
 */
template<typename xpu>
class ConvNet : public INNet {
 public:
  // initialize the network
  ConvNet(int batch_size, int insize, int nchannel, int ksize, int kstride, int psize, int num_out)
      :rnd(0), ksize(ksize), kstride(kstride), psize(psize) {
    // setup stream
    Stream<xpu> *stream = NewStream<xpu>();
    ninput.set_stream(stream);
    nhidden.set_stream(stream);
    nhiddenbak.set_stream(stream);
    npool.set_stream(stream);
    npoolbak.set_stream(stream);
    nflat.set_stream(stream);
    nout.set_stream(stream);
    hbias.set_stream(stream); g_hbias.set_stream(stream);
    obias.set_stream(stream);  g_obias.set_stream(stream);
    Ki2h.set_stream(stream);  g_Ki2h.set_stream(stream);
    Wh2o.set_stream(stream);   g_Wh2o.set_stream(stream);
    tmp_col.set_stream(stream);
    tmp_dst.set_stream(stream);
    // setup nodes
    ninput.Resize(Shape4(batch_size, 1, insize, insize));
    nhidden.Resize(Shape4(batch_size, nchannel, (insize - ksize)/kstride+1, (insize -ksize)/kstride+1));
    nhiddenbak.Resize(nhidden.shape_);
    npool.Resize(Shape4(batch_size, nchannel, (nhidden.size(2)+1-psize)/psize, (nhidden.size(3)+1-psize)/psize));
    npoolbak.Resize(npool.shape_);
    nflat.Resize(Shape2(batch_size, npool.size(1)*npool.size(2)*npool.size(3)));
    nout.Resize(Shape2(batch_size, num_out));
    // setup bias
    hbias.Resize(Shape1(nchannel)); g_hbias.Resize(hbias.shape_);
    obias.Resize(Shape1(num_out));  g_obias.Resize(obias.shape_);
    hbias = 0.0f; obias = 0.0f;
    // setup weights
    Ki2h.Resize(Shape2(nchannel, ksize*ksize));  g_Ki2h.Resize(Ki2h.shape_);
    Wh2o.Resize(Shape2(nflat.size(1), num_out));   g_Wh2o.Resize(Wh2o.shape_);
    rnd.SampleGaussian(&Ki2h, 0, 0.01f);
    rnd.SampleGaussian(&Wh2o, 0, 0.01f);

    printf("conv=%d, pool=%d\n", nhidden.size(3), npool.size(3));
  }
  virtual ~ConvNet() {}
  // forward propagation
  virtual void Forward(const Tensor<cpu, 4, real_t>& inbatch, Tensor<cpu, 2, real_t> &oubatch) {
    index_t batch_size = inbatch.size(0);
    // copy data to input layer
    Copy(ninput, inbatch, ninput.stream_);
    // first layer, conv, use stride=2
    ConvForward(ninput, Ki2h, nhidden, ksize, kstride, tmp_col, tmp_dst);
    // add bias
    nhidden += broadcast<1>(hbias, nhidden.shape_);
    // activation, relu, backup activation in nhidden
    nhidden = F<relu>(nhidden);
    Copy(nhiddenbak, nhidden, nhiddenbak.stream_);
    // max pooling
    npool = pool<red::maximum>(nhiddenbak, npool[0][0].shape_, psize, psize, psize);
    Copy(npoolbak, npool, npoolbak.stream_);
    // flat
    nflat = reshape(npool, nflat.shape_);
    // second layer fullc
    nout = dot(nflat, Wh2o);
    nout += repmat(obias, batch_size);
    // softmax calculation
    Softmax(nout, nout);
    // copy result out
    Copy(oubatch, nout, nout.stream_);
  }
  // back propagation
  virtual void Backprop(const Tensor<cpu, 2, real_t>& gradout) {
    // copy gradient to output layer
    Copy(nout, gradout, nout.stream_);
    // calc grad of final layer
    g_obias = sum_rows(nout);
    g_Wh2o  = dot(nflat.T(), nout);
    // backprop to previous layer
    nflat = dot(nout, Wh2o.T());
    npool = reshape(nflat, npool.shape_);
    // backprop pooling layer
    nhiddenbak = unpool<red::maximum>(nhiddenbak, npoolbak, npool, psize, psize, psize);
    // calculate gradient of relu layer
    nhidden = F<relu_grad>(nhidden) * nhiddenbak;
    // calc grad of layer 1
    g_hbias = sumall_except_dim<1>(nhidden);
    ConvBackWard(nhidden, Ki2h, g_Ki2h, ninput, ksize, kstride, tmp_col, tmp_dst);
  }
  // update weight
  virtual void Update(void) {
    // run SGD
    const float eta = 0.1;
    const float wd = 0.00001;
    // update weight
    Ki2h -= eta * (wd * Ki2h + g_Ki2h);
    Wh2o -= eta * (wd * Wh2o + g_Wh2o);
    // no regularization for bias
    hbias-= eta * g_hbias;
    obias-= eta * g_obias;
  }
 private:
  // forward convolution, tmp_col and tmp_dst are helper structure
  inline static void ConvForward(const Tensor<xpu, 4, real_t> &in,
                                 const Tensor<xpu, 2, real_t> &kernel,
                                 Tensor<xpu, 4, real_t> &out,
                                 int ksize, int kstride,
                                 TensorContainer<xpu, 2, real_t> &tmp_col,
                                 TensorContainer<xpu, 2, real_t> &tmp_dst) {
    index_t oheight  = (in.size(2) - ksize)/kstride + 1;
    index_t owidth   = (in.size(3) - ksize)/kstride + 1;
    index_t nbatch   = in.size(0);
    index_t nchannel = out.size(1);
    // we directly unpack all local patches and do a dot product
    // this cost lots of memory, normally for large image, only unpack several image at a time
    tmp_col.Resize(Shape2(in.size(1)*ksize*ksize, nbatch*oheight*owidth));
    tmp_dst.Resize(Shape2(nchannel, nbatch*oheight*owidth));
    // unpack local patches , stride=1
	tmp_col = unpack_patch2col(in, ksize, ksize, kstride, kstride, 1, 1);
    tmp_dst = dot(kernel, tmp_col);
    // reshape, then swap axis, we chain equations together
    out = swapaxis<1,0>(reshape(tmp_dst, Shape4(nchannel, nbatch, oheight, owidth)));
  }
  // backward convolution, calculate gradient of kernel, and backprop back to in
  inline static void ConvBackWard(const Tensor<xpu, 4, real_t> &out,
                                  const Tensor<xpu, 2, real_t> &kernel,
                                  Tensor<xpu, 2, real_t> &g_kernel,
                                  Tensor<xpu, 4, real_t> &in,
                                  int ksize, int kstride,
                                  TensorContainer<xpu, 2, real_t> &tmp_col,
                                  TensorContainer<xpu, 2, real_t> &tmp_dst) {
    index_t oheight  = (in.size(2) - ksize)/kstride + 1;
    index_t owidth   = (in.size(3) - ksize)/kstride + 1;
    index_t nbatch   = in.size(0);
    index_t nchannel = out.size(1);
    // we directly unpack all local patches and do a dot product
    // this cost lots of memory, normally for large image, only unpack several image at a time
    tmp_col.Resize(Shape2(in.size(1) * ksize * ksize,
                          nbatch * oheight * owidth));
    tmp_dst.Resize(Shape2(nchannel, nbatch * oheight * owidth));
    // unpack local patches
    tmp_col = unpack_patch2col(in, ksize, ksize, kstride, kstride, 1, 1);
    tmp_dst = reshape(swapaxis<1,0>(out), tmp_dst.shape_);
    g_kernel = dot(tmp_dst, tmp_col.T());
        // backpropgation: not necessary for first layer, but included anyway
    tmp_col = dot(kernel.T(), tmp_dst);
    in = pack_col2patch(tmp_col, in.shape_, ksize, ksize, kstride, kstride, 1, 1);
  }
 private:
  // random seed generator
  Random<xpu, real_t> rnd;
  // kernel size, pooling size
  int ksize, kstride, psize;
  // nodes in neural net
  TensorContainer<xpu, 4, real_t> ninput, nhidden, nhiddenbak, npool, npoolbak;
  TensorContainer<xpu, 2, real_t> nflat, nout;
  // temp helper structure
  TensorContainer<xpu, 2, real_t> tmp_col, tmp_dst;
  // hidden bias, gradient
  TensorContainer<xpu, 1, real_t> hbias, obias, g_hbias, g_obias;
  // weight, gradient: Ki2h is actually convoltuion kernel, with shape=(num_channel,ksize*ksize)
  TensorContainer<xpu, 2, real_t> Ki2h,  Wh2o, g_Ki2h, g_Wh2o;
};

// helper function to get the max inde
inline int MaxIndex(Tensor<cpu, 1, real_t> pred) {
  int maxidx = 0;
  for (index_t i = 1; i < pred.size(0); ++i) {
    if(pred[i] > pred[maxidx]) maxidx = (int)i;
  }
  return maxidx;
}

int main(int argc, char *argv[]) {
  if(argc < 2) {
    printf("Usage: cpu or gpu\n"); return 0;
  }
  srand(0);
  // settings
  int batch_size = 100;
  int insize = 28;
  int nchannel = 10;
  int ksize = 5;
  int kstride = 1;
  int psize = 2;
  int num_out = 10;

  // choose which version to use
  INNet *net;
  if (!strcmp(argv[1], "gpu")) {
    InitTensorEngine<gpu>();
    net = new ConvNet<gpu>(batch_size, insize, nchannel, ksize, kstride, psize, num_out);
  } else {
    InitTensorEngine<cpu>();
    net = new ConvNet<cpu>(batch_size, insize, nchannel, ksize, kstride, psize, num_out);
  }

  // temp output layer
  TensorContainer<cpu, 2, real_t> pred;
  pred.Resize(Shape2(batch_size, num_out));

  // label
  std::vector<int> ytrain, ytest;
  // data
  TensorContainer<cpu, 2, real_t> xtrain_, xtest_;
  LoadMNIST("train-images-idx3-ubyte", "train-labels-idx1-ubyte", ytrain, xtrain_, true);
  LoadMNIST("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", ytest, xtest_, false);

  TensorContainer<cpu, 4, real_t> xtrain(Shape4(xtrain_.size(0), 1, insize, insize));
  TensorContainer<cpu, 4, real_t> xtest(Shape4(xtest_.size(0),  1, insize, insize));
  xtrain = reshape(xtrain_, xtrain.shape_);
  xtest = reshape(xtest_, xtest.shape_);

  int num_iter = 20;

  for (int i = 0; i < num_iter; ++ i) {
    // training
    for (index_t j = 0; j + batch_size <= xtrain.size(0); j += batch_size) {
      net->Forward(xtrain.Slice(j, j + batch_size), pred);
      // set gradient into pred
      for (int k = 0; k < batch_size; ++ k) {
        pred[k][ ytrain[k+j] ] -= 1.0f;
      }
      // scale gradient by batchs zie
      pred *= 1.0f / batch_size;
      // run backprop
      net->Backprop(pred);
      // update net parameters
      net->Update();
    }
    // evaluation
    long nerr = 0;
    for (index_t j = 0; j + batch_size <= xtest.size(0); j += batch_size) {
      net->Forward(xtest.Slice(j, j + batch_size), pred);
      for (int k = 0; k < batch_size; ++ k) {
        nerr += MaxIndex(pred[k]) != ytest[j+k];
      }
    }
    printf("round %d: test-err=%f\n", i, (float)nerr/xtest.size(0));
  }
  delete net;

  if (!strcmp(argv[1], "gpu")) {
    ShutdownTensorEngine<gpu>();
  } else {
    ShutdownTensorEngine<cpu>();
  }
  return 0;
}
