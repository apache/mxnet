/*!
 *  Copyright (c) 2014 by Contributors
 * \file ps_dist-inl.h
 * \brief distributed version of PS
 *
 * \author Tianqi Chen, Mu Li
 */
#ifndef MSHADOW_PS_DIST_INL_H_ // NOLINT(*)
#define MSHADOW_PS_DIST_INL_H_ // NOLINT(*)

#include <vector>
#include "./mshadow_ps.h"
#include "./ps_local-inl.h"

#if MSHADOW_DIST_PS
#include "parameter/kv_layer.h"
namespace mshadow {
namespace ps {

/**
 * @brief bridge IModelUpdater to KVLayerUpdater
 */
template<typename DType>
class UpdaterWrapper {
 public:
  explicit UpdaterWrapper(IModelUpdater<DType> * updater)
      : updater_(updater) { }
  ~UpdaterWrapper() { delete updater_; }

  /// @brief initialize the data
  void Init(int id, size_t size, DType* data) {
    updater_->InitModel(id, data, size);
  }

  /// @brief update the model by using received data
  void Update(int id, size_t size, const DType* recv_data, DType* data) {
    updater_->Update(id, (DType*)recv_data, size);  // NOLINT(*)
  }
 private:
  IModelUpdater<DType> *updater_;
};


template<typename xpu, typename DType>
class DistModel : public LocalModel<xpu, DType> {
 public:
  // parent type
  typedef LocalModel<xpu, DType> Parent;

  // initialize the parameter server
  virtual void Init(const std::vector<int> &devices) {
    Parent::Init(devices);
    if (this->custom_server != NULL) {
      delete this->custom_server;
      this->custom_server = NULL;
    }
  }
  virtual ~DistModel(void) {
  }

 protected:
  // do nothing
  virtual void InitCustomerServer(void) {
  }
  virtual void ServerInitKey(Tensor<cpu, 2> weight, int key) {
    // this is called when key get initialized for the first time
    // weight can be used to hold the model that pulled back
    // use this to initialize the key on serverside
    shared_model_.Pull(
        ::ps::Parameter::Request(key), weight.dptr_, weight.MSize(),
        [this, weight, key]() {
          // call PullReady to notify LocalServer pulling is ready
          this->PullReady(weight, key);
        });
  }
  // override this function, to use parameter server
  virtual void HandlePushFinish(Tensor<cpu, 3, DType> data,
                                int key) {
    // summation the data fron all devices
    LocalModel<xpu, DType>::ReduceSum(data);

    // push and pull
    Tensor<cpu, 2> sendrecv = data[0];
    CHECK_EQ(data[0].CheckContiguous(), true) << "data must be contiguous";

    int ts = shared_model_.Push(
        ::ps::Parameter::Request(key), sendrecv.dptr_, sendrecv.MSize(), false);

    // let this pull request wait the push finish at the server node
    shared_model_.Pull(
        ::ps::Parameter::Request(key, -1, {ts}), sendrecv.dptr_, sendrecv.MSize(),
        [this, sendrecv, key]() {
          // call PullReady to notify LocalServer pulling is ready
          this->PullReady(sendrecv, key);
        });
  }

 private:
  ::ps::KVLayer<DType, UpdaterWrapper<DType> > shared_model_;
};


template<typename DType>
class MShadowServerNode {
 public:
  // conf: get from the flag -app_conf
  MShadowServerNode(int argc, char *argv[]) {
    IModelUpdater<DType> *updater = CreateModelUpdater<DType>();
    updater->InitUpdater(::ps::MyRank(), argc, argv);

    UpdaterWrapper<DType> *wrapper = new UpdaterWrapper<DType>(updater);
    typedef ::ps::KVLayer<DType, UpdaterWrapper<DType> > PSServer;
    PSServer *shared_model_ = new PSServer();
    shared_model_->set_updater(wrapper);
    ::ps::Postoffice::instance().manager().TransferCustomer(
         CHECK_NOTNULL(shared_model_));
  }
  virtual ~MShadowServerNode() { }
};

// NOTE: do not add PS::CreateServer here add it in the program that uses
// mshadow-ps
}  // namespace ps
}  // namespace mshadow
#endif
#endif  // MSHADOW_PS_DIST_INL_H_  NOLINT(*)
