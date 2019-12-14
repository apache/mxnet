/*!
 *  Copyright (c) 2014 by Contributors
 * \file ps_local-inl.h
 * \brief local multi-threading implementation of PS abstraction
 *
 * \author Tianqi Chen, Mu Li
 */
#ifndef MSHADOW_PS_LOCAL_INL_H_  // NOLINT(*)
#define MSHADOW_PS_LOCAL_INL_H_  // NOLINT(*)
#include <map>
#include <utility>
#include <string>
#include <vector>
#if defined(_OPENMP)
#include <omp.h>
#ifdef _MSC_VER
typedef int ms_omp_uint;
#else
typedef unsigned ms_omp_uint;
#endif
#endif

#include "./thread.h"
#include "./thread_util.h"

namespace mshadow {
namespace ps {
// multi-threaded implementation of
template<typename xpu, typename DType>
class LocalModel : public ISharedModel<xpu, DType> {
 public:
  // redefine callback function
  typedef typename ISharedModel<xpu, DType>::CallbackFunction
  CallbackFunction;
  // constructor
  LocalModel(void) {
    init_end = 0;
    perdev_pull_thread = 1;
    perdev_push_thread = 1;
    use_fifo_push_queue = 0;
    bigarray_bound = 1000 * 1000;
    nthread_reduction = 8;
    use_pin_memory = 1;
    test_on_server = 0;
    update_on_server = 0;
    destroy_signal = false;
    custom_server = NULL;
  }
  // destructor
  virtual ~LocalModel(void) {
    this->Destroy();
  }
  inline void Destroy(void) {
    if (init_end != 0) {
      destroy_signal = true;
      for (size_t i = 0; i < push_queues.size(); ++i) {
        push_queues[i].Abort(1);
      }
      for (size_t i = 0; i < pull_queues.size(); ++i) {
        pull_queues[i].Abort(1);
      }
      for (size_t i = 0; i < thread_push_handler.size(); ++i) {
        thread_push_handler[i].Join();
      }
      for (size_t i = 0; i < thread_pull_handler.size(); ++i) {
        thread_pull_handler[i].Join();
      }
      for (size_t i = 0; i < push_queues.size(); ++i) {
        push_queues[i].Destroy();
      }
      push_map.Destroy();
      push_lock.Destroy();
      for (size_t i = 0; i < pull_queues.size(); ++i) {
        pull_queues[i].Destroy();
      }
      pull_map.Destroy();
      request_lock.Destroy();
      wait_lock.Destroy();
      wait_cond.Destroy();
      init_end = 0;
    }
    if (custom_server != NULL) {
      delete custom_server;
      custom_server = NULL;
    }
  }
  virtual void SetParam(const char *name, const char *val) {
    int key;
    if (sscanf(name, "push_op[%d]", &key) == 1) {
      if (!strcmp(val, "gather")) {
        request_lock.Lock();
        push_operation[key] = kGather;
        request_lock.Unlock();
        return;
      }
      if (!strcmp(val, "sum")) {
        push_operation[key] = kSum; return;
      }
      LOG(FATAL) << "unknown push operation " << val;
    }
    if (!strcmp(name, "reduce_thread")) {
      nthread_reduction = atoi(val);
    }
    if (!strcmp(name, "use_pin_memory")) {
      use_pin_memory = atoi(val);
    }
    if (!strcmp(name, "bigarray_bound")) {
      bigarray_bound = static_cast<size_t>(atol(val));
    }
    if (!strcmp(name, "pull_thread")) {
      if (!strcmp(val, "ndev")) {
        perdev_pull_thread = 1;
      } else if (!strcmp(val, "one")) {
        perdev_pull_thread = 0;
      } else {
        LOG(FATAL) << "invalid value for parameter pull_thread," << " can only be ndev or one";
      }
    }
    if (!strcmp(name, "push_thread")) {
      if (!strcmp(val, "ndev")) {
        perdev_push_thread = 1;
      } else if (!strcmp(val, "one")) {
        perdev_push_thread = 0;
      } else {
        LOG(FATAL) << "invalid value for parameter push_thread," << " can only be ndev or one";
      }
    }
    if (!strcmp(name, "update_on_server")) {
      update_on_server = atoi(val);
    }
    if (!strcmp(name, "test_on_server")) {
      test_on_server = atoi(val);
    }
    // ignore message parameter
    if (!strncmp(name, "msg:", 4)) return;
    cfgvec.push_back(std::make_pair(std::string(name),
                                    std::string(val)));
  }
  virtual void PullWait(int key, int devid) {
    const int wid = GetWorkIndex(devid);
    PullEntry *p = pull_map.Get(key);
    if (p == NULL || p->wait.size() == 0) return;
    PullEntry &e = *p;
    // wake up waiters if any
    CHECK_EQ(e.wait.size(), devices.size()) << "PullWait: must initialize the wait";
    PullWaitRecord &w = e.wait[wid];
    if (!w.finished) {
      wait_lock.Lock();
      w.nwait += 1;
      while (!w.finished) {
        wait_cond.Wait(&wait_lock);
      }
      w.nwait -= 1;
      CHECK_GE(w.nwait, 0) << "boundary check";
      wait_lock.Unlock();
    }
  }
  virtual void Init(const std::vector<int> &devices) {
    CHECK_EQ(init_end, 0) << "LocalServer.Init can only call Init once";
    CHECK_NE(devices.size(), 0) << "LocalServer.Init: must at least contain 1 devices";
    this->devices = devices;
    destroy_signal = false;
    // initialize device id to local index
    dev2index.clear();
    for (size_t i = 0; i < devices.size(); ++i) {
      int devid = devices[i];
      CHECK_GE(devid, 0) << "device id must be bigger than 0";
      if (devid >= static_cast<int>(dev2index.size())) {
        dev2index.resize(devid + 1, -1);
      }
      dev2index[devid] = static_cast<int>(i);
    }
    // allocate space
    pull_stream.resize(devices.size());
    push_stream.resize(devices.size());
    // initialize all the thread related things
    if (perdev_push_thread != 0) {
      push_queues.resize(devices.size());
    } else {
      push_queues.resize(1);
    }
    for (size_t i = 0; i < push_queues.size(); ++i) {
      push_queues[i].Init(use_fifo_push_queue != 0);
    }
    push_map.Init();
    push_lock.Init();
    pull_map.Init();
    request_lock.Init();
    wait_lock.Init();
    wait_cond.Init();
    if (perdev_pull_thread != 0) {
      pull_queues.resize(devices.size());
    } else {
      pull_queues.resize(1);
    }
    for (size_t i = 0; i < pull_queues.size(); ++i) {
      pull_queues[i].Init();
    }
    // initialize the thread
    if (perdev_push_thread != 0) {
      thread_push_handler.resize(devices.size());
      for (size_t i = 0; i < devices.size(); ++i) {
        std::pair<LocalModel*, size_t> *p
            = new std::pair<LocalModel*, size_t>();
        *p = std::make_pair(this, i);
        thread_push_handler[i].Start(PushLocalThread, p);
      }
    } else {
      thread_push_handler.resize(1);
      thread_push_handler[0].Start(PushGlobalThread, this);
    }
    // initialize pull handler
    if (perdev_pull_thread != 0) {
      thread_pull_handler.resize(devices.size());
      for (size_t i = 0; i < devices.size(); ++i) {
        std::pair<LocalModel*, size_t> *p
            = new std::pair<LocalModel*, size_t>();
        *p = std::make_pair(this, i);
        thread_pull_handler[i].Start(PullLocalThread, p);
      }
    } else {
      thread_pull_handler.resize(1);
      thread_pull_handler[0].Start(PullGlobalThread, this);
    }
    this->InitCustomerServer();
    this->init_end = 1;
  }

  // set weight
  virtual void SetWeight_(Tensor<xpu, 2, DType> data,
                          int key,
                          int devid) {
    PushEntry &e = push_map.GetRef(key);
    Stream<xpu> s;
    push_lock.Lock();
    mshadow::Copy(e.weight, data, &s);
    push_lock.Unlock();
  }
  virtual void CheckWeight_(Tensor<xpu, 2, DType> data,
                            int key,
                            int devid) {
    CHECK_NE(test_on_server, 0) << "must be in pair debug mode";
    PushEntry &e = push_map.GetRef(key);
    mshadow::TensorContainer<cpu, 2, DType> tmp(false);
    tmp.Resize(data.shape_);
    Stream<xpu> s;
    push_lock.Lock();
    // copy data
    mshadow::Copy(tmp, data, &s);
    index_t count = tmp.shape_.Size();
    double diff = 0.0, ssum = 0.0, maxdiff = 0.0;
    index_t mxidx = 0;
    for (index_t i = 0; i < count; ++i) {
      double d = std::abs(tmp.dptr_[i] - e.weight.dptr_[i]);
      if (d > maxdiff) {
        maxdiff = d; mxidx = i;
      }
      diff += d;
      ssum += std::abs(tmp.dptr_[i]);
    }
    push_lock.Unlock();
    // relative absolute error
    double rerr = diff / ssum;
    if (rerr > 1e-5 || diff != diff) {
      fprintf(stderr, "PSLocal:key=%d,dev=%d: err=%f, maxd[%u]=%f, diff=%f, ssum=%f\n",
              key, devid, rerr, mxidx, maxdiff, diff, ssum);
    } else {
      fprintf(stderr, "PSLocal:key=%d,dev=%d:check pass\n", key, devid);
    }
  }

 protected:
  /*! \brief operation performed locally in PS */
  enum LocalOp {
    /*! \brief take sum of all devices over the same key */
    kSum = 0,
    /*!
     * \brief concatenate(gather),
     *  the tensors in all devices with same key
     */
    kGather = 1
  };
  virtual void InitKey_(Shape<2> shape,
                        int key, int devid) {
    this->InitPullMap(key);
    this->InitPushMap(key, shape);
  }
  virtual void Push_(Tensor<xpu, 2, DType> data,
                     int key, int devid, int priority) {
    PullEntry &e = pull_map.GetRef(key);
    e.req[GetWorkIndex(devid)].ready = false;
    if (perdev_push_thread != 0) {
      int wid = GetWorkIndex(devid);
      push_queues[wid].Push(PullTask(data, key, devid), priority);
    } else {
      push_queues[0].Push(PullTask(data, key, devid), priority);
    }
  }
  virtual void PullReq_(Tensor<xpu, 2, DType> data,
                        int key, int devid, int priority,
                        CallbackFunction callback,
                        void *callback_arg) {
    PullEntry &e = pull_map.GetRef(key);
    CHECK_EQ(e.req.size(), devices.size()) << "PullReq: must initialize the key, req";
    CHECK_EQ(e.wait.size(), devices.size()) << "PullReq: must initialize the key, wait";
    const int wid = GetWorkIndex(devid);
    PullReqRecord &r = e.req[wid];
    r.dest = data;
    r.priority = priority;
    r.callback = callback;
    r.callback_arg = callback_arg;
    // reset pull request finish mark
    wait_lock.Lock();
    e.wait[wid].finished = false;
    wait_lock.Unlock();
    // check ready event
    request_lock.Lock();
    CHECK_EQ(!r.pending, true) << "key = " << key
      << "cannot send duplicate pull request before it finishes";
    if (e.req[wid].ready) {
      if (perdev_pull_thread != 0) {
        pull_queues[wid].Push(std::make_pair(key, devid));
      } else {
        pull_queues[0].Push(std::make_pair(key, devid));
      }
    } else {
      r.pending = true;
    }
    request_lock.Unlock();
  }
  /*!
   * \brief called to notify that the data is ready for pull
   * \param data the data that can be pulled back
   * \param the key of the data
   */
  virtual void PullReady(Tensor<cpu, 2> data, int key) {
    PullEntry &e = pull_map.GetRef(key);
    CHECK_EQ(e.req.size(), devices.size()) << "PullReady: must initialize the key, req";
    request_lock.Lock();
    e.src = data;
    for (index_t i = 0; i < e.req.size(); ++i) {
      e.req[i].ready = true;
      if (e.req[i].pending) {
        if (perdev_pull_thread != 0) {
          pull_queues[i].Push(std::make_pair(key, devices[i]));
        } else {
          pull_queues[0].Push(std::make_pair(key, devices[i]));
        }
        e.req[i].pending = false;
      }
    }
    request_lock.Unlock();
  }
  virtual void ServerInitKey(Tensor<cpu, 2> weight, int key) {
    if (custom_server != NULL) {
      // intialize server, and ready for pullback
      custom_server->InitModel(key, weight.dptr_, weight.MSize());
      if (update_on_server != 0) {
        this->PullReady(weight, key);
      }
    }
  }
  /*!
   * \brief event handler for push finish
   *  called when all the data with same key comes int
   * \param data the buffer holds the data in all devices
   * \param key the key of the data
   */
  virtual void HandlePushFinish(Tensor<cpu, 3, DType> data,
                                int key) {
    // LOG(ERROR) << dbstr(data);
    LocalOp op = kSum;
    typename std::map<int, LocalOp>::const_iterator
        it = push_operation.find(key);
    if (it != push_operation.end() && it->first == key) {
      op = it->second;
    }
    // customized server
    if (custom_server != NULL) {
      this->ReduceSum(data);
      custom_server->Update(key, data[0].dptr_, data[0].MSize());
      if (update_on_server != 0) {
        PushEntry &e = push_map.GetRef(key);
        this->PullReady(e.weight, key);
      } else {
        CHECK_NE(test_on_server, 0) << "test mode";
        this->PullReady(data[0], key);
      }
      return;
    }
    switch (op) {
      case kSum: {
        this->ReduceSum(data);
        this->PullReady(data[0], key);
        return;
      }
      case kGather: {
        this->PullReady(data.FlatTo2D(), key);
        return;
      }
      default: LOG(FATAL) << "unknown LocalOp";
    }
  }
  /*!
   * \brief event handler for reduce finish
   *  called when all the data with same key finishes the reduction
   * \param data the buffer holds the reduction result
   * \param key the key of the data
   */
  inline void HandleReduceFinish(Tensor<cpu, 2, DType> data,
                                 int key) {
    if (custom_server != NULL) {
      custom_server->Update(key, data.dptr_, data.MSize());
      if (update_on_server != 0) {
        PushEntry &e = push_map.GetRef(key);
        this->PullReady(e.weight, key);
      } else {
        CHECK_NE(test_on_server, 0) << "test mode";
        this->PullReady(data, key);
      }
    } else {
      this->PullReady(data, key);
    }
  }
  virtual void InitCustomerServer(void) {
    if (update_on_server != 0 || test_on_server != 0) {
      custom_server = CreateModelUpdater<DType>();
      for (size_t j = 0; j < cfgvec.size(); ++j) {
        custom_server->SetParam(cfgvec[j].first.c_str(),
                                cfgvec[j].second.c_str());
      }
      custom_server->InitUpdater(0, 0, NULL);
    }
  }

 protected:
  // customized server
  IModelUpdater<DType> *custom_server;
  // whether use fifo push queue
  int use_fifo_push_queue;

  // perform sum reduction
  inline void ReduceSum(Tensor<cpu, 3, DType> data) {
    #if defined(_OPENMP)
    if (data[0].MSize() >= bigarray_bound &&
        nthread_reduction != 0) {
      ms_omp_uint ntask = static_cast<ms_omp_uint>(data.size(1));
      #pragma omp parallel for schedule(static) num_threads(nthread_reduction)
      for (ms_omp_uint j = 0; j < ntask; ++j) {
        for (index_t i = 1; i < data.size(0); ++i) {
          data[0][j] += data[i][j];
        }
      }
    } else  //NOLINT(*)
      #endif
    {
      for (index_t i = 1; i < data.size(0); ++i) {
        data[0] += data[i];
      }
    }
  }

 private:
  /*! \brief task running */
  struct PullTask {
    /*! \brief the task data source */
    Tensor<xpu, 2, DType> data;
    /*! \brief the key to the tensor */
    int key;
    /*!
     * \brief the device id, (key,devid),
     * uniquely identifies a mem location
     */
    int devid;
    PullTask(void) {}
    PullTask(Tensor<xpu, 2, DType> data, int key, int devid)
        : data(data), key(key), devid(devid) {}
  };
  /*! \brief data structure to hold temporal push result */
  struct PushEntry {
    // temporal space to hold input data
    Tensor<cpu, 4, DType> data;
    // temporal space to hold weight, if needed
    Tensor<cpu, 2, DType> weight;
    // indicator whether the certain devices is already copied in
    std::vector<bool> copied;
    // number of data copied in
    int num_copied;
    // version number of data used to hold incomming data in push
    int copyin_version;
    // use pinned memory
    bool pin_memory;
    // constructor
    PushEntry(void)
        : copyin_version(0) {
      weight.dptr_ = NULL;
    }
    ~PushEntry(void) {
      if (data.dptr_ != NULL) {
        if (pin_memory) {
          mshadow::FreeHost<xpu>(&data);
          if (weight.dptr_ != NULL) {
            mshadow::FreeHost<xpu>(&weight);
          }
        } else {
          mshadow::FreeSpace(&data);
          if (weight.dptr_ != NULL) {
            mshadow::FreeSpace(&weight);
          }
        }
      }
    }
    // constructor
    inline void Init(int ndevice, Shape<2> shape,
                     bool pin_memory, bool need_weight) {
      this->pin_memory = pin_memory;
      data.shape_ = Shape4(2, ndevice, shape[0], shape[1]);
      weight.shape_ = shape;
      if (pin_memory) {
        mshadow::AllocHost<xpu>(&data);
        if (need_weight) mshadow::AllocHost<xpu>(&weight);
      } else {
        mshadow::AllocSpace(&data, false);
        if (need_weight) mshadow::AllocSpace(&weight);
      }
      CHECK_EQ(data.CheckContiguous(), true) << "Data must be contiguous";
      CHECK(!need_weight || weight.CheckContiguous()) << "Weight must be contiguous";
      num_copied = 0;
      copied.resize(ndevice, false);
    }
  };
  // a record to remember things related to pull request
  struct PullReqRecord {
    // whether this record contains a pending request
    // whether pull is ready to go
    bool ready;
    // waiting for pull ready
    bool pending;
    // the destination to pull data into
    Tensor<xpu, 2, DType> dest;
    // the priority of the
    int priority;
    // callback function
    CallbackFunction *callback;
    // argument for callback
    void *callback_arg;
    PullReqRecord(void) : ready(false), pending(false) {
    }
  };
  // a record to help handle pullwait
  struct PullWaitRecord {
    // number of thread that waits for the request to finish
    int nwait;
    // the request was finished
    bool finished;
    PullWaitRecord(void)
        : nwait(0), finished(true) {
      // set finished to true so pull without pull request returns
    }
  };
  /*! \brief data structure to hold pull request */
  struct PullEntry {
    // data to be pulled back
    Tensor<cpu, 2, DType> src;
    // pullrequest record
    std::vector<PullReqRecord> req;
    // whether there is thread waiting on this event
    std::vector<PullWaitRecord> wait;
    PullEntry(void) {
    }
  };
  // signal to notify all the thread about class destruction
  bool destroy_signal;
  // vector of devices
  std::vector<int> devices;
  // device index to local index
  std::vector<int> dev2index;
  //----- data structure used to support push ----
  // stream used by push thread each device for memcpy
  std::vector<Stream<xpu>*> push_stream;
  // the queue used for push task
  std::vector<utils::ThreadPQueue<PullTask> > push_queues;
  // thread to handle push task
  std::vector<utils::Thread> thread_push_handler;
  // lock to lock push field
  utils::Mutex push_lock;
  // the map of push buffer
  utils::ThreadSafeMap<PushEntry> push_map;
  // customized local reduction operation
  std::map<int, LocalOp> push_operation;
  //----- data structure used to support pull ----
  // the queue used for pull task
  std::vector<utils::ThreadPQueue<std::pair<int, int> > > pull_queues;
  // stream used by pull thread each device for memcpy
  std::vector<Stream<xpu>*> pull_stream;
  // the map to store pull status
  utils::ThreadSafeMap<PullEntry> pull_map;
  // thread to handle pull task
  std::vector<utils::Thread> thread_pull_handler;
  // lock to lock request field
  utils::Mutex request_lock;
  // lock to lock wait field
  utils::Mutex wait_lock;
  // conditional variable to do waiting
  utils::ConditionVariable wait_cond;
  // ---------configurations of server-------
  int init_end;
  // whether perform update on serverside
  int update_on_server;
  // debug option
  int test_on_server;
  // use pinned memory
  int use_pin_memory;
  // number of reduction thread
  int nthread_reduction;
  // the threshold for big array
  size_t bigarray_bound;
  // whether use pull thread per device
  int perdev_pull_thread;
  // whether use push thread per device
  int perdev_push_thread;
  /*! \brief history of configurations */
  std::vector< std::pair<std::string, std::string> > cfgvec;
  // push handler
  inline void PushProc(utils::ThreadPQueue<PullTask> *queue) {
    while (!destroy_signal) {
      PullTask tsk;
      if (queue->Pop(&tsk)) {
        const int wid = GetWorkIndex(tsk.devid);
        PushEntry &e = push_map.GetRef(tsk.key);
        CHECK_EQ(e.data[0][0].shape_, tsk.data.shape_)
          << "Tensor with same key must share same shape "
          << e.data[0][0].shape_
          << " vs "
          << tsk.data.shape_;
        CHECK_EQ(!e.copied[wid], true) << "data inconsistency";
        // start copy
        SetDevice<xpu>(tsk.devid);
        Copy(e.data[e.copyin_version][wid], tsk.data, push_stream[wid]);
        // wait till the copy finishes
        push_stream[wid]->Wait();
        // mark copied
        e.copied[wid] = true;
        push_lock.Lock();
        e.num_copied += 1;
        int cp_version = e.copyin_version;
        bool push_finish = e.num_copied >= static_cast<int>(devices.size());
        if (push_finish) {
          // switch version
          e.copyin_version = (e.copyin_version + 1) % e.data.size(0);
          std::fill(e.copied.begin(), e.copied.end(), false);
          e.num_copied = 0;
        }
        push_lock.Unlock();
        if (push_finish) {
          this->HandlePushFinish(e.data[cp_version], tsk.key);
        }
      } else {
        CHECK_EQ(destroy_signal, true) << "abort but not destroy";
      }
    }
  }
  inline void PushHandlerGlobal(void) {
    // allocate stream resources
    for (size_t i = 0; i < devices.size(); ++i) {
      SetDevice<xpu>(devices[i]);
      push_stream[i] = NewStream<xpu>(devices[i]);
    }
    this->PushProc(&push_queues[0]);
    // free resources
    for (size_t i = 0; i < devices.size(); ++i) {
      SetDevice<xpu>(devices[i]);
      DeleteStream(push_stream[i]);
    }
  }
  inline void PushHandlerLocal(size_t tid) {
    CHECK_LT(tid, devices.size()) << "threadid exceed boundary";
    CHECK_EQ(push_queues.size(), devices.size()) << "must have one pull_queue per device";
    // allocate stream resources
    SetDevice<xpu>(devices[tid]);
    push_stream[tid] = NewStream<xpu>(devices[tid]);
    this->PushProc(&push_queues[tid]);
    SetDevice<xpu>(devices[tid]);
    DeleteStream(push_stream[tid]);
  }
  /*!\brief entry point of loader thread */
  inline static MSHADOW_THREAD_PREFIX PushGlobalThread(void *pthread) {
    static_cast<LocalModel*>(pthread)->PushHandlerGlobal();
    utils::ThreadExit(NULL);
    return NULL;
  }
  inline static MSHADOW_THREAD_PREFIX PushLocalThread(void *arg) {
    std::pair<LocalModel*, size_t> *p
        = static_cast<std::pair<LocalModel*, size_t>*>(arg);
    p->first->PushHandlerLocal(p->second);
    delete p;
    return NULL;
  }
  // push handler procedure
  inline void PullProc(utils::ThreadPQueue<std::pair<int, int> > *queue) {
    while (!destroy_signal) {
      std::pair<int, int> tsk;
      if (queue->Pop(&tsk)) {
        const int key = tsk.first;
        const int devid = tsk.second;
        const int wid = GetWorkIndex(devid);
        PullEntry &e = pull_map.GetRef(key);
        {
          // handle request
          CHECK_EQ(e.req.size(), devices.size()) << "PullHandler: must initialize the key, req";
          PullReqRecord &r = e.req[wid];
          SetDevice<xpu>(devid);
          Copy(r.dest, e.src, pull_stream[wid]);
          // callback, if any
          if (r.callback != NULL) {
            (*r.callback)(pull_stream[wid], r.callback_arg);
          }
          // wait till the operation finishes
          pull_stream[wid]->Wait();
        }
        {
          // wake up waiters if any
          CHECK_EQ(e.wait.size(), devices.size()) << "PullHandler, must initialize the key, req";
          PullWaitRecord &w = e.wait[wid];
          wait_lock.Lock();
          w.finished = true;
          if (w.nwait != 0) {
            wait_cond.Broadcast();
          }
          wait_lock.Unlock();
        }
      } else {
        CHECK_EQ(destroy_signal, true) << "abort but not destroy";
      }
    }
  }
  // use one thread for all pull actions
  inline void PullHandlerGlobal(void) {
    // allocate stream resources
    for (size_t i = 0; i < devices.size(); ++i) {
      SetDevice<xpu>(devices[i]);
      pull_stream[i] = NewStream<xpu>(devices[i]);
    }
    this->PullProc(&pull_queues[0]);
    // free resources
    for (size_t i = 0; i < devices.size(); ++i) {
      SetDevice<xpu>(devices[i]);
      DeleteStream(pull_stream[i]);
    }
  }
  inline void PullHandlerLocal(size_t tid) {
    CHECK_LT(tid, devices.size()) << "threadid exceed boundary";
    CHECK_EQ(pull_queues.size(), devices.size()) << "must have one pull_queue per device";
    // allocate stream resources
    SetDevice<xpu>(devices[tid]);
    pull_stream[tid] = NewStream<xpu>(devices[tid]);
    this->PullProc(&pull_queues[tid]);
    SetDevice<xpu>(devices[tid]);
    DeleteStream(pull_stream[tid]);
  }
  /*!\brief entry point of pull thread, one thread for all devices */
  inline static MSHADOW_THREAD_PREFIX PullGlobalThread(void *arg) {
    static_cast<LocalModel*>(arg)->PullHandlerGlobal();
    return NULL;
  }
  inline static MSHADOW_THREAD_PREFIX PullLocalThread(void *arg) {
    std::pair<LocalModel*, size_t> *p
        = static_cast<std::pair<LocalModel*, size_t>*>(arg);
    p->first->PullHandlerLocal(p->second);
    delete p;
    return NULL;
  }
  // get internal index of device
  inline int GetWorkIndex(int devid) const {
    CHECK(devid >= 0 &&
          devid < static_cast<int>(dev2index.size()) &&
          dev2index[devid] >= 0) << "Push: invalid devid";
    return dev2index[devid];
  }
  // functions to handle pull
  inline void InitPullMap(int key) {
    pull_map.Init(key);
    PullEntry &e = pull_map.GetRef(key);
    request_lock.Lock();
    // must recheck after lock
    if (e.req.size() == 0) {
      e.req.resize(devices.size(), PullReqRecord());
    }
    request_lock.Unlock();
    // check wait map
    wait_lock.Lock();
    // must recheck after lock
    if (e.wait.size() == 0) {
      e.wait.resize(devices.size(), PullWaitRecord());
    }
    wait_lock.Unlock();
  }
  // functions to handle pull
  inline void InitPushMap(int key, Shape<2> shape) {
    push_map.Init(key);
    PushEntry &e = push_map.GetRef(key);
    push_lock.Lock();
    if (e.copied.size() == 0) {
      e.Init(devices.size(), shape,
             use_pin_memory != 0,
             update_on_server != 0 || test_on_server != 0);
    }
    this->ServerInitKey(e.weight, key);
    push_lock.Unlock();
  }
};
}  // namespace ps
}  // namespace mshadow
#endif // MSHADOW_PS_LOCAL_INL_H_  NOLINT(*)
