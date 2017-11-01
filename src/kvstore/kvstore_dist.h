/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/**
 * @file   kvstore_dist.h
 * @brief  distributed implementation based on ps-lite
 */
#ifndef MXNET_KVSTORE_KVSTORE_DIST_H_
#define MXNET_KVSTORE_KVSTORE_DIST_H_
#include <string>
#include <vector>
#include <algorithm>
#include <utility>
#include "./kvstore_local.h"
#include "mxnet/engine.h"
#include "ps/ps.h"
#include "./kvstore_dist_server.h"
#include "../../ps-lite/src/infiniband_van.h"
#include "../../ps-lite/src/PHubAllocator.h"
#if MKL_EXPERIMENTAL == 1
#include <mkl_memory.h>
#include "../operator/mkl/mkl_memory-inl.h"
#include "../operator/mkl/mkl_util-inl.h"
#endif
namespace mxnet {
namespace kvstore {

/**
 * \brief distributed kvstore
 *
 * it's the server node's job to control the data consistency among all
 * workers. see details on \ref ServerHandle::Start
 */
class KVStoreDist : public KVStoreLocal {
 public:
  explicit KVStoreDist(bool use_device_comm)
      : KVStoreLocal(use_device_comm), ps_worker_(nullptr), server_(nullptr) {
    if (IsWorkerNode()) {
      ps_worker_ = new ps::KVWorker<real_t>(0);
      ps::StartAsync("mxnet\0");
      if (!ps::Postoffice::Get()->is_recovery()) {
        ps::Postoffice::Get()->Barrier(
          ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);
      }
    }
    bigarray_bound_ = dmlc::GetEnv("MXNET_KVSTORE_BIGARRAY_BOUND", 512 * 1024);
    CHECK(bigarray_bound_ % 1024 == 0);
    pinned_ctx_ = Context::CPU();
    log_verbose_ = dmlc::GetEnv("MXNET_KVSTORE_DIST_ROW_SPARSE_VERBOSE", false);
    printf("[note]Keys with element larger than %d will be splitted\n", bigarray_bound_);
  }

  virtual ~KVStoreDist() {
    Engine::Get()->WaitForAll();
    if (IsWorkerNode()) {
      if (barrier_before_exit_) {
        Barrier("KVStore Finalizer");
        if (get_rank() == 0) {
          // stop the executor at servers
          SendCommandToServers(kStopServer, "");
        }
      }
      ps::Finalize(barrier_before_exit_);
      delete ps_worker_;
    }
  }

  void set_updater(const Updater& updater) override {
    CHECK(updater) << "invalid updater";
    if (IsServerNode()) {
      CHECK_NOTNULL(server_)->set_updater(updater);
    } else {
      updater_ = updater;
    }
  }

  void Barrier(std::string name) {
    ps::Postoffice::Get()->Barrier(ps::kWorkerGroup);
  }

  void SendCommandToServers(int cmd_id,
                            const std::string& cmd_body) override {
    CHECK_NOTNULL(ps_worker_);
    ps_worker_->Wait(ps_worker_->Request(cmd_id, cmd_body, ps::kServerGroup));
  }

  int get_group_size() const override { return ps::NumWorkers(); }

  int get_rank() const override { return ps::MyRank(); }

  int get_num_dead_node(int node_id, int timeout) const override {
    int number = 0;
    auto dead_nodes = ps::Postoffice::Get()->GetDeadNodes(timeout);
    const auto& watch_nodes = ps::Postoffice::Get()->GetNodeIDs(node_id);
    std::unordered_set<int> watch_set(watch_nodes.begin(), watch_nodes.end());
    for (int r : dead_nodes) {
      if (watch_set.find(r) != watch_set.end()) number++;
    }
    return number;
  }

  void RunServer(const Controller& controller) override {
    CHECK(!IsWorkerNode());
    if (IsServerNode()) {
      server_ = new KVStoreDistServer();
      server_->set_controller(controller);
    }

    ps::StartAsync("mxnet_server\0");
    if (!ps::Postoffice::Get()->is_recovery()) {
      ps::Postoffice::Get()->Barrier(
        ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);
    }//initialization call.

    //this is the wait for all key size setup call
    if (!ps::Postoffice::Get()->is_recovery()) {
	ps::Postoffice::Get()->Barrier(
	    ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler, "WorkerKeyPopulation");
    }

    //now call to setup infiniband qps and etc.

    if (IsServerNode())
	LOG(INFO) << "[IMPORTANT BOOTSTRAP]Server about to call OnKeyPopulated!";

    ps::Postoffice::Get()->van()->OnKeyPopulated();
    if (!ps::Postoffice::Get()->is_recovery()) {
	ps::Postoffice::Get()->Barrier(
	    ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler, "OnKeyPopulated");
    }

    
    if (server_) server_->Run();
    ps::Finalize();
    if (server_) {
      delete server_;
    }
    server_ = nullptr;
  }

 private:

  //translate a physical key into a virtual key
  std::vector<int> VirtualKeyTranslation;
  //size of each virtual key
  std::vector<int> VirtualKeySizes;
  //translates a virtual key into a physical key.
  std::vector<int> PhysicalKeyTranslation;
  std::vector<std::vector<NDArray*>> PhysicalKeyPullAddress;

  void PopulateVirtualKey(std::vector<int>& size)
  {
      int nextIndex = 0;
      for (int i = 0; i < size.size(); i++)
      {
	  int pieces = 1;
	  if (size[i] > bigarray_bound_)
	  {
	      pieces = (int)ceil(1.0 * size[i] / bigarray_bound_);
	  }
	  for (int p = 0; p < pieces; p++)
	  {
	      int sz = 0;
	      if (pieces == 1)
	      {
		  sz = size[i];
		  VirtualKeySizes.push_back(sz);
		  //printf("key = %d, chunk = %d, size = %d\n", i, p, sz);
	      }
	      else if (p != pieces - 1)
	      {
		  //mid part. this is bigarray_bound
		  VirtualKeySizes.push_back(bigarray_bound_);
	      }
	      else
	      {
		  //last piece. shall not be 0.
		  //CHECK((size[i] % bigarray_bound_) != 0);
		  VirtualKeySizes.push_back(size[i] - bigarray_bound_ * p);
	      }
	      PhysicalKeyTranslation.push_back(i);
	      //vkey = VirtualKeySizes.size() - 1
	      //pkey = i.
	  }
	  /*if(Environment::Get()->find("MXNET_PROFILER_EXTENDED_SUPPORT") != NULL)
	        {
		std::string kInfo = "V=";
		kInfo += std::to_string(VirtualKeySizes.size() - 1);
		kInfo += "P=";
		kInfo += std::to_string(i);
		std::string pushStr = "KVPush ";
		pushStr += kInfo;
		std::string pullStr = "KVPull ";
		pullStr += kInfo;
		extendedProfilerSupportPush[i] = pushStr;
		extendedProfilerSupportPull[i] = pullStr;
		}*/
	      
	  VirtualKeyTranslation.push_back(nextIndex);
	  nextIndex += pieces;
      }
      //for(int i = 0; i < VirtualKeySizes.size();i++)
      //    printf("virtual key =  %d size = %d\n",i,VirtualKeySizes[i] * sizeof(float));
      //recalculate. 
      CHECK(PhysicalKeyTranslation.size() == VirtualKeySizes.size());
      CHECK(VirtualKeyTranslation.size() <= PhysicalKeyTranslation.size());
  }

  int RetrievePhysicalKeyFromVirtualKey(int vk)
  {
      CHECK(PhysicalKeyTranslation.size() > 0);
      return PhysicalKeyTranslation[vk];
  }

  int RetrieveVirtualKeySizeFromVirtualKey(int kv)
  {
      CHECK(VirtualKeySizes.size() > 0);
      return VirtualKeySizes[kv];
  }

  int RetrieveVirtualKeyFromPhysicalKey(int key, int part)
  {
      CHECK(VirtualKeyTranslation.size() > 0);
      return VirtualKeyTranslation[key] + part;
  }


  void InitImpl(const std::vector<int>& keys,
                const std::vector<NDArray>& values) override {
    CheckUnique(keys);
    if(ps::Postoffice::Get()->van()->HasFeature(ps::Van::SupportsKeyChunking) == false)
    {
	//ZMQ.
	if(keys[0] < 0)
	{
	    if(ps::Postoffice::Get()->is_recovery() == false)
	    {
		ps::Postoffice::Get()->Barrier(
		    ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler, keys[0] == -1 ? "WorkerKeyPopulation" : "OnKeyPopulated");
		return;
	    }
	}
	for (size_t i = 0; i < keys.size(); ++i) {
	    comm_->Init(keys[i], values[i].storage_type(), values[i].shape(), values[i].dtype());
	}
	if (get_rank() == 0) {
	    printf("Worker 0 is pushing key %d\n", keys[0]);
	    Push_(keys, values, 0, false);
	    // wait until the push is finished
	    for (const auto& v : values) {
		v.WaitToWrite();
	    }
	    // wait until the push is finished
	    for (const int key : keys) {
		comm_buf_[key].WaitToWrite();
	    }
	    printf("Worker 0 has finished pushing key %d\n", keys[0]);	    
	} else {
	    // do nothing
	}
	if (!ps::Postoffice::Get()->is_recovery()) {
	    Barrier("Key Init " + std::to_string(keys[0]));
	}
    }
    else
    {
	//LOG(INFO)<<"Worker Important Bootstrap K = "<<keys[0];
	if (keys.size() != 1 || keys[0] >= 0)
	{
	    //copy whatever code it was originally.
	    CheckUnique(keys);
	    for (size_t i = 0; i < keys.size(); ++i) {
		//if extended profiler is turned on, we need to assign profiler names.
		//const char* opr_name 
		comm_->Init(keys[i], values[i].storage_type(), values[i].shape(), values[i].dtype());
		//first chunk the key if value is too large.
		//int chunks = (int)ceil(1.0  * values[i].shape().Size() / bigarray_bound_);
		//previously we registered the buffer, but we havent resize those yet.
		//remember, buf is per physical key.
		auto& buf = comm_buf_[keys[i]];
		if (buf.is_none())
		{
		    //some keys may not be trainable, but may still be pushed.
		    buf = NDArray(values[i].shape(), Context::CPU());
		}
		else
		{
		    //printf("key resizing key=%d, size=%d\n",keys[i],values[i].shape().Size());
		    buf.ReshapeInternalExact(values[i].shape());
		}
		// printf("initializing key = %d\n",keys[0]);

	    }
	    if (get_rank() == 0) {
		//PHUB auto chunk. No need to chunk for it.
		Push_(keys, values, 0, false);
		// wait until the push is finished
		for (const auto& v : values) {
		    v.WaitToWrite();
		}
		//the above is not sufficient because we copied value to comm_buf, which should in turn be waited for.
		for (auto k : keys)
		{
		    CHECK(comm_buf_[k].is_none() == false);
		    comm_buf_[k].WaitToWrite();
		}

		//in fact, should wait on 
		//printf("rank 0 init %d ready\n", keys[0]);
		LOG(INFO)<<"worker 0 setting up "<<keys.size()<<" keys";
	    }
	    else {
		// do nothing
	    }
	    if (!ps::Postoffice::Get()->is_recovery()) {
		Barrier("rank 0 initialization " + std::to_string(keys[0]));
	    }
	    //printf("rank %d finished key %d\n", get_rank(), keys[0]);
	}
	else if (keys[0] == -1)
	{
	    //this is the initialization code
	    //everyone, setup your key counts!
	    //first figure out how many keys are there.

	    auto keyCounts = values.size();
	    std::vector<int> physicalKeySizes;
	    for (int i = 0; i < keyCounts; i++)
	    {
		physicalKeySizes.push_back(values[i].shape().Size());
	    }
	    PhysicalKeyPullAddress.resize(keyCounts);
	    PopulateVirtualKey(physicalKeySizes);
	    //first, populate PK to VK translation

	    auto allocator = PHubAllocator::Get();
	    CHECK(allocator->IsInitialized() == false);
	    //make sure it's not initialized.
	    //now create a map that allocator uses.
	    //note that we need VIRTUAL keys here.
	    std::unordered_map<int, int> tempVKBuf;
	    for (size_t i = 0; i < VirtualKeySizes.size(); i++)
	    {
		tempVKBuf[i] = VirtualKeySizes[i] * sizeof(float); //allocator speaks bytes, not element counts
	    }

	    //initialize verbs if possible
	    //initialize my verbs.
	    int socketCnt = 1;
	    if (ps::Postoffice::Get()->van()->HasFeature(Van::NativeInfiniband))
	    {
		auto pVan = (InfiniBandVan*)ps::Postoffice::Get()->van();
		int qpCnt = pVan->QPCountOverride > 0 ? pVan->QPCountOverride : (int)tempVKBuf.size();
		//don't have a prefered interface.
		//see if DirectConnect is specified.
		auto dConn = Environment::Get()->find("IB_DIRECT_CONNECT") != NULL;
		pVan->verbs = new Verbs(qpCnt, pVan->GetUnderlyingWorkerThreadCount(),
					(int)ps::Postoffice::Get()->van()->my_node().id, tempVKBuf, dConn, "");
		socketCnt = pVan->verbs->SocketCount;
	    }
	    allocator->Init(tempVKBuf, false, 1, sizeof(ps::MetaSlim), socketCnt, VirtualKeySizes.size() == physicalKeySizes.size());
	    //We have not registered buffer yet.
	    ps_worker_->PHUBDeterministicCallbacks.resize(physicalKeySizes.size(), nullptr);
	    ps_worker_->PHUBDeterministicChunkCounters.resize(physicalKeySizes.size(), 0);
	    ps_worker_->PHUBVirtualToPhysicalKeyMapping = PhysicalKeyTranslation;

	    if (get_rank() == 0)
	    {
		//printf("[warning] PhysicalKey=%d, VirtualKey=%d. pushed to server\n", physicalKeySizes.size(), VirtualKeySizes.size());
		//bypass engine.
		CHECK(sizeof(float) == sizeof(int));
		auto sKeys = ps::SArray<ps::Key>(1, -1);
		auto sVSizes = ps::SArray<real_t>((real_t*)VirtualKeySizes.data(), VirtualKeySizes.size(), false);
		auto sLen = ps::SArray<int>(1, VirtualKeySizes.size());
		//ZPush is used because comm_ isnt initialized.
		ps_worker_->Wait((ps_worker_)->ZPush(sKeys, sVSizes, sLen));
	    }

	    //initialize comm buffers.
	    for (int i = 0; i < keyCounts; i++)
	    {
		auto& buf = comm_buf_[i];
		//defensive
		CHECK(buf.is_none());
		if (buf.is_none()) {
		    //??? what would be the shape?
		    //we currently do not know. just create the same size, the resize it
		    //when we know its exact size later.
		    //Here, instead of using the default allocator, we must use PHUBAllocator if we want 0 copy.
		    //we can learn how to create a NDArray from an existing buffer from kvstore_dist_server.h
		    size_t len;
		    //make sure the allocated buffer is at least as large.
		    auto vk = RetrieveVirtualKeyFromPhysicalKey(i, 0);
		    auto socketIdx = 0;
		    if (ps::Postoffice::Get()->van()->HasFeature(Van::NativeInfiniband))
		    {
			auto verbs = ((InfiniBandVan*)ps::Postoffice::Get()->van())->verbs;
			socketIdx = verbs->Helper_Worker_GetEndpointFromKey(i).SocketIdx;

		    }
		    auto workerBuf = allocator->WorkerKVBuffer(vk, socketIdx, len);
		    //CHECK(len >= cntDbg);
		    //the buffer len allocated should be larger than key (larger for padding reasons).
		    //also the size of PHSYICAL key is used because all virtual keys are contiguous in Worker.
		    TBlob recv_blob((real_t*)workerBuf, // NOLINT(*)
                                    values[i].shape(), cpu::kDevMask);
		    buf = NDArray(recv_blob, 0);

		    //buf = NDArray(values[i].shape(), Context::CPU());
		}
		CHECK(buf.is_none() == false);

	    }

	    //now i know there are many keys.
	    for (int i = 0; i < keyCounts; i++)
	    {
		auto& buf = comm_buf_[i];
		CHECK(buf.is_none() == false);
		CHECK(buf.shape().Size() == values[i].shape().Size());
		//i need to somehow communicate with IBVerbs.
		int chunks = (int)ceil(1.0 * physicalKeySizes[i] / bigarray_bound_);
		//you only need to register once and prefer registering with larger chunks.
		real_t* currentBufferStart = (real_t*)buf.data().dptr_;
		real_t* currentRecvBufferStart = (real_t*)values[i].data().dptr_;
		auto remaining = physicalKeySizes[i];
		for (int c = 0; c < chunks; c++)
		{
		    int curr = remaining >= bigarray_bound_ ? bigarray_bound_ : remaining;
		    if (chunks == 1)
			curr = physicalKeySizes[i];
		    auto vk = RetrieveVirtualKeyFromPhysicalKey(i, c);
		    //printf("[Worker0]VK = %d has size %d, read location = %p, write location = %p\n", vk, curr, currentBufferStart, currentRecvBufferStart);
		    //printf("Worker registration self key=%d\n",vk);
		    CHECK(currentBufferStart != NULL);
		    //Note these two buffers are identical, because we're using comm_buf to synchronzie.
		    //This violates the previous assumption that we can deliver buffer directly without a copy, but since there is GPU a copy is forced.
		    ps::Postoffice::Get()->van()->SetKeySize(vk, sizeof(real_t) * curr, (uint64_t)currentBufferStart, (uint64_t)currentBufferStart);
		    CHECK(curr == RetrieveVirtualKeySizeFromVirtualKey(vk)) << " vk=" << vk << " " << curr << " vs " << RetrieveVirtualKeySizeFromVirtualKey(vk);
		    //... redundant calculations. But initialize code only run once.
		    remaining -= curr;
		    currentBufferStart += curr;
		    currentRecvBufferStart += curr;
		}
		CHECK(remaining == 0);
		//assuming there's only one server.
		//CHECK_EQ(ps::Postoffice::Get()->num_servers(), 1);
		//printf("[%d]Worker local key population\n",ps::Postoffice::Get()->van()->my_node().id);
	    }
	    //so hacky. but this is to signal OnKeyPopulated need to be called.
	    if (!ps::Postoffice::Get()->is_recovery()) {
		ps::Postoffice::Get()->Barrier(
		    ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler, "WorkerKeyPopulation");
	    }
	}
	else if (keys[0] == -2)
	{
	    //everyone, now initiate phase 1 initialization of infiniband.
	    ps::Postoffice::Get()->van()->OnKeyPopulated();
	    if (!ps::Postoffice::Get()->is_recovery()) {
		ps::Postoffice::Get()->Barrier(ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler, "OnKeyPopulated");
	    }
	}
	else
	{
	    CHECK(false);
	}
    }
  }

  void PushImpl(const std::vector<int>& keys,
                const std::vector<NDArray>& values,
                int priority) override {
    Push_(keys, values, priority, true);
  }

  void PullImpl(const std::vector<int>& keys,
                const std::vector<NDArray*>& values,
                int priority) override {
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray*> > grouped_vals;
    GroupKVPairsPull(keys, values, &uniq_keys, &grouped_vals);

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      int key = uniq_keys[i];
      // use the same array for merging to guarantee that pull always happens
      // after the previous push on this key
      auto& recv_buf = comm_buf_[key];
      const auto storage_type = grouped_vals[i][0]->storage_type();
      CHECK_EQ(storage_type, kDefaultStorage)
               << "Expected stype of value to be kDefaultStorage";
      if (recv_buf.is_none()) {
        // it may happen for the first time a no-rank-0 worker pull the weight.
        recv_buf = NDArray(grouped_vals[i][0]->shape(), pinned_ctx_,
                           true, grouped_vals[i][0]->dtype());
      }
      real_t* data = static_cast<real_t*>(recv_buf.data().dptr_);
      size_t size = recv_buf.shape().Size();
      
      if (ps::Postoffice::Get()->van()->HasFeature(ps::Van::PullRequestElision) == true)
      {
	  //This pull is PHYSICAL key.
	  CHECK(uniq_keys.size() == 1);
	  if (PhysicalKeyPullAddress.at(key).size() == 0)
	  {
	      //make sure this van is not fully initialized. otherwise logic issue
	      CHECK(ps::Postoffice::Get()->van()->FullyInitialized() == false);
	      for (auto addr : values)
	      {
		  PhysicalKeyPullAddress.at(key).push_back(addr);
	      }
	  }
	  else
	  {
	      //safely continue.
	      //another way to do this is to still issue a pull if mismatch.
	      //but i dont think the address is really changing.
	      //CHECK(values.size() == 1);
	      CHECK(values.size() == PhysicalKeyPullAddress.at(key).size());
	      for (size_t i = 0; i < values.size(); i++)
	      {
		  CHECK(values[i] == PhysicalKeyPullAddress.at(key).at(i));
	      }
	      //printf("key = %d elision success. \n", key);
	      continue;
	  }
	  //CHECK that elision logic is correct.
      }
      auto pull_from_servers = [this, key, data,size](
          RunContext rctx, Engine::CallbackOnComplete cb) {
        // convert to ps keys
	  //size_t size = recv_buf.shape().Size();
        PSKV& pskv = EncodeKey(key, size);
#if MKL_EXPERIMENTAL == 1
        mkl_set_tblob_eager_mode(recv_buf.data());
#endif
        //real_t* data = recv_buf.data().dptr<real_t>();
        // false means not to delete data when SArray is deleted
        auto vals = new ps::SArray<real_t>(data, size, false);
        // issue pull
        CHECK_NOTNULL(ps_worker_)->ZPull(
	    pskv.keys, vals, &pskv.lens, kDefaultPushPull, [pskv, vals, cb](){ delete vals; cb(); });
      };

      CHECK_NOTNULL(Engine::Get())->PushAsync(
          pull_from_servers,
          pinned_ctx_,
          {},
          {recv_buf.var()},
          FnProperty::kNormal,
          priority,
          PROFILER_MESSAGE("KVStoreDistDefaultPull"));

      comm_->Broadcast(key, recv_buf, grouped_vals[i], priority);
    }
  }

  void PullRowSparseImpl(const std::vector<int>& keys,
                         const std::vector<std::pair<NDArray*, NDArray>>& val_rowids,
                         int priority = 0) override {
    std::vector<int> uniq_keys;
    std::vector<std::vector<std::pair<NDArray*, NDArray>>> grouped_val_rowids;
    GroupKVPairsPullRsp(keys, val_rowids, &uniq_keys, &grouped_val_rowids);

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      int key = uniq_keys[i];
      // use the same array for merging to guarantee that pull always happens
      // after the previous push on this key
      auto& recv_buf = comm_buf_[key];
      auto& grouped_val_rowid = grouped_val_rowids[i];
      const auto storage_type = grouped_val_rowid[0].first->storage_type();
      CHECK_EQ(storage_type, kRowSparseStorage)
               << "expected kRowSparseStorage, but got " << storage_type;
      if (recv_buf.is_none()) {
        // it may happen for the first time a no-rank-0 worker pull the weight.
        recv_buf = NDArray(storage_type, grouped_val_rowid[0].first->shape(),
                           pinned_ctx_, true, grouped_val_rowid[0].first->dtype());
      }
      auto &target_val_rowids = grouped_val_rowids[i];
      const size_t num_vals = target_val_rowids.size();
      size_t num_rows = 0;
      // TODO(haibin) refactor this for loop
      for (size_t i = 0; i < num_vals; i++) {
        auto &row_id = target_val_rowids[i].second;
        NDArray indices(row_id.shape(), pinned_ctx_, false, mshadow::kInt64);
        CopyFromTo(row_id, &indices, 0);
        Unique(&indices, priority);
        target_val_rowids[i].second = indices;
        num_rows += indices.shape().Size();
      }
      if (num_vals > 1) {
        // TODO(haibin) aggregate over all unique indices
        LOG(FATAL) << "RowSparsePull with multiple values is not implemented yet";
      } else {
        auto& indices = target_val_rowids[0].second;
        PullRowSparse_(key, recv_buf, indices, priority);
        comm_->BroadcastRowSparse(key, recv_buf, grouped_val_rowid, num_vals == 1, priority);
      }
    }
  }

  void Push_(const std::vector<int>& keys,
             const std::vector<NDArray>& values,
             int priority,
             bool do_merge) {
    // first aggregate the values over keys
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray> > grouped_vals;
    GroupKVPairsPush(keys, values, &uniq_keys, &grouped_vals);

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      // merge over devcies
      int key = uniq_keys[i];
      auto& vals = grouped_vals[i];
      NDArray merged = do_merge ? comm_->Reduce(key, vals, priority) : vals[0];
      /*if(keys[0] == 0)
      {
	  printf("worker side. value = %s\n", ((NDArray*)&values[0])->Summarize().c_str());
	  printf("worker side. vals[0] = %s\n", vals[0].Summarize().c_str());
	  }*/

      auto& send_buf = comm_buf_[key];
      const auto storage_type = merged.storage_type();
      if (merged.ctx().dev_mask() == cpu::kDevMask) {
	  //don't use this shortcut.
	  if (ps::Postoffice::Get()->van()->HasFeature(ps::Van::WorkerSidePushPullZeroCopy))
	  {
	      CHECK(send_buf.is_none() == false);
	  }
	  else if(send_buf.is_none())
	  {
	      send_buf = NDArray(merged.shape(), pinned_ctx_, true, merged.dtype());
	  }
	  //CopyFromTo(merged, &send_buf);
	  // Start of a push doesn't guarantee that the previous pushes are completed.
	  // This shouldn't affect training of networks though because training involves
	  // a sequence of push, pull, then push. This imposes ordering that the
	  // second push happens after the first pull, and the pull happens after first push.
	  //send_buf = merged;  // avoid memory copy, PHUB: disable this shortcut.
      } else {
        if (send_buf.is_none()) {
          if (storage_type == kDefaultStorage) {
            send_buf = NDArray(merged.shape(), pinned_ctx_, true, merged.dtype());
          } else {
            send_buf = NDArray(storage_type, merged.shape(), pinned_ctx_, true, merged.dtype());
          }
        }
        //CopyFromTo(merged, &send_buf);
      }
      CopyFromTo(merged, &send_buf);
      //if(key == 0)
      //{
//	  printf("worker side. vals[0] = %s, vals_cnt = %d\n", vals[0].Summarize().c_str(), vals.size());
//	  printf("worker side. send_buf = %s\n", send_buf.Summarize().c_str());
      //    }
      //now, each key has only one corresponding merge buffer.
      // push to servers
      size_t size = send_buf.shape().Size();
      real_t* data = static_cast<real_t*>(send_buf.data().dptr_);
      //printf("worker ZPushing determining storage type %d\n", kDefaultStorage);
	      
      if (storage_type == kDefaultStorage) 
      {
	  //printf("worker ZPushing Key %d\n", key);

	  auto push_to_servers =
	      [this, key, data, size](RunContext rctx, Engine::CallbackOnComplete cb) {
	      // convert to ps keys
	      //size_t size = send_buf.shape().Size();
	      PSKV& pskv = EncodeKey(key, size);
	      
#if MKL_EXPERIMENTAL == 1
	      mkl_set_tblob_eager_mode(send_buf.data());
#endif
	      //real_t* data = send_buf.data().dptr<real_t>();
	      // do push. false means no delete
	      ps::SArray<real_t> _vals(data, size, false);
	      //pskv contains potentially many keys.
	      //all keys put their stuff in the SArray(vals).
	      //the locations are clearly marked in pskv.lens.
	      CHECK_NOTNULL(ps_worker_)->ZPush(
		  pskv.keys, _vals, pskv.lens, 0, [cb]() { cb(); });
	  };
	  if (ps::Postoffice::Get()->van()->HasFeature(ps::Van::PullRequestElision) == true &&
	      ps::Postoffice::Get()->van()->FullyInitialized() == true)
	  {
	      Engine::Get()->PushAsync(
		  push_to_servers,
		  pinned_ctx_,
		  {},
		  {send_buf.var()},
		  FnProperty::kNormal,
		  priority,
		  PROFILER_MESSAGE("KVStoreDistDefaultPush"));
	      CHECK(PhysicalKeyPullAddress.at(key).size() != 0);
	      //queue another broadcast. this won't happen until push is done, and by which time 
	      //with pull request elision enabled vans send_buf will be populated.
	      comm_->Broadcast(key, send_buf, PhysicalKeyPullAddress.at(key), priority);
	  }
	  else
	  {
	      Engine::Get()->PushAsync(
		  push_to_servers,
		  pinned_ctx_,
		  { send_buf.var() },
		  {},
		  FnProperty::kNormal,
		  priority,
		  PROFILER_MESSAGE("KVStoreDistDefaultPush"));
	      //PUll elision disabled means it can be shared for read.
	  }
      }
      else if (storage_type == kRowSparseStorage) 
      {
	  PushRowSparse(key, send_buf, priority);
      }
      else
      {
	  LOG(FATAL) << "unknown storage type";
      }
    }
  }

  // pull row sparse weight into `recv_buf` based on indices given by `indices`
  void PullRowSparse_(const int key, const NDArray& recv_buf,
                      const NDArray& indices, int priority) {
    using namespace rowsparse;
    auto pull_from_servers = [this, key, recv_buf, indices]
                             (RunContext rctx, Engine::CallbackOnComplete cb) {
      // allocate memory for the buffer
      size_t num_rows = indices.shape().Size();
      recv_buf.CheckAndAlloc({mshadow::Shape1(num_rows)});
#if MKL_EXPERIMENTAL == 1
      mkl_set_tblob_eager_mode(recv_buf.data());
#endif
      real_t* data = recv_buf.data().dptr<real_t>();
      const auto offsets = indices.data().dptr<int64_t>();
      const auto unit_len = recv_buf.shape().ProdShape(1, recv_buf.shape().ndim());
      const int64_t size = num_rows * unit_len;
       // convert to ps keys in row sparse format
      PSKV& pskv = EncodeRowSparseKey(key, size, num_rows, offsets,
                                      unit_len, recv_buf.shape()[0]);
      if (this->log_verbose_) {
        LOG(INFO) << "worker " << get_rank() << " pull lens: " << pskv.lens << " keys: "
                  << pskv.keys << " size: " << size;
      }
      auto vals = new ps::SArray<real_t>(data, size, false);
      // copy indices to recv_buf. this needs to be done before ZPull
      // because after pull is done, the callback function returns and locks are released.
      // at this point, later functions may access the indices variable while copy happens
      mshadow::Copy(recv_buf.aux_data(kIdx).FlatTo1D<cpu, int64_t>(),
                    indices.data().FlatTo1D<cpu, int64_t>());
      CHECK_NOTNULL(ps_worker_)->ZPull(pskv.keys, vals, &pskv.lens, kRowSparsePushPull,
        [vals, cb]() { delete vals; cb(); });
    };
    CHECK_NOTNULL(Engine::Get())->PushAsync(
        pull_from_servers,
        pinned_ctx_,
        {indices.var()},
        {recv_buf.var()},
        FnProperty::kNormal,
        priority,
        PROFILER_MESSAGE("KVStoreDistRowSparsePull"));
  }

  // push row sparse gradient
  void PushRowSparse(int key, const NDArray &send_buf, int priority) {
    using namespace rowsparse;
    auto push_to_servers = [this, key, send_buf]
                           (RunContext rctx, Engine::CallbackOnComplete cb) {
#if MKL_EXPERIMENTAL == 1
      mkl_set_tblob_eager_mode(send_buf.data());
#endif
      real_t* data = send_buf.data().dptr<real_t>();
      const int64_t num_rows = send_buf.aux_shape(kIdx)[0];
      const auto offsets = send_buf.aux_data(kIdx).dptr<int64_t>();
      const auto unit_len = send_buf.shape().ProdShape(1, send_buf.shape().ndim());
      const int64_t size = num_rows * unit_len;

       // convert to ps keys in row sparse format
      PSKV& pskv = EncodeRowSparseKey(key, size, num_rows, offsets,
                                      unit_len, send_buf.shape()[0]);
      if (this->log_verbose_) {
        LOG(INFO) << "worker " << get_rank() << " push lens: " << pskv.lens << " keys: "
                  << pskv.keys << " size: " << size;
      }
      ps::SArray<real_t> vals(data, size, false);
      CHECK_NOTNULL(ps_worker_)->ZPush(pskv.keys, vals, pskv.lens, kRowSparsePushPull, [cb]() {
        cb();
      });
    };
    Engine::Get()->PushAsync(
        push_to_servers,
        pinned_ctx_,
        {send_buf.var()},
        {},
        FnProperty::kNormal,
        priority,
        PROFILER_MESSAGE("KVStoreDistRowSparsePush"));
  }

  /**
   * \brief check if the keys are all unique
   */
  void CheckUnique(const std::vector<int>& keys) {
    auto keys_copy = keys;
    auto last = std::unique(keys_copy.begin(), keys_copy.end());
    CHECK_EQ(static_cast<size_t>(std::distance(keys_copy.begin(), last)),
             static_cast<size_t>(keys.size()));
  }

  /**
   * \brief struct for ps keys and lens
   */
  struct PSKV {
    ps::SArray<ps::Key> keys;  // n keys
    ps::SArray<int> lens;  // the length of the i-th value
    int size;
  };

  /**
   * \brief cache all key partitions
   */
  std::unordered_map<int, PSKV> ps_kv_;

  /**
   * \brief serizelize EncodeRowSparseKey and EncodeKey
   */
  std::mutex mu_;

  /**
   * \brief convert to keys in ps
   */
  inline PSKV& EncodeKey(int key, size_t size) {
    mu_.lock();
    PSKV& pskv = ps_kv_[key];
    mu_.unlock();

    if (!pskv.keys.empty()) {
	CHECK_EQ(static_cast<size_t>(pskv.size), size) << "The value size cannot be changed key = " << key
						       << " currval = "  << pskv.size 
						       << " expected = " << size;
    } else {
      auto krs = ps::Postoffice::Get()->GetServerKeyRanges();
      int num_servers = krs.size();
      CHECK_GT(num_servers, 0);

      // a simple heuristic for load balance
      if (size < bigarray_bound_)
      {
	  if (ps::Postoffice::Get()->van()->HasFeature(ps::Van::SupportsKeyChunking))
	  {
	      pskv.keys.push_back(RetrieveVirtualKeyFromPhysicalKey(key, 0));
	  }
	  else
	  {
	      // send it to a single random picked server
	      int server = (key * 9973) % num_servers;
	      ps::Key ps_key = krs[server].begin() + key;
	      CHECK_LT(ps_key, krs[server].end());
	      pskv.keys.push_back(ps_key);
	  }
	  pskv.lens.push_back(size);
	  pskv.size = size;
      } 
      else
      {
	  if (ps::Postoffice::Get()->van()->HasFeature(ps::Van::SupportsKeyChunking))
	  {
	      pskv.size = 0;
	      int chunks = (int)ceil(1.0 * size / bigarray_bound_);
	      int remaining = size;
	      for (int i = 0; i < chunks; i++)
	      {
		  ps::Key ps_key = RetrieveVirtualKeyFromPhysicalKey(key, i);
		  pskv.keys.push_back(ps_key);
		  int curr = remaining >= bigarray_bound_ ? bigarray_bound_ : remaining;
		  pskv.lens.push_back(curr);
		  remaining -= curr;
		  pskv.size = size;//not sure whether we need, but carry information around.
	      }
	      CHECK(remaining == 0);
	  }
	  else
	  {
	      // parition it to all servers
	      pskv.size = 0;
	      for (int i = 0; i < num_servers; ++i) 
	      {
		  size_t part_size =
		      static_cast<size_t>(round(static_cast<double>(size)/num_servers*(i+1))) -
		      static_cast<size_t>(round(static_cast<double>(size)/num_servers*i));
		  ps::Key ps_key = krs[i].begin() + key;
		  CHECK_LT(ps_key, krs[i].end());
		  pskv.keys.push_back(ps_key);
		  pskv.lens.push_back(part_size);
		  pskv.size += part_size;
	      }
	  }
	  CHECK_EQ(static_cast<size_t>(pskv.size), size);
      }
    }
    return pskv;
  }

  // Note: this encoding method for row sparse keys doesn't allow cross-layer batching
  inline PSKV& EncodeRowSparseKey(const int key, const int64_t size, const int64_t num_rows,
                                  const int64_t *offsets, const size_t unit_len,
                                  const int64_t total_num_rows) {
    using namespace common;
    mu_.lock();
    PSKV& pskv = ps_kv_[key];
    mu_.unlock();
    pskv.keys.clear();
    pskv.lens.clear();
    // TODO(haibin) cache this information
    auto krs = ps::Postoffice::Get()->GetServerKeyRanges();
    int num_servers = krs.size();
    CHECK_GT(num_servers, 0);

    if (total_num_rows * unit_len >= bigarray_bound_) {
      pskv.size = 0;
      int64_t start_row = 0;
      // parition it to all servers
      for (int i = 0; i < num_servers; ++i) {
        ps::Key master_key = krs[i].begin() + key;
        pskv.keys.push_back(master_key);
        pskv.lens.push_back(0);
        if (offsets && size > 0) {
          // calculate partition ranges
          int64_t part_num_rows =
            llround(static_cast<double>(total_num_rows) / num_servers * (i + 1)) -
            llround(static_cast<double>(total_num_rows) / num_servers * i);
          auto end_row = start_row + part_num_rows;
          // search for offsets in [start_row, end_row)
          auto lb = std::lower_bound(offsets, offsets + num_rows, start_row);
          auto ub = std::upper_bound(offsets, offsets + num_rows, end_row - 1);
          for (auto offset = lb; offset < ub; offset++) {
            ps::Key ps_key = krs[i].begin() + key + (*offset - start_row);
            CHECK_LT(ps_key, krs[i].end());
            pskv.keys.push_back(ps_key);
            pskv.lens.push_back(unit_len);
            pskv.size += unit_len;
          }
          start_row = end_row;
        }
      }
      CHECK_EQ(static_cast<size_t>(pskv.size), size);
    } else {
      // send it to a single random picked server
      int server = (key * 9973) % num_servers;
      ps::Key master_key = krs[server].begin() + key;
      pskv.keys.push_back(master_key);
      pskv.lens.push_back(0);
      for (int64_t i = 0; i < num_rows; i++) {
        ps::Key ps_key = krs[server].begin() + key + offsets[i];
        CHECK_LT(ps_key, krs[server].end());
        pskv.keys.push_back(ps_key);
        pskv.lens.push_back(unit_len);
      }
      pskv.size = size;
    }
    return pskv;
  }


  /**
   * \brief for worker to push and pull data
   */
  ps::KVWorker<real_t>* ps_worker_;
  /**
   * \brief the server handle
   */
  KVStoreDistServer* server_;
  /**
   * \brief threshold for partition
   */
  size_t bigarray_bound_;
  /// \brief send & recver buffer
  std::unordered_map<int, NDArray> comm_buf_;
  bool log_verbose_;
};

}  // namespace kvstore
}  // namespace mxnet


#endif  // MXNET_KVSTORE_KVSTORE_DIST_H_
