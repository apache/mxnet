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

/*!
 * Copyright (c) 2015 by Contributors
 * \file mxnet_node.h
 * \brief implement mxnet nodes
 */
#ifndef MXNET_KVSTORE_KVSTORE_DIST_SERVER_H_
#define MXNET_KVSTORE_KVSTORE_DIST_SERVER_H_
#include <queue>
#include <string>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <functional>
#include <future>
#include <vector>
#include "ps/ps.h"
#include "mxnet/kvstore.h"
#include "../operator/tensor/elemwise_binary_op-inl.h"
#include "../operator/tensor/init_op.h"

namespace mxnet {
namespace kvstore {

enum class CommandType {
  kController, kStopServer, kSyncMode, kSetGradientCompression
};

enum class DataHandleMode {
  kDefaultPushPull, kRowSparsePushPull, kCompressedPushPull
};

struct DataHandleType {
  DataHandleMode mode;
  int dtype;
};

/*!
 * Uses Cantor pairing function to generate a unique number given two numbers.
 * Ref: https://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function
 * @param dtype
 * @param mode
 * @return
 */
static int GetCommandType(DataHandleMode mode, int d) {
  int m = static_cast<int>(mode);
  return (((m + d) * (m + d + 1)) / 2) + d;
}

static DataHandleType DepairDataHandleType(int z) {
  int w = std::floor((std::sqrt(8 * z + 1) - 1)/2);
  int t = ((w * w) + w) / 2;

  int y = z - t;
  int x = w - y;
  CHECK_GE(x, 0);
  CHECK_GE(y, 0);
  DataHandleType type;
  type.mode = static_cast<DataHandleMode>(x);
  type.dtype = y;
  return type;
}

/**
 * \brief executor runs a function using the thread called \ref Start
 */
class Executor {
 public:
  /**
   * \brief start the executor
   */
  void Start() {
    std::unique_lock<std::mutex> lk(mu_);
    while (true) {
      cond_.wait(lk, [this]{return !queue_.empty();});
      Block blk = std::move(queue_.front());
      queue_.pop();
      lk.unlock();

      if (blk.f) {
        blk.f(); blk.p->set_value();
      } else {
        blk.p->set_value(); break;
      }
      lk.lock();
    }
  }

  /**
   * \brief function
   */
  typedef std::function<void()> Func;

  /**
   * \brief let the thread called \ref Start to exec a function. threadsafe
   */
  void Exec(const Func& func) {
    Block blk(func);
    auto fut = blk.p->get_future();
    {
      std::lock_guard<std::mutex> lk(mu_);
      queue_.push(std::move(blk));
      cond_.notify_one();
    }
    fut.wait();
  }

  /**
   * \brief stop the thread, threadsafe
   */
  void Stop() {
    Exec(Func());
  }

 private:
  struct Block {
  explicit Block(const Func& func) : f(func), p(std::make_shared<std::promise<void>>()) { }
    Func f;
    std::shared_ptr<std::promise<void>> p;
  };
  std::queue<Block> queue_;
  std::mutex mu_;
  std::condition_variable cond_;
};

class KVStoreDistServer {
 public:
  KVStoreDistServer() {
    using namespace std::placeholders;
    ps_server_ = new ps::KVServer<char>(0);
    static_cast<ps::SimpleApp*>(ps_server_)->set_request_handle(
        std::bind(&KVStoreDistServer::CommandHandle, this, _1, _2));
    ps_server_->set_request_handle(
        std::bind(&KVStoreDistServer::DataHandleEx, this, _1, _2, _3));
    sync_mode_ = false;
    gradient_compression_ = std::make_shared<GradientCompression>();
    log_verbose_ = dmlc::GetEnv("MXNET_KVSTORE_DIST_ROW_SPARSE_VERBOSE", false);
  }

  ~KVStoreDistServer() {
    delete ps_server_;
  }

  void set_controller(const KVStore::Controller& controller) {
    CHECK(controller);
    controller_ = controller;
  }

  void set_updater(const KVStore::Updater& updater)  {
    CHECK(updater);
    updater_ = updater;
  }

  /**
   * \brief blocked until received the command \a kSyncMode
   */
  void Run() {
    exec_.Start();
  }

 private:
  struct MergeBuf {
    std::vector<ps::KVMeta> request;
    NDArray array;
    NDArray temp_array;
  };

  void CommandHandle(const ps::SimpleData& recved, ps::SimpleApp* app) {
    CommandType recved_type = static_cast<CommandType>(recved.head);
    if (recved_type == CommandType::kStopServer) {
      exec_.Stop();
    } else if (recved_type == CommandType::kSyncMode) {
      sync_mode_ = true;
    } else if (recved_type == CommandType::kSetGradientCompression) {
      gradient_compression_->DecodeParams(recved.body);
    } else {
      // this uses value 0 for message id from frontend
      // let the main thread to execute ctrl, which is necessary for python
      exec_.Exec([this, recved]() {
          CHECK(controller_);
          controller_(recved.head, recved.body);
        });
    }
    app->Response(recved);
  }

  void DataHandleEx(const ps::KVMeta& req_meta,
                    const ps::KVPairs<char>& req_data,
                    ps::KVServer<char>* server) {
    DataHandleType type = DepairDataHandleType(req_meta.cmd);
    switch (type.mode) {
      case DataHandleMode::kRowSparsePushPull:
        DataHandleRowSparse(type, req_meta, req_data, server);
        break;
      case DataHandleMode::kCompressedPushPull:
        DataHandleCompressed(type, req_meta, req_data, server);
        break;
      case DataHandleMode::kDefaultPushPull:
        DataHandleDefault(type, req_meta, req_data, server);
        break;
    }
  }

  inline void ApplyUpdates(const int key, const int dtype, MergeBuf *merged, NDArray *stored,
                           ps::KVServer<char>* server) {
    if (merged->request.size() == (size_t) ps::NumWorkers()) {
      // let the main thread to execute updater_, which is necessary for python
      if (updater_) {
        exec_.Exec([this, key, merged, stored](){
          CHECK(updater_);
          updater_(key, merged->array, stored);
        });
      } else {
        // if no updater, just copy
        CopyFromTo(merged->array, stored);
      }
      // better to cast once and store than for each pull
      // we don't need to wait on this because stored wont go out of scope
      if (dtype != mshadow::kFloat32) {
        auto& stored_dtype = store_[key].arr_dtype;
        CopyFromTo(*stored, &stored_dtype, 0);
      }
      if (log_verbose_)  {
        LOG(INFO) << "sync response to " << merged->request.size() << " workers";
      }
      for (const auto& req : merged->request) {
        server->Response(req);
      }
      merged->request.clear();
      stored->WaitToRead();
    } else {
      merged->array.WaitToRead();
    }
  }

  void DecodeRowIds(const ps::SArray<ps::Key> &keys, int64_t *indices,
                    const int64_t master_key, const int64_t num_rows) {
    indices[0] = 0;
    for (int64_t i = 1; i <= num_rows; i++) {
      int key = DecodeKey(keys[i]);
      auto row_id = key - master_key;
      indices[i - 1] = row_id;
    }
  }

  void AccumulateRowSparseGrads(const NDArray& recved_realt, MergeBuf* merged) {
    NDArray out(kRowSparseStorage, merged->array.shape(), Context());
    std::vector<Engine::VarHandle> const_vars;
    const_vars.push_back(recved_realt.var());
    const_vars.push_back(merged->array.var());
    // accumulate row_sparse gradients
    // TODO(haibin) override + operator for row_sparse NDArray
    // instead of calling BinaryComputeRspRsp directly
    using namespace mshadow;
    Engine::Get()->PushAsync(
    [recved_realt, merged, out](RunContext ctx, Engine::CallbackOnComplete on_complete) {
      op::ElemwiseBinaryOp::ComputeEx<cpu, op::mshadow_op::plus>(
      {}, {}, {recved_realt, merged->array}, {kWriteTo}, {out});
      on_complete();
    }, recved_realt.ctx(), const_vars, {out.var()},
    FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);
    CopyFromTo(out, &(merged->array), 0);
  }

  void RowSparsePullResponse(int master_key, int dtype, size_t num_rows,
                             const ps::KVMeta& req_meta,
                             const ps::KVPairs<char>& req_data,
                             ps::KVServer<char>* server) {
    if (log_verbose_) LOG(INFO) << "pull: " << master_key;
    ps::KVPairs<char> response;
    if (num_rows == 0) {
      std::vector<int> lens(req_data.keys.size(), 0);
      response.keys = req_data.keys;
      response.lens.CopyFrom(lens.begin(), lens.end());
      server->Response(req_meta, response);
      return;
    }
    const NDArray& stored = (dtype == mshadow::kFloat32) ? store_[master_key].arr_fp32 :
                                                           store_[master_key].arr_dtype;
    CHECK(!stored.is_none()) << "init " << master_key << " first";
    if (dtype != mshadow::kFloat32) stored.WaitToRead();
    auto shape = stored.shape();
    auto unit_len = shape.ProdShape(1, shape.ndim());
    int num_bytes = mshadow::mshadow_sizeof(dtype);
    const char* data = (char *) stored.data().dptr_;
    auto len = unit_len * num_rows * num_bytes;
    // concat values
    response.vals.resize(len);
    #pragma omp parallel for
    for (size_t i = 1; i <= num_rows; i++) {
      int key = DecodeKey(req_data.keys[i]);
      int64_t row_id = key - master_key;
      const auto src = data + row_id * unit_len * num_bytes; 
      auto begin = (i - 1) * unit_len * num_bytes;
      auto end = i * unit_len * num_bytes;
      response.vals.segment(begin, end).CopyFrom(src, unit_len * num_bytes);
    }
    // setup response
    response.keys = req_data.keys;
    std::vector<int> lens(req_data.keys.size(), unit_len);
    lens[0] = 0;
    response.lens.CopyFrom(lens.begin(), lens.end());
    server->Response(req_meta, response);
  }

  void InitRowSparseStored(DataHandleType type,
                           int master_key,
                           size_t num_rows,
                           const ps::KVMeta& req_meta,
                           const ps::KVPairs<char>& req_data,
                           ps::KVServer<char>* server) {
    auto& stored = store_[master_key].arr_fp32;
    auto& stored_dtype = store_[master_key].arr_dtype;
    int num_bytes = mshadow::mshadow_sizeof(type.dtype);
    auto unit_len = req_data.lens[1] / num_bytes;
    CHECK_GT(unit_len, 0);
    size_t ds[] = {num_rows, (size_t) unit_len};
    TShape dshape(ds, ds + 2);
    CHECK_EQ(req_data.vals.size(), num_rows * unit_len * num_bytes);

    TBlob recv_blob;
    MSHADOW_REAL_TYPE_SWITCH(type.dtype, DType, {
      recv_blob = TBlob((DType*)req_data.vals.data(), // NOLINT(*)
      dshape, cpu::kDevMask);
    })
    NDArray recved = NDArray(recv_blob, 0);
    stored = NDArray(kRowSparseStorage, dshape, Context(), false,
                     mshadow::DataType<real_t>::kFlag);
    if (type.dtype != mshadow::kFloat32) {
      stored_dtype = NDArray(kRowSparseStorage, dshape, Context(), false,
                             type.dtype);
    }
    Engine::Get()->PushAsync(
    [recved, stored](RunContext ctx, Engine::CallbackOnComplete on_complete) {
      NDArray rsp = stored;
      stored.CheckAndAlloc({mshadow::Shape1(recved.shape()[0])});
      mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
      using namespace mxnet::op;
      nnvm::dim_t nnr = rsp.shape()[0];
      MSHADOW_IDX_TYPE_SWITCH(rsp.aux_type(rowsparse::kIdx), IType, {
        IType* idx = rsp.aux_data(rowsparse::kIdx).dptr<IType>();
        mxnet_op::Kernel<PopulateFullIdxRspKernel, cpu>::Launch(s, nnr, idx);
      });
      if (recved.data().type_flag_ != mshadow::kFloat32) {
        MSHADOW_TYPE_SWITCH(recved.data().type_flag_, SrcDType, {
          rsp.data().FlatTo1D<cpu, float>() =
          mshadow::expr::tcast<float>(recved.data().FlatTo1D<cpu, SrcDType>());
        });
      } else {
        mshadow::Copy(rsp.data().FlatTo1D<cpu, float>(),
                      recved.data().FlatTo1D<cpu, float>(), s);
      }
      on_complete();
    }, recved.ctx(), {recved.var()}, {stored.var()},
    FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);
    if (type.dtype != mshadow::kFloat32) {
      CopyFromTo(stored, stored_dtype);
    }
    stored.WaitToRead();
    server->Response(req_meta);
  }

  void DataHandleRowSparse(DataHandleType type, const ps::KVMeta& req_meta,
                           const ps::KVPairs<char>& req_data,
                           ps::KVServer<char>* server) {
    int master_key = DecodeKey(req_data.keys[0]);
    auto num_rows = req_data.keys.size() - 1;
    auto& stored = store_[master_key].arr_fp32;
    auto& stored_dtype = store_[master_key].arr_dtype;
    if (req_meta.push) {
      CHECK_GT(req_data.lens.size(), 0) << "req_data.lens cannot be empty";
      CHECK_EQ(req_data.lens[0], 0);
      if (stored.is_none()) {
        if (log_verbose_) LOG(INFO) << "initial push: " << master_key;
        // initialization
        CHECK_GT(num_rows, 0) << "init with empty data is not supported";
        InitRowSparseStored(type, master_key, num_rows, req_meta, req_data, server);
        return;
      }
      // synced push
      if (sync_mode_) {
        if (log_verbose_) LOG(INFO) << "sync push: " << master_key << " " << req_data.keys;
        auto& merged = merge_buf_[master_key];
        if (merged.array.is_none()) {
          merged.array = NDArray(kRowSparseStorage, stored.shape(), Context());
          merged.temp_array = NDArray(kRowSparseStorage, stored.shape(), Context());
        }
        if (num_rows == 0) {
          // reset to zeros
          if (merged.request.size() == 0) {
            merged.array = NDArray(kRowSparseStorage, stored.shape(), Context());
          } else {
            // nothing to aggregate
          }
          merged.request.push_back(req_meta);
          ApplyUpdates(master_key, type.dtype, &merged,  &stored, server);
          return;
        }
        int num_bytes= mshadow::mshadow_sizeof(type.dtype);
        auto unit_len = req_data.lens[1] / num_bytes;
        CHECK_GT(unit_len, 0);
        // indices
        std::vector<int64_t> indices(num_rows);
        DecodeRowIds(req_data.keys, indices.data(), master_key, num_rows);

        // data
        TBlob idx_blob(indices.data(), mshadow::Shape1(num_rows), cpu::kDevMask);
        size_t ds[] = {(size_t) num_rows, (size_t) unit_len};
        TShape dshape(ds, ds + 2);
        TBlob recv_blob;
        MSHADOW_REAL_TYPE_SWITCH(type.dtype, DType, {
          recv_blob = TBlob((DType*)req_data.vals.data(), // NOLINT(*)
          dshape, cpu::kDevMask);
        })
        // row_sparse NDArray
        NDArray recved(kRowSparseStorage, stored.shape(), recv_blob, {idx_blob}, 0);

        if (merged.request.size() == 0) {
          CopyFromTo(recved, &merged.array, 0);
        } else {
          if (type.dtype != mshadow::kFloat32) {
            CopyFromTo(recved, merged.temp_array);
            AccumulateRowSparseGrads(merged.temp_array, &merged);
          } else {
            AccumulateRowSparseGrads(recved, &merged);
          }
        }
        merged.request.push_back(req_meta);
        ApplyUpdates(master_key, type.dtype, &merged,  &stored, server);
      } else {
        // async push
        if (log_verbose_) LOG(INFO) << "async push: " << master_key;
        if (num_rows == 0) {
          server->Response(req_meta);
          return;
        }
        auto& merged = merge_buf_[master_key];
        auto unit_len = req_data.lens[1];
        CHECK_GT(unit_len, 0);
        // indices
        std::vector<int64_t> indices(num_rows);
        DecodeRowIds(req_data.keys, indices.data(), master_key, num_rows);
        TBlob idx_blob(indices.data(), mshadow::Shape1(num_rows), cpu::kDevMask);
        size_t ds[] = {(size_t) num_rows, (size_t) unit_len};
        TShape dshape(ds, ds + 2);
        TBlob recv_blob;
        MSHADOW_REAL_TYPE_SWITCH(type.dtype, DType, {
          recv_blob = TBlob((DType*)req_data.vals.data(), // NOLINT(*)
          dshape, cpu::kDevMask);
        })
        NDArray recved(kRowSparseStorage, stored.shape(), recv_blob, {idx_blob}, 0);
        NDArray recved_realt;
        if (type.dtype == mshadow::kFloat32) {
          recved_realt = recved;
        } else {
          if (merged.temp_array.is_none()) {
            merged.temp_array = NDArray(kRowSparseStorage, stored.shape(), Context());
          }
          CopyFromTo(recved, merged.temp_array);
          recved_realt = merged.temp_array;
        }
        exec_.Exec([this, master_key, &recved_realt, &stored](){
            CHECK(updater_);
            updater_(master_key, recved_realt, &stored);
          });
        server->Response(req_meta);
        stored.WaitToRead();
      }
    } else {
      RowSparsePullResponse(master_key, type.dtype, num_rows, req_meta, req_data, server);
    }
  }

  void DefaultStorageResponse(int key,
                              int dtype,
                              const ps::KVMeta& req_meta,
                              const ps::KVPairs<char> &req_data,
                              ps::KVServer<char>* server) {
    ps::KVPairs<char> response;
    const NDArray& stored = (dtype == mshadow::kFloat32) ? store_[key].arr_fp32 :
                                                           store_[key].arr_dtype;
    CHECK(!stored.is_none()) << "init " << key << " first";
    int num_bytes = mshadow::mshadow_sizeof(dtype);
    auto len = stored.shape().Size() * num_bytes;
    response.keys = req_data.keys;
    response.lens = {len};
    // TODO(mli) try to remove this CopyFrom
    response.vals.CopyFrom(static_cast<const char*>(stored.data().dptr_), len);
    server->Response(req_meta, response);
  }

  void DataHandleCompressed(DataHandleType type,
                            const ps::KVMeta& req_meta,
                            const ps::KVPairs<char> &req_data,
                            ps::KVServer<char>* server) {
    CHECK_EQ(type.dtype, mshadow::kFloat32)
      << "Gradient compression is currently supported for fp32 only";
    if (req_meta.push) {
      // there used several WaitToRead, this is because \a recved's memory
      // could be deallocated when this function returns. so we need to make sure
      // the operators with \a NDArray are actually finished

      // first for dummy key which represents original size of array, whose len is 0
      CHECK_EQ(req_data.keys.size(), (size_t)2);
      CHECK_EQ(req_data.lens.size(), (size_t)2);
      CHECK_EQ(req_data.vals.size(), (size_t)req_data.lens[1]);

      int original_size = DecodeKey(req_data.keys[0]);
      int key = DecodeKey(req_data.keys[1]);
      auto& stored = store_[key].arr_fp32;

      size_t ds[] = {(size_t)req_data.lens[1] / mshadow::mshadow_sizeof(type.dtype)};
      TShape dshape(ds, ds + 1);
      TBlob recv_blob((real_t*) req_data.vals.data(), // NOLINT(*)
                      dshape, cpu::kDevMask);
      NDArray recved = NDArray(recv_blob, 0);

      NDArray decomp_buf = decomp_buf_[key];
      dshape = TShape{(int64_t) original_size};

      if (decomp_buf.is_none()) {
        decomp_buf = NDArray(dshape, Context());
      }

      if (stored.is_none()) {
        stored = NDArray(dshape, Context());
        gradient_compression_->Dequantize(recved, &stored, 0);
        server->Response(req_meta);
        stored.WaitToRead();
      } else if (sync_mode_) {
        // synced push
        auto& merged = merge_buf_[key];
        if (merged.array.is_none()) {
          merged.array = NDArray(dshape, Context());
        }
        if (merged.request.size() == 0) {
          gradient_compression_->Dequantize(recved, &merged.array, 0);
        } else {
          gradient_compression_->Dequantize(recved, &decomp_buf, 0);
          merged.array += decomp_buf;
        }
        merged.request.push_back(req_meta);
        ApplyUpdates(key, type.dtype, &merged, &stored, server);
      } else {
        // async push
        gradient_compression_->Dequantize(recved, &decomp_buf, 0);
        exec_.Exec([this, key, &decomp_buf, &stored]() {
          CHECK(updater_);
          updater_(key, decomp_buf, &stored);
        });
        server->Response(req_meta);
        stored.WaitToRead();
      }
    } else {       // pull
      CHECK_EQ(req_data.keys.size(), (size_t)1);
      CHECK_EQ(req_data.lens.size(), (size_t)0);
      int key = DecodeKey(req_data.keys[0]);
      DefaultStorageResponse(key, type.dtype, req_meta, req_data, server);
    }
  }

  void DataHandleDefault(DataHandleType type, const ps::KVMeta& req_meta,
                         const ps::KVPairs<char> &req_data,
                         ps::KVServer<char>* server) {
    // do some check
    CHECK_EQ(req_data.keys.size(), (size_t)1);
    if (req_meta.push) {
      CHECK_EQ(req_data.lens.size(), (size_t)1);
      CHECK_EQ(req_data.vals.size(), (size_t)req_data.lens[0]);
    }
    int key = DecodeKey(req_data.keys[0]);
    auto& stored = store_[key].arr_fp32;
    auto& stored_dtype = store_[key].arr_dtype;
    // there used several WaitToRead, this is because \a recved's memory
    // could be deallocated when this function returns. so we need to make sure
    // the operators with \a NDArray are actually finished
    if (req_meta.push) {
      size_t ds[] = {(size_t) req_data.lens[0] / mshadow::mshadow_sizeof(type.dtype)};
      TShape dshape(ds, ds + 1);
      TBlob recv_blob;
      MSHADOW_REAL_TYPE_SWITCH(type.dtype, DType, {
        recv_blob = TBlob((DType*)req_data.vals.data(), // NOLINT(*)
                          dshape, cpu::kDevMask);
      })
      NDArray recved = NDArray(recv_blob, 0);
      if (stored.is_none()) {
        // initialization
        // stored is real_t
        stored = NDArray(dshape, Context(), false, mshadow::DataType<real_t>::kFlag);
        if (type.dtype != mshadow::kFloat32) {
          stored_dtype = NDArray(dshape, Context(), false, type.dtype);
          // no need to wait on stored_dtype because stored will be in scope
        }
        CopyFromTo(recved, &stored, 0);
        if (type.dtype != mshadow::kFloat32) {
          CopyFromTo(stored, &stored_dtype, 0);
        }
        server->Response(req_meta);
        stored.WaitToRead();
      } else if (sync_mode_) {
        // synced push
        auto& merged = merge_buf_[key];
        if (merged.array.is_none()) {
          merged.array = NDArray(dshape, Context(), false, mshadow::DataType<real_t>::kFlag);
          merged.temp_array = NDArray(dshape, Context(), false, mshadow::DataType<real_t>::kFlag);
        }
        if (merged.request.size() == 0) {
          CopyFromTo(recved, merged.array);
        } else {
          if (type.dtype == mshadow::DataType<real_t>::kFlag) {
            merged.array += recved;
          } else {
            CopyFromTo(recved, merged.temp_array);
            merged.array += merged.temp_array;
          }
        }
        merged.request.push_back(req_meta);
        ApplyUpdates(key, type.dtype, &merged, &stored, server);
      } else {
        // async push
        exec_.Exec([this, key, &recved, &stored](){
            CHECK(updater_);
            updater_(key, recved, &stored);
          });
        server->Response(req_meta);
        if (type.dtype != mshadow::kFloat32) {
          CopyFromTo(stored, &stored_dtype, 0);
        }
        stored.WaitToRead();
      }
    } else {
      DefaultStorageResponse(key, type.dtype, req_meta, req_data, server);
    }
  }

  int DecodeKey(ps::Key key) {
    auto kr = ps::Postoffice::Get()->GetServerKeyRanges()[ps::MyRank()];
    return key - kr.begin();
  }


  /**
   * \brief user defined mode for push
   */
  bool sync_mode_;
  KVStore::Controller controller_;
  KVStore::Updater updater_;

  /**
   * \brief Server always works with float32 (realt) array as stored,
   * but when datatype for a particular key is not float32, then server
   * stores a cast of `arr_fp32` in `arr_dtype` so that pulls can be responded to without delay
   */
  struct StoredArr {
    NDArray arr_fp32;
    NDArray arr_dtype;
  };

  /**
   * \brief store_ contains the value at kvstore for each key
   */
  std::unordered_map<int, StoredArr> store_;

  /**
   * \brief merge_buf_ is a buffer used if sync_mode is true. It represents
   * values from different workers being merged. The store will be updated
   * to this value when values from all workers are pushed into this buffer.
   */
  std::unordered_map<int, MergeBuf> merge_buf_;

  /**
   * \brief decomp_buf_ is a buffer into which compressed values are
   * decompressed before merging to the store. used when compress_!='none'
   */
  std::unordered_map<int, NDArray> decomp_buf_;

  Executor exec_;
  ps::KVServer<char>* ps_server_;

  // whether to LOG verbose information
  bool log_verbose_;

  /**
   * \brief gradient compression object.
   * starts with none, used after SetGradientCompression sets the type
   * currently there is no support for unsetting gradient compression
   */
  std::shared_ptr<kvstore::GradientCompression> gradient_compression_;
};

}  // namespace kvstore
}  // namespace mxnet

#endif  // MXNET_KVSTORE_KVSTORE_DIST_SERVER_H_
