#if MXNET_USE_MPI_DIST_KVSTORE

#include <unordered_map>
#include <queue>
#include <thread>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <iostream>
#include <mpi.h>

#include "mxnet/base.h"
#include "mxnet/ndarray.h"
#include "mxnet/engine.h"
#include "dmlc/logging.h"
#include "mpi_message.pb.h"
#include "mpi_collectives.h"
#include "mpi_wrapper.h"
#include "mpi_util.h"

using namespace mxnet::kvstore;

const std::string INT_PREFIX("INT:");
const std::string STR_PREFIX("STR:");

namespace {

struct CollectiveOpRecord {
  int rank;

  std::string key;

  MPIDataType dtype;

  mxnet::NDArray *val_in;

  mxnet::NDArray *val_out;

  int root_rank;

  mxnet::engine::CallbackOnComplete callback;
};

typedef std::unordered_map<std::string, CollectiveOpRecord> NDArrayTable;

typedef std::unordered_map<std::string, std::vector<MPIRequest> > MessageTable;

struct MPIGlobalState {
  std::atomic_flag initialized_flag = ATOMIC_FLAG_INIT;

  std::condition_variable cv;

  bool initialization_done = false;

  int init_status;

  std::mutex mu;

  NDArrayTable ndarray_table;

  std::queue<MPIRequest> message_queue;

  std::thread background_thread;

  bool shut_down = false;

  std::unique_ptr<MessageTable> message_table;

  int rank = 0;

  int local_rank = 0;

  int size = 1;

  int device = -1;

  mxnet::Context pinned_ctx;

~MPIGlobalState() {
  if (background_thread.joinable()) {
    shut_down = true;
    background_thread.join();
  }
}

};

static MPIGlobalState mpi_global;

//static std::unordered_map<std::string, mxnet::NDArray> mpi_comm_buf;

#define RANK_ZERO 0

#define TAG_NOTIFY 1

bool IncrementNDArrayCount(
  std::unique_ptr<MessageTable>& message_table,
  MPIRequest msg, int mpi_size)
{
  auto name = msg.key_name();
  auto table_iter = message_table->find(name);
  if (table_iter == message_table->end()) {
    message_table->emplace(name, std::vector<MPIRequest>({msg}));
    MXMPI_DEBUG(mpi_global.rank, "Insert new message key [%s] from rank[%d] into message table!\n", name.c_str(), msg.request_rank());
    table_iter = message_table->find(name);
  } else {
    MXMPI_DEBUG(mpi_global.rank, "Insert existing message key [%s] from rank[%d] into message table!\n", name.c_str(), msg.request_rank());
    table_iter->second.push_back(msg);
  }

  int count = table_iter->second.size();
  MXMPI_DEBUG(mpi_global.rank, "Message Key [%s] count [%d]\n", name.c_str(), count);
  return count == mpi_size;
}

int DataTypeToMPIType(int ndarray_dtype, MPIDataType *mpi_dtype)
{
  if (ndarray_dtype == mshadow::kFloat32) {
    *mpi_dtype = MX_MPI_FLOAT32;
  } else if (ndarray_dtype == mshadow::kInt32) {
    *mpi_dtype = MX_MPI_INT32;
  } else if (ndarray_dtype == mshadow::kInt64) {
    *mpi_dtype = MX_MPI_INT64;
  } else {
    return -1;
  }
  return 0;
}

MPIResponse ConstructMPIResponse(std::unique_ptr<MessageTable>& message_table,
                                 std::string name)
{
  bool error = false;
  auto it = message_table->find(name);
  assert(it != message_table->end());

  std::vector<MPIRequest> requests = it->second;
  assert(requests.size() > 0);

  std::ostringstream error_message_stream;

  auto data_type = requests[0].value_type();
  for (unsigned int i = 1; i < requests.size(); i++) {
    auto request_type = requests[i].value_type();
    if (data_type != request_type) {
      error = true;
      error_message_stream
        << "Mismatched data types: One rank had type "
        << MPIDataType_Name(data_type)
        << ", but another rank had type "
        << MPIDataType_Name(request_type)
        << ".";
      break;
    }
  }

  auto message_type = requests[0].request_type();
  for (unsigned int i = 1; i < requests.size(); i++) {
    if (error) {
      break;
    }
    auto request_type = requests[i].request_type();
    if (message_type != request_type) {
      error = true;
      error_message_stream
        << "Mismatched MPI operations: One rank did an "
        << message_type
        << ", but another rank did an "
        << request_type
        << ".";
      break;
    }
  }

  //TODO Check value shape for all reduce and all gather

  MPIResponse response;
  response.set_key_name(name);
  if (error) {
    std::string error_message = error_message_stream.str();
    response.set_response_type(MPIResponse::ERROR);
    response.set_error_message(error_message);
  } else {
    auto response_type = MPIResponse::ERROR;
    if (message_type == MPIRequest::ALLREDUCE) {
      response_type = MPIResponse::ALLREDUCE;
    } else if (message_type == MPIRequest::ALLGATHER) {
      response_type = MPIResponse::ALLGATHER;
    } else {
      response_type = MPIResponse::BROADCAST;
    }
    response.set_response_type(response_type);
  }

  // Clear all queued up requests for this name. They are now taken care of
  // by the constructed MPI response.
  message_table->erase(it);

  return response;
}

void PerformCollectiveOp(NDArrayTable &ndarray_table, MPIResponse response)
{
  mxnet::NDArray *input_array;
  mxnet::NDArray *output_array;
  mxnet::engine::CallbackOnComplete callback;
  int root_rank;
  bool on_gpu;
  {
    std::lock_guard<std::mutex> guard(mpi_global.mu);
    auto name = response.key_name();
    auto iter = ndarray_table.find(name);
    assert(iter != ndarray_table.end());

    assert(response.response_type() == MPIResponse::ALLREDUCE ||
           response.response_type() == MPIResponse::ALLGATHER ||
           response.response_type() == MPIResponse::BROADCAST ||
           response.response_type() == MPIResponse::ERROR);

    CollectiveOpRecord record = iter->second;
    input_array = record.val_in;
    output_array = record.val_out;
    callback = record.callback;
    root_rank = record.root_rank;
    ndarray_table.erase(iter);
  }

  const int dev_in  = input_array->ctx().dev_mask();
  if (response.response_type() == MPIResponse::ALLREDUCE) {
    const int dev_out = output_array->ctx().dev_mask();
    // We only support the case in ndarray and out ndarray
    // share the same device type currently in dist_sync_mpi.
    if (dev_in != dev_out) {
      LOG(FATAL) << "input and output ndarray with mixed device"
                 << "(One CPU the other GPU or vice versa) "
                 << "is not supported in kvstore with type dist_sync_mpi.";
    }
  }

  if (dev_in == mshadow::cpu::kDevMask) {
    on_gpu = false;
  } else {
#if MXNET_USE_CUDA
    if (dev_in == mshadow::gpu::kDevMask) {
      on_gpu = true;
    } else {
      LOG(FATAL) << "Unknown ndarray device type " << dev_in;
    }
#else
    LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
#endif
  }

  auto dtype = input_array->dtype();
  int ret;
  std::string mpi_ops;
  if (response.response_type() == MPIResponse::ALLREDUCE) {
    mpi_ops = "allreduce";
    if (dtype == mshadow::kFloat32) {
      if (!on_gpu) {
        ret = MPI_Wrapper<false, float>::AllReduce(input_array, output_array, MPI_FLOAT);
      } else {
        ret = MPI_Wrapper<true, float>::AllReduce(input_array, output_array, MPI_FLOAT);
      }
    } else if (dtype == mshadow::kInt32) {
      if (!on_gpu) {
        ret = MPI_Wrapper<false, int>::AllReduce(input_array, output_array, MPI_INT);
      } else {
        ret = MPI_Wrapper<true, int>::AllReduce(input_array, output_array, MPI_INT);
      }
    } else {
      LOG(FATAL) << "rank[" << mpi_global.rank << "]:" << "Not supported datatype:"
                 << dtype << " of ndarray with name " << response.key_name();
    }
  } else if (response.response_type() == MPIResponse::BROADCAST) {
    mpi_ops = "broadcast";
    if (dtype == mshadow::kFloat32) {
      if (!on_gpu) {
        ret = MPI_Wrapper<false, float>::Broadcast(input_array, root_rank, MPI_FLOAT);
      } else {
        ret = MPI_Wrapper<true, float>::Broadcast(input_array, root_rank, MPI_FLOAT);
      }
    } else if (dtype == mshadow::kInt32) {
      if (!on_gpu) {
        ret = MPI_Wrapper<false, int>::Broadcast(input_array, root_rank, MPI_INT);
      } else {
        ret = MPI_Wrapper<true, int>::Broadcast(input_array, root_rank, MPI_INT);
      }
    } else {
      LOG(FATAL) << "rank[" << mpi_global.rank << "]:" << "Not supported datatype:"
                 << dtype << " of ndarray with name " << response.key_name();
    }
  } else {
    LOG(FATAL) << "rank[" << mpi_global.rank << "]:" << "Invalid MPI response type:"
               << response.response_type();
  }
  if (ret != 0) {
    LOG(FATAL) << "rank[" << mpi_global.rank << "]:" << "MPI Operation " << mpi_ops
               << " failed at ndarray with name " << response.key_name();
  }
  callback();
}

void BackgroundThreadLoop()
{
  auto init_result = MPI_Init(NULL, NULL);
  if (init_result != MPI_SUCCESS) {
    mpi_global.init_status = -1;
    LOG(FATAL) << "MPI_Initialization Failure!";
    mpi_global.initialization_done = true;
    mpi_global.cv.notify_all();
    return;
  } else {
    mpi_global.init_status = 0;
  }

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  bool is_coordinator = rank == 0;

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MPI_Comm local_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
  int local_rank;
  MPI_Comm_rank(local_comm, &local_rank);

  mpi_global.rank = rank;
  mpi_global.local_rank = local_rank;
  mpi_global.size = size;
  mpi_global.initialization_done = true;

  mpi_global.cv.notify_all();

  if (is_coordinator) {
    mpi_global.message_table =
      std::unique_ptr<MessageTable>(new MessageTable());
  }

  bool should_shut_down = false;
  do {
    // TODO: Eliminate the need for thread sleep by making all activity
    // depend on other activity (e.g. condition or MPI waits).
    std::this_thread::sleep_for (std::chrono::milliseconds(1));

    // Copy the data structures from global state under this lock.
    // However, don't keep the lock for the rest of the loop, so that
    // enqueued stream callbacks can continue.
    std::queue<MPIRequest> message_queue;
    {
      std::lock_guard<std::mutex> guard(mpi_global.mu);
      while (!mpi_global.message_queue.empty()) {
        MPIRequest message = mpi_global.message_queue.front();
        mpi_global.message_queue.pop();
        message_queue.push(message);
      }
    }

    // Collect all tensors that are ready to be reduced. Record them in the
    // tensor count table (rank zero) or send them to rank zero to be
    // recorded (everyone else).
    std::vector<std::string> ready_to_reduce;
    while (!message_queue.empty()) {
      // Pop the first available message message
      MPIRequest message = message_queue.front();
      message_queue.pop();

      if (is_coordinator) {
        bool reduce = IncrementNDArrayCount(mpi_global.message_table,
                                           message, size);
        if (reduce) {
          MXMPI_DEBUG(mpi_global.rank, "Push back ndarray with key [%s] to ready_to_reduce!\n", message.key_name().c_str());
          ready_to_reduce.push_back(message.key_name());
        }
      } else {
        std::string encoded_message;
        message.SerializeToString(&encoded_message);
        MPI_Send(encoded_message.c_str(), encoded_message.length() + 1,
                 MPI_BYTE, RANK_ZERO, TAG_NOTIFY, MPI_COMM_WORLD);
        MXMPI_DEBUG(mpi_global.rank, "MPI_Send message %s!\n", encoded_message.c_str());
      }
    }

    // Rank zero has put all its own tensors in the tensor count table.
    // Now, it should count all the tensors that are coming from other
    // ranks at this tick. It should keep getting tensors until it gets a
    // DONE message from all the other ranks.
    if (is_coordinator) {
      // Count of DONE messages. Keep receiving messages until the number
      // of messages is equal to the number of processes. Initialize to
      // one since the coordinator is effectively done.
      int completed_ranks = 1;
      while (completed_ranks != size) {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, TAG_NOTIFY, MPI_COMM_WORLD, &status);

        // Find number of characters in message (including zero byte).
        int source_rank = status.MPI_SOURCE;
        int msg_length;
        MPI_Get_count(&status, MPI_BYTE, &msg_length);

        // If the length is zero, this is a DONE message.
        if (msg_length == 0) {
          completed_ranks++;
          MPI_Recv(NULL, 0, MPI_BYTE, source_rank, TAG_NOTIFY,
                   MPI_COMM_WORLD, &status);
          continue;
        }

        // Get tensor name from MPI into an std::string.
        char* buffer = new char[msg_length];
        MPI_Recv(buffer, msg_length, MPI_BYTE, source_rank,
                 TAG_NOTIFY, MPI_COMM_WORLD, &status);
        std::string received_data(buffer);
        delete[] buffer;

        MPIRequest received_message;
        received_message.ParseFromString(received_data);
        auto received_name = received_message.key_name();

        bool reduce = IncrementNDArrayCount(
                        mpi_global.message_table, received_message, size);
        if (reduce) {
          MXMPI_DEBUG(mpi_global.rank, "Push back ndarray with key [%s] to ready_to_reduce!\n", received_name.c_str());
          ready_to_reduce.push_back(received_name);
        }
      }

      // At this point, rank zero should have a fully updated tensor
      // count table and should know all the tensors that need to be
      // reduced or gathered, and everyone else should have sent all
      // their information to rank zero. We can now do reductions and
      // gathers; rank zero will choose which ones and in what order,
      // and will notify the other ranks before doing each reduction.
      for (size_t i = 0; i < ready_to_reduce.size(); i++) {
        // Notify all nodes which tensor we'd like to reduce now
        auto name = ready_to_reduce[i];
        MPIResponse response = ConstructMPIResponse(mpi_global.message_table, name);
        std::string encoded_response;
        response.SerializeToString(&encoded_response);
        for (int r = 1; r < size; r++) {
          MPI_Send(encoded_response.c_str(),
                   encoded_response.length() + 1,
                   MPI_BYTE, r, TAG_NOTIFY, MPI_COMM_WORLD);
        }

        // Perform the reduction. All nodes should end up performing
        // the same reduction.
        PerformCollectiveOp(mpi_global.ndarray_table, response);
      }

      // Notify all nodes that we are done with the reductions for this
      // tick.
      MPIResponse done_response;
      done_response.set_response_type(mpi_global.shut_down ?
                                      MPIResponse::SHUTDOWN : MPIResponse::DONE);

      std::string encoded_response;
      done_response.SerializeToString(&encoded_response);

      for (int r = 1; r < size; r++) {
        MPI_Send(encoded_response.c_str(),
                 encoded_response.length() + 1,
                 MPI_BYTE, r, TAG_NOTIFY, MPI_COMM_WORLD);
      }
      if (mpi_global.shut_down) {
        should_shut_down = true;
      }
    } else {
      // Notify the coordinator that this node is done sending messages.
      // A DONE message is encoded as a zero-length message.
      MPI_Send(NULL, 0, MPI_BYTE, RANK_ZERO, TAG_NOTIFY, MPI_COMM_WORLD);

      // Receive names for tensors to reduce from rank zero. Once we
      // receive a empty DONE message, stop waiting for more names.
      while (true) {
        MPI_Status status;
        MPI_Probe(0, TAG_NOTIFY, MPI_COMM_WORLD, &status);

        // Find number of characters in message (including zero byte).
        int msg_length;
        MPI_Get_count(&status, MPI_BYTE, &msg_length);

        // Get tensor name from MPI into an std::string.
        char* buffer = new char[msg_length];
        MPI_Recv(buffer, msg_length, MPI_BYTE, 0,
                 TAG_NOTIFY, MPI_COMM_WORLD, &status);
        std::string received_message(buffer);
        delete[] buffer;

        MPIResponse response;
        response.ParseFromString(received_message);
        if (response.response_type() == MPIResponse::DONE) {
          // No more messages this tick
          break;
        } else if (response.response_type() == MPIResponse::SHUTDOWN) {
        // No more messages this tick, and the background thread
        // should shut down
          should_shut_down = true;
          break;
        } else {
          // Process the current message
          PerformCollectiveOp(mpi_global.ndarray_table, response);
        }
      }
    }
  } while (!should_shut_down);

  MPI_Finalize();
}

int InitializeMPIOnce(bool gpu)
{
  if (mpi_global.initialized_flag.test_and_set())
    return mpi_global.init_status;

  mpi_global.device = -1;
  mpi_global.pinned_ctx = mxnet::Context::CPUPinned(0);

  mpi_global.background_thread = std::thread(BackgroundThreadLoop);
  std::unique_lock<std::mutex> lock(mpi_global.mu);
  mpi_global.cv.wait(lock);
  if (!mpi_global.initialization_done) {
    mpi_global.init_status = -1;
  }

  MXMPI_DEBUG(mpi_global.rank, "MPI Initialization Done!\n");
  return mpi_global.init_status;
}

int IsMPIInitialized()
{
  if (!mpi_global.initialization_done) {
    return 0;
  }
  return 1;
}

void EnqueueCollective(CollectiveOpRecord record,
                       MPIRequest::RequestType rtype,
                       mxnet::Engine::CallbackOnComplete cb)
{
  record.callback = cb;
  MPIRequest message;
  message.set_request_rank(record.rank);
  message.set_key_name(record.key);
//TODO: Need to set value type from NDArray
  message.set_value_type(MX_MPI_FLOAT32);
  message.set_request_type(rtype);
  if (rtype == MPIRequest::BROADCAST) {
    message.set_root_rank(record.root_rank);
  }

  std::lock_guard<std::mutex> guard(mpi_global.mu);
  mpi_global.ndarray_table.emplace(record.key, record);
  mpi_global.message_queue.push(message);
  MXMPI_DEBUG(mpi_global.rank, "Enqueue ndarray key [%s] to message queue!\n", record.key.c_str());
}

};

namespace mxnet {
namespace kvstore {


int MXMPIGetMpiSize(int *ret)
{
  if (IsMPIInitialized()) {
    *ret = mpi_global.size;
    return 0;
  }
  return -1;
}

int MXMPIGetMpiRank(int *ret)
{
  if (IsMPIInitialized()) {
    *ret = mpi_global.rank;
    return 0;
  }
  return -1;
}

int MXMPIInit()
{
  return InitializeMPIOnce(false);
}

int MXMPIGetLocalRank(int *ret)
{
  if (IsMPIInitialized()) {
    *ret = mpi_global.local_rank;
    return 0;
  }
  return -1;
}

int MXMPIAllReduceImpl(const std::vector<std::string> &v_keys,
                       const std::vector<mxnet::NDArray*> &v_invals,
                       const std::vector<mxnet::NDArray*> &v_outvals,
                       int priority)
{
  size_t len = v_keys.size();
  for (size_t i = 0; i < len; ++i) {
    CollectiveOpRecord record;
    record.key = v_keys[i];
    record.rank = mpi_global.rank;
    record.val_in = v_invals[i];
    record.val_out = v_outvals[i];
    MXMPI_DEBUG(mpi_global.rank, "MXMPIAllReduceImpl insert one record key [%s]!\n", record.key.c_str());

    auto all_reduce_async_fn = [record]
    (mxnet::RunContext rctx, mxnet::Engine::CallbackOnComplete cb) {
      EnqueueCollective(record, MPIRequest::ALLREDUCE, cb);
    };
    if (v_invals[i]->var() != v_outvals[i]->var()) {
      CHECK_NOTNULL(mxnet::Engine::Get())->PushAsync(
        all_reduce_async_fn,
        mpi_global.pinned_ctx,
        {record.val_in->var()},
        {record.val_out->var()},
        mxnet::FnProperty::kNormal,
        priority, "KVSTORE PUSHPULL");
    } else {
      CHECK_NOTNULL(mxnet::Engine::Get())->PushAsync(
        all_reduce_async_fn,
        mpi_global.pinned_ctx,
        {},
        {record.val_out->var()},
        mxnet::FnProperty::kNormal,
        priority, "KVSTORE PUSHPULL");
    }
  }
  return 0;
}

int MXMPIAllReduce(const std::vector<int> &keys,
                   const std::vector<mxnet::NDArray*> &in_values,
                   const std::vector<mxnet::NDArray*> &out_values,
                   int priority)
{
  std::vector<std::string> v_keys;
  std::string key_prefix = INT_PREFIX;
  std::string new_key;
  size_t idx = 0;
  for (auto& key: keys) {
    // To simplify original logic for group key value, we rename the original
    // duplicated key and make every key unique now.
    size_t count = countNth(keys, key, idx);
    new_key = key_prefix + std::to_string(key) + std::to_string(count);
    v_keys.push_back(new_key);
    idx++;
  }
  return MXMPIAllReduceImpl(v_keys, in_values, out_values, priority);
}

int MXMPIAllReduceEx(const std::vector<std::string> &keys,
                     const std::vector<mxnet::NDArray*> &in_values,
                     const std::vector<mxnet::NDArray*> &out_values,
                     int priority)
{
  std::vector<std::string> v_keys;
  std::string key_prefix = STR_PREFIX;
  std::string new_key;
  size_t idx = 0;
  for (auto& key: keys) {
    // To simplify original logic for group key value, we rename the original
    // duplicated key and make every key unique now.
    size_t count = countNth(keys, key, idx);
    new_key = key_prefix + key + std::to_string(count);
    v_keys.push_back(new_key);
    idx++;
  }
  return MXMPIAllReduceImpl(v_keys, in_values, out_values, priority);
}

int MXMPIBroadcastImpl(const std::vector<std::string> &v_keys,
                       const std::vector<mxnet::NDArray*> &v_invals,
                       int root_rank,
                       int priority)
{
  size_t len = v_keys.size();
  for (size_t i = 0; i < len; ++i) {
    CollectiveOpRecord record;
    record.key = v_keys[i];
    record.rank = mpi_global.rank;
    record.root_rank = root_rank;
    record.val_in = v_invals[i];
    MXMPI_DEBUG(mpi_global.rank, "MXMPIBroadCastImpl insert one record key [%s]!\n", record.key.c_str());

    auto broadcast_async_fn = [record]
    (mxnet::RunContext rctx, mxnet::Engine::CallbackOnComplete cb) {
      EnqueueCollective(record, MPIRequest::BROADCAST, cb);
    };
    CHECK_NOTNULL(mxnet::Engine::Get())->PushAsync(
      broadcast_async_fn,
      mpi_global.pinned_ctx,
      {},
      {record.val_in->var()},
      mxnet::FnProperty::kNormal,
      priority, "KVSTORE BROADCAST");
  }
  return 0;
}

int MXMPIBroadcast(const std::vector<int> &keys,
                   const std::vector<mxnet::NDArray*> &values,
                   int root_rank,
                   int priority)
{
  std::vector<std::string> v_keys;
  std::string key_prefix = INT_PREFIX;
  std::string new_key;
  size_t idx = 0;
  for (auto& key: keys) {
    // To simplify original logic for group key value, we rename the original
    // duplicated key and make every key unique now.
    size_t count = countNth(keys, key, idx);
    new_key = key_prefix + std::to_string(key) + std::to_string(count);
    v_keys.push_back(new_key);
    idx++;
  }
  return MXMPIBroadcastImpl(v_keys, values, root_rank, priority);
}

int MXMPIBroadcastEx(const std::vector<std::string> &keys,
                     const std::vector<mxnet::NDArray*> &values,
                     int root_rank,
                     int priority)
{
  std::vector<std::string> v_keys;
  std::string key_prefix = STR_PREFIX;
  std::string new_key;
  size_t idx = 0;
  for (auto& key: keys) {
    // To simplify original logic for group key value pairs, we rename the original
    // duplicated key and make every key unique now.
    size_t count = countNth(keys, key, idx);
    new_key = key_prefix + key + std::to_string(count);
    v_keys.push_back(new_key);
    idx++;
  }
  return MXMPIBroadcastImpl(v_keys, values, root_rank, priority);
}

int MXMPIAllGather(const std::vector<int> &keys,
                   const std::vector<mxnet::NDArray*> &values,
                   int priority)
{
  return 0;
}

int MXMPIAllGatherEx(const std::vector<std::string> &keys,
                     const std::vector<mxnet::NDArray*> &values,
                     int priority)
{
  return 0;
}

} // end of namespace kvstore

} // end of namespace mxnet
#endif
