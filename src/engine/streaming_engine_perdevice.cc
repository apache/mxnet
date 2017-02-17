/*!
 * Copyright (c) 2017 by Contributors
 * \file streaming_engine_perdevice.cc
 * \brief Streaming engine that schedules in a fixed number of streams for GPU devices
 *        and a fixed number of threads for CPU devices. Allows scheduling without synchronizing
 *        streams in between kernels for improved efficiency of smaller kernels.
 * \author Martin Jakobsson (martin.jacobsson-at-gmail.com) - based on threaded_engine_perdevice.cc
 */

#define MXNET_USE_MOODYCAMEL

#include <dmlc/base.h>
#include <dmlc/omp.h>
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <dmlc/concurrency.h>
#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <cuda_runtime.h>
#include <vector>
#include <functional>
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <random>
#include <condition_variable>
#include "./thread_pool.h"
#include "../common/lazy_alloc_array.h"
#include "../common/utils.h"
#include "./engine_impl.h"
#include "./profiler.h"
#include "../common/object_pool.h"
#include "concurrency_ext.h"

// #define MXNET_USE_NVTX

#if MXNET_USE_CUDA
#include <cuda_runtime_api.h>
#ifdef MXNET_USE_NVTX
#include <nvToolsExt.h>
#include <nvToolsExtCuda.h>
#endif  // MXNET_USE_NVTX
#endif  // MXNET_USE_CUDA

#if MXNET_USE_PROFILER
#warning Streaming per device engine does not support MXNET_USE_PROFILER!
#endif

#ifdef MXNET_USE_NVTX
static void NVTX_RANGE_START(std::string s, uint32_t col = 0x88FF00FF) {
#   if MXNET_USE_CUDA
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color =  col;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = s.c_str();
    nvtxRangePushEx(&eventAttrib);
#   endif  // MXNET_USE_CUDA
}

static void NVTX_RANGE_END() {
#   if MXNET_USE_CUDA
    nvtxRangePop();
#   endif  // MXNET_USE_CUDA
}

#define IFNVTX(x) x
#define NVTX_MARK(a) nvtxMarkA(a)
#define NVTX_NAMETHREAD(a, b) nvtxNameOsThreadA(a, b)
#define NVTX_NAMESTREAM(a, b) nvtxNameCuStreamA(a, b)
#else

#define IFNVTX(x)
#define NVTX_RANGE_START(a, b)
#define NVTX_RANGE_END()
#define NVTX_MARK(a)
#define NVTX_NAMETHREAD(a, b)
#define NVTX_NAMESTREAM(a, b)
#endif

namespace mxnet {
namespace engine {
// Imports / shorter names
template<typename T> using shared_ptr  = std::shared_ptr<T>;
template<typename T> using vector      = std::vector<T>;
template<typename T> using unique_lock = std::unique_lock<T>;
template<typename T> using lock_guard  = std::lock_guard<T>;
template<typename T> using LazyAllocArray = common::LazyAllocArray<T>;
using Spinlock = dmlc::Spinlock;
using string = std::string;

static auto constexpr kFIFO = dmlc::ConcurrentQueueType::kFIFO;
static auto constexpr kPriority = dmlc::ConcurrentQueueType::kPriority;

#if ENGINE_DEBUG
Spinlock cout_m_;  ///< For debug output
#endif  // ENGINE_DEBUG

// Forward declarations
class Variable;
struct Operator;
struct Operation;
class StreamingEnginePerDevice;

// Main queue type
#ifdef MXNET_USE_MOODYCAMEL
using EventQueue = dmlc::LocklessConcurrentBlockingQueue<shared_ptr<Operation>>;
#else
using EventQueue = dmlc::ConcurrentBlockingQueueWithLock<shared_ptr<Operation>, Spinlock>;
#endif  // MXNET_USE_MOODYCAMEL

/// Operator in engine.
/// All operations scheduled in the engine references an operator.
struct Operator final : public Opr, public common::ObjectPoolAllocatable<Operator> {
    Engine::AsyncFn fn;              ///< The function to be invoked each time.
    vector<Variable*> const_vars;    ///< The variables this operator will read from.
    vector<Variable*> mutable_vars;  ///< The variables this operator will write to
    FnProperty prop;                 ///< The property of the operator
    string opr_name;            ///< The name of the operator

    // Whether this is an temporary operator that can be deleted right after the operator completed.
    // Used for ops such as copies, assignments, etc.
    bool temporary { false };

    // Cast a Opr pointer to Operator pointer
    inline static Operator* CastFromBase(Opr* ptr) { return ptr->Cast<Operator>(); }

#   if ENGINE_DEBUG

 public:
    int id { 0 };  ///< Unique id for operator
    Operator() {
        id = counter++;
        LOCKOUTPUT
        LOG(INFO) << "New Operator id: " << id << std::endl;
    }
    ~Operator() {
        LOCKOUTPUT
        LOG(INFO) << "Deleted Operator id: " << id << std::endl;
    }
    static int counter;  ///< Unique id assignment counter
#   endif  // ENGINE_DEBUG
};

#if ENGINE_DEBUG
int Operator::counter = 0;
#endif

/// Variable in engine, returned externally as handle
class Variable final : public Var, public common::ObjectPoolAllocatable<Variable> {
 public:
    /// Cast a Var pointer to Variable pointer
    /// \param ptr pointer from base.
    /// \return a casted pointer.
    inline static Variable* CastFromBase(Var* ptr) { return ptr->Cast<Variable>(); }

#   if ENGINE_DEBUG
    static int counter;  ///< Unique id assignment counter
    int id { 0 };        ///< Unique id

    Variable() {
        id = counter++;
        opsReadingThisVariable.reserve(5);
        LOCKOUTPUT
        LOG(INFO) << "New variable: " << id << std::endl;
    }

    ~Variable() {
        LOCKOUTPUT
        LOG(INFO) << "Deleted variable: " << id << std::endl;
    }
#   endif  // ENGINE_DEBUG

    /// Mark this variable to be deleted the next time a referencing operation finishes.
    inline void SetToDelete() { to_delete_ = true; }

    /// Return whether this variable should be deleted the next
    /// time a referencing operation finishes.
    /// \return status of deletion
    inline bool MarkedForDeletion() { return to_delete_; }

    /// Add operator as a reader of this variable
    void addReading(const shared_ptr<Operation> &o) { opsReadingThisVariable.push_back(o); }

    /// Remove operator as a reader of this variable
    void removeReading(const shared_ptr<Operation> &o) {
        auto &&it = find(opsReadingThisVariable.begin(), opsReadingThisVariable.end(), o);
        if (it != opsReadingThisVariable.end()) {
            *it = opsReadingThisVariable.back();
            opsReadingThisVariable.pop_back();
        }
    }

    /// Iterate over all current readers of this variable
    void iterateReaders(std::function<void(const shared_ptr<Operation> &)> fn) {
        for (auto &&o : opsReadingThisVariable) fn(o);
    }

    /// Remove all reading and writing ops from variable and replace with a single operator
    void setWriting(const shared_ptr<Operation> &o) {
        // We are now the only op to depend on for this variable from now on
        opsReadingThisVariable.clear();
        lastWriting_ = o;
    }

    /// Return last writing operation of this variable
    const shared_ptr<Operation> &lastWriting() { return lastWriting_; }

    /// Remove a variable as reader of this variable
    int removeWriting(const shared_ptr<Operation> &o) {
        if (lastWriting_ == o) {
            lastWriting_ = nullptr;
            return 1;
        }
        return 0;
    }

 private:
    /// Current operations reading this variable
    std::vector<shared_ptr<Operation>> opsReadingThisVariable;

    /// Last operation writing to this variable
    shared_ptr<Operation> lastWriting_;

    std::atomic<bool> to_delete_{false};  ///< If true, delete after operation completes.

 public:
    Spinlock v_m;  ///< Lock for variable
};

#if ENGINE_DEBUG
int Variable::counter = 0;
#endif

enum OperationType {
    kCPUOp = 0,         ///< Denotes a CPU operation,
    kGPUOp = 1,         ///< Denotes a GPU operation (incl. copies to/from GPU)
    kNumOperationTypes
};

struct StreamRunner {
    using EventQueue = dmlc::LocklessConcurrentBlockingQueue<shared_ptr<Operation>>;

    using PEventQueue = std::unique_ptr<EventQueue>;
    using LThread = LazyAllocArray<std::thread>;

    LThread op_threads;                ///< threads launching ops, one per stream
    LThread event_threads;             ///< threads that waits on GPU events for this device
    vector<PEventQueue> op_queues;     ///< op queues for ops on all streams for this device
    vector<PEventQueue> event_queues;  ///< task queues for event on all streams for this
                                       ///< device (event are pushed after op is launched)
    vector<shared_ptr<Operation>> lastInQueue;  ///< last element on queue
    std::mt19937 generator;            ///< for generating randomness

#   if MXNET_USE_CUDA
    vector<shared_ptr<mshadow::Stream<gpu>>>      streams;  ///< gpu streams for this block
#   endif  // MXNET_USE_CUDA

    /// Lock for event_queues / lastInQueue synchronization
    Spinlock qm_;

    ~StreamRunner() noexcept(false) {
        for (int i = 0; i <    op_queues.size();  i++) op_queues[i]->SignalForKill();
        for (int i = 0; i < event_queues.size();  i++) event_queues[i]->SignalForKill();

        auto joinThread = [](int i, std::thread *t){ t->join(); };
        event_threads.ForEach(joinThread);
        op_threads   .ForEach(joinThread);
    }
};


/// Operation block in scheduler
/// When an external handle to a Operator is pushed to the engine,
/// internally an Operation is put in data structures
struct Operation : public common::ObjectPoolAllocatable<Operation>,
                   public std::enable_shared_from_this<Operation> {
 public:
    // (The following members are set once and will not change during the
    //  lifetime of Operation and thus need not be protected by lock)

    Operator* opr{nullptr};      ///< Pointer to information on performing real operation
    Context ctx;                 ///< The context this operation
    int priority;                ///< priority of the function
    bool profiling{false};       ///< indicate whether to profile this operation
    string name;                 ///< Name of operation

    std::atomic<int> sid { 0 };  ///< Stream identifier of selected stream to run on

    enum OpState {
        Initialized = -1,
        Launched    = 0,
        Completed   = 1
    };

    std::atomic<OpState> state;

    /// (0 = depends on launch, 1 = depends on completion);
    std::array<std::vector<shared_ptr<Operation>>, 2> dependencies;

    /// How many operation dependencies that has been fulfilled (0 = launched, 1 = completed)
    std::array<int, 2> depsFulfilled;

    /// operations that depend on this operation
    std::array<std::vector<shared_ptr<Operation>>, 2> dependents;

 public:
    /// Determines whether operation executes on CPU or GPU
    /// \return the type of the operation
    int opType() { return ctx.dev_mask() == cpu::kDevMask ? kCPUOp : kGPUOp; }

    /// Determines whether operations is a copy operation
    /// \return true if operation is a copy operation to or from GPU
    bool isCopy() {
        return (opr->prop == FnProperty::kCopyFromGPU ||
                opr->prop == FnProperty::kCopyToGPU);
    }

    shared_ptr<Operation> nonFinished() {
        // Step 2 - identify whether any crucial dep isnt finished
        for (auto &&o : dependencies[Launched]) {
            if (o->state >= Completed) continue;  ///< Verified Completed already, no need to wait
            if (o->sid   == sid)       continue;  ///< Scheduled on same stream, no need to wait
            return o;  // Need to wait for completion of o
        }
        return nullptr;
    }

    Operation() {
        id = counter++;

        // Save some time
        dependents[Launched].reserve(5);
        dependents[Completed].reserve(5);
        for (int i = 0; i < kNumOperationTypes; i++) dependencies[i].reserve(5);
    }

    ~Operation() {
#       if ENGINE_DEBUG
        {
            LOCKOUTPUT
            string opString = opType() == kGPUOp ? "GPU" : "CPU";
            LOG(INFO) << "Destructing " << opString << " op id:" << id << " - ";
            LOG(INFO) << name << std::endl;
        }
        isDeleted = true;
#       endif  // ENGINE_DEBUG
    }

    /// Add operation to scheduler
    /// \return if operation should be scheduled immediately
    bool addToScheduler() {
        {
            lock_guard<Spinlock> lock { op_m };

            for (auto &&i : opr->const_vars) {
                lock_guard<Spinlock> lock { i->v_m };

                // Must wait for all ops writing read variables to finish!
                const shared_ptr<Operation> &o = i->lastWriting();
                if (o != 0) dependencies[o->opType() == kGPUOp ? Launched : Completed].push_back(o);

                // Mark that variables are processed by this op
                i->addReading(shared_from_this());
            }

            for (auto &&i : opr->mutable_vars) {
                lock_guard<Spinlock> lock { i->v_m };

                // Must wait for all ops writing written variables to finish!
                const shared_ptr<Operation> &o = i->lastWriting();
                if (o != 0) dependencies[o->opType() == kGPUOp ? Launched : Completed].push_back(o);

                // Writing i, so must wait for all ops that reads variable
                auto fn = [this](const shared_ptr<Operation> &o) {
                    dependencies[o->opType() == kGPUOp ? Launched : Completed].push_back(o);
                };
                i->iterateReaders(fn);

                // Mark that variables are processed by this op
                i->setWriting(shared_from_this());
            }
        }

        /// Now add as dependent to other ops.
        /// As soon as added, depsFulfilled can be updated and op scheduled!
        for (int dt = Operation::Launched; dt <= Operation::Completed; dt++) {
            for (auto &&o : dependencies[dt]) {
                // Lock in id order to avoid deadlock!
                lock_guard<Spinlock> lock { o->op_m }, lock2 { op_m };

                o->dependents[dt].push_back(shared_from_this());
                if (o->state >= dt) depsFulfilled[dt]++;
            }
        }

        {  /// Schedule immediately?
            lock_guard<Spinlock> lock { op_m };
            return dependencies[Launched ].size() == depsFulfilled[Operation::Launched ] &&
                   dependencies[Completed].size() == depsFulfilled[Operation::Completed];
        }
    }

    /// Returns operations to be scheduled as a result of given operation reaching new state
    /// \param Operation that reached new state
    /// \return Dependent operations to be scheduled as a result
    vector<shared_ptr<Operation>> opsToBeScheduled(OpState newState) {
        if (newState == Operation::Completed) {  // Remove operation from variables
            for (auto&& i : opr->const_vars) {
                lock_guard<Spinlock> lock { i->v_m };
                i->removeReading(shared_from_this());
            }

            for (auto&& i : opr->mutable_vars) {
                lock_guard<Spinlock> lock { i->v_m };
                i->removeWriting(shared_from_this());
            }
        }

        // (Can now no longer have new dependents added!)

        lock_guard<Spinlock> lock { op_m };
        state = newState;

        vector<shared_ptr<Operation>> scheduled;
        scheduled.reserve(5);

        for (auto&& o : dependents[newState]) {
            lock_guard<Spinlock> lock { o->op_m };

            o->depsFulfilled[newState]++;

#           if ENGINE_DEBUG
            {
                LOCKOUTPUT
                string opString = o->opType() == kGPUOp ? "GPU" : "CPU";
                LOG(INFO) << id << " Increasing depsFulfilled (";
                LOG(INFO) << (opType() == kGPUOp ? "GPU" : "CPU") << ") of " << opString;
                LOG(INFO) << " op id:" << o->id << " - " << o->name << " to ";
                LOG(INFO) << o->depsFulfilled[newState == Launched ? 0 : 1] << std::endl;
                LOG(INFO) << o->id << " depsFulfilled[Operation::Launched] now ";
                LOG(INFO) << o->depsFulfilled[Operation::Launched] << std::endl;
                LOG(INFO) << o->id << " depsFulfilled[Operation::Completed] now ";
                LOG(INFO) << o->depsFulfilled[Operation::Completed] << std::endl;
            }
#           endif  // ENGINE_DEBUG

            if (o->depsFulfilled[Operation::Launched ] == o->dependencies[Launched ].size() &&
                o->depsFulfilled[Operation::Completed] == o->dependencies[Completed].size())
                scheduled.push_back(o);
        }

        if (newState == Operation::Completed) {
            // Clear dependencies, allowing release of older operations
            dependencies[Launched ].clear();
            dependencies[Completed].clear();
            dependents[Launched ].clear();
            dependents[Completed].clear();
        }

        // Generally schedule by priority
        auto prio = [](const shared_ptr<Operation> &o1,
                       const shared_ptr<Operation> &o2) { return o1->priority > o2->priority; };
        sort(scheduled.begin(), scheduled.end(), prio);
        return scheduled;
    }

    int id;                     ///< Unique int id for operation
    static std::atomic<int> counter;         ///< Unique int id counter

    std::vector<shared_ptr<Operation>> getDeps() {
        std::vector<shared_ptr<Operation>> deps;
        for (int i = 0; i < 2; i++) {
            deps.insert(deps.end(), dependencies[i].begin(), dependencies[i].end());
        }
        return deps;
    }

    Spinlock op_m;  ///< Lock for operation
    bool isDeleted  { false };  ///< Whether operation is deleted - for debug only
};

std::atomic<int> Operation::counter { 0 };

/// StreamingEnginePerDevice uses per device threads and multiple streams for GPUs.
/// The policy of this Engine:
///  - Execute Async operation immediately if pushed from Pusher.
///  - Use a fixed amount of streams to schedule operations in parallel on GPU
///  - Use fixed amount of threads for each device. For GPU devices, threads
///    are used to launch kernels in parallel on any of the the fixed amount of streams
///    according to a simple scheduling algorithm.
///  - Use special threads and streams for copy operations.
///
/// A unique ability of this engine is launching of GPU kernels
/// without synchronizing streams in between, yielding
/// significant speedups for small kernels launched in sequence due
/// to the removed latency between synchronization and the next kernel launch.
class StreamingEnginePerDevice : public Engine {
 public:
    void NotifyShutdown() override { shutdown_phase_.store(true); }

    StreamingEnginePerDevice() noexcept(false) {
        NVTX_NAMETHREAD((uint32_t)(size_t) (pthread_self()), "Streaming Engine Main thread");

        // Hold shared objects to object pool so to not destruct early
        objpool_opr_ref_    = common::ObjectPool<Operator >::_GetSharedRef();
        objpool_op_ref_     = common::ObjectPool<Operation>::_GetSharedRef();
        objpool_var_ref_    = common::ObjectPool<Variable >::_GetSharedRef();

        // Number of threads calling user code to actually run a (GPU) operation
        // (For GPU ops, launching kernels on a device and stream)
        gpu_streams_        = dmlc::GetEnv("MXNET_GPU_NSTREAMS", 32);

        noCores = 1;

        if (gpu_streams_ % noCores != 0) {
            throw std::runtime_error("Number of streams must be a multiple of number of cores!");
        }

        // Create CPU task runner
        cpu_worker_nthreads_      = dmlc::GetEnv("MXNET_CPU_WORKER_NTHREADS", 2);
        int cpu_priority_nthreads = dmlc::GetEnv("MXNET_CPU_PRIORITY_NTHREADS", 4);
        auto cpuWorker = [this] { this->CPUWorker(cpu_priority_worker_.get()); };
        cpu_priority_worker_.reset(new TaskRunner<kPriority>());
        cpu_priority_worker_->pool.reset(new ThreadPool(cpu_priority_nthreads, cpuWorker));

        // (GPU task runners are created lazily later per device)
    }

    ~StreamingEnginePerDevice() noexcept(false) {
        cpu_normal_workers_.Clear();
        cpu_priority_worker_.reset(nullptr);

        {
            lock_guard<Spinlock> lock { finished_m_ };
            kill_.store(true);
        }
        finished_cv_.notify_all();
    }

    // External API
    Variable* NewVariable() override { return Variable::New(); }

    // External API
    Operator* NewOperator(StreamingEnginePerDevice::AsyncFn fn,
                             vector<VarHandle> const& const_vars,
                             vector<VarHandle> const& mutable_vars,
                             FnProperty prop,
                             const char* opr_name) override {
        auto ret = Operator::New();
        ret->opr_name = opr_name == nullptr ? "<unknown>" : opr_name;
        ret->fn = std::move(fn);
        ret->prop = prop;
        ret->const_vars.resize(const_vars.size());
        ret->mutable_vars.resize(mutable_vars.size());

        // Store copies of variable pointers
        std::transform(const_vars.begin(), const_vars.end(),
                       ret->const_vars.begin(),
                       Variable::CastFromBase);
        std::transform(mutable_vars.begin(), mutable_vars.end(),
                       ret->mutable_vars.begin(),
                       Variable::CastFromBase);

        return ret;
    }

    // External API
    void DeleteOperator(OprHandle oph) override {
        Operator* opr = Operator::CastFromBase(oph);

        // All dependencies must be completed
        vector<VarHandle> deps;
        deps.reserve(opr->const_vars.size() + opr->mutable_vars.size());
        deps.insert(deps.end(), opr->const_vars.begin(),   opr->const_vars.end());
        deps.insert(deps.end(), opr->mutable_vars.begin(), opr->mutable_vars.end());

        auto op = [opr](RunContext) { Operator::Delete(opr); };
        this->PushSync(op, Context::CPU(), {}, deps, FnProperty::kAsync, 0,
                       PROFILER_MESSAGE("DeleteOperator"));
    }

    // External API
    void Push(OprHandle oph, Context exec_ctx, int priority, bool profiling) override {
        Operator* opr = Operator::CastFromBase(oph);
        shared_ptr<Operation> op = shared_ptr<Operation>(Operation::New(), Operation::Delete);
        op->name = opr->opr_name;
        op->opr = opr;
        op->ctx = exec_ctx;
        op->priority = priority;
        op->profiling = profiling;
        op->state = Operation::Initialized;
        op->depsFulfilled[Operation::Launched] = 0;
        op->depsFulfilled[Operation::Completed] = 0;

        bool schedule = false;
        {
            lock_guard<Spinlock> lock2{finished_m_};
            schedule = op->addToScheduler();

#           if ENGINE_DEBUG
            {
                std::vector<shared_ptr<Operation>> deps;
                deps = op->getDeps();

                LOCKOUTPUT
                LOG(INFO) << "Added op: " << op->id << " - " << op->name << std::endl;
                LOG(INFO) << "Is waiting for: " << std::endl;
                for (auto &&o : deps) LOG(INFO) << "  " << o->id << " - " << o->name << std::endl;
            }
#           endif  // ENGINE_DEBUG

            pending_++;
        }

        // Schedule if can be run straight away
        if (schedule) this->PushToExecute(op, true);

        NVTX_MARK((string("Push ") + op->name).c_str());
    }

    // External API
    void PushAsync(AsyncFn fn, Context exec_ctx,
                   vector<VarHandle> const& const_vars,
                   vector<VarHandle> const& mutable_vars,
                   FnProperty prop,
                   int priority,
                   const char* opr_name) override {
        // Create temporary operator with given function
        Operator *opr = NewOperator(std::move(fn), const_vars, mutable_vars, prop, opr_name);
        opr->temporary = true;

        Push(opr, exec_ctx, priority, false);
    }

    // External API
    void DeleteVariable(SyncFn delete_fn,
                        Context exec_ctx,
                        VarHandle var) override {
        Variable* threaded_var = Variable::CastFromBase(var);

        auto op = [delete_fn, threaded_var, this](RunContext ctx) {
            threaded_var->SetToDelete();
            delete_fn(ctx);
        };

        this->PushSync(op, exec_ctx, {}, {var}, FnProperty::kAsync, 0,
                       PROFILER_MESSAGE("DeleteVariable"));
    }

    // External API
    void WaitForVar(VarHandle var) override {
        // Flush an op through and wait for it to finish
        std::atomic<bool> done{false};
        auto op = [this, &done](RunContext) {
            {
                lock_guard<Spinlock> lock { finished_m_ };
                done.store(true);
            }
            finished_cv_.notify_all();
        };

        this->PushSync(op, Context::CPU(), {var}, {}, FnProperty::kNormal, 0,
                       PROFILER_MESSAGE("WaitForVar"));
        {
            unique_lock<Spinlock> lock { finished_m_ };
            auto condition = [this, &done]() { return done.load() || kill_.load(); };
            finished_cv_.wait(lock, condition);
        }
    }

    // External API
    void WaitForAll() override {
        unique_lock<Spinlock> lock { finished_m_ };
        auto condition = [this]() { return pending_.load() == 0 || kill_.load(); };
        finished_cv_.wait(lock, condition);
    }

    // External API
    bool SupportsAsynchronousKernelExecution() override { return true; }

 protected:
    /// Call this function to actually execute an op
    /// \param run_ctx runtime context used to execute the function.
    /// \param op the op to be executed
    void ExecuteOperation(RunContext run_ctx, shared_ptr<Operation> op) {
        CallbackOnComplete callback;
        auto &&opp = new shared_ptr<Operation>(op);
        bool isAsync = run_ctx.get_stream<gpu>() == nullptr;
        if      (isAsync)                callback = this->CreateCallback(OnCompletedStatic,   opp);
        else if (op->opType() == kGPUOp) callback = this->CreateCallback(OnGPULaunchedStatic, opp);
        else                             callback = this->CreateCallback(OnCompletedStatic,   opp);

        if (shutdown_phase_) {
            callback();
        } else {
            try {
                #ifdef MXNET_USE_NVTX
                string opstr = (op->opType() == kCPUOp ? string("CPU") : string("GPU"));
                string rname = opstr + string(" op: ") + op->name;
                NVTX_RANGE_START(rname, op->opType() == kCPUOp ? 0x88FF00FF : 0xFF8800FF);
                #endif
                op->opr->fn(run_ctx, callback);
            }
            catch(dmlc::Error &e) {
                string what = e.what();
                if (what.find("driver shutting down") == string::npos &&
                    !shutdown_phase_) {
                    LOG(FATAL) << e.what() << "\n" <<
                    "A fatal error occurred in an asynchronous engine operation. "
                    "If you do not know what caused this error, "
                    "you can try to set an environment variable MXNET_ENGINE_TYPE "
                    "to NaiveEngine and run with a debugger (i.e. gdb). "
                    "This will force all operations to be synchronous and "
                    "a backtrace will give you the series of calls that lead "
                    "to this error. Remember to remove the environment variable"
                    "MXNET_ENGINE_TYPE after debugging.";
                    exit(0);
                }
            }
        }
    }

 private:
    template<dmlc::ConcurrentQueueType type> struct TaskRunner;

    // make_unique implementation mimicking standard
    template<typename T, typename... Args>
    std::unique_ptr<T> make_unique_cust(Args&&... args) {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }

    StreamRunner *GetStreamRunner(int dev_id) {
        auto runner = new StreamRunner();

#       if MXNET_USE_CUDA
        // Deallocator for streams
        auto streamDeleter = [](mshadow::Stream<gpu> *s) {
#           if ENGINE_DEBUG
            LOCKOUTPUT
            LOG(INFO) << "Deleting stream " << s->stream_ << std::endl;
#           endif  // ENGINE_DEBUG

            MSHADOW_CATCH_ERROR(mshadow::DeleteStream<gpu>(s));
        };

        for (int i = 0; i < gpu_streams_; i++) {
            auto &&s = mshadow::NewStream<gpu>(true, MXNET_USE_CUDNN != 0);
            runner->streams.push_back(shared_ptr<mshadow::Stream<gpu>>(s, streamDeleter));
            string streamName = string("Stream ") + std::to_string(i);
            NVTX_NAMESTREAM(runner->streams[i]->stream_, streamName.c_str());

#           if ENGINE_DEBUG
            {
                LOCKOUTPUT
                LOG(INFO) << "Created stream " << runner->streams[i]->stream_ << std::endl;
            }
#           endif  // ENGINE_DEBUG
        }

        // Set up stream queues and event detectors. Threads are set up lazily.

        runner->op_queues   .resize(noCores);
        for (int i = 0; i < noCores; i++) {
            runner->op_queues[i]    = make_unique_cust<EventQueue>();
        }

        runner->event_queues.resize(gpu_streams_);
        runner->lastInQueue .resize(gpu_streams_);

        // Start event threads
        for (int sid = 0; sid < gpu_streams_; sid++) {
            runner->event_queues[sid] = make_unique_cust<EventQueue>();

            auto &&sr = runner;
            auto eventWorker   = [this, dev_id, sr, sid] () { this->EventWorker(dev_id, sr, sid); };
            auto eventThread   = [&eventWorker]() { return new std::thread(eventWorker); };
            sr->event_threads.Get(sid, eventThread);
        }

#       endif  // MXNET_USE_CUDA

        return runner;
    }

    template<typename T>
    int bestStream(const T &block, const shared_ptr<Operation> &op) {
        lock_guard<Spinlock> lock { op->op_m };

        int best = -1;

        for (auto &&d : op->dependencies[Operation::Launched]) {  // Try tailing on dependencies
            if (d->state >= Operation::Completed) continue;  // No need to schedule on this

            if (block->lastInQueue[d->sid] == d) {  // Tail possible
                best = d->sid;
                break;
            }
        }

        if (best == -1) {  // Schedule on any free stream
#           if ENGINE_DEBUG
            {
                LOCKOUTPUT
                LOG(INFO) << "No immediate dependency found for op: "
                << op->id << " - " << op->name << std::endl;
            }
#           endif  // ENGINE_DEBUG

            for (int sid = 0; sid < block->event_queues.size(); sid++) {
                if (block->lastInQueue[sid] == nullptr ||
                    block->lastInQueue[sid]->state == Operation::Completed
                    /*|| block->lastInQueue[sid]->dependents[0].size() == 0*/) {
                    best = sid;
                }
            }
        }

        if (best == -1) {  // Schedule on stream with running dependency
            for (auto &&d : op->dependencies[Operation::Launched]) {
                if (d->state < Operation::Completed) {
                    best = d->sid;
                    break;
                }
            }
        }

        // Schedule on oldest stream instead?

//        if (best == -1) { // Schedule on smallest stream
//            int bestPenalty = INT_MAX;
//            for (int sid = 0; sid < block->event_queues.size(); sid++) {
//                int penalty = block->event_queues[sid]->size_approx();
//                if (bestPenalty > penalty) {
//                    bestPenalty = penalty;
//                    best = sid;
//                }
//            }
//        }

        if (best == -1) best = randomGPUStream(block);  ///< Schedule on random stream

#       if ENGINE_DEBUG
        {
            LOCKOUTPUT
            LOG(INFO) << "Scheduling op: " << op->id << " - " << op->name;
            LOG(INFO) << " on: " << best << std::endl;
        }
#       endif  // ENGINE_DEBUG

        return best;
    }

    /// Push the opr block to execution queue to be executed.
    /// This function is implemented by the corresponding subclass
    /// for specific policy.
    /// \param op The operation to be executed
    /// \param pusher_thread whether the caller is the thread that calls push
    virtual void PushToExecute(const shared_ptr<Operation> &op, bool pusher_thread) {
#       if ENGINE_DEBUG
        {
            std::vector<shared_ptr<Operation>> deps = op->getDeps();

            LOCKOUTPUT
            LOG(INFO) << "PushToExecute op: " << op->id << " - " << op->name << std::endl;
            LOG(INFO) << "Was waiting for: " << std::endl;
            for (auto &&o : deps) {
                LOG(INFO) << "  " << o->id << " - " << o->name << std::endl;
                if (o->opType() == kGPUOp) assert(o->state >= Operation::Launched);
                if (o->opType() == kCPUOp) assert(o->state >= Operation::Completed);
            }
        }
#       endif  // ENGINE_DEBUG

        const Context& ctx = op->ctx;
        if (op->opr->prop == FnProperty::kAsync &&
            pusher_thread) {  // Execute straight away, bypassing worker pools (if not in one)
            if (op->opType() == kGPUOp) {
#               if MXNET_USE_CUDA
                MSHADOW_CATCH_ERROR(mshadow::SetDevice<gpu>(ctx.dev_id));
#               endif  // MXNEY_USE_CUDA
            }
            RunContext run_ctx; run_ctx.stream = nullptr;
            this->ExecuteOperation(run_ctx, op);

        } else {
            if (op->opType() == kCPUOp) {  // CPU execution
                if (op->opr->prop == FnProperty::kCPUPrioritized) {
                    cpu_priority_worker_->task_queue.Push(op, op->priority);
                } else {
                    int nthread = cpu_worker_nthreads_;
                    auto newCPUWorkerPool = [this, nthread]() {
                        auto runner = new TaskRunner<kFIFO>();
                        auto cpuWorker = [this, runner] () { this->CPUWorker(runner); };
                        runner->pool.reset(new ThreadPool(nthread, cpuWorker));
                        return runner;
                    };

                    auto &&tq = cpu_normal_workers_.Get(ctx.dev_id, newCPUWorkerPool)->task_queue;
                    tq.Push(op, op->priority);
                }
            } else {  // GPU execution.
                int dev_id = ctx.dev_id;
                auto msr  = [this, dev_id]() { return GetStreamRunner(dev_id); };
                auto &&sr = gpu_stream_runners_.Get(dev_id, msr);

                lock_guard<Spinlock> lock(sr->qm_);
                op->sid    = bestStream(sr, op);

                int core = op->sid % noCores;
                sr->op_queues[core]->Push(op);
                sr->lastInQueue[op->sid] = op;

                // Create launch thread if nonexistant
                auto gpuWorker   = [this, dev_id, sr, core] () {
                    this->GPUWorker(dev_id, sr, core);
                };
                auto gpuThread   = [&gpuWorker]() { return new std::thread(gpuWorker); };
                sr->op_threads.Get(core, gpuThread);
            }
        }
    }

    std::atomic<int>  pending_{0};              ///< Number of pending operations.
    std::atomic<bool> kill_{false};             ///< whether we want to kill the waiters
    std::atomic<bool> shutdown_phase_{false};   ///< whether it is during shutdown phase

    // Mutex and condition_variable, used to Notify waits for single or all variables.
    Spinlock finished_m_;
    std::condition_variable_any finished_cv_;

    /// Holding a shared_ptr to the object pool to prevent it from being destructed too early
    /// See also #309 (https://github.com/dmlc/mxnet/issues/309)
    shared_ptr<common::ObjectPool<Operator>>       objpool_opr_ref_;
    shared_ptr<common::ObjectPool<Operation>>      objpool_op_ref_;
    shared_ptr<common::ObjectPool<Variable>>       objpool_var_ref_;

    /// Each CPU has a corresponding TaskRunner
    template<dmlc::ConcurrentQueueType type>
    struct TaskRunner {
        /// task queue on this task
        dmlc::ConcurrentBlockingQueueWithLock<shared_ptr<Operation>, Spinlock, type> task_queue;

        /// thread pool that works on this task
        std::unique_ptr<ThreadPool> pool;

        ~TaskRunner() noexcept(false) { task_queue.SignalForKill(); }
    };

    int cpu_worker_nthreads_;  ///< number of concurrent thread cpu worker uses
    int gpu_streams_;          ///< number of concurrent thread each gpu copy worker uses

    int noCores;

    LazyAllocArray<TaskRunner<kFIFO          >> cpu_normal_workers_;   ///< cpu worker
    std::unique_ptr<TaskRunner<kPriority     >> cpu_priority_worker_;  ///< cpu priority worker
    LazyAllocArray<StreamRunner               > gpu_stream_runners_;   ///< stream runner for GPU

    template<typename T>
    int randomGPUStream(const T &block) {
        std::uniform_int_distribution<int> distribution(0, gpu_streams_ - 1);
        return distribution(block->generator);
    }

    StreamRunner *getStreamRunner(int dev_id) {
        auto msr  = [this, dev_id]() { return GetStreamRunner(dev_id); };
        return gpu_stream_runners_.Get(dev_id, msr);
    }

    /// Detector for GPU events signaling operations have finished
    /// \param dev_id The device id of the worker.
    /// \param block The task block of the worker.
    /// \param stream id to detect events on
    inline void EventWorker(int dev_id,
                            StreamRunner *block,
                            int si) {
#       if MXNET_USE_CUDA
        string threadName = string("Event Worker thread") + std::to_string(si);
        NVTX_NAMETHREAD((uint32_t)(size_t) (pthread_self()), threadName.c_str());

        // Wait for events on queues until exit
        mshadow::SetDevice<gpu>(dev_id);

        vector<shared_ptr<Operation>> ops;
        shared_ptr<Operation> op;
        while (block->event_queues[si]->Pop(&op)) {
            ops.clear();
            ops.push_back(op);

            for (int i = 0; i < 10; i++) {  // Burst
                if (block->event_queues[si]->ApproxSize() > 0) {
                    block->event_queues[si]->Pop(&op);
                    ops.push_back(op);
                } else { break; }
            }

            for (int i = 0; i < ops.size(); i++) {
                shared_ptr<Operation> &op = ops[i];

//                NVTX_RANGE_END();
#               if MXNET_USE_NVTX
                string range_name = string("opsToBeScheduled ") + op->name +
                                    std::to_string(static_cast<int>(static_cast<size_t>op->opr));
#               endif
                NVTX_RANGE_START(rangeName.c_str() , 0x00008800);

                // Launch further ops if possible
                vector<shared_ptr<Operation>> scheduled = op->opsToBeScheduled(Operation::Launched);
                NVTX_RANGE_END();

                NVTX_RANGE_START("PushToExecute", 0x00008800);

//                NVTX_RANGE_START("PushToExecute", 0x00008800);
                if (scheduled.size() > 0) {
                    for (auto &&o : scheduled) this->PushToExecute(o, false);
                }
//                NVTX_RANGE_END();

                NVTX_RANGE_END();
            }

            // Wait for completion
            auto &&stream = mshadow::Stream<gpu>::GetStream(block->streams[si].get());
            MSHADOW_CUDA_CALL(cudaStreamSynchronize(stream));
            for (int i = 0; i < ops.size(); i++) OnComplete(ops[i]);
        }
#       endif  // MXNET_USE_CUDA
    }

    /// GPU worker that performs operations on a certain device.
    /// \param dev_id The device id of the worker.
    /// \param block The task block of the worker.
    inline void GPUWorker(int dev_id,
                          StreamRunner *block,
                          int si) {
#       if MXNET_USE_CUDA
        mshadow::SetDevice<gpu>(dev_id);

        string threadName = string("GPU Worker thread") + std::to_string(si);
        NVTX_NAMETHREAD((uint32_t)(size_t) (pthread_self()), threadName.c_str());

        auto sr = getStreamRunner(dev_id);

        shared_ptr<Operation> op;
        while (block->op_queues[si]->Pop(&op)) {
#           if ENGINE_DEBUG
            {
                LOCKOUTPUT
                LOG(INFO) << "GPUWorker starting op: " << op->id << " - " << op->name << std::endl;
            }
#           endif  // ENGINE_DEBUG

            NVTX_RANGE_START("nonFinished", 0x88FF00FF);
            {
                bool delay = false;
                for (auto &&o : op->dependencies[Operation::Launched]) {
                    lock_guard<Spinlock> lock { o->op_m }, lock2 { op->op_m };

                    if (o->state >= Operation::Completed) continue;  ///< Verified Completed already
                    if (o->sid   == op->sid)              continue;  ///< Scheduled on same stream

                    // If we cannot schedule, add dependency on completion of some
                    // incomplete op not on same stream and reschedule at that point
                    op->dependencies[Operation::Completed].push_back(o);
                    o->dependents[Operation::Completed].push_back(op);

#                   if ENGINE_DEBUG
                    {
                        LOCKOUTPUT
                        LOG(INFO) << "GPUWorker rescheduling op: " << op->id << " - "
                        LOG(INFO) << op->name << std::endl;
                        LOG(INFO) << "after op: " << o->id << " - " << o->name << std::endl;
                        LOG(INFO) << "with opr: " << o->opr->id << std::endl;
                        LOG(INFO) << "with opr-temp: " << o->opr->temporary << std::endl;
                        LOG(INFO) << "with state: " << o->state << std::endl;
                    }
#                   endif  // ENGINE_DEBUG

                    delay = true;
                    break;
                }
                if (delay) continue;
            }
            NVTX_RANGE_END();

#           if ENGINE_DEBUG
            {
                LOCKOUTPUT
                LOG(INFO) << "GPUWorker deps done for op: " << op->id;
                LOG(INFO) << " - " << op->name << std::endl;
            }
#           endif  // ENGINE_DEBUG

            RunContext run_ctx;
            run_ctx.stream = sr->streams[op->sid].get();
            this->ExecuteOperation(run_ctx, op);

#           if ENGINE_DEBUG
            {
                LOCKOUTPUT
                LOG(INFO) << "GPUWorker after exec for op: " << op->id;
                LOG(INFO) << " - " << op->name << std::endl;
            }
#           endif  // ENGINE_DEBUG
        }

#       if ENGINE_DEBUG
        {
            LOCKOUTPUT
            LOG(INFO) << "GPUWorker finished " << std::endl;
        }
#       endif  // ENGINE_DEBUG

#       endif  // MXNET_USE_CUDA
    }

    /// CPU worker that performs operations on CPU.
    /// \param block The task block of the worker.
    template<dmlc::ConcurrentQueueType type>
    inline void CPUWorker(TaskRunner<type> *block) {
        RunContext run_ctx; run_ctx.stream = nullptr;

        NVTX_NAMETHREAD((uint32_t)(size_t) (pthread_self()), "CPU Worker thread");

        // execute tasks until exit
        shared_ptr<Operation> op;
        auto* task_queue = &(block->task_queue);
        while (task_queue->Pop(&op)) {
#           if ENGINE_DEBUG
            {
                LOCKOUTPUT
                LOG(INFO) << "CPUWorker starting op: " << op->id << " - " << op->name << std::endl;
            }
#           endif  // ENGINE_DEBUG
            this->ExecuteOperation(run_ctx, op);
        }

#       if ENGINE_DEBUG
        {
            LOCKOUTPUT
            LOG(INFO) << "CPUWorker finished " << std::endl;
        }
#       endif  // ENGINE_DEBUG
    }

    /// Stub for calling OnGPULaunched
    static void OnGPULaunchedStatic(Engine *engine, void *op_) {
        shared_ptr<Operation> *ob = static_cast<shared_ptr<Operation>*>(op_);
        static_cast<StreamingEnginePerDevice*>(engine)->OnGPULaunched(*ob);
        delete ob;
    }

    /// Called after launching of GPU operation
    /// \param op operation that was launched
    inline void OnGPULaunched(const shared_ptr<Operation> &op) {
        #ifdef MXNET_USE_NVTX
        string opstr = (op->opType() == kCPUOp ? string("CPU") : string("GPU"));
        string rname = opstr + string(" op: ") + op->name + string(" launch callback");

        NVTX_RANGE_END();
//        NVTX_RANGE_START(rname, (op->opType() == kCPUOp ? 0x88FF00FF : 0xFF8800FF) + 0x00008800);

        #endif

#       if ENGINE_DEBUG
        {
            LOCKOUTPUT
            string opType = op->opType() == kGPUOp ? "GPU" : "CPU";
            LOG(INFO) << "Launched " << opType << " op: " << op->id;
            LOG(INFO) << " - " << op->name << std::endl;
        }
#       endif  // ENGINE_DEBUG

#       if MXNET_USE_CUDA
        {  // Operation has launched, so add to event queue to track finish
            auto sr = getStreamRunner(op->ctx.dev_id);
            auto &eq = sr->event_queues;
            int sid = op->sid;

            // Add to queue
            #ifdef MXNET_USE_NVTX
            string addr = std::to_string(static_cast<int>(static_cast<size_t>op->opr));
            string s = string("Enqueued ") + op->name + addr;
            #endif
            NVTX_RANGE_START("threads", 0x00008800);
            eq[sid]->Push(std::move(op));
            NVTX_RANGE_END();
            NVTX_MARK(s.c_str());
        }

//        NVTX_RANGE_END();

#       endif  // MXNET_USE_CUDA
    }

    /// Stub for calling OnGPULaunched
    static void OnCompletedStatic(Engine *engine, void *op_) {
        shared_ptr<Operation> *ob = static_cast<shared_ptr<Operation>*>(op_);
        static_cast<StreamingEnginePerDevice*>(engine)->OnComplete(*ob);
        delete ob;
    }

    /// Called after operation completed (GPU or CPU)
    /// \param op operation that was completed
    inline void OnComplete(const shared_ptr<Operation> &op) {
        if (op->opType() == kCPUOp) NVTX_RANGE_END();

        bool rstarted = false;
        if (op->dependents[Operation::Completed].size() > 0) {
            string opstr = (op->opType() == kCPUOp ? string("CPU") : string("GPU"));
            string rname = opstr + string(" op: ") + op->name + string(" complete callback");
            IFNVTX(uint32_t opcol = (op->opType() == kCPUOp ? 0x88FF00FF : 0xFF8800FF);)
            NVTX_RANGE_START(rname, opcol + 0x0000FF00);
            rstarted = true;
        }

#       if ENGINE_DEBUG
        {
            LOCKOUTPUT
            string opType = op->opType() == kGPUOp ? "GPU" : "CPU";
            LOG(INFO) << "Completed " << opType << " op: " << op->id;
            LOG(INFO) << " - " << op->name << std::endl;
        }
#       endif  // ENGINE_DEBUG

        vector<shared_ptr<Operation>> scheduled = op->opsToBeScheduled(Operation::Completed);
        if (scheduled.size() > 0) {
            for (auto &&o : scheduled) this->PushToExecute(o, false);
        }

        // Delete variables marked for deletion
        for (auto &&i : op->opr->mutable_vars) if (i->MarkedForDeletion()) Variable::Delete(i);

        int npending;
        {
            lock_guard<Spinlock> lock { finished_m_ };
            npending = --pending_;
        }
        DCHECK_GE(npending, 0);  // invariant
        if (npending == 0) finished_cv_.notify_all();

        if (op->opr->temporary) Operator::Delete(op->opr);

        if (rstarted) NVTX_RANGE_END();
    }

    /// Disallow copy construction and assignment.
    DISALLOW_COPY_AND_ASSIGN(StreamingEnginePerDevice);
};

Engine *CreateStreamingEnginePerDevice() { return new StreamingEnginePerDevice(); }

}  // namespace engine
}  // namespace mxnet
