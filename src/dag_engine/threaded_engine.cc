#include <queue>
#include <memory>
#include <tuple>
#include <utility>
#include <atomic>
#include <thread>
#include <random>

#include <dmlc/logging.h>
#include <mxnet/dag_engine.h>
#include "../common/spin_lock.h"
#include "../common/concurrent_blocking_queue.h"

using namespace std;

namespace mxnet {

#define DEFAULT_NUM_WORKER_THREADS 4

class ThreadedEngine : public DAGEngine {
 public:
  ThreadedEngine(int numthreads = DEFAULT_NUM_WORKER_THREADS): numthreads_(numthreads) {
    for(int i = 0; i < numthreads; ++i) {
      worker_queues_.push_back(new ConcurrentBlockingQueue<OpDescr*>());
      workers_.emplace_back(&ThreadedEngine::WorkerRoutine, this, i);
    }
  }
  ~ThreadedEngine() {
    for(int i = 0; i < numthreads_; ++i) {
      worker_queues_[i]->SignalForKill();
      delete worker_queues_[i];
      workers_[i].join();
    }
  }
  void Push(AsyncOp exec_fun,
            Context exec_ctx,
            const vector<Variable> &use_vars,
            const vector<Variable> &mutate_vars) override {
    shared_ptr<OpDescr> opd( new OpDescr{exec_fun, exec_ctx, use_vars, mutate_vars},
        [this] (OpDescr* o) { this->OnDepsResolved(o); } );
    for( Variable v : use_vars ) { // read
      VarDescr* vard = static_cast<VarDescr*>(v); // safe to cast here
      spin_lock(&vard->lock);
      if (vard->rw < 0) {
        vard->waitings.push(make_pair(opd, DepType::kRead));
      } else {
        ++vard->rw;
      }
      spin_unlock(&vard->lock);
    }
    for( Variable v : mutate_vars ) { // write
      VarDescr* vard = static_cast<VarDescr*>(v); // safe to cast here
      spin_lock(&vard->lock);
      if (vard->rw != 0) {
        vard->waitings.push(make_pair(opd, DepType::kWrite));
      } else {
        vard->rw = -1;
      }
      spin_unlock(&vard->lock);
    }
  }
  void Push(Op exec_fun,
            Context exec_ctx,
            const vector<Variable> &use_vars,
            const vector<Variable> &mutate_vars) override {
    this->Push([exec_fun](RunContext ctx, Callback on_complete) {
        exec_fun(ctx); on_complete();
      }, exec_ctx, use_vars, mutate_vars);
  }
  void PushDelete(Op delete_fun, Context exec_ctx, Variable var) override {
    // TODO
    this->Push([delete_fun, var] (RunContext ctx) {
          delete_fun(ctx);
          delete static_cast<VarDescr*>(var); // TODO use variable pool instead
        }, exec_ctx, {}, {var});
  }
  Variable NewVar() override {
    // in practice return a ptr to a cell
    // that have the info about the variable
    // use ptr directly instead of ID because this avoids an indirect mapping
    // TODO use variable pool instead
    VarDescr* vd = new VarDescr;
    vd->lock = SPINLOCK_INITIALIZER;
    vd->rw = 0;
    return vd;
  }
  void WaitForVar(Variable var) override {
    // TODO
  }
  void WaitForAll() override {
    // TODO
  }
 private:
  enum class DepType {
    kRead = 0,
    kWrite,
    kDelete,
  };
  struct OpDescr {
    AsyncOp op;
    Context exec_ctx;
    vector<Variable> read_vars;
    vector<Variable> write_vars;
  };
  struct VarDescr {
    spinlock lock;
    int rw; // a semaphore-like count
            // if rw > 0, the variable has several readers and the number
            //   means how many operators are currently reading it;
            // if rw < 0, the varaible has one writer (should be -1)
    queue<pair<shared_ptr<OpDescr>, DepType>> waitings;
  };
  void TriggerWaiting(VarDescr* vard) {
    // ATTENTION: this function should be called with vard->lock held.
    CHECK(vard->rw == 0) << "the variable should be free during triggering";
    if(!vard->waitings.empty()) {
      // pop all reads first
      while(vard->waitings.front().second == DepType::kRead) {
        vard->waitings.pop();
        ++vard->rw;
      }
      if (vard->rw == 0) {
        // pop the next write
        vard->waitings.pop();
        vard->rw = -1;
      }
    }
  }
  void OnOpFinished(OpDescr* opd) {
    CHECK(opd) << "completing a nullptr op!";
    for(Variable v : opd->read_vars) {
      VarDescr* vard = static_cast<VarDescr*>(v); // safe to cast here
      spin_lock(&vard->lock);
      CHECK(vard->rw > 0) << "incorrect rw count (reader):" << vard->rw;
      if(--vard->rw == 0) {
        TriggerWaiting(vard);
      }
      spin_unlock(&vard->lock);
    }
    for(Variable v : opd->write_vars) {
      VarDescr* vard = static_cast<VarDescr*>(v); // safe to cast here
      spin_lock(&vard->lock);
      CHECK(vard->rw == -1) << "incorrect rw count (writer):" << vard->rw;
      vard->rw = 0;
      TriggerWaiting(vard);
      spin_unlock(&vard->lock);
    }
    delete opd; // delete the operator
  }
  RunContext GetRunContext(const Context& ctx) {
    // TODO
    return RunContext();
  }
  void OnDepsResolved(OpDescr* opd) {
    static default_random_engine generator;
    static uniform_int_distribution<int> distribution(0, numthreads_ - 1);
    int thrid = distribution(generator);
    //LOG(INFO) << "schedule operator " << opd << " to thread #" << thrid;
    worker_queues_[thrid]->Push(opd);
  }
  void WorkerRoutine(int thrid) {
    OpDescr* opd = nullptr;
    while(! worker_queues_[thrid]->Pop(opd)) {
      //LOG(INFO) << "worker thread #" << thrid << " got operator " << opd;
      opd->op(GetRunContext(opd->exec_ctx), [this, opd] () { this->OnOpFinished(opd); });
      opd = nullptr;
    }
  }
 private:
  const int numthreads_;
  vector<ConcurrentBlockingQueue<OpDescr*>*> worker_queues_;
  vector<thread> workers_;
};

// implements the singleton factory
DAGEngine* DAGEngine::Get() {
  static ThreadedEngine engine;
  return &engine;
}
}  // namespace mxnet
