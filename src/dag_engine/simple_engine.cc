#include <dmlc/logging.h>
#include <mxnet/dag_engine.h>
namespace mxnet {
class SimpleEngine : public DAGEngine {
 public:

  virtual void Push(AsyncOp exec_fun,
                    Context exec_ctx,
                    const std::vector<Variable> &use_vars,
                    const std::vector<Variable> &mutate_vars) {
    // cannot schedule async using naive way because deps are not captured
    LOG(FATAL) << "cannot schedule async operations";
  }
  virtual void Push(Op exec_fun,
                    Context exec_ctx,
                    const std::vector<Variable> &use_vars,
                    const std::vector<Variable> &mutate_vars) {
    if (exec_ctx.dev_mask == gpu::kDevMask) {
      ctx_.stream = &stream;
      mshadow::SetDevice<gpu>(exec_ctx.dev_id);
      exec_fun(ctx_);
    } else {
      exec_fun(ctx_);
    }
  }
  virtual void PushDelete(Op delete_fun,
                          Context exec_ctx,
                          Variable var) {
    this->Push(delete_fun, exec_ctx, {}, {var});
  }
  virtual Variable NewVar() {
    // in practice return a ptr to a cell
    // that have the info about the variable
    // use ptr directly instead of ID because this avoids an indirect mapping
    return NULL;
  }

 private:
  RunContext ctx_;
  mshadow::Stream<gpu> stream;
};
// implements the singleton factory
DAGEngine* DAGEngine::Get() {
  static SimpleEngine engine;
  return &engine;
}
}  // namespace mxnet
