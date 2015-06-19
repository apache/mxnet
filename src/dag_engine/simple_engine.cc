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
    exec_fun(RunContext());
  }
  virtual void PushDelete(Op delete_fun, Variable var) {
    delete_fun(RunContext());
  }
  virtual Variable NewVar() {
    // in practice return a ptr to a cell
    // that have the info about the variable
    // use ptr directly instead of ID because this avoids an indirect mapping
    return NULL;
  }  
};
// implements the singleton factory
DAGEngine* DAGEngine::Get() {
  static SimpleEngine engine;
  return &engine;
}
}  // namespace mxnet
