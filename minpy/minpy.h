#include <memory>

extern "C" {

// Entry point. Here we return immediately but save the operation in our own
// sequence. The output array will have correct shape but invalid `Chunk`.
int MXImperativeInvoke(AtomicSymbolCreator creator, int num_inputs,
                       NDArrayHandle* inputs, int* num_outputs,
                       NDArrayHandle** outputs, int num_params,
                       char const** param_keys, char const** param_vals);

// When these two functions are called, flush JIT sequence to make sure data is
// truly ready.
int MXNDArrayWaitToRead(NDArrayHandle handle);
int MXNDArrayWaitToWrite(NDArrayHandle handle);
}

namespace mxnet {
namespace minpy {

class ImperativeRuntime : public Singleton<ImperativeRuntime> {
 public:
  // Python-side utility functions.
  void EnableJIT();
  void DisableJIT();
  void EnableAutograd();
  void DisableAutograd();
  void StrictEvaluate();

  struct ComputingRecord {
    using DelayedFunction = std::variant<
        FCompute, std::pair<std::shared_ptr<Operator>, std::vector<uint32_t>>>;
    DelayedFunction delayed_function;
    nnvm::Op op;
    nnvm::NodeAttrs attrs;
    Context ctx;
    std::vector<engine::VarHandle> read_vars;
    std::vector<engine::VarHandle> write_vars;
    std::vector<Resource> requested;
    std::vector<NDArray> ndinputs;
    std::vector<NDArray> ndoutputs;
  };

  void Invoke(ComputingRecord record);

 private:
  ImperativeRuntime() = default;
  virtual ~ImperativeRuntime() = default;

  void PushAutogradRecord(ComputingRecord record);
  void PushJITRecord(ComputingRecord record);

  void FlushJITSequence();

  void FlushGradSequence();

  // We own the following two components.
  AutogradComponent autograd_component_{};
  JITComponent jit_component_{};
  std::vector<ComputingRecord> autograd_sequence_{};
  std::vector<ComputingRecord> jit_sequence_{};

  bool jit_enabled_ = false;
  bool autograd_enabled_ = false;
};  // class ImperativeRuntime

}  // namespace minpy
}  // namespace mxnet
